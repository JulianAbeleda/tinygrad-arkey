#!/usr/bin/env python3
"""Unified lifecycle/stream tracer for generated and hand prefill WMMA kernels.

This is intentionally structural: it inspects final RDNA3 Inst fields, not
disassembly text, and it does not launch the GPU.
"""
from __future__ import annotations

import argparse, json, os, sys
from collections import Counter
from dataclasses import replace
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad import Device, Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.renderer.isa import amd as amd_isa
from tinygrad.uop.ops import Ops, UOp

from extra.qk.prefill import native_isa_l4_stream_probe as sp
from extra.qk.prefill.wmma import build_gemm_pipe, build_gemm_lds2, default_lds2_lifecycle_template
from extra.qk.prefill_v2_schedule_search import _compile_native_program


def _generated_insts(m_up: int, target: str) -> tuple[list[Any], dict[str, Any]]:
  a = Tensor.empty(64, 64, dtype="half")
  b = Tensor.empty(64, 64, dtype="half")
  lin = (a @ b.transpose()).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  opts = (Opt(OptOps.TC, axis=0, arg=(0, 0, 1)),) + (Opt(OptOps.UPCAST, axis=0, arg=4),) * m_up
  ast = ast.replace(arg=replace(ast.arg, opts_to_apply=opts))
  ren = AMDISARenderer(Target.parse(target))
  prg = to_program(ast, ren)
  lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
  final_uops = sp._final_stream(ren, lin_uop.src)
  return sp._insts_from_uops(final_uops), {"program": str(prg.arg), "tail_off": "generated: UOps -> isel -> regalloc -> waitcnt/scheduler -> Inst"}


def _generated_active_insts(args: argparse.Namespace, shape: tuple[int, int]) -> tuple[list[Any], dict[str, Any]]:
  u0, u1 = shape
  amd_isa.DBUF_D3A_AUDIT_LOG.clear()
  prg = _compile_native_program(args.m, args.n, args.k, u0, u1, args.loc, args.unr)
  lin_uop = [u for u in prg.src if u.op is Ops.LINEAR][0]
  ren = AMDISARenderer(Target.parse(args.target))
  final_uops = sp._final_stream(ren, lin_uop.src)
  try:
    from extra.qk.prefill import a_fragment_alias_probe as afp
    byte_trace = afp._analyze(sp._insts_from_uops(final_uops), prg)
  except Exception as e:
    byte_trace = {"error": f"{type(e).__name__}: {e}"}
  return sp._insts_from_uops(final_uops), {
    "program": str(prg.arg),
    "shape": f"{u0}x{u1}",
    "u0": u0,
    "u1": u1,
    "loc": args.loc,
    "unr": args.unr,
    "tail_off": "generated active prefill: _compile_native_program -> isel -> regalloc -> waitcnt/scheduler -> Inst",
    "ds_byte_window_rows": {"stores": byte_trace.get("ds_store_rows", []), "loads": byte_trace.get("ds_load_rows", []),
                            "status_counts": byte_trace.get("ds_window_status_counts", {}), "error": byte_trace.get("error")},
  }


def _hand_insts(kind: str, args: argparse.Namespace) -> tuple[list[Any], dict[str, Any]]:
  if kind == "hand-pipe":
    insts = build_gemm_pipe(args.m, args.n, args.k, args.tm, args.tn)
    meta = {"builder": f"build_gemm_pipe({args.m},{args.n},{args.k},{args.tm},{args.tn})"}
  elif kind == "hand-lds2":
    insts = build_gemm_lds2(args.m, args.n, args.k, args.waves_m, args.waves_n, args.wm, args.wn,
                            args.bk, args.pad, args.dbuf, args.plra, args.plrab, args.leanaddr, args.dshalf)
    meta = {"builder": (
      f"build_gemm_lds2({args.m},{args.n},{args.k},{args.waves_m},{args.waves_n},"
      f"{args.wm},{args.wn},{args.bk},{args.pad},{args.dbuf})"
    )}
    template = default_lds2_lifecycle_template(args.dbuf)
    meta["hand_lifecycle_oracle"] = {
      "source": "extra/qk/prefill/wmma.py::default_lds2_lifecycle_template",
      "meaning": "hand LDS2 assigns producers by logical DBUF slot before instruction emission",
      "prologue": [(s.op, s.slot) for s in template.prologue],
      "body": [(s.op, s.slot) for s in template.body],
      "tail": [(s.op, s.slot) for s in template.tail],
      "producer_rule": "compute(slot N) consumes stores from the most recent completed coop_store(slot N) after a barrier; body stores the opposite slot for a future compute",
    }
  else:
    raise ValueError(kind)
  meta["tail_off"] = "hand: Python builder -> fixed Inst list"
  return insts, meta


def _op_rows(insts: list[Any]) -> dict[str, list[dict[str, Any]]]:
  fields = {
    "global_load_b128": ("vdst", "addr", "vaddr", "saddr"),
    "global_load_u16": ("vdst", "addr", "vaddr", "saddr"),
    "global_store_b16": ("addr", "data", "saddr"),
    "ds_store_b128": ("addr", "data0", "data1", "data2", "data3"),
    "ds_store_b64": ("addr", "data0", "data1"),
    "ds_store_b32": ("addr", "data0"),
    "ds_store_b16": ("addr", "data0"),
    "ds_load_b128": ("vdst", "addr", "data0", "data1", "data2", "data3"),
    "s_barrier": tuple(),
    "s_waitcnt": tuple(),
    sp.WMMA_NAME: ("vdst", "src0", "src1", "src2"),
  }
  return {name: sp._interesting_rows(insts, name, fields.get(name, tuple())) for name in sp.TRACK_NAMES}


def _waits(insts: list[Any]) -> list[dict[str, Any]]:
  out = []
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    simm16 = sp._waitcnt_simm16(inst)
    if simm16 is not None: out.append({"idx": idx, **sp._decode_waitcnt(simm16), "text": str(inst)})
  return out


def _waits_per_wmma(waits: list[dict[str, Any]], wmma_indices: list[int]) -> list[dict[str, Any]]:
  rows = []
  prev = -1
  for ordinal, idx in enumerate(wmma_indices):
    selected = [w for w in waits if prev < w["idx"] < idx]
    rows.append({
      "wmma_ordinal": ordinal,
      "wmma_idx": idx,
      "wait_count_since_prev_wmma": len(selected),
      "wait_indices": [w["idx"] for w in selected],
      "vmcnt_sequence": [w["vmcnt"] for w in selected],
      "lgkmcnt_sequence": [w["lgkmcnt"] for w in selected],
    })
    prev = idx
  return rows


def _span_regs(span: dict[str, int] | None) -> set[int]:
  if span is None or span.get("kind") != "v": return set()
  return set(range(span["lo"], span["hi"] + 1))


def _def_regs(inst: Any) -> set[int]:
  if isinstance(inst, tuple): return set()
  spans = sp._field_spans(inst)
  name = sp._mn(inst).lower()
  defs: set[int] = set()
  for field in ("vdst", "vdata", "sdst", "sdata"):
    span = spans.get(field)
    if span is not None and span.get("kind") == "v": defs |= _span_regs(span)
  # Stores consume data registers; their "data*" operands are not defs.
  if "store" in name: defs.clear()
  return defs


def _producer_name_before(insts: list[Any], upto: int, reg: int) -> str:
  for idx in range(upto - 1, -1, -1):
    inst = insts[idx]
    if reg in _def_regs(inst): return sp._mn(inst)
  return "unassigned"


def _classify_operand_before(insts: list[Any], upto: int, span: dict[str, int] | None) -> str:
  regs = _span_regs(span)
  if not regs: return "unknown"
  names = {_producer_name_before(insts, upto, r) for r in regs}
  if names == {"ds_load_b128"}: return "ds_load_b128"
  if names == {"global_load_b128"}: return "global_load_b128"
  if names <= {"global_load_b128", "global_load_u16"}: return "global_load"
  if names == {"unassigned"}: return "unassigned"
  return "mixed:" + ",".join(sorted(names))


def _wmma_origins_before(insts: list[Any], wmma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  out = []
  for row in wmma_rows:
    spans = row["spans"]
    out.append({
      "idx": row["idx"],
      "src0": _classify_operand_before(insts, row["idx"], spans.get("src0")),
      "src1": _classify_operand_before(insts, row["idx"], spans.get("src1")),
      "src2": "accumulator",
    })
  return out


def _pipeline_phase(idx: int, wmma_indices: list[int]) -> str:
  if not wmma_indices: return "no_wmma"
  if idx < wmma_indices[0]: return "prologue"
  if idx > wmma_indices[-1]: return "tail"
  return "body"


def _pipeline_store_key(row: dict[str, Any]) -> str:
  try: return sp._addr_key(row)
  except Exception:
    span = row.get("spans", {}).get("addr")
    if span is None: return "addr_unknown"
    return f"{span.get('kind')}:{span.get('lo')}:{span.get('hi')}:{row.get('text', '')}"


def _pipeline_load_key(row: dict[str, Any]) -> str:
  return _pipeline_store_key(row)


def _pipeline_spans_overlap(a: dict[str, int] | None, b: dict[str, int] | None) -> bool:
  if a is None or b is None or a.get("kind") != b.get("kind"): return False
  return not (int(a["hi"]) < int(b["lo"]) or int(b["hi"]) < int(a["lo"]))


def _pipeline_span_key(span: dict[str, int] | None) -> str:
  if span is None: return "?"
  return f"{span.get('kind')}{span.get('lo')}:{span.get('hi')}"


def _pipeline_store_source_key(row: dict[str, Any], global_rows: list[dict[str, Any]]) -> str:
  data = row.get("spans", {}).get("data0")
  prev = [g for g in global_rows if g["idx"] < row["idx"] and _pipeline_spans_overlap(g.get("spans", {}).get("vdst"), data)]
  if not prev: return "source_unknown"
  g = prev[-1]
  return f"saddr={_pipeline_span_key(g.get('spans', {}).get('saddr'))}|vaddr={_pipeline_span_key(g.get('spans', {}).get('addr'))}"


def _dbuf_pipeline_construction_audit(ops: dict[str, list[dict[str, Any]]], wmma_indices: list[int]) -> dict[str, Any]:
  store_rows = ops.get("ds_store_b128", [])
  load_rows = ops.get("ds_load_b128", [])
  global_rows = ops.get("global_load_b128", [])
  phases = ("prologue", "body", "tail")
  stores_by_phase = {p: [] for p in phases}
  loads_by_phase = {p: [] for p in phases}
  for row in store_rows:
    if (phase := _pipeline_phase(row["idx"], wmma_indices)) in stores_by_phase:
      stores_by_phase[phase].append(row)
  for row in load_rows:
    if (phase := _pipeline_phase(row["idx"], wmma_indices)) in loads_by_phase:
      loads_by_phase[phase].append(row)
  store_keys = {p: [_pipeline_store_key(r) for r in rows] for p, rows in stores_by_phase.items()}
  key_sets = {p: set(keys) for p, keys in store_keys.items()}
  source_by_phase_key = {p: {} for p in phases}
  for phase, rows in stores_by_phase.items():
    for row in rows:
      source_by_phase_key[phase].setdefault(_pipeline_store_key(row), set()).add(_pipeline_store_source_key(row, global_rows))
  prologue_body_overlap = sorted(key_sets["prologue"] & key_sets["body"])
  source_mismatch = []
  for k in prologue_body_overlap:
    pro_src, body_src = source_by_phase_key["prologue"].get(k, set()), source_by_phase_key["body"].get(k, set())
    if pro_src and body_src and pro_src.isdisjoint(body_src):
      source_mismatch.append({"window": k, "prologue_sources": sorted(pro_src), "body_sources": sorted(body_src)})
  body_first_store = min((r["idx"] for r in stores_by_phase["body"]), default=None)
  body_loads_before_body_store = []
  load_keys_before_body_store: set[str] = set()
  if body_first_store is not None:
    body_loads_before_body_store = [r["idx"] for r in loads_by_phase["body"] if r["idx"] < body_first_store]
    load_keys_before_body_store = {_pipeline_load_key(r) for r in loads_by_phase["prologue"] + loads_by_phase["body"]
                                   if r["idx"] < body_first_store}
  warmup_required_overlap = sorted(set(prologue_body_overlap) & load_keys_before_body_store)
  steady_state_body_produced_overlap = sorted(set(prologue_body_overlap) - load_keys_before_body_store)
  if prologue_body_overlap:
    verdict = "physical_window_overlap_requires_epoch_reaching_def"
  elif stores_by_phase["body"]:
    verdict = "body_staging_without_physical_overlap"
  else:
    verdict = "no_body_staging"
  return {
    "verdict": verdict,
    "note": "same addr-register LDS window across prologue/body is pipeline/alias evidence, not a redundancy proof",
    "store_counts": {p: len(stores_by_phase[p]) for p in phases},
    "load_counts": {p: len(loads_by_phase[p]) for p in phases},
    "unique_store_windows": {p: len(key_sets[p]) for p in phases},
    "prologue_body_physical_window_overlap_count": len(prologue_body_overlap),
    "prologue_body_physical_window_overlap_sample": prologue_body_overlap[:16],
    "prologue_body_source_mismatch_count": len(source_mismatch),
    "prologue_body_source_mismatch_sample": source_mismatch[:8],
    "body_first_store_idx": body_first_store,
    "body_loads_before_first_body_store_count": len(body_loads_before_body_store),
    "body_loads_before_first_body_store_sample": body_loads_before_body_store[:16],
    "warmup_required_overlap_count": len(warmup_required_overlap),
    "warmup_required_overlap_sample": warmup_required_overlap[:16],
    "steady_state_body_produced_overlap_count": len(steady_state_body_produced_overlap),
    "steady_state_body_produced_overlap_sample": steady_state_body_produced_overlap[:16],
    "pipeline_epoch_candidate": bool(steady_state_body_produced_overlap),
    "construction_invariant": (
      "Do not delete prologue stores from physical-window equality. Build/peel/predicate epochs so only warmup "
      "epochs are emitted in the prologue, or prove MemorySSA-style that a body store reaches every consumer first "
      "with the same runtime epoch and a barrier in between."
    ),
  }


def _barrier_epoch(idx: int, barrier_indices: list[int]) -> int:
  return sum(1 for b in barrier_indices if b < idx)


def _latest_overlapping_load(inst_idx: int, span: dict[str, int] | None, load_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
  rows = [r for r in load_rows if r["idx"] < inst_idx and _pipeline_spans_overlap(r.get("spans", {}).get("vdst"), span)]
  return rows[-1] if rows else None


def _lds_reaching_def_map(ops: dict[str, list[dict[str, Any]]], wmma_indices: list[int], byte_rows: dict[str, Any] | None=None) -> dict[str, Any]:
  barriers = [r["idx"] for r in ops.get("s_barrier", [])]
  stores = sorted(ops.get("ds_store_b128", []), key=lambda r: r["idx"])
  loads = sorted(ops.get("ds_load_b128", []), key=lambda r: r["idx"])
  byte_store = {r["idx"]: r for r in (byte_rows or {}).get("stores", []) if r.get("op") == "ds_store_b128"}
  byte_load = {r["idx"]: r for r in (byte_rows or {}).get("loads", []) if r.get("op") == "ds_load_b128"}
  def row_key(row: dict[str, Any], table: dict[int, dict[str, Any]], fallback) -> str:
    br = table.get(row["idx"])
    if br is not None and br.get("window_status") == "known": return str(br.get("window"))
    return fallback(row)
  latest_store: dict[str, dict[str, Any]] = {}
  load_rows = []
  si = 0
  for ld in loads:
    while si < len(stores) and stores[si]["idx"] < ld["idx"]:
      latest_store[row_key(stores[si], byte_store, _pipeline_store_key)] = stores[si]
      si += 1
    key = row_key(ld, byte_load, _pipeline_load_key)
    st = latest_store.get(key)
    load_rows.append({
      "load_idx": ld["idx"],
      "load_key": key,
      "key_source": "normalized_byte_window" if byte_load.get(ld["idx"], {}).get("window_status") == "known" else "addr_family",
      "load_epoch": _barrier_epoch(ld["idx"], barriers),
      "producer_store_idx": None if st is None else st["idx"],
      "producer_epoch": None if st is None else _barrier_epoch(st["idx"], barriers),
      "barrier_between": False if st is None else any(st["idx"] < b < ld["idx"] for b in barriers),
      "status": "covered" if st is not None else "missing_store",
    })
  load_by_idx = {r["load_idx"]: r for r in load_rows}
  wmma_rows = []
  for ordinal, wm in enumerate(ops.get(sp.WMMA_NAME, [])):
    a_load = _latest_overlapping_load(wm["idx"], wm.get("spans", {}).get("src0"), loads)
    b_load = _latest_overlapping_load(wm["idx"], wm.get("spans", {}).get("src1"), loads)
    a_map = None if a_load is None else load_by_idx.get(a_load["idx"])
    b_map = None if b_load is None else load_by_idx.get(b_load["idx"])
    wmma_rows.append({
      "wmma_ordinal": ordinal,
      "wmma_idx": wm["idx"],
      "a_load_idx": None if a_load is None else a_load["idx"],
      "a_store_idx": None if a_map is None else a_map["producer_store_idx"],
      "a_status": "missing_load" if a_load is None else a_map["status"],
      "b_load_idx": None if b_load is None else b_load["idx"],
      "b_store_idx": None if b_map is None else b_map["producer_store_idx"],
      "b_status": "missing_load" if b_load is None else b_map["status"],
    })
  covered = [r for r in load_rows if r["status"] == "covered"]
  missing = [r for r in load_rows if r["status"] != "covered"]
  no_barrier = [r for r in covered if not r["barrier_between"]]
  return {
    "key_strength": "normalized_ds_byte_window_when_available_else_final_stream_addr_family",
    "limitation": "if store/load address-base registers differ and no normalized byte window is available, final-stream matching can report missing even when hand-builder lifecycle is correct",
    "byte_window_status_counts": None if byte_rows is None else byte_rows.get("status_counts", {}),
    "byte_window_error": None if byte_rows is None else byte_rows.get("error"),
    "load_count": len(load_rows),
    "covered_load_count": len(covered),
    "missing_load_count": len(missing),
    "covered_without_barrier_count": len(no_barrier),
    "load_rows_sample": load_rows[:24],
    "missing_load_rows_sample": missing[:24],
    "wmma_rows_sample": wmma_rows[:24],
    "load_rows": load_rows,
    "wmma_rows": wmma_rows,
    "wmma_missing_a_count": sum(1 for r in wmma_rows if r["a_status"] != "covered"),
    "wmma_missing_b_count": sum(1 for r in wmma_rows if r["b_status"] != "covered"),
  }


def _bytes(insts: list[Any]) -> int:
  total = 0
  for inst in insts:
    if isinstance(inst, tuple): continue
    total += len(inst.to_bytes())
  return total


def _active_cadence_report(ops: dict[str, list[dict[str, Any]]], overlap: dict[str, Any],
                           waits_per_wmma: list[dict[str, Any]]) -> dict[str, Any]:
  regions = overlap.get("global_work_regions_json", [])
  region_rows = []
  for r in regions:
    if not isinstance(r, dict): continue
    region_rows.append({
      "label": r["label"],
      "start": r["start"],
      "end": r["end"],
      "strict": r["strict"],
      "global_load_b128": r["global_load_b128_count"],
      "ds_store_b128": r["ds_store_b128_count"],
      "barrier": r["s_barrier_count"],
      "ds_load_b128": r["ds_load_b128_count"],
      "wmma_at_end": r["end"] if r["label"].startswith("between_wmma_") else None,
      "has_future_slot_work_before_current_compute": bool(r["strict"] and (r["global_load_b128_count"] or r["ds_store_b128_count"])),
    })
  packed_chain_visible = (
    len(ops["global_load_b128"]) > 0 and len(ops["ds_store_b128"]) > 0 and
    len(ops["s_barrier"]) > 0 and len(ops["ds_load_b128"]) > 0 and
    len(ops[sp.WMMA_NAME]) > 0
  )
  scalar_fallback_counts = {
    "ds_store_b16": len(ops["ds_store_b16"]),
    "ds_store_b32": len(ops["ds_store_b32"]),
    "ds_store_b64": len(ops["ds_store_b64"]),
  }
  return {
    "packed_global_to_lds_to_wmma_visible": packed_chain_visible,
    "packed_chain": "global_load_b128 -> ds_store_b128 -> barrier -> ds_load_b128 -> WMMA",
    "future_slot_work_before_current_compute": any(r["has_future_slot_work_before_current_compute"] for r in region_rows),
    "regions": region_rows,
    "scalar_lds_fallback_counts": scalar_fallback_counts,
    "scalar_lds_fallback_total": sum(scalar_fallback_counts.values()),
    "waits_per_wmma": waits_per_wmma,
  }


def _report(label: str, insts: list[Any], meta: dict[str, Any], full_rows: bool) -> dict[str, Any]:
  mns = [sp._mn(x) for x in insts if not isinstance(x, tuple)]
  ops = _op_rows(insts)
  widx = [x["idx"] for x in ops[sp.WMMA_NAME]]
  overlap = sp._collect_regions(ops, widx)
  waits = _waits(insts)
  waits_by_wmma = _waits_per_wmma(waits, widx)
  origins = _wmma_origins_before(insts, ops[sp.WMMA_NAME])
  lds_families = sp._summarize_lds_addresses(ops)
  operand_families = sp._wmma_lds_operand_families(insts, ops[sp.WMMA_NAME])
  dbuf = sp._dbuf_gate_summary(ops, overlap, lds_families, operand_families, origins)
  dbuf_compile_audit = sp._dbuf_d3a_compile_audit_summary(list(amd_isa.DBUF_D3A_AUDIT_LOG))
  origin_counts = Counter((x["src0"], x["src1"]) for x in origins)
  report = {
    "label": label,
    **meta,
    "shared_floor": "Inst list -> assemble_linear -> ELF -> AMDProgram/HSA launch -> GPU",
    "instruction_total": len(mns),
    "byte_count": _bytes(insts),
    "instruction_counts": dict(sorted(Counter(mns).items())),
    "track_counts": {k: len(v) for k, v in ops.items()},
    "wmma_indices": widx,
    "global_load_b128_indices": [x["idx"] for x in ops["global_load_b128"]],
    "ds_store_b128_indices": [x["idx"] for x in ops["ds_store_b128"]],
    "ds_load_b128_indices": [x["idx"] for x in ops["ds_load_b128"]],
    "barrier_indices": [x["idx"] for x in ops["s_barrier"]],
    "waitcnt_summary": {
      "count": len(waits),
      "vmcnt_sequence": [x["vmcnt"] for x in waits],
      "lgkmcnt_sequence": [x["lgkmcnt"] for x in waits],
      "nonfull_count": len([x for x in waits if x["vmcnt"] < 0x3F or x["lgkmcnt"] < 0x3F]),
      "per_wmma_avg": round(len(waits) / len(widx), 3) if widx else 0.0,
    },
    "waits_per_wmma": waits_by_wmma,
    "cadence": sp._cadence_summary(overlap),
    "active_shape_dbuf_cadence": _active_cadence_report(ops, overlap, waits_by_wmma),
    "global_work_between_wmmas": overlap["global_work_between_regions"],
    "wmma_operand_origin_counts": {f"{a}/{b}": n for (a, b), n in sorted(origin_counts.items())},
    "lds_address_families": {
      "store_family_count": lds_families["store_family_count"],
      "load_family_count": lds_families["load_family_count"],
      "store_load_intersection_count": lds_families["store_load_intersection_count"],
    },
    "dbuf_gate_summary": dbuf,
    "dbuf_pipeline_construction_audit": _dbuf_pipeline_construction_audit(ops, widx),
    "lds_reaching_def_map": _lds_reaching_def_map(ops, widx, meta.get("ds_byte_window_rows")),
    "dbuf_d3a_compile_audit": dbuf_compile_audit,
  }
  if full_rows:
    report["track_rows"] = ops
    report["waitcnt"] = waits
    report["wmma_operand_origins"] = origins
    report["wmma_lds_operand_families"] = operand_families
    report["lds_address_families_full"] = lds_families
  return report


def _print_compact(report: dict[str, Any]) -> None:
  if not report.get("ok", True):
    print(f"\n== {report['label']} ==")
    print(f"ERROR: {report['error']}")
    return
  tc = report["track_counts"]
  print(f"\n== {report['label']} ==")
  print(report["tail_off"])
  print(f"shared_floor: {report['shared_floor']}")
  if "program" in report: print(f"program: {report['program']}")
  if "builder" in report: print(f"builder: {report['builder']}")
  print(f"insts={report['instruction_total']} bytes={report['byte_count']} "
        f"global_b128={tc.get('global_load_b128', 0)} ds_store_b128={tc.get('ds_store_b128', 0)} "
        f"ds_load_b128={tc.get('ds_load_b128', 0)} barriers={tc.get('s_barrier', 0)} "
        f"wmma={tc.get(sp.WMMA_NAME, 0)}")
  print(f"wmma_operand_origins={report['wmma_operand_origin_counts']}")
  print(f"global_work_between_wmmas={len(report['global_work_between_wmmas'])} "
        f"{report['global_work_between_wmmas'][:8]}")
  print(f"cadence={report['cadence']}")
  active = report["active_shape_dbuf_cadence"]
  print(f"active_dbuf packed_chain={active['packed_global_to_lds_to_wmma_visible']} "
        f"future_slot_before_compute={active['future_slot_work_before_current_compute']} "
        f"scalar_lds_fallback_total={active['scalar_lds_fallback_total']} "
        f"waits_per_wmma_avg={report['waitcnt_summary']['per_wmma_avg']}")
  print(f"active_dbuf regions={active['regions'][:6]}")
  print(f"lds_families={report['lds_address_families']}")
  rd = report["lds_reaching_def_map"]
  print(f"lds_reaching_def covered={rd['covered_load_count']}/{rd['load_count']} "
        f"missing={rd['missing_load_count']} no_barrier={rd['covered_without_barrier_count']} "
        f"wmma_missing_a={rd['wmma_missing_a_count']} wmma_missing_b={rd['wmma_missing_b_count']}")
  if "hand_lifecycle_oracle" in report:
    print(f"hand_lifecycle={report['hand_lifecycle_oracle']['producer_rule']}")
  print(f"dbuf_D7={report['dbuf_gate_summary']['D7_scheduler_readiness']}")


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--kind", choices=("generated", "hand-pipe", "hand-lds2", "all"), default="all")
  p.add_argument("--active-generated", action="store_true",
                 help="compile real prefill generated active shapes with _compile_native_program instead of the small m-up matmul")
  p.add_argument("--shapes", default="2,2;4,2;2,4", help="semicolon-separated active generated shapes, e.g. '2,2;4,2;2,4'")
  p.add_argument("--target", default="AMD:ISA:gfx1100")
  p.add_argument("--m-up", type=int, default=2)
  p.add_argument("--m", type=int, default=64)
  p.add_argument("--n", type=int, default=64)
  p.add_argument("--k", type=int, default=128)
  p.add_argument("--loc", type=int, default=2)
  p.add_argument("--unr", type=int, default=2)
  p.add_argument("--tm", type=int, default=2)
  p.add_argument("--tn", type=int, default=4)
  p.add_argument("--waves-m", type=int, default=1)
  p.add_argument("--waves-n", type=int, default=1)
  p.add_argument("--wm", type=int, default=2)
  p.add_argument("--wn", type=int, default=4)
  p.add_argument("--bk", type=int, default=32)
  p.add_argument("--pad", type=int, default=0)
  p.add_argument("--dbuf", type=int, default=1)
  p.add_argument("--plra", type=int, default=0)
  p.add_argument("--plrab", type=int, default=0)
  p.add_argument("--leanaddr", type=int, default=0)
  p.add_argument("--dshalf", type=int, default=0)
  p.add_argument("--json", action="store_true")
  p.add_argument("--full-rows", action="store_true")
  p.add_argument("--s10-lds-route-trace", action="store_true",
                 help="emit the S10 LDS primitive route trace/proof instead of instruction lifecycle rows")
  p.add_argument("--route-trace-out", default="",
                 help="optional path for the S10 LDS route trace artifact, e.g. bench/prefill-s10-lds2-ownership/route-trace.json")
  args = p.parse_args()

  if args.s10_lds_route_trace:
    from pathlib import Path
    from extra.qk.prefill_graph_gemm_route import prefill_lds_primitive_route_trace
    report = prefill_lds_primitive_route_trace(args.n, args.k, role="ffn_gate_up", primitive_opt_in=True)
    if args.route_trace_out:
      path = Path(args.route_trace_out)
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2) if args.json or not args.route_trace_out else args.route_trace_out)
    return

  getenv.cache_clear()
  to_program_cache.clear()
  reports = []
  if args.active_generated:
    for item in args.shapes.split(";"):
      if not item.strip(): continue
      a, b = item.split(",", 1)
      shape = (int(a), int(b))
      try:
        insts, meta = _generated_active_insts(args, shape)
        reports.append(_report(f"generated-active-{shape[0]}x{shape[1]}", insts, meta, args.full_rows))
      except Exception as e:
        reports.append({"label": f"generated-active-{shape[0]}x{shape[1]}", "ok": False, "error": f"{type(e).__name__}: {e}"})
  else:
    kinds = ("generated", "hand-pipe", "hand-lds2") if args.kind == "all" else (args.kind,)
    for kind in kinds:
      try:
        if kind == "generated": insts, meta = _generated_insts(args.m_up, args.target)
        else: insts, meta = _hand_insts(kind, args)
        reports.append(_report(kind, insts, meta, args.full_rows))
      except Exception as e:
        reports.append({"label": kind, "ok": False, "error": f"{type(e).__name__}: {e}"})

  if args.json:
    print(json.dumps(reports if len(reports) > 1 else reports[0], indent=2))
  else:
    for r in reports: _print_compact(r)


if __name__ == "__main__":
  main()
