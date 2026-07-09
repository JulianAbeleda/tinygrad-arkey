#!/usr/bin/env python3
"""JSON structural probe for the native-ISA 4x4 route-shaped WMMA stream."""
from __future__ import annotations

import argparse, json, os, re, sys, traceback
from collections import Counter
from dataclasses import replace
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad import Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import AddrSpace
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
from tinygrad.renderer.amd.dsl import Reg
from tinygrad.renderer.isa import amd as amd_isa
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp


ENV_FLAGS = (
  "PREFILL_DBUF", "AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG",
  "PREFILL_TC_LOCAL_STAGE", "PREFILL_TC_LOCAL_STAGE_WITH_LOCAL", "PREFILL_TC_LOCAL_STAGE_B_TILEKEY",
  "PREFILL_LDS_PACK_WITHLOCAL_B128", "PREFILL_DBUF_D3A_POST", "PREFILL_DBUF_D3A_AUDIT", "PREFILL_DBUF_D3A_STAGE_A",
  "PREFILL_DBUF_D3A_STAGE_B",
)
REG_FIELDS = ("vdst", "src0", "src1", "src2", "addr", "vaddr", "saddr", "sdst", "sdata", "data0", "data1", "data2", "data3", "data", "vdata")
WMMA_NAME = "v_wmma_f32_16x16x16_f16"
TRACK_NAMES = (
  "global_load_b128",
  "global_load_u16",
  "ds_store_b128",
  "ds_store_b64",
  "ds_store_b32",
  "ds_store_b16",
  "ds_load_b128",
  "global_store_b16",
  "s_barrier",
  "s_waitcnt",
  WMMA_NAME,
)
BROAD_STAGING_NAMES = ("global_load_b128", "ds_store_b128", "ds_store_b64", "ds_store_b32", "ds_store_b16")
LDS_OP_NAMES = ("ds_store_b128", "ds_store_b64", "ds_store_b32", "ds_store_b16", "ds_load_b128")


def _span_key(rows: list[dict[str, Any]]) -> set[int]:
  return {r["idx"] for r in rows}


def _sorted_regions(labels: list[tuple[str, int, int, bool]]) -> list[dict[str, int | str | bool]]:
  return [{"label": label, "start": start, "end": end, "strict": strict} for label, start, end, strict in labels]


def _collect_by_region(rows: list[dict[str, Any]], start: int, end: int, strict: bool) -> list[dict[str, Any]]:
  if strict:
    return [r for r in rows if start < r["idx"] < end]
  return [r for r in rows if start <= r["idx"] < end]


def _apply_env(args: argparse.Namespace) -> dict[str, str | None]:
  overrides = {
    "PREFILL_DBUF": args.prefill_dbuf,
    "AMD_ISA_WAITCNT_TARGETED": args.targeted_waitcnt,
    "AMD_ISA_WMMA_B128_FRAG": args.b128_frag,
  }
  for key, value in overrides.items():
    if value is not None: os.environ[key] = str(value)
  getenv.cache_clear()
  to_program_cache.clear()
  return {k: os.environ.get(k) for k in ENV_FLAGS}


def _route_ast(m_up: int):
  a = Tensor.empty(64, 64, dtype="half")
  b = Tensor.empty(64, 64, dtype="half")
  lin = (a @ b.transpose()).schedule_linear()
  ast = [u for u in lin.toposort() if u.op is Ops.SINK][0]
  opts = (Opt(OptOps.TC, axis=0, arg=(0, 0, 1)),) + (Opt(OptOps.UPCAST, axis=0, arg=4),) * m_up
  return ast.replace(arg=replace(ast.arg, opts_to_apply=opts))


def _mn(arg: Any) -> str:
  if isinstance(arg, UOp): arg = arg.arg
  return "marker" if isinstance(arg, tuple) else str(arg).split("(", 1)[0]


def _insts_from_uops(uops: list[UOp]) -> list[Any]:
  return [u.arg if isinstance(u, UOp) else u for u in uops]


def _final_stream(ren: AMDISARenderer, lin_src: tuple[UOp, ...]) -> list[UOp]:
  uops = list(lin_src)
  if getenv("AMD_ISA_SCHED", 1): uops = ren._schedule(uops)
  return ren._resolve_labels(ren._insert_waitcnt(uops))


def _reg_span(reg: Any, size: int | None=None) -> dict[str, int] | None:
  if not isinstance(reg, Reg): return None
  n = size if size is not None else reg.sz
  kind, base = ("v", reg.offset - 256) if 256 <= reg.offset < 512 else ("s", reg.offset)
  return {"kind": kind, "lo": base, "hi": base + n - 1, "n": n}


def _field_spans(inst: Any) -> dict[str, dict[str, int]]:
  out: dict[str, dict[str, int]] = {}
  op_regs = getattr(inst, "op_regs", {})
  for fname, _field in getattr(inst, "_fields", ()):
    if fname not in REG_FIELDS: continue
    if (span := _reg_span(getattr(inst, fname, None), op_regs.get(fname))) is not None: out[fname] = span
  return out


def _markers(label: str, uops: list[UOp]) -> dict[str, Any]:
  rows = []
  for idx, u in enumerate(uops):
    arg = u.arg if isinstance(u, UOp) else u
    if isinstance(arg, tuple):
      row = {"idx": idx, "kind": arg[0], "arg": repr(arg[1:])}
      if arg[0] == "audit_dbuf_d3a_stage" and len(arg) > 1:
        try: row["payload"] = dict(arg[1])
        except Exception: pass
      rows.append(row)
  return {"stream": label, "count": len(rows), "rows": rows}

def _dbuf_d3a_audit_summary(marker_reports: list[dict[str, Any]], audited_rows: list[dict[str, Any]] | None=None) -> dict[str, Any]:
  rows = [r for m in marker_reports for r in m.get("rows", []) if r.get("kind") == "audit_dbuf_d3a_stage"]
  final_rows = [r for m in marker_reports if m.get("stream") == "final" for r in m.get("rows", []) if r.get("kind") == "audit_dbuf_d3a_stage"]
  store_rows = [r for r in (audited_rows or []) if r.get("audit_payload") is not None]
  payloads = [r.get("payload", {}) for r in final_rows] + [r.get("audit_payload", {}) for r in store_rows]
  roles = sorted({p.get("role", "").strip("'") for p in payloads if p.get("role") is not None})
  slots = sorted({p.get("dbuf_slot") for p in payloads if p.get("dbuf_slot") is not None})
  nbufs = sorted({p.get("nbuf") for p in payloads if p.get("nbuf") is not None})
  ok_payloads = [p for p in payloads if p.get("ok") is True]
  final_count = len(final_rows) + len(store_rows)
  return {
    "marker_count_all_streams": len(rows),
    "final_marker_count": len(final_rows),
    "audited_store_count": len(store_rows),
    "ok_marker_count": len(ok_payloads),
    "roles": roles,
    "dbuf_slots": slots,
    "nbufs": nbufs,
    "ok": bool(final_count) and len(ok_payloads) == final_count and set(roles) >= {"A", "B"} and any(str(n).isdigit() and int(n) > 1 for n in nbufs),
    "rows": final_rows,
    "audited_store_rows": store_rows,
  }

def _dbuf_d3a_compile_audit_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  roles = sorted({str(p.get("role", "")).strip("'") for p in rows if p.get("role") is not None})
  slots = sorted({p.get("dbuf_slot") for p in rows if p.get("dbuf_slot") is not None})
  nbufs = sorted({p.get("nbuf") for p in rows if p.get("nbuf") is not None})
  ok_rows = [p for p in rows if p.get("ok") is True]
  stage_key_rows = [p for p in rows if p.get("kind") == "stage_key_audit"]
  suppress_rows = [p for p in rows if p.get("kind") == "stage_key_suppress_decision"]
  suppressed_rows = [p for p in suppress_rows if p.get("suppressed") is True]
  by_slot: dict[str, set[str]] = {}
  by_key: dict[str, set[str]] = {}
  for p in stage_key_rows:
    slot = repr(p.get("slot"))
    key = str(p.get("strong_key"))
    source = str(p.get("source"))
    by_slot.setdefault(slot, set()).add(source)
    by_key.setdefault(key, set()).add(source)
  weak_alias_slots = {k: sorted(v) for k, v in by_slot.items() if len(v) > 1}
  strong_key_collisions = {k: sorted(v) for k, v in by_key.items() if len(v) > 1}
  return {
    "compile_audit_count": len(rows),
    "ok_audit_count": len(ok_rows),
    "roles": roles,
    "dbuf_slots": slots,
    "nbufs": nbufs,
    "ok": bool(rows) and len(ok_rows) == len(rows) and set(roles) >= {"A", "B"} and any(str(n).isdigit() and int(n) > 1 for n in nbufs),
    "stage_key_audit_count": len(stage_key_rows),
    "stage_key_weak_alias_slot_count": len(weak_alias_slots),
    "stage_key_weak_alias_slot_sample": dict(list(sorted(weak_alias_slots.items()))[:8]),
    "stage_key_strong_collision_count": len(strong_key_collisions),
    "stage_key_strong_collision_sample": dict(list(sorted(strong_key_collisions.items()))[:8]),
    "stage_key_rejects_weak_aliases": bool(weak_alias_slots) and not strong_key_collisions,
    "stage_key_suppress_decision_count": len(suppress_rows),
    "stage_key_suppressed_count": len(suppressed_rows),
    "stage_key_suppressed_sample": suppressed_rows[:8],
    "rows": rows,
  }


def _waitcnt_simm16(inst: Any) -> int | None:
  if _mn(inst) != "s_waitcnt": return None
  return getattr(inst, "simm16", getattr(inst, "sim16", None))


def _decode_waitcnt(simm16: int) -> dict[str, int]:
  return {"simm16": simm16, "vmcnt": (simm16 >> 10) & 0x3F, "lgkmcnt": (simm16 >> 4) & 0x3F, "expcnt": simm16 & 0x7}


def _audit_payload_from_tag(tag: Any) -> dict[str, Any] | None:
  if not (isinstance(tag, tuple) and len(tag) == 2 and tag[0] == "audit_dbuf_d3a_stage"): return None
  try: return dict(tag[1])
  except Exception: return None

def _interesting_rows(insts: list[Any], name: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
  rows = []
  for idx, raw in enumerate(insts):
    inst = raw.arg if isinstance(raw, UOp) else raw
    if isinstance(inst, tuple) or _mn(inst) != name: continue
    spans = {k: v for k, v in _field_spans(inst).items() if k in fields}
    row = {"idx": idx, "spans": spans, "text": str(inst)}
    if isinstance(raw, UOp) and (payload := _audit_payload_from_tag(raw.tag)) is not None: row["audit_payload"] = payload
    rows.append(row)
  return rows


def _range_regs(span: dict[str, int]) -> range:
  return range(span["lo"], span["hi"] + 1)


def _trailing_imm(text: str) -> int:
  m = re.search(r",\s*(-?\d+)\)$", text)
  return int(m.group(1)) if m else 0


def _addr_key(row: dict[str, Any]) -> str:
  span = row["spans"].get("addr")
  if span is None: return "addr_unknown"
  return f"{span['kind']}{span['lo']}:{_trailing_imm(row['text'])}"


def _addr_key_with_def(insts: list[Any], row: dict[str, Any]) -> str:
  base = _addr_key(row)
  span = row["spans"].get("addr")
  if span is None or span.get("kind") != "v": return base
  defs = _vreg_def_history(insts)
  parts = []
  for reg in _range_regs(span):
    candidates = [x for x in defs.get(("v", reg), []) if x < row["idx"]]
    didx = candidates[-1] if candidates else None
    if didx is None: parts.append(f"v{reg}=undef")
    else: parts.append(f"v{reg}@{didx}:{_mn(insts[didx])}")
  return base + "|" + ",".join(parts)


def _span_label(span: dict[str, int] | None) -> str:
  if span is None: return "?"
  return f"{span['kind']}{span['lo']}..{span['hi']}"


def _regs_overlap(a: dict[str, int] | None, b: dict[str, int] | None) -> bool:
  if a is None or b is None or a["kind"] != b["kind"]: return False
  return not (a["hi"] < b["lo"] or b["hi"] < a["lo"])


def _addr_span_key(row: dict[str, Any]) -> str:
  span = row["spans"].get("addr") or row["spans"].get("vaddr")
  return f"{_span_label(span)}:{_trailing_imm(row['text'])}"


def _nearest_def_rows(insts: list[Any], defs: dict[tuple[str, int], int], span: dict[str, int] | None,
                      names: set[str], limit: int=8) -> list[dict[str, Any]]:
  if span is None or span["kind"] != "v": return []
  rows = []
  for reg in _range_regs(span):
    didx = defs.get(("v", reg))
    if didx is None or isinstance(insts[didx], tuple) or _mn(insts[didx]) not in names: continue
    row = {"idx": didx, "name": _mn(insts[didx]), "spans": _field_spans(insts[didx]), "text": str(insts[didx])}
    if row not in rows: rows.append(row)
    if len(rows) >= limit: break
  return rows


def _vreg_def_history(insts: list[Any]) -> dict[tuple[str, int], list[int]]:
  defs: dict[tuple[str, int], list[int]] = {}
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    for span in (s for f, s in _field_spans(inst).items() if f == "vdst" and s["kind"] == "v"):
      for reg in _range_regs(span):
        defs.setdefault(("v", reg), []).append(idx)
  return defs


def _nearest_def_rows_before(insts: list[Any], hist: dict[tuple[str, int], list[int]], span: dict[str, int] | None,
                             before_idx: int, names: set[str], limit: int=8) -> list[dict[str, Any]]:
  if span is None or span["kind"] != "v": return []
  rows = []
  for reg in _range_regs(span):
    candidates = [x for x in hist.get(("v", reg), []) if x < before_idx]
    didx = candidates[-1] if candidates else None
    if didx is None or isinstance(insts[didx], tuple) or _mn(insts[didx]) not in names: continue
    row = {"idx": didx, "name": _mn(insts[didx]), "spans": _field_spans(insts[didx]), "text": str(insts[didx])}
    if row not in rows: rows.append(row)
    if len(rows) >= limit: break
  return rows


def _store_source_kind(insts: list[Any], defs: dict[tuple[str, int], int], span: dict[str, int] | None) -> str:
  if span is None or span["kind"] != "v": return "unknown"
  names = set()
  for reg in _range_regs(span):
    didx = defs.get(("v", reg))
    names.add("unassigned" if didx is None or isinstance(insts[didx], tuple) else _mn(insts[didx]))
  if any(x == WMMA_NAME for x in names): return "wmma_accumulator"
  if any(x.startswith("v_cvt") for x in names): return "converted_accumulator"
  return "+".join(sorted(names)) if names else "unknown"


def _global_operand_family(insts: list[Any], hist: dict[tuple[str, int], list[int]], span: dict[str, int] | None,
                           before_idx: int) -> dict[str, Any]:
  loads = []
  for row in _nearest_def_rows_before(insts, hist, span, before_idx, {"global_load_b128", "global_load_u16"}):
    loads.append({"idx": row["idx"], "name": row["name"], "vdst": row["spans"].get("vdst"),
                  "addr": row["spans"].get("addr") or row["spans"].get("vaddr"), "addr_key": _addr_span_key(row)})
  return {"load_count": len(loads), "loads": loads}


def wmma_chain_trace(final: list[Any], max_rows: int | None=None) -> dict[str, Any]:
  """Diagnostic-only native-ISA WMMA chain trace; logical rows are best-effort when metadata is absent."""
  wmma_rows = _interesting_rows(final, WMMA_NAME, ("vdst", "src0", "src1", "src2"))
  gl_rows = _interesting_rows(final, "global_load_b128", ("vdst", "addr", "vaddr", "saddr"))
  gu16_rows = _interesting_rows(final, "global_load_u16", ("vdst", "addr", "vaddr", "saddr"))
  st_rows = _interesting_rows(final, "global_store_b16", ("data", "vdata", "addr", "vaddr", "saddr"))
  defs = _vreg_definers(final)
  hist = _vreg_def_history(final)
  store_summary = []
  for si, store in enumerate(st_rows):
    src = store["spans"].get("data") or store["spans"].get("vdata")
    store_summary.append({"store_idx": si, "inst_idx": store["idx"], "source": src,
                          "address": store["spans"].get("addr") or store["spans"].get("vaddr"),
                          "addr_key": _addr_span_key(store), "source_kind": _store_source_kind(final, defs, src)})
  chain = []
  for wi, row in enumerate(wmma_rows):
    spans = row["spans"]
    related_stores = [s for s in store_summary if _regs_overlap(spans.get("vdst"), s["source"]) or s["source_kind"] in {"wmma_accumulator", "converted_accumulator"}]
    chain.append({"wmma_idx": wi, "inst_idx": row["idx"], "src0": spans.get("src0"), "src1": spans.get("src1"),
                  "src2": spans.get("src2"), "vdst": spans.get("vdst"),
                  "src0_global_origin": _global_operand_family(final, hist, spans.get("src0"), row["idx"]),
                  "src1_global_origin": _global_operand_family(final, hist, spans.get("src1"), row["idx"]),
                  "global_store_b16": related_stores[:8]})
  repeated: dict[str, list[int]] = {}
  for row in chain: repeated.setdefault(_span_label(row["vdst"]), []).append(row["wmma_idx"])
  return {"wmma_count": len(wmma_rows), "global_load_b128_count": len(gl_rows), "global_load_u16_count": len(gu16_rows),
          "global_store_b16_count": len(st_rows),
          "lds_op_count": sum(1 for x in final if not isinstance(x, tuple) and _mn(x).startswith("ds_")),
          "repeated_vdst_spans": {k: v for k, v in repeated.items() if len(v) > 1},
          "chain": chain if max_rows is None else chain[:max_rows],
          "global_store_b16": store_summary if max_rows is None else store_summary[:max_rows],
          "note": "origin rows are physical-register definers after scheduling/regalloc; logical output rows are unavailable unless carried by upstream metadata"}


def _summarize_lds_addresses(ops: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
  by_op = {}
  all_keys = set()
  for name in LDS_OP_NAMES:
    keys: dict[str, list[int]] = {}
    for row in ops[name]:
      key = _addr_key(row)
      keys.setdefault(key, []).append(row["idx"])
      all_keys.add(key)
    by_op[name] = {
      "family_count": len(keys),
      "families": [{"addr_key": key, "count": len(indices), "indices": indices} for key, indices in sorted(keys.items())],
    }
  load_keys = {_addr_key(row) for row in ops["ds_load_b128"]}
  store_keys = {_addr_key(row) for name in ("ds_store_b128", "ds_store_b64", "ds_store_b32", "ds_store_b16") for row in ops[name]}
  return {
    "visible": bool(all_keys),
    "family_count": len(all_keys),
    "store_family_count": len(store_keys),
    "load_family_count": len(load_keys),
    "store_load_intersection_count": len(store_keys & load_keys),
    "by_op": by_op,
  }


def _vreg_definers(insts: list[Any]) -> dict[tuple[str, int], int]:
  defs: dict[tuple[str, int], int] = {}
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    for span in (s for f, s in _field_spans(inst).items() if f == "vdst" and s["kind"] == "v"):
      for reg in _range_regs(span):
        defs[("v", reg)] = idx
  return defs


def _vreg_defs_by_op(insts: list[Any], names: set[str]) -> dict[tuple[str, int], int]:
  defs: dict[tuple[str, int], int] = {}
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple) or _mn(inst) not in names: continue
    span = _field_spans(inst).get("vdst")
    if span is None or span["kind"] != "v": continue
    for reg in _range_regs(span):
      defs[("v", reg)] = idx
  return defs


def _trace_vreg_origin(insts: list[Any], defs: dict[tuple[str, int], int], reg: int, memo: dict[tuple[str, int], set[str]]) -> set[str]:
  key = ("v", reg)
  if key in memo: return memo[key]
  def_idx = defs.get(key)
  if def_idx is None:
    memo[key] = {"unassigned"}
    return memo[key]
  inst = insts[def_idx]
  if isinstance(inst, tuple):
    memo[key] = {"tuple"}
    return memo[key]
  inst_name = _mn(inst)
  if inst_name in {"global_load_b128", "global_load_u16", "ds_load_b128"}:
    memo[key] = {inst_name}
    return memo[key]
  if inst_name.startswith("v_mov"):
    span = _field_spans(inst).get("src0")
    if span is None or span["kind"] != "v":
      memo[key] = {"other"}
      return memo[key]
    vals: set[str] = set()
    for src in _range_regs(span):
      vals |= _trace_vreg_origin(insts, defs, src, memo)
    memo[key] = vals
    return vals
  memo[key] = {"other"}
  return memo[key]


def _classify_vreg_origin(origins: set[str]) -> str:
  if origins == {"global_load_b128"}:
    return "global_load_b128"
  if origins == {"ds_load_b128"}:
    return "ds_load_b128"
  if origins == {"unassigned"}:
    return "unassigned"
  if origins <= {"global_load_b128", "global_load_u16"} and "ds_load_b128" not in origins:
    return "global_load"
  if origins <= {"ds_load_b128", "global_load_b128", "global_load_u16"} and "global_load_b128" in origins:
    return "mixed_global_lds"
  if origins <= {"other", "tuple"}:
    return "other"
  return "mixed"


def _wmma_operand_origins(insts: list[Any], wmma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  defs = _vreg_definers(insts)
  memo: dict[tuple[str, int], set[str]] = {}
  out = []
  for row in wmma_rows:
    spans = row["spans"]
    src_origin = {}
    for key in ("src0", "src1"):
      span = spans.get(key)
      if span is None or span["kind"] != "v":
        src_origin[key] = "unknown"
      else:
        origin = set[str]()
        for reg in _range_regs(span):
          origin |= _trace_vreg_origin(insts, defs, reg, memo)
        src_origin[key] = _classify_vreg_origin(origin)
    src_origin["src2"] = "accumulator"
    out.append({"idx": row["idx"], "src0": src_origin["src0"], "src1": src_origin["src1"], "src2": src_origin["src2"]})
  return out


def _wmma_lds_operand_families(insts: list[Any], wmma_rows: list[dict[str, Any]]) -> dict[str, Any]:
  ds_defs = _vreg_defs_by_op(insts, {"ds_load_b128"})
  ds_rows_by_idx = {
    idx: {"idx": idx, "spans": _field_spans(insts[idx]), "text": str(insts[idx])}
    for idx in sorted(set(ds_defs.values()))
  }
  operand_families: dict[str, dict[str, list[int]]] = {"src0": {}, "src1": {}}
  rows = []
  for row in wmma_rows:
    operand_row: dict[str, Any] = {"idx": row["idx"]}
    for operand in ("src0", "src1"):
      span = row["spans"].get(operand)
      load_indices = sorted({ds_defs[("v", reg)] for reg in _range_regs(span) if ("v", reg) in ds_defs}) if span else []
      families = []
      for load_idx in load_indices:
        key = _addr_key_with_def(insts, ds_rows_by_idx[load_idx])
        operand_families[operand].setdefault(key, []).append(row["idx"])
        families.append({"load_idx": load_idx, "addr_key": key})
      operand_row[operand] = families
    rows.append(operand_row)
  by_operand = {
    operand: {
      "family_count": len(fams),
      "families": [{"addr_key": key, "wmma_indices": sorted(set(indices))} for key, indices in sorted(fams.items())],
    } for operand, fams in operand_families.items()
  }
  return {"by_operand": by_operand, "rows": rows}


def _cadence_summary(overlap: dict[str, Any]) -> dict[str, Any]:
  regions = overlap.get("global_work_regions_json", [])
  prologue = next((r for r in regions if r.get("label") == "prologue"), {})
  tail = next((r for r in regions if r.get("label") == "tail"), {})
  body = [r for r in regions if r.get("strict")]
  body_with_staging = [r["label"] for r in body if r.get("global_load_b128_count", 0) > 0 or
                       r.get("ds_store_b128_count", 0) > 0 or r.get("ds_store_b64_count", 0) > 0 or
                       r.get("ds_store_b32_count", 0) > 0 or r.get("ds_store_b16_count", 0) > 0]
  body_with_current_lds_load = [r["label"] for r in body if r.get("ds_load_b128_count", 0) > 0]
  return {
    "prologue_has_staging": prologue.get("global_work_count", 0) > 0,
    "body_region_count": len(body),
    "body_has_next_slot_work": bool(body_with_staging),
    "body_regions_with_next_slot_work": body_with_staging,
    "body_regions_with_current_slot_lds_load": body_with_current_lds_load,
    "tail_region_present": bool(tail),
    "tail_has_no_staging": bool(tail) and tail.get("global_work_count", 0) == 0,
    "wmma_region_count": overlap.get("wmma_count", 0),
  }


def _dbuf_gate_summary(ops: dict[str, list[dict[str, Any]]], overlap: dict[str, Any], lds_addr: dict[str, Any],
                       operand_families: dict[str, Any], origins: list[dict[str, Any]]) -> dict[str, Any]:
  cadence = _cadence_summary(overlap)
  src0_slots = operand_families["by_operand"]["src0"]["family_count"]
  src1_slots = operand_families["by_operand"]["src1"]["family_count"]
  origins_lds = bool(origins) and all(row.get("src0") == "ds_load_b128" and row.get("src1") == "ds_load_b128" for row in origins)
  scalar_lds = len(ops["ds_store_b16"]) + len(ops["ds_store_b32"])
  byte_identity = lds_addr.get("byte_window_identity", {})
  byte_ok = byte_identity.get("verdict") == "covered"
  def_sensitive = any("|" in fam["addr_key"] for operand in operand_families["by_operand"].values() for fam in operand["families"])
  weak_addr_family_ok = lds_addr["store_family_count"] >= 2 and lds_addr["load_family_count"] >= 2 and src0_slots >= 1 and src1_slots >= 1
  strict_two_operand_ok = byte_ok or (lds_addr["store_family_count"] >= 2 and lds_addr["load_family_count"] >= 2 and src0_slots >= 2 and src1_slots >= 2)
  d3_ok = cadence["prologue_has_staging"] and cadence["body_has_next_slot_work"] and cadence["tail_region_present"]
  d7_ok = strict_two_operand_ok and d3_ok and origins_lds and scalar_lds == 0
  return {
    "D2_two_slot_identity": {
      "ok": strict_two_operand_ok,
      "weak_addr_family_ok": weak_addr_family_ok,
      "proof_strength": "normalized_ds_byte_windows" if byte_ok else "addr_register_def_sensitive_family" if def_sensitive else "addr_register_family_only",
      "byte_window_verdict": byte_identity.get("verdict", "not_available"),
      "byte_window_status_counts": lds_addr.get("byte_window_status_counts", {}),
      "store_family_count": lds_addr["store_family_count"],
      "load_family_count": lds_addr["load_family_count"],
      "src0_lds_family_count": src0_slots,
      "src1_lds_family_count": src1_slots,
      "strict_requirement": "store/load families >= 2 and both WMMA operands observe >= 2 LDS load address families",
      "note": "families are instruction-operand addr register plus immediate offset; this is structural identity, not resolved LDS byte values",
    },
    "D3_cadence": {"ok": d3_ok, **cadence},
    "D7_scheduler_readiness": {
      "ok": d7_ok,
      "wmma_operands_from_lds": origins_lds,
      "scalar_lds_store_count": scalar_lds,
      "next_slot_work_near_compute": cadence["body_has_next_slot_work"],
      "reason": "ready" if d7_ok else "missing one or more of slot identity, body staging overlap, LDS operand origins, or scalar-store cleanliness",
    },
  }


def _dregs_in_src_cone(u: UOp, seen: set[UOp]) -> list[UOp]:
  out: list[UOp] = []
  for sr in u.src:
    if sr.op is Ops.DEFINE_REG: out.append(sr)
    elif sr.op is Ops.AFTER and sr not in seen:
      seen.add(sr)
      out += _dregs_in_src_cone(sr, seen)
  return out


def _resource_summary(prg: UOp, lds_limit_bytes: int) -> dict[str, Any]:
  sink = prg.src[0]
  local_bytes, reg_bytes, reclaimable_reg_bytes = 0, 0, 0
  define_rows: list[dict[str, Any]] = []
  lid_threads: dict[str, int] = {}
  pinned_dreg, lds_dreg = set(), set()
  for u in sink.toposort():
    if u.op is not Ops.INS: continue
    arg = str(u.arg)
    if "ACCUM" in arg: pinned_dreg.update(_dregs_in_src_cone(u, set()))
    elif "DS_LOAD" in arg or "DS_STORE" in arg: lds_dreg.update(_dregs_in_src_cone(u, set()))
  for u in sink.toposort():
    if u.op in (Ops.DEFINE_LOCAL, Ops.DEFINE_REG):
      nbytes = u.ptrdtype.size * u.ptrdtype.base.itemsize
      is_reg = u.ptrdtype.addrspace == AddrSpace.REG
      is_reclaimable = is_reg and u in pinned_dreg and u not in lds_dreg
      if is_reg:
        reg_bytes += nbytes
        if is_reclaimable: reclaimable_reg_bytes += nbytes
      else:
        local_bytes += nbytes
      define_rows.append({
        "op": u.op.name,
        "addrspace": str(u.ptrdtype.addrspace),
        "bytes": nbytes,
        "reclaimable_accumulator_reg": is_reclaimable,
        "dtype": str(u.ptrdtype),
      })
    elif u.op is Ops.SPECIAL and isinstance(u.arg, str) and u.arg.startswith("lidx"):
      lid_threads[u.arg] = int(u.src[0].arg)
  n_threads = 1
  for v in lid_threads.values(): n_threads *= v
  reg_accum_enabled = getenv("AMD_ISA_REG_ACCUM", 0)
  effective_reg_bytes = reg_bytes - reclaimable_reg_bytes if reg_accum_enabled else reg_bytes
  estimated_group_segment = local_bytes + effective_reg_bytes * n_threads
  unreclaimed_group_segment = local_bytes + reg_bytes * n_threads
  binary_group_segment = None
  for u in prg.src:
    if u.op is Ops.BINARY:
      try: binary_group_segment = group_segment_fixed_size_from_elf(u.arg)
      except Exception: binary_group_segment = None
      break
  return {
    "lds_limit_bytes": lds_limit_bytes,
    "n_threads": n_threads,
    "local_bytes": local_bytes,
    "reg_bytes_per_thread": reg_bytes,
    "reclaimable_reg_bytes_per_thread": reclaimable_reg_bytes,
    "reg_accum_reclaim_enabled": bool(reg_accum_enabled),
    "effective_reg_bytes_per_thread": effective_reg_bytes,
    "group_segment_estimated_bytes": estimated_group_segment,
    "group_segment_unreclaimed_bytes": unreclaimed_group_segment,
    "binary_group_segment_bytes": binary_group_segment,
    "over_limit": estimated_group_segment > lds_limit_bytes,
    "unreclaimed_over_limit": unreclaimed_group_segment > lds_limit_bytes,
    "define_rows": define_rows,
  }


def _failure_dbuf_gate_summary(error: Exception) -> dict[str, Any]:
  return {
    "D2_two_slot_identity": {"ok": False, "reason": "no final instruction stream; compile/render failed before LDS operands were visible"},
    "D3_cadence": {"ok": False, "reason": "no final instruction stream; compile/render failed before WMMA regions were visible"},
    "D7_scheduler_readiness": {"ok": False, "reason": f"{type(error).__name__}: {error}"},
  }


def _collect_regions(ops: dict[str, list[dict[str, Any]]], widx: list[int]) -> dict[str, Any]:
  region_defs: list[tuple[str, int, int, bool]] = []
  n = max([x["idx"] for rows in ops.values() for x in rows], default=-1) + 1
  if not widx:
    return {
      "wmma_count": 0,
      "wmma_regions": [],
      "global_work_regions": [],
      "global_work_overlap": False,
      "global_work_between_overlap": False,
      "global_work_between_regions": [],
      "current_lds_load_regions": [],
      "global_work_regions_json": {"no_wmma": True},
    }
  if widx:
    region_defs.append(("prologue", 0, widx[0], False))
    for left, right in zip(widx, widx[1:]):
      region_defs.append((f"between_wmma_{left}_{right}", left, right, True))
    region_defs.append(("tail", widx[-1], n, False))
  regions = []
  overlap = False
  between_overlap = []
  for label, start, end, strict in region_defs:
    by_name = {}
    for name, rows in ops.items():
      selected = _collect_by_region(rows, start, end, strict)
      by_name[name] = {
        "count": len(selected),
        "indices": [x["idx"] for x in selected],
        "rows": selected,
      }
    global_work_count = sum(by_name[name]["count"] for name in BROAD_STAGING_NAMES if name in by_name)
    overlap = overlap or global_work_count > 0
    if strict and global_work_count:
      between_overlap.append(label)
    regions.append({
      "label": label,
      "start": start,
      "end": end,
      "strict": strict,
      "global_work_count": global_work_count,
      "global_load_b128_count": by_name["global_load_b128"]["count"],
      "ds_store_b128_count": by_name["ds_store_b128"]["count"],
      "ds_store_b16_count": by_name["ds_store_b16"]["count"],
      "ds_store_b32_count": by_name["ds_store_b32"]["count"],
      "ds_store_b64_count": by_name["ds_store_b64"]["count"],
      "ds_load_b128_count": by_name["ds_load_b128"]["count"],
      "s_barrier_count": by_name["s_barrier"]["count"],
      "s_waitcnt_count": by_name["s_waitcnt"]["count"],
    })
  staging_rows = sorted(_span_key(ops["global_load_b128"] + ops["ds_store_b128"] + ops["ds_store_b64"] +
                                  ops["ds_store_b32"] + ops["ds_store_b16"]))
  lds_load_rows = sorted(_span_key(ops["ds_load_b128"]))
  global_windows = []
  lds_load_windows = []
  for reg in regions:
    if reg["start"] <= reg["end"]:
      selected_staging = [i for i in staging_rows if reg["start"] <= i < reg["end"]] if not reg["strict"] else \
                         [i for i in staging_rows if reg["start"] < i < reg["end"]]
      selected_lds_load = [i for i in lds_load_rows if reg["start"] <= i < reg["end"]] if not reg["strict"] else \
                          [i for i in lds_load_rows if reg["start"] < i < reg["end"]]
      global_windows.append({
        "label": reg["label"],
        "indices": selected_staging,
        "count": len(selected_staging),
      })
      lds_load_windows.append({
        "label": reg["label"],
        "indices": selected_lds_load,
        "count": len(selected_lds_load),
      })
  return {
    "wmma_count": len(widx),
    "wmma_regions": _sorted_regions(region_defs),
    "global_work_regions": global_windows,
    "current_lds_load_regions": lds_load_windows,
    "global_work_overlap": overlap,
    "global_work_between_overlap": bool(between_overlap),
    "global_work_between_regions": between_overlap,
    "global_work_regions_json": regions,
  }


def _branches(insts: list[Any]) -> list[dict[str, Any]]:
  return [{"idx": i, "name": _mn(x), "text": str(x)} for i, x in enumerate(insts) if not isinstance(x, tuple) and "branch" in _mn(x)]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
  env = _apply_env(args)
  ren = AMDISARenderer(Target.parse(args.target))
  report: dict[str, Any] = {"target": args.target, "m_up": args.m_up, "env": env, "ok": False}
  amd_isa.DBUF_D3A_AUDIT_LOG.clear()
  prg = to_program(_route_ast(args.m_up), ren)
  lin = [u for u in prg.src if u.op is Ops.LINEAR][0]
  pre_uops = list(lin.src)
  scheduled_uops = ren._schedule(pre_uops) if getenv("AMD_ISA_SCHED", 1) else pre_uops
  final_uops = _final_stream(ren, lin.src)
  final = _insts_from_uops(final_uops)
  mns = [_mn(x) for x in final if not isinstance(x, tuple)]
  counts = dict(sorted(Counter(mns).items()))
  op_fields = {
    "global_load_b128": ("vdst", "addr", "vaddr", "saddr"),
    "global_load_u16": ("vdst", "addr", "vaddr", "saddr"),
    "ds_store_b128": ("addr", "data0", "data1", "data2", "data3"),
    "ds_store_b64": ("addr", "data0", "data1"),
    "ds_store_b32": ("addr", "data0"),
    "ds_store_b16": ("addr", "data0"),
    "ds_load_b128": ("vdst", "addr", "data0", "data1", "data2", "data3"),
    "global_store_b16": ("data", "vdata", "addr", "vaddr", "saddr"),
    "s_barrier": tuple(),
    "s_waitcnt": tuple(),
    WMMA_NAME: ("vdst", "src0", "src1", "src2"),
  }
  ops: dict[str, list[dict[str, Any]]] = {}
  for name in TRACK_NAMES:
    ops[name] = _interesting_rows(final_uops, name, op_fields.get(name, tuple()))
  bidx, widx = [x["idx"] for x in ops["global_load_b128"]], [x["idx"] for x in ops[WMMA_NAME]]
  waits = []
  for idx, inst in enumerate(final):
    if isinstance(inst, tuple): continue
    if (simm16 := _waitcnt_simm16(inst)) is not None:
      waits.append({"idx": idx, **_decode_waitcnt(simm16), "text": str(inst)})
  waitcnt_summary = {
    "count": len(waits),
    "full": len([x for x in waits if x["vmcnt"] == 0x3F and x["lgkmcnt"] == 0x3F and x["expcnt"] == 0x7]),
    "targeted": len([x for x in waits if x["vmcnt"] < 0x3F or x["lgkmcnt"] < 0x3F]),
    "vmcnt_sequence": [x["vmcnt"] for x in waits],
    "lgkmcnt_sequence": [x["lgkmcnt"] for x in waits],
  }
  waitcnt_nonfull = [x for x in waits if (x["vmcnt"] < 0x3F or x["lgkmcnt"] < 0x3F)]
  overlap = _collect_regions(ops, widx)
  wmma_operand_origins = _wmma_operand_origins(final, ops[WMMA_NAME])
  lds_address_families = _summarize_lds_addresses(ops)
  try:
    from extra.qk.prefill import a_fragment_alias_probe as afp
    byte_trace = afp._analyze(final, prg)
    lds_address_families["byte_window_identity"] = byte_trace.get("a_fragment_window_identity", {})
    lds_address_families["byte_window_status_counts"] = byte_trace.get("ds_window_status_counts", {})
    lds_address_families["a_byte_store_windows"] = byte_trace.get("a_store_windows", [])
    lds_address_families["a_byte_inconclusive_rows"] = byte_trace.get("a_inconclusive_window_rows", [])
    lds_address_families["a_byte_missing_store_rows"] = byte_trace.get("a_missing_store_window_rows", [])
    lds_address_families["a_byte_alias_windows"] = byte_trace.get("a_alias_windows", [])
  except Exception as e:
    lds_address_families["byte_window_identity"] = {"verdict": "unknown", "ok": False, "reason": f"{type(e).__name__}: {e}"}
  wmma_lds_operand_families = _wmma_lds_operand_families(final, ops[WMMA_NAME])
  dbuf_gate_summary = _dbuf_gate_summary(ops, overlap, lds_address_families, wmma_lds_operand_families, wmma_operand_origins)
  resource_summary = _resource_summary(prg, args.lds_limit_bytes)
  marker_reports = [_markers("pre_schedule", pre_uops), _markers("scheduled_pre_resolve", scheduled_uops), _markers("final", final_uops)]
  dbuf_d3a_audit = _dbuf_d3a_audit_summary(marker_reports, ops["ds_store_b128"])
  dbuf_d3a_compile_audit = _dbuf_d3a_compile_audit_summary(list(amd_isa.DBUF_D3A_AUDIT_LOG))
  report.update({
    "ok": True,
    "program": str(prg.arg),
    "instruction_total": len(mns),
    "instruction_counts": counts,
    "track_counts": {k: len(v) for k, v in ops.items()},
    "track_rows": ops,
    "waitcnt": waits,
    "waitcnt_summary": waitcnt_summary,
    "waitcnt_nonfull": waitcnt_nonfull,
    "markers": marker_reports,
    "branches": _branches(final),
    "b128": {"count": len(ops["global_load_b128"]), "indices": bidx, "rows": ops["global_load_b128"]},
    "wmma": {"count": len(ops[WMMA_NAME]), "indices": widx, "rows": ops[WMMA_NAME]},
    "wmma_chain_trace": wmma_chain_trace(final),
    "global_staging": {
      "global_load_b128": {"count": len(ops["global_load_b128"]), "indices": bidx, "rows": ops["global_load_b128"]},
      "ds_store_b128": {"count": len(ops["ds_store_b128"]), "indices": [x["idx"] for x in ops["ds_store_b128"]], "rows": ops["ds_store_b128"]},
      "ds_store_b64": {"count": len(ops["ds_store_b64"]), "indices": [x["idx"] for x in ops["ds_store_b64"]], "rows": ops["ds_store_b64"]},
      "ds_store_b32": {"count": len(ops["ds_store_b32"]), "indices": [x["idx"] for x in ops["ds_store_b32"]], "rows": ops["ds_store_b32"]},
      "ds_store_b16": {"count": len(ops["ds_store_b16"]), "indices": [x["idx"] for x in ops["ds_store_b16"]], "rows": ops["ds_store_b16"]},
      "ds_load_b128": {"count": len(ops["ds_load_b128"]), "indices": [x["idx"] for x in ops["ds_load_b128"]], "rows": ops["ds_load_b128"]},
      "s_barrier": {"count": len(ops["s_barrier"]), "indices": [x["idx"] for x in ops["s_barrier"]], "rows": ops["s_barrier"]},
      "s_waitcnt": {"count": len(ops["s_waitcnt"]), "indices": [x["idx"] for x in ops["s_waitcnt"]], "rows": ops["s_waitcnt"]},
    },
    "b128_overlap": overlap,
    "wmma_operand_origins": wmma_operand_origins,
    "lds_address_families": lds_address_families,
    "wmma_lds_operand_families": wmma_lds_operand_families,
    "dbuf_gate_summary": dbuf_gate_summary,
    "dbuf_d3a_audit": dbuf_d3a_audit,
    "dbuf_d3a_compile_audit": dbuf_d3a_compile_audit,
    "resource_summary": resource_summary,
  })
  return report


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--target", default="AMD:ISA:gfx1100")
  p.add_argument("--prefill-dbuf", choices=("0", "1"), help="override PREFILL_DBUF for this run")
  p.add_argument("--targeted-waitcnt", choices=("0", "1"), help="override AMD_ISA_WAITCNT_TARGETED for this run")
  p.add_argument("--b128-frag", choices=("0", "1"), help="override AMD_ISA_WMMA_B128_FRAG for this run")
  p.add_argument("--m-up", type=int, default=2, help="number of repeated UPCAST(axis=0,arg=4) opts; default 2 is 4x4")
  p.add_argument("--lds-limit-bytes", type=int, default=65536, help="LDS group-segment limit used for resource classification")
  p.add_argument("--indent", type=int, default=2)
  args = p.parse_args()
  try:
    report = build_report(args)
  except Exception as e:
    report = {"ok": False, "target": args.target, "m_up": args.m_up, "env": {k: os.environ.get(k) for k in ENV_FLAGS},
              "error": type(e).__name__, "message": str(e),
              "dbuf_gate_summary": _failure_dbuf_gate_summary(e),
              "traceback": traceback.format_exc().splitlines()[-12:]}
  print(json.dumps(report, indent=args.indent, sort_keys=True))


if __name__ == "__main__":
  main()
