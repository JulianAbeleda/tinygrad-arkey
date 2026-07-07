#!/usr/bin/env python3
"""JSON structural probe for the native-ISA 4x4 route-shaped WMMA stream."""
from __future__ import annotations

import argparse, json, os, sys, traceback
from collections import Counter
from dataclasses import replace
from typing import Any

sys.path.insert(0, os.getcwd())

from tinygrad import Tensor
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.amd.dsl import Reg
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, UOp


ENV_FLAGS = ("PREFILL_DBUF", "AMD_ISA_WAITCNT_TARGETED", "AMD_ISA_WMMA_B128_FRAG")
REG_FIELDS = ("vdst", "src0", "src1", "src2", "addr", "vaddr", "saddr", "sdst", "sdata", "data0", "data1", "data2", "data3", "vdata")
WMMA_NAME = "v_wmma_f32_16x16x16_f16"
TRACK_NAMES = (
  "global_load_b128",
  "global_load_u16",
  "ds_store_b128",
  "ds_store_b32",
  "ds_store_b16",
  "ds_load_b128",
  "s_barrier",
  "s_waitcnt",
  WMMA_NAME,
)
BROAD_STAGING_NAMES = ("global_load_b128", "ds_store_b128", "ds_load_b128", "ds_store_b32", "ds_store_b16")


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
    if isinstance(arg, tuple): rows.append({"idx": idx, "kind": arg[0], "arg": repr(arg[1:])})
  return {"stream": label, "count": len(rows), "rows": rows}


def _waitcnt_simm16(inst: Any) -> int | None:
  if _mn(inst) != "s_waitcnt": return None
  return getattr(inst, "simm16", getattr(inst, "sim16", None))


def _decode_waitcnt(simm16: int) -> dict[str, int]:
  return {"simm16": simm16, "vmcnt": (simm16 >> 10) & 0x3F, "lgkmcnt": (simm16 >> 4) & 0x3F, "expcnt": simm16 & 0x7}


def _interesting_rows(insts: list[Any], name: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
  rows = []
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple) or _mn(inst) != name: continue
    spans = {k: v for k, v in _field_spans(inst).items() if k in fields}
    rows.append({"idx": idx, "spans": spans, "text": str(inst)})
  return rows


def _range_regs(span: dict[str, int]) -> range:
  return range(span["lo"], span["hi"] + 1)


def _vreg_definers(insts: list[Any]) -> dict[tuple[str, int], int]:
  defs: dict[tuple[str, int], int] = {}
  for idx, inst in enumerate(insts):
    if isinstance(inst, tuple): continue
    for span in (s for f, s in _field_spans(inst).items() if f == "vdst" and s["kind"] == "v"):
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
      "ds_load_b128_count": by_name["ds_load_b128"]["count"],
      "s_barrier_count": by_name["s_barrier"]["count"],
      "s_waitcnt_count": by_name["s_waitcnt"]["count"],
    })
  global_rows = sorted(_span_key(ops["global_load_b128"] + ops["ds_store_b128"] + ops["ds_load_b128"]
                                + ops["ds_store_b32"] + ops["ds_store_b16"]))
  global_windows = []
  for reg in regions:
    if reg["start"] <= reg["end"]:
      global_windows.append({
        "label": reg["label"],
        "indices": [i for i in global_rows if reg["start"] <= i < reg["end"]] if not reg["strict"] else [i for i in global_rows if reg["start"] < i < reg["end"]],
        "count": len([i for i in global_rows if reg["start"] <= i < reg["end"]]) if not reg["strict"] else len([i for i in global_rows if reg["start"] < i < reg["end"]]),
      })
  return {
    "wmma_count": len(widx),
    "wmma_regions": _sorted_regions(region_defs),
    "global_work_regions": global_windows,
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
    "ds_store_b32": ("addr", "data0"),
    "ds_store_b16": ("addr", "data0"),
    "ds_load_b128": ("vdst", "addr", "data0", "data1", "data2", "data3"),
    "s_barrier": tuple(),
    "s_waitcnt": tuple(),
    WMMA_NAME: ("vdst", "src0", "src1", "src2"),
  }
  ops: dict[str, list[dict[str, Any]]] = {}
  for name in TRACK_NAMES:
    ops[name] = _interesting_rows(final, name, op_fields.get(name, tuple()))
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
    "markers": [_markers("pre_schedule", pre_uops), _markers("scheduled_pre_resolve", scheduled_uops), _markers("final", final_uops)],
    "branches": _branches(final),
    "b128": {"count": len(ops["global_load_b128"]), "indices": bidx, "rows": ops["global_load_b128"]},
    "wmma": {"count": len(ops[WMMA_NAME]), "indices": widx, "rows": ops[WMMA_NAME]},
    "global_staging": {
      "global_load_b128": {"count": len(ops["global_load_b128"]), "indices": bidx, "rows": ops["global_load_b128"]},
      "ds_store_b128": {"count": len(ops["ds_store_b128"]), "indices": [x["idx"] for x in ops["ds_store_b128"]], "rows": ops["ds_store_b128"]},
      "ds_store_b32": {"count": len(ops["ds_store_b32"]), "indices": [x["idx"] for x in ops["ds_store_b32"]], "rows": ops["ds_store_b32"]},
      "ds_store_b16": {"count": len(ops["ds_store_b16"]), "indices": [x["idx"] for x in ops["ds_store_b16"]], "rows": ops["ds_store_b16"]},
      "ds_load_b128": {"count": len(ops["ds_load_b128"]), "indices": [x["idx"] for x in ops["ds_load_b128"]], "rows": ops["ds_load_b128"]},
      "s_barrier": {"count": len(ops["s_barrier"]), "indices": [x["idx"] for x in ops["s_barrier"]], "rows": ops["s_barrier"]},
      "s_waitcnt": {"count": len(ops["s_waitcnt"]), "indices": [x["idx"] for x in ops["s_waitcnt"]], "rows": ops["s_waitcnt"]},
    },
    "b128_overlap": overlap,
    "wmma_operand_origins": wmma_operand_origins,
  })
  return report


def main() -> None:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--target", default="AMD:ISA:gfx1100")
  p.add_argument("--prefill-dbuf", choices=("0", "1"), help="override PREFILL_DBUF for this run")
  p.add_argument("--targeted-waitcnt", choices=("0", "1"), help="override AMD_ISA_WAITCNT_TARGETED for this run")
  p.add_argument("--b128-frag", choices=("0", "1"), help="override AMD_ISA_WMMA_B128_FRAG for this run")
  p.add_argument("--m-up", type=int, default=2, help="number of repeated UPCAST(axis=0,arg=4) opts; default 2 is 4x4")
  p.add_argument("--indent", type=int, default=2)
  args = p.parse_args()
  try:
    report = build_report(args)
  except Exception as e:
    report = {"ok": False, "target": args.target, "m_up": args.m_up, "env": {k: os.environ.get(k) for k in ENV_FLAGS},
              "error": type(e).__name__, "message": str(e),
              "traceback": traceback.format_exc().splitlines()[-12:]}
  print(json.dumps(report, indent=args.indent, sort_keys=True))


if __name__ == "__main__":
  main()
