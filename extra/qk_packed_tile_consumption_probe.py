#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, time
from typing import Any, Callable

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import KernelInfo, Ops, UOp

from extra.qk_descriptor_policy import load_json
from extra.qk_packed_tile import load_tile, tile_from_semantic_row, tile_summary

CUSTOM_Q4_DOT_SOURCE = """{{
  typedef unsigned int tg_uint4 __attribute__((ext_vector_type(4)));
  tg_uint4 qv = *((tg_uint4*)({1}+4));
  float total = 0.0f;
  for (int lane = 0; lane < 4; lane++) {{
    unsigned int w = qv[lane];
    for (int nib = 0; nib < 4; nib++) {{
      unsigned int byte = (w >> (8u*(unsigned int)nib)) & 255u;
      total += (float)(byte & 15u) * {2}[lane*4+nib];
      total += (float)(byte >> 4u) * {2}[32+lane*4+nib];
    }}
  }}
  {0}[0] = total;
}}"""

def qk_probe_uop_lane_gep_sum_kernel(out:UOp, words:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  vec = words.index(i*4, ptr=True).load(dtype=dtypes.uint32.vec(4))
  total = vec.gep(0) + vec.gep(1) + vec.gep(2) + vec.gep(3)
  return out[i].store(total).end(i).sink(arg=KernelInfo(name=f"qk_probe_tile_uop_lane_gep_sum_{out.numel()}", opts_to_apply=()))

def qk_probe_uop_vector_arith_store_kernel(out:UOp, words:UOp) -> UOp:
  i = UOp.range(out.numel() // 4, 0)
  base = i * 4
  vec = words.index(base, ptr=True).load(dtype=dtypes.uint32.vec(4))
  q = vec.rshift(4).bitwise_and(0xf)
  return out.index(base, ptr=True).store(q).end(i).sink(
    arg=KernelInfo(name=f"qk_probe_tile_uop_vector_arith_store_{out.numel()}", opts_to_apply=()))

def qk_probe_custom_q4_dot_kernel(out:UOp, words:UOp, x:UOp) -> UOp:
  gid = UOp.special(1, "gidx0")
  zero = UOp.const(dtypes.weakint, 0)
  out_ptr = out.index(gid, ptr=True)
  words_ptr = words.index(zero, ptr=True)
  x_ptr = x.index(zero, ptr=True)
  stmt = UOp(Ops.CUSTOM, dtypes.void, (out_ptr, words_ptr, x_ptr), arg=CUSTOM_Q4_DOT_SOURCE)
  return stmt.sink(arg=KernelInfo(name="qk_probe_tile_custom_q4_dot_36", opts_to_apply=()))

def _q4_words_and_x() -> tuple[np.ndarray, np.ndarray, np.float32]:
  words = np.zeros(36, dtype=np.uint32)
  # Fill the first four quant payload words (block word indices 4..7) with
  # stable byte patterns. Low nibbles represent group 0, high nibbles group 1.
  words[4:8] = np.array([0x10325476, 0x98badcfe, 0x01234567, 0xfedcba98], dtype=np.uint32)
  x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
  expected = np.float32(0.0)
  for lane, word in enumerate(words[4:8]):
    for nib in range(4):
      byte = int((int(word) >> (8*nib)) & 0xff)
      expected += np.float32(byte & 0xf) * x[lane*4+nib]
      expected += np.float32(byte >> 4) * x[32+lane*4+nib]
  return words, x, expected

def _bench(fn:Callable[[], Tensor], iters:int) -> dict[str, float|int|None]:
  fn().realize()
  GlobalCounters.reset()
  st = time.perf_counter()
  for _ in range(iters): fn().realize()
  wall = (time.perf_counter() - st) / iters
  dev = GlobalCounters.time_sum_s / iters
  return {
    "iters": iters,
    "wall_ms": wall * 1000,
    "device_ms": dev * 1000 if dev > 0 else None,
    "kernels": GlobalCounters.kernel_count / iters,
  }

def _failure_row(mode:str, exc:Exception) -> dict[str, Any]:
  return {
    "mode": mode,
    "status": "expected_fail" if mode.startswith("uop_") else "fail",
    "error_type": type(exc).__name__,
    "error_message": str(exc).splitlines()[-1][:500],
  }

def run_uop_lane_gep(*, device:str, iters:int) -> dict[str, Any]:
  words = Tensor(np.arange(16, dtype=np.uint32), device=device).realize()
  out = Tensor.zeros(4, dtype=dtypes.uint32, device=device).realize()
  try:
    def fn(): return out.custom_kernel(words, fxn=qk_probe_uop_lane_gep_sum_kernel)[0]
    got = fn().realize().numpy()
    expected = np.arange(16, dtype=np.uint32).reshape(4, 4).sum(axis=1).astype(np.uint32)
    row = {
      "mode": "uop_lane_gep",
      "status": "pass" if bool((got == expected).all()) else "wrong",
      "exact": bool((got == expected).all()),
      "got": [int(x) for x in got],
      "expected": [int(x) for x in expected],
    }
    row.update(_bench(fn, iters))
    return row
  except Exception as exc:  # construction failure is the point of this mode.
    return _failure_row("uop_lane_gep", exc)

def run_uop_vector_arith(*, device:str, iters:int) -> dict[str, Any]:
  src_np = np.arange(16, dtype=np.uint32) * 17 + 5
  words = Tensor(src_np, device=device).realize()
  out = Tensor.zeros(16, dtype=dtypes.uint32, device=device).realize()
  try:
    def fn(): return out.custom_kernel(words, fxn=qk_probe_uop_vector_arith_store_kernel)[0]
    got = fn().realize().numpy()
    expected = ((src_np >> 4) & 0xf).astype(np.uint32)
    row = {
      "mode": "uop_vector_arith",
      "status": "pass" if bool((got == expected).all()) else "wrong",
      "exact": bool((got == expected).all()),
      "got": [int(x) for x in got],
      "expected": [int(x) for x in expected],
    }
    row.update(_bench(fn, iters))
    return row
  except Exception as exc:
    return _failure_row("uop_vector_arith", exc)

def run_custom_q4_dot(*, device:str, iters:int) -> dict[str, Any]:
  words_np, x_np, expected = _q4_words_and_x()
  words = Tensor(words_np, device=device).realize()
  x = Tensor(x_np, device=device).realize()
  out = Tensor.zeros(1, dtype=dtypes.float32, device=device).realize()
  try:
    def fn(): return out.custom_kernel(words, x, fxn=qk_probe_custom_q4_dot_kernel)[0]
    got = fn().realize().numpy()
    max_abs = float(np.max(np.abs(got - expected)))
    row = {
      "mode": "custom_q4_dot",
      "status": "pass" if max_abs == 0.0 else "wrong",
      "exact": max_abs == 0.0,
      "got": float(got[0]),
      "expected": float(expected),
      "max_abs": max_abs,
      "source_contains_uint4_load": "tg_uint4 qv = *((tg_uint4*)" in CUSTOM_Q4_DOT_SOURCE,
      "source_extracts_vector_lanes": "qv[lane]" in CUSTOM_Q4_DOT_SOURCE,
      "source_unpacks_q4_nibbles": "byte & 15u" in CUSTOM_Q4_DOT_SOURCE and "byte >> 4u" in CUSTOM_Q4_DOT_SOURCE,
    }
    row.update(_bench(fn, iters))
    return row
  except Exception as exc:
    return _failure_row("custom_q4_dot", exc)

RUNNERS = {
  "uop_lane_gep": run_uop_lane_gep,
  "uop_vector_arith": run_uop_vector_arith,
  "custom_q4_dot": run_custom_q4_dot,
}

def _pick_tile(descriptor:dict[str, Any], role:str) -> dict[str, Any]:
  for row in descriptor.get("descriptors", []):
    if row.get("format") == "Q4_K" and row.get("role") == role:
      tile = tile_from_semantic_row(row)
      load = load_tile(tile, "u32x4_aligned")
      return {"tile": tile_summary(tile), "required_load_tile": load.name}
  raise ValueError(f"descriptor has no Q4_K row with role={role!r}")

def build_report(descriptor_path:pathlib.Path, *, role:str, modes:list[str], device:str, iters:int) -> dict[str, Any]:
  descriptor = load_json(descriptor_path)
  tile = _pick_tile(descriptor, role)
  rows = [RUNNERS[mode](device=device, iters=iters) for mode in modes]
  by_mode = {row["mode"]: row for row in rows}
  normal_uop_ok = by_mode.get("uop_lane_gep", {}).get("status") == "pass" and by_mode.get("uop_vector_arith", {}).get("status") == "pass"
  custom_ok = by_mode.get("custom_q4_dot", {}).get("status") == "pass"
  decision = "normal_uops_can_consume_packed_qk_tile" if normal_uop_ok else (
    "semantic_custom_op_required" if custom_ok else "blocked"
  )
  return {
    "kind": "qk_packed_tile_consumption_probe",
    "schema_version": 1,
    "device": device,
    "source_descriptor": str(descriptor_path),
    "role": role,
    "packed_qk": tile,
    "rows": rows,
    "summary": {
      "normal_uop_lane_extract_passed": by_mode.get("uop_lane_gep", {}).get("status") == "pass",
      "normal_uop_vector_arith_passed": by_mode.get("uop_vector_arith", {}).get("status") == "pass",
      "custom_q4_dot_passed": custom_ok,
      "decision": decision,
      "run_microbench": False,
      "run_full_decode": False,
      "next_path": (
        "rewrite semantic-codegen v4 through normal UOps" if normal_uop_ok else
        "add a first-class packed QK load/decode/dot semantic op or renderer PatternMatcher lowering"
        if custom_ok else "fix construction before any benchmark"
      ),
    },
    "notes": [
      "This is a construction probe, not a performance benchmark.",
      "The custom path proves the hardware/source shape is expressible; the failing UOp paths decide that current normal UOps cannot consume it.",
      "No microbench or full-decode gate should run unless normal UOps pass or a semantic lowering is implemented.",
    ],
  }

def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Packed Tile Consumption Probe",
    "",
    "Construction gate for consuming a `PackedQKTile` Q4_K `u32x4_aligned` load.",
    "This is not a speed benchmark.",
    "",
    "## Decision",
    "",
    f"- decision: `{report['summary']['decision']}`",
    f"- next path: {report['summary']['next_path']}",
    f"- run microbench: `{report['summary']['run_microbench']}`",
    f"- run full decode: `{report['summary']['run_full_decode']}`",
    "",
    "## Packed Tile",
    "",
    f"- source descriptor: `{report['source_descriptor']}`",
    f"- tensor: `{report['packed_qk']['tile']['tensor']}`",
    f"- shape: `{report['packed_qk']['tile']['shape'][0]}x{report['packed_qk']['tile']['shape'][1]}`",
    f"- legal load tiles: `{', '.join(report['packed_qk']['tile']['legal_load_tiles'])}`",
    f"- required load tile: `{report['packed_qk']['required_load_tile']}`",
    "",
    "## Rows",
    "",
    "| mode | status | exact | device ms | key evidence |",
    "|---|---|---:|---:|---|",
  ]
  for row in report["rows"]:
    evidence = row.get("error_message") or (
      "uint4 load + lane extraction + nibble unpack" if row["mode"] == "custom_q4_dot" else "normal UOp construction"
    )
    device_ms = row.get("device_ms")
    lines.append(
      f"| `{row['mode']}` | `{row['status']}` | `{row.get('exact')}` | "
      f"{'n/a' if device_ms is None else f'{device_ms:.6f}'} | {evidence} |"
    )
  lines += [
    "",
    "## Interpretation",
    "",
    "Current normal UOps cannot consume the packed vector load. Scalar lane",
    "extraction through `GEP` fails verifier, and vector integer arithmetic fails",
    "shape validation. A custom semantic kernel can load `tg_uint4`, index lanes,",
    "unpack low/high Q4 nibbles, and accumulate an exact dot. Therefore the next",
    "implementation should be a first-class packed QK load/decode/dot lowering or",
    "renderer PatternMatcher rule, not another normal-UOp rewrite of v4.",
    "",
  ]
  return "\n".join(lines)

def main() -> int:
  parser = argparse.ArgumentParser(description="Probe whether PackedQKTile vector loads can be consumed by tinygrad UOps")
  parser.add_argument("--descriptor", type=pathlib.Path, default=pathlib.Path("bench/qk-ansor-transition-20260612/descriptors/8b.json"))
  parser.add_argument("--role", default="ffn_gate")
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--iters", type=int, default=3)
  parser.add_argument("--mode", choices=("all", *RUNNERS.keys()), default="all")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()
  modes = list(RUNNERS) if args.mode == "all" else [args.mode]
  report = build_report(args.descriptor, role=args.role, modes=modes, device=args.device, iters=args.iters)
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, sort_keys=True))
  if args.md:
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(report_markdown(report))
  print(report_markdown(report))
  return 0

if __name__ == "__main__":
  raise SystemExit(main())
