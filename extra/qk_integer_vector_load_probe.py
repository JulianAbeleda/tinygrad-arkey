#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, time
from typing import Any

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.helpers import GlobalCounters
from tinygrad.uop.ops import KernelInfo, Ops, UOp


def qk_probe_scalar_copy_kernel(out:UOp, src:UOp) -> UOp:
  i = UOp.range(out.numel(), 0)
  return out[i].store(src[i]).end(i).sink(arg=KernelInfo(name=f"qk_probe_scalar_u32_copy_{out.numel()}", opts_to_apply=()))


def qk_probe_uop_vec_request_copy_kernel(out:UOp, src:UOp) -> UOp:
  i = UOp.range(out.numel() // 4, 0)
  base = i * 4
  vec = src.index(base, ptr=True).load(dtype=dtypes.uint32.vec(4))
  # This intentionally asks the normal UOp path for a uint32x4 load/store. The
  # AMD CStyle path currently lowers it to one scalar lane, which this probe
  # records instead of hiding behind a passing scalar fallback.
  return out.index(base, ptr=True).store(vec).end(i).sink(
    arg=KernelInfo(name=f"qk_probe_uop_vec_request_u32x4_copy_{out.numel()}", opts_to_apply=()))


CUSTOM_UINT4_SOURCE = """{{
  typedef unsigned int tg_uint4 __attribute__((ext_vector_type(4)));
  tg_uint4 v = *((tg_uint4*)({1}));
  *((tg_uint4*)({0})) = v;
}}"""


def qk_probe_custom_uint4_copy_kernel(out:UOp, src:UOp) -> UOp:
  gid = UOp.special(out.numel() // 4, "gidx0")
  out_ptr = out.index(gid * 4, ptr=True)
  src_ptr = src.index(gid * 4, ptr=True)
  stmt = UOp(Ops.CUSTOM, dtypes.void, (out_ptr, src_ptr), arg=CUSTOM_UINT4_SOURCE)
  return stmt.sink(arg=KernelInfo(name=f"qk_probe_custom_uint4_copy_{out.numel()}", opts_to_apply=()))


KERNELS = {
  "scalar": qk_probe_scalar_copy_kernel,
  "uop_vec_request": qk_probe_uop_vec_request_copy_kernel,
  "custom_uint4": qk_probe_custom_uint4_copy_kernel,
}


def _bench(fn, iters:int) -> dict[str, float | int | None]:
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


def run_mode(mode:str, *, n_words:int, iters:int, device:str) -> dict[str, Any]:
  expected = np.arange(n_words, dtype=np.uint32)
  src = Tensor(expected, device=device).realize()
  out = Tensor.zeros(n_words, dtype=dtypes.uint32, device=device).realize()

  def copy():
    return out.custom_kernel(src, fxn=KERNELS[mode])[0]

  got = copy().realize().numpy()
  mismatch = np.nonzero(got != expected)[0]
  first_bad = int(mismatch[0]) if mismatch.size else None
  row = {
    "mode": mode,
    "device": device,
    "n_words": n_words,
    "copy_exact": bool(mismatch.size == 0),
    "mismatch_count": int(mismatch.size),
    "first_mismatch": first_bad,
    "first_values": [int(x) for x in got[:min(16, got.size)]],
    "expected_first_values": [int(x) for x in expected[:min(16, expected.size)]],
  }
  row.update(_bench(copy, iters))
  if mode == "uop_vec_request":
    row["interpretation"] = "Normal UOp uint32.vec(4) load/store request; exact copy proves all four lanes survived lowering."
  elif mode == "custom_uint4":
    row["interpretation"] = "Raw custom C can force a uint4 load/store when it supplies its own vector typedef."
  else:
    row["interpretation"] = "Scalar correctness baseline."
  return row


def build_report(modes:list[str], *, n_words:int, iters:int, device:str) -> dict[str, Any]:
  rows = [run_mode(mode, n_words=n_words, iters=iters, device=device) for mode in modes]
  by_mode = {row["mode"]: row for row in rows}
  return {
    "kind": "qk_integer_vector_load_probe",
    "schema_version": 1,
    "device": device,
    "n_words": n_words,
    "rows": rows,
    "summary": {
      "scalar_exact": bool(by_mode.get("scalar", {}).get("copy_exact")),
      "uop_vec_request_exact": bool(by_mode.get("uop_vec_request", {}).get("copy_exact")),
      "custom_uint4_exact": bool(by_mode.get("custom_uint4", {}).get("copy_exact")),
      "normal_uop_uint4_load_supported": bool(by_mode.get("uop_vec_request", {}).get("copy_exact")),
      "raw_custom_uint4_escape_supported": bool(by_mode.get("custom_uint4", {}).get("copy_exact")),
    },
    "notes": [
      "This is a capability probe, not a Q4_K performance benchmark.",
      "The UOp vector-request path is considered supported only if it copies all four uint32 lanes exactly.",
      "The custom_uint4 mode demonstrates what raw custom C can force; it is not evidence that BEAM/search can emit the load.",
    ],
  }


def report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# QK Integer Vector Load Probe",
    "",
    "Capability probe for integer vector global loads on the AMD path.",
    "",
    "## Summary",
    "",
    f"- device: `{report['device']}`",
    f"- n_words: `{report['n_words']}`",
    f"- normal UOp uint4 load supported: `{report['summary']['normal_uop_uint4_load_supported']}`",
    f"- raw custom uint4 escape supported: `{report['summary']['raw_custom_uint4_escape_supported']}`",
    "",
    "| mode | exact | mismatches | first mismatch | device ms | interpretation |",
    "|---|---:|---:|---:|---:|---|",
  ]
  for row in report["rows"]:
    device_ms = row.get("device_ms")
    lines.append(
      f"| `{row['mode']}` | `{row['copy_exact']}` | {row['mismatch_count']} | "
      f"{'n/a' if row['first_mismatch'] is None else row['first_mismatch']} | "
      f"{'n/a' if device_ms is None else f'{device_ms:.6f}'} | {row['interpretation']} |"
    )
  lines += [
    "",
    "First values:",
    "",
  ]
  for row in report["rows"]:
    lines.append(f"- `{row['mode']}` got `{row['first_values']}` expected `{row['expected_first_values']}`")
  lines.append("")
  return "\n".join(lines)


def main() -> int:
  parser = argparse.ArgumentParser(description="Probe whether tinygrad AMD codegen preserves uint32x4 global loads")
  parser.add_argument("--device", default="AMD")
  parser.add_argument("--n-words", type=int, default=4096)
  parser.add_argument("--iters", type=int, default=5)
  parser.add_argument("--mode", choices=("all", *KERNELS.keys()), default="all")
  parser.add_argument("--json", type=pathlib.Path)
  parser.add_argument("--md", type=pathlib.Path)
  args = parser.parse_args()
  if args.n_words % 4 != 0: raise ValueError("--n-words must be divisible by 4")
  modes = list(KERNELS) if args.mode == "all" else [args.mode]
  report = build_report(modes, n_words=args.n_words, iters=args.iters, device=args.device)
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
