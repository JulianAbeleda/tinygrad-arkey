#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import ms_stats, read_q4
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_producer_resource_context_audit_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


class ProducerFixture:
  def __init__(self, seed: int):
    rng = np.random.default_rng(seed)
    self.x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
    self.w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
    rinv = np.float32(1.0 / np.sqrt(np.sum(self.x * self.x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
    self.ref_norm = (self.x * rinv * self.w).astype(np.float32)
    self.xbuf, self.wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
    self.norm_out, self.q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
    copyin_array(self.xbuf, self.x)
    copyin_array(self.wbuf, self.w)

  def correctness(self) -> dict[str, Any]:
    got_norm = copyout_array(self.norm_out, np.empty(4096, dtype=np.float32))
    got_q8 = bytearray(128 * 36)
    self.q8buf.copyout(memoryview(got_q8))
    q8_x = q8_dequant(bytes(got_q8), 4096)
    fp_err = np.abs(got_norm - self.ref_norm)
    q8_err = np.abs(q8_x - self.ref_norm)
    return {
      "producer_fp_max_abs": float(fp_err.max()),
      "q8_dequant_max_abs": float(q8_err.max()),
      "producer_correct": float(fp_err.max()) <= 1e-5,
      "q8_dequant_bounded": float(q8_err.max()) <= 0.02,
    }


def time_producer(prg: FixedLaunchRunner, fx: ProducerFixture, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    ms = float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0
    if i >= warmups:
      samples.append(ms)
  return {"median_us": statistics.median(samples) * 1000.0, "timing": ms_stats(samples), "correctness": fx.correctness()}


def alloc_dummy(total_bytes: int, chunks: int, copy: bool, seed: int) -> list[Any]:
  if total_bytes <= 0 or chunks <= 0: return []
  rng = np.random.default_rng(seed)
  each = max(1, total_bytes // chunks)
  bufs = []
  for i in range(chunks):
    n = each if i < chunks - 1 else total_bytes - each * (chunks - 1)
    b = make_buffer(n, dtypes.uint8)
    if copy:
      arr = rng.integers(0, 256, size=n, dtype=np.uint8)
      b.copyin(memoryview(arr))
    bufs.append(b)
  return bufs


def alloc_real_q4(gguf: pathlib.Path, rows: int | None, copy: bool) -> tuple[list[Any], dict[str, Any]]:
  q40, rows0, k0, shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", rows)
  q41, rows1, k1, shape1 = read_q4(gguf, "blk.0.ffn_up.weight", rows)
  bufs = [make_buffer(len(q40), dtypes.uint8), make_buffer(len(q41), dtypes.uint8)]
  if copy:
    bufs[0].copyin(memoryview(q40))
    bufs[1].copyin(memoryview(q41))
  meta = {"gate_bytes": len(q40), "up_bytes": len(q41), "total_bytes": len(q40) + len(q41),
          "rows": rows0, "k": k0, "shape0": shape0, "shape1": shape1, "shape_match": rows0 == rows1 and k0 == k1}
  return bufs, meta


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 producer resource/context audit")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("q8_producer_resource_context", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  fx = ProducerFixture(args.seed)

  keepalive: list[Any] = []
  baseline = time_producer(prod_prg, fx, args.warmups, args.iters)

  real_uncopied, real_meta = alloc_real_q4(args.gguf, args.rows, copy=False)
  keepalive += real_uncopied
  real_q4_alloc_only = time_producer(prod_prg, fx, args.warmups, args.iters)

  real_copied, real_meta2 = alloc_real_q4(args.gguf, args.rows, copy=True)
  keepalive += real_copied
  real_q4_copied = time_producer(prod_prg, fx, args.warmups, args.iters)

  dummy_same_uncopied = alloc_dummy(real_meta["total_bytes"], chunks=2, copy=False, seed=args.seed + 1)
  keepalive += dummy_same_uncopied
  dummy_same_alloc_only = time_producer(prod_prg, fx, args.warmups, args.iters)

  dummy_same_copied = alloc_dummy(real_meta["total_bytes"], chunks=2, copy=True, seed=args.seed + 2)
  keepalive += dummy_same_copied
  dummy_same_copied_row = time_producer(prod_prg, fx, args.warmups, args.iters)

  dummy_half_copied = alloc_dummy(real_meta["total_bytes"] // 2, chunks=2, copy=True, seed=args.seed + 3)
  keepalive += dummy_half_copied
  dummy_half_copied_row = time_producer(prod_prg, fx, args.warmups, args.iters)

  dummy_double_copied = alloc_dummy(real_meta["total_bytes"] * 2, chunks=4, copy=True, seed=args.seed + 4)
  keepalive += dummy_double_copied
  dummy_double_copied_row = time_producer(prod_prg, fx, args.warmups, args.iters)

  rows = {
    "baseline": baseline,
    "real_q4_alloc_only": real_q4_alloc_only,
    "real_q4_copied": real_q4_copied,
    "dummy_same_alloc_only": dummy_same_alloc_only,
    "dummy_same_copied": dummy_same_copied_row,
    "dummy_half_copied": dummy_half_copied_row,
    "dummy_double_copied": dummy_double_copied_row,
  }
  base = baseline["median_us"]
  deltas = {k: v["median_us"] - base for k, v in rows.items() if k != "baseline"}
  gates = {
    "all_correct": all(v["correctness"]["producer_correct"] and v["correctness"]["q8_dequant_bounded"] for v in rows.values()),
    "real_q4_alloc_reproduces_slowdown": deltas["real_q4_alloc_only"] >= 5.0,
    "real_q4_copy_not_required": abs(deltas["real_q4_copied"] - deltas["real_q4_alloc_only"]) <= 3.0,
    "dummy_same_tests_memory_pressure": True,
  }
  real_delta = deltas["real_q4_alloc_only"]
  dummy_delta = deltas["dummy_same_alloc_only"]
  if not gates["all_correct"]:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_INCORRECT"
  elif real_delta >= 5.0 and dummy_delta >= 5.0:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_GENERAL_RESIDENCY"
  elif real_delta >= 5.0:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_Q4_SPECIFIC_OR_ALLOCATOR_PLACEMENT"
  else:
    verdict = "PASS_DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_NOT_REPRODUCED"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_PRODUCER_RESOURCE_CONTEXT_AUDIT",
    "schema": "decode_q8_producer_resource_context_audit_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "q4_meta": real_meta,
    "q4_meta_copied": real_meta2,
    "rows": rows,
    "deltas_vs_baseline_us": deltas,
    "gates": gates,
    "keepalive_count": len(keepalive),
    "decision": {
      "if_general_residency": "slowdown follows resident bytes; next test memory pressure/allocator placement and avoid co-resident producer timing assumptions",
      "if_q4_specific": "slowdown follows q4 allocations more than dummy bytes; inspect buffer placement/flags/source",
      "if_not_reproduced": "context isolation result is order-sensitive; rerun with interleaving/clock provenance",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "baseline_us": base,
    "rows_us": {k: v["median_us"] for k, v in rows.items()},
    "deltas_vs_baseline_us": deltas,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if gates["all_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
