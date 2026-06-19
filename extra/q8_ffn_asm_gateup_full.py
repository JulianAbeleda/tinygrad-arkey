#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, time

import numpy as np

from tinygrad import GlobalCounters, Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.realize import run_linear
from extra.q8_ffn_asm_fullrow_reduce import HIDDEN, Q8_BYTES, build_fullrow_reduce
from extra.q8_ffn_fast_artifact_probe import read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows, q8_blocks
from extra.q8_ffn_hcq_artifact import q8_dequant

def pctile(xs:list[float], p:float) -> float:
  ys = sorted(xs)
  return ys[min(len(ys)-1, max(0, round((len(ys)-1)*p)))]

def stats_ms(xs:list[float]) -> dict:
  return {
    "samples_ms": [round(x, 6) for x in xs],
    "min_ms": round(min(xs), 6),
    "median_ms": round(statistics.median(xs), 6),
    "mean_ms": round(statistics.fmean(xs), 6),
    "p10_ms": round(pctile(xs, 0.10), 6),
    "p90_ms": round(pctile(xs, 0.90), 6),
    "max_ms": round(max(xs), 6),
  }

def main() -> None:
  ap = argparse.ArgumentParser(description="B2b10 real-GGUF full fused gate/up ASM consumer")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=30)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/asm_gateup_full.json"))
  args = ap.parse_args()

  rng = np.random.default_rng(args.seed)
  x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
  q8_host = np.frombuffer(q8_blocks(x), dtype=np.uint8).copy()
  q8_x = q8_dequant(q8_host.tobytes(), 4096)

  q40, rows, k, shape0 = read_q4(args.gguf, "blk.0.ffn_gate.weight", HIDDEN)
  q41, rows1, k1, shape1 = read_q4(args.gguf, "blk.0.ffn_up.weight", HIDDEN)
  if rows != HIDDEN or rows1 != HIDDEN or k != 4096 or k1 != 4096: raise ValueError((rows, rows1, k, k1))
  ref0, ref1 = q4_ref_rows(q40, rows, k, q8_x), q4_ref_rows(q41, rows, k, q8_x)

  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
  gate_words = Tensor(np.frombuffer(q40, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  up_words = Tensor(np.frombuffer(q41, dtype=np.uint32).copy(), dtype=dtypes.uint32, device="AMD").contiguous().realize()
  q8 = Tensor(q8_host, dtype=dtypes.uint8, device="AMD").contiguous().realize()
  gate, up, *_ = Tensor.custom_kernel(gate, up, gate_words, up_words, q8, fxn=build_fullrow_reduce)[:2]
  linear = gate.schedule_linear()

  samples = []
  for i in range(args.warmups + args.iters):
    GlobalCounters.reset()
    t0 = time.perf_counter()
    run_linear(linear)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    device_ms = GlobalCounters.time_sum_s * 1000.0
    if i >= args.warmups: samples.append(device_ms if device_ms > 0 else elapsed_ms)

  got0, got1 = gate.numpy().astype(np.float32), up.numpy().astype(np.float32)
  err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
  result = {
    "date": "2026-06-19",
    "phase": "B2b10_real_gguf_gateup_full",
    "route": "tinygrad_Ops.PROGRAM_AMD_DSL_full_fused_gateup_consumer",
    "shape": {"gate": shape0, "up": shape1},
    "rows": rows,
    "q8_bytes": int(Q8_BYTES),
    "timing": stats_ms(samples),
    "correctness": {
      "gate_max_abs": float(err0.max()),
      "gate_mean_abs": float(err0.mean()),
      "up_max_abs": float(err1.max()),
      "up_mean_abs": float(err1.mean()),
    },
    "gates": {
      "gate_correct_lte_2e_3": float(err0.max()) <= 2e-3,
      "up_correct_lte_2e_3": float(err1.max()) <= 2e-3,
      "consumer_lte_60us": statistics.median(samples) <= 0.060,
      "no_external_artifact": True,
    },
  }
  result["verdict"] = "PASS" if all(result["gates"].values()) else "FAIL"
  result["next"] = "If correctness passes but perf fails, B2b reaches the decode ownership kill gate."
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps({"out": str(args.out), "verdict": result["verdict"], "correctness": result["correctness"],
                    "median_us": result["timing"]["median_ms"] * 1000.0, "gates": result["gates"]}, indent=2))

if __name__ == "__main__":
  main()
