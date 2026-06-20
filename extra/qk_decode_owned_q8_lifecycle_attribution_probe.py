#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
from types import SimpleNamespace
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, ms_stats, perf_gateup, read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_attribution_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


class Fixture:
  def __init__(self, gguf: pathlib.Path, rows_arg: int | None, seed: int):
    rng = np.random.default_rng(seed)
    self.x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
    self.w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
    rinv = np.float32(1.0 / np.sqrt(np.sum(self.x * self.x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
    self.ref_norm = (self.x * rinv * self.w).astype(np.float32)

    self.xbuf, self.wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
    self.norm_out, self.q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
    copyin_array(self.xbuf, self.x)
    copyin_array(self.wbuf, self.w)

    self.q40, self.rows, self.k, self.shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", rows_arg)
    self.q41, rows1, k1, self.shape1 = read_q4(gguf, "blk.0.ffn_up.weight", rows_arg)
    if self.rows != rows1 or self.k != k1:
      raise ValueError("gate/up shape mismatch")
    self.q4b0, self.q4b1 = make_buffer(len(self.q40), dtypes.uint8), make_buffer(len(self.q41), dtypes.uint8)
    self.dst0, self.dst1 = make_buffer(self.rows, dtypes.float32), make_buffer(self.rows, dtypes.float32)
    self.q4b0.copyin(memoryview(self.q40))
    self.q4b1.copyin(memoryview(self.q41))

  def producer_correctness(self) -> dict[str, Any]:
    got_norm = copyout_array(self.norm_out, np.empty(4096, dtype=np.float32))
    got_q8 = bytearray(128 * 36)
    self.q8buf.copyout(memoryview(got_q8))
    q8_x = q8_dequant(bytes(got_q8), 4096)
    fp_err, q8_err = np.abs(got_norm - self.ref_norm), np.abs(q8_x - self.ref_norm)
    return {
      "q8_x": q8_x,
      "producer_fp_max_abs": float(fp_err.max()),
      "q8_dequant_max_abs": float(q8_err.max()),
      "producer_correct": float(fp_err.max()) <= 1e-5,
      "q8_dequant_bounded": float(q8_err.max()) <= 0.02,
    }

  def consumer_correctness(self, q8_x: np.ndarray) -> dict[str, Any]:
    ref0, ref1 = q4_ref_rows(self.q40, self.rows, self.k, q8_x), q4_ref_rows(self.q41, self.rows, self.k, q8_x)
    got0 = copyout_array(self.dst0, np.empty(self.rows, dtype=np.float32))
    got1 = copyout_array(self.dst1, np.empty(self.rows, dtype=np.float32))
    err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
    return {
      "gate_max_abs": float(err0.max()),
      "gate_mean_abs": float(err0.mean()),
      "up_max_abs": float(err1.max()),
      "up_mean_abs": float(err1.mean()),
      "gate_correct": float(err0.max()) <= 2e-3,
      "up_correct": float(err1.max()) <= 2e-3,
    }


def time_producer(prg: FixedLaunchRunner, fx: Fixture, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    ms = float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0
    if i >= warmups:
      samples.append(ms)
  corr = fx.producer_correctness()
  q8_x = corr.pop("q8_x")
  return {"timing": ms_stats(samples), "median_us": statistics.median(samples) * 1000.0, "q8_x": q8_x, "correctness": corr}


def time_gateup(prg: FixedLaunchRunner, fx: Fixture, q8_x: np.ndarray, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    ms = float(prg(fx.dst0._buf, fx.dst1._buf, fx.q4b0._buf, fx.q4b1._buf, fx.q8buf._buf, wait=True)) * 1000.0
    if i >= warmups:
      samples.append(ms)
  return {
    "timing": ms_stats(samples),
    "median_us": statistics.median(samples) * 1000.0,
    "correctness": fx.consumer_correctness(q8_x),
  }


def main() -> int:
  ap = argparse.ArgumentParser(description="Owned q8 lifecycle attribution timing ladder")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  scope = load("bench/qk-decode-primitive-transfer/decode_owned_q8_lifecycle_attribution_scope_result.json", {})
  mixed = load("bench/qk-decode-primitive-transfer/decode_owned_q8_mixed_lifecycle_result.json", {})
  target_us = float(mixed.get("target_lifecycle_us") or 115.24)

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("owned_q8_lifecycle_attr_producer", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  fx = Fixture(args.gguf, args.rows, args.seed)

  producer_before_gateup_load = time_producer(prod_prg, fx, args.warmups, args.iters)
  q8_x = producer_before_gateup_load.pop("q8_x")

  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  gateup_prg = FixedLaunchRunner(dev.runtime("owned_q8_lifecycle_attr_gateup", gateup_blob), (args.rows, 2, 1), (32, 4, 1))

  producer_after_gateup_load = time_producer(prod_prg, fx, args.warmups, args.iters)
  q8_x = producer_after_gateup_load.pop("q8_x")
  consumer_after_owned_producer = time_gateup(gateup_prg, fx, q8_x, args.warmups, args.iters)
  producer_after_consumer = time_producer(prod_prg, fx, args.warmups, args.iters)
  q8_x = producer_after_consumer.pop("q8_x")
  consumer_after_second_producer = time_gateup(gateup_prg, fx, q8_x, args.warmups, args.iters)

  perf_args = SimpleNamespace(gguf=args.gguf, rows=args.rows, seed=args.seed, warmups=args.warmups, iters=args.iters, producer_threads=256)
  mixed_perf = perf_gateup(perf_args, prod_prg, gateup_prg)

  controlled_lifecycle_us = producer_after_gateup_load["median_us"] + consumer_after_owned_producer["median_us"]
  second_lifecycle_us = producer_after_consumer["median_us"] + consumer_after_second_producer["median_us"]
  producer_inflation_us = producer_after_gateup_load["median_us"] - producer_before_gateup_load["median_us"]
  consumer_vs_prior_mixed_us = consumer_after_owned_producer["median_us"] - ((mixed.get("perf_gateup") or {}).get("gateup_consumer") or {}).get("median_ms", 0.0) * 1000.0

  gates = {
    "scope_ready": scope.get("gate_pass") is True,
    "producer_before_correct": producer_before_gateup_load["correctness"]["producer_correct"],
    "producer_after_correct": producer_after_gateup_load["correctness"]["producer_correct"],
    "consumer_after_correct": consumer_after_owned_producer["correctness"]["gate_correct"] and consumer_after_owned_producer["correctness"]["up_correct"],
    "mixed_correct": mixed_perf["gates"].get("producer_correct") and mixed_perf["gates"].get("gate_correct") and mixed_perf["gates"].get("up_correct"),
  }
  interpretation = {
    "producer_load_perturbation_us": producer_inflation_us,
    "controlled_lifecycle_us": controlled_lifecycle_us,
    "second_controlled_lifecycle_us": second_lifecycle_us,
    "mixed_perf_gateup_lifecycle_us": mixed_perf["gate_up_lifecycle_us"],
    "target_lifecycle_us": target_us,
    "consumer_delta_vs_prior_mixed_us": consumer_vs_prior_mixed_us,
  }
  if not all(gates.values()):
    verdict = "BLOCKED_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_INCORRECT"
  elif controlled_lifecycle_us <= target_us:
    verdict = "PASS_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_COMPOSITION_OK"
  else:
    verdict = "BLOCKED_DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION_COMPOSITION_SLOW"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_OWNED_Q8_LIFECYCLE_ATTRIBUTION",
    "schema": "decode_owned_q8_lifecycle_attribution_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "rows": {
      "producer_before_gateup_load": producer_before_gateup_load,
      "producer_after_gateup_load": producer_after_gateup_load,
      "consumer_after_owned_producer": consumer_after_owned_producer,
      "producer_after_consumer": producer_after_consumer,
      "consumer_after_second_producer": consumer_after_second_producer,
      "perf_gateup_full_mixed": mixed_perf,
    },
    "interpretation": interpretation,
    "gates": gates,
    "decision": {
      "if_composition_ok": "The previous mixed failure is harness/order sensitive; rerun promotion gate with controlled ordering.",
      "if_composition_slow": "The lifecycle gap is real in-process composition, so next scope should isolate runtime/cache/queue effects or build fully owned consumer.",
      "if_incorrect": "Fix attribution harness correctness before using timing.",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "producer_before_gateup_load_us": producer_before_gateup_load["median_us"],
    "producer_after_gateup_load_us": producer_after_gateup_load["median_us"],
    "consumer_after_owned_producer_us": consumer_after_owned_producer["median_us"],
    "controlled_lifecycle_us": controlled_lifecycle_us,
    "perf_gateup_lifecycle_us": mixed_perf["gate_up_lifecycle_us"],
    "target_lifecycle_us": target_us,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if all(gates.values()) else 1


if __name__ == "__main__":
  raise SystemExit(main())
