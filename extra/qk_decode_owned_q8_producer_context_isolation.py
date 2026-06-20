#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, statistics, subprocess
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, ms_stats, read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_producer_context_isolation_result.json"


def load(rel: str, default: Any = None) -> Any:
  p = ROOT / rel
  return json.loads(p.read_text()) if p.exists() else default


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
      "q8_x": q8_x,
      "producer_fp_max_abs": float(fp_err.max()),
      "q8_dequant_max_abs": float(q8_err.max()),
      "producer_correct": float(fp_err.max()) <= 1e-5,
      "q8_dequant_bounded": float(q8_err.max()) <= 0.02,
    }


class GateUpFixture:
  def __init__(self, gguf: pathlib.Path, rows_arg: int | None):
    self.q40, self.rows, self.k, self.shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", rows_arg)
    self.q41, rows1, k1, self.shape1 = read_q4(gguf, "blk.0.ffn_up.weight", rows_arg)
    if self.rows != rows1 or self.k != k1:
      raise ValueError("gate/up shape mismatch")
    self.q4b0, self.q4b1 = make_buffer(len(self.q40), dtypes.uint8), make_buffer(len(self.q41), dtypes.uint8)
    self.dst0, self.dst1 = make_buffer(self.rows, dtypes.float32), make_buffer(self.rows, dtypes.float32)
    self.q4b0.copyin(memoryview(self.q40))
    self.q4b1.copyin(memoryview(self.q41))

  def correctness(self, q8_x: np.ndarray) -> dict[str, Any]:
    ref0 = q4_ref_rows(self.q40, self.rows, self.k, q8_x)
    ref1 = q4_ref_rows(self.q41, self.rows, self.k, q8_x)
    got0 = copyout_array(self.dst0, np.empty(self.rows, dtype=np.float32))
    got1 = copyout_array(self.dst1, np.empty(self.rows, dtype=np.float32))
    err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
    return {
      "gate_max_abs": float(err0.max()),
      "up_max_abs": float(err1.max()),
      "gate_correct": float(err0.max()) <= 2e-3,
      "up_correct": float(err1.max()) <= 2e-3,
    }


def time_producer(prg: FixedLaunchRunner, fx: ProducerFixture, warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    ms = float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0
    if i >= warmups:
      samples.append(ms)
  corr = fx.correctness()
  q8_x = corr.pop("q8_x")
  return {"median_us": statistics.median(samples) * 1000.0, "timing": ms_stats(samples), "correctness": corr, "q8_x": q8_x}


def time_gateup(prg: FixedLaunchRunner, pfx: ProducerFixture, gfx: GateUpFixture, q8_x: np.ndarray,
                warmups: int, iters: int) -> dict[str, Any]:
  samples: list[float] = []
  for i in range(warmups + iters):
    ms = float(prg(gfx.dst0._buf, gfx.dst1._buf, gfx.q4b0._buf, gfx.q4b1._buf, pfx.q8buf._buf, wait=True)) * 1000.0
    if i >= warmups:
      samples.append(ms)
  return {"median_us": statistics.median(samples) * 1000.0, "timing": ms_stats(samples), "correctness": gfx.correctness(q8_x)}


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode owned q8 producer context isolation")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  lifecycle = load("bench/qk-decode-primitive-transfer/decode_lifecycle_cross_apply_gate_result.json", {})
  target_lifecycle_us = float(lifecycle.get("target_lifecycle_us") or 115.24)
  target_producer_us = float(lifecycle.get("producer_lifecycle_us") or 30.54)

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("owned_q8_context_iso_producer", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  pfx = ProducerFixture(args.seed)

  producer_only = time_producer(prod_prg, pfx, args.warmups, args.iters)
  q8_x = producer_only.pop("q8_x")

  gfx = GateUpFixture(args.gguf, args.rows)
  producer_after_q4_buffers = time_producer(prod_prg, pfx, args.warmups, args.iters)
  q8_x = producer_after_q4_buffers.pop("q8_x")

  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  gateup_prg = FixedLaunchRunner(dev.runtime("owned_q8_context_iso_gateup", gateup_blob), (gfx.rows, 2, 1), (32, 4, 1))
  producer_after_gateup_load = time_producer(prod_prg, pfx, args.warmups, args.iters)
  q8_x = producer_after_gateup_load.pop("q8_x")

  gateup_after_producer = time_gateup(gateup_prg, pfx, gfx, q8_x, args.warmups, args.iters)
  producer_after_gateup_exec = time_producer(prod_prg, pfx, args.warmups, args.iters)
  q8_x = producer_after_gateup_exec.pop("q8_x")
  gateup_second = time_gateup(gateup_prg, pfx, gfx, q8_x, args.warmups, args.iters)
  producer_after_second_gateup_exec = time_producer(prod_prg, pfx, args.warmups, args.iters)
  producer_after_second_gateup_exec.pop("q8_x")

  rows = {
    "producer_only": producer_only,
    "producer_after_q4_buffers": producer_after_q4_buffers,
    "producer_after_gateup_program_load": producer_after_gateup_load,
    "gateup_after_producer": gateup_after_producer,
    "producer_after_gateup_execution": producer_after_gateup_exec,
    "gateup_second": gateup_second,
    "producer_after_second_gateup_execution": producer_after_second_gateup_exec,
  }
  for row in rows.values():
    if "correctness" in row and "producer_correct" in row["correctness"]:
      row["correctness"].pop("q8_x", None)

  lifecycle_us = producer_after_gateup_load["median_us"] + gateup_after_producer["median_us"]
  deltas = {
    "q4_buffer_delta_us": producer_after_q4_buffers["median_us"] - producer_only["median_us"],
    "program_load_delta_us": producer_after_gateup_load["median_us"] - producer_after_q4_buffers["median_us"],
    "post_gateup_exec_delta_us": producer_after_gateup_exec["median_us"] - producer_after_gateup_load["median_us"],
    "post_second_gateup_exec_delta_us": producer_after_second_gateup_exec["median_us"] - producer_after_gateup_exec["median_us"],
    "controlled_lifecycle_us": lifecycle_us,
    "target_lifecycle_us": target_lifecycle_us,
    "controlled_lifecycle_gap_us": lifecycle_us - target_lifecycle_us,
  }
  gates = {
    "producer_only_correct": producer_only["correctness"]["producer_correct"],
    "producer_after_q4_correct": producer_after_q4_buffers["correctness"]["producer_correct"],
    "producer_after_load_correct": producer_after_gateup_load["correctness"]["producer_correct"],
    "producer_after_exec_correct": producer_after_gateup_exec["correctness"]["producer_correct"],
    "gateup_correct": gateup_after_producer["correctness"]["gate_correct"] and gateup_after_producer["correctness"]["up_correct"],
    "producer_context_lte_cross_apply": producer_after_gateup_load["median_us"] <= target_producer_us,
    "controlled_lifecycle_lte_target": lifecycle_us <= target_lifecycle_us,
  }
  if not all(v for k, v in gates.items() if not k.endswith("_lte_target") and k != "producer_context_lte_cross_apply"):
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_INCORRECT"
  elif gates["controlled_lifecycle_lte_target"]:
    verdict = "PASS_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_LIFECYCLE_READY"
  elif gates["producer_context_lte_cross_apply"]:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_CONSUMER_OR_SUM_GAP"
  else:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_CONTEXT_ISOLATION_PRODUCER_CONTEXT_SLOW"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_PRODUCER_CONTEXT_ISOLATION",
    "schema": "decode_q8_producer_context_isolation_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "shape": {"rows": gfx.rows, "k": gfx.k, "producer_n": 4096},
    "rows": rows,
    "deltas": deltas,
    "gates": gates,
    "decision": {
      "if_lifecycle_ready": "rerun lifecycle promotion with this controlled ordering",
      "if_producer_context_slow": "producer cost is already context-sensitive; next work should inspect HCQ/cache/resource state, not consumer schedule",
      "if_consumer_or_sum_gap": "producer is no longer the blocker; inspect lifecycle sum/consumer variance",
      "if_incorrect": "fix harness correctness before using timing",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "producer_only_us": producer_only["median_us"],
    "producer_after_q4_buffers_us": producer_after_q4_buffers["median_us"],
    "producer_after_gateup_program_load_us": producer_after_gateup_load["median_us"],
    "producer_after_gateup_execution_us": producer_after_gateup_exec["median_us"],
    "gateup_after_producer_us": gateup_after_producer["median_us"],
    "controlled_lifecycle_us": lifecycle_us,
    "target_lifecycle_us": target_lifecycle_us,
    "deltas": deltas,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if all(v for k, v in gates.items() if not k.endswith("_lte_target") and k != "producer_context_lte_cross_apply") else 1


if __name__ == "__main__":
  raise SystemExit(main())
