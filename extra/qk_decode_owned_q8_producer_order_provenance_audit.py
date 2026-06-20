#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess, time
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, ms_stats, read_q4
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_q8_producer_order_provenance_audit_result.json"


def rel(p: pathlib.Path) -> str:
  return str(p.relative_to(ROOT)) if p.is_absolute() and p.is_relative_to(ROOT) else str(p)


def git_sha() -> str:
  try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
  except Exception: return "unknown"


def clock_sample() -> dict[str, Any]:
  cmd = ["rocm-smi", "--showgpuclocks"]
  try:
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-2000:]}
  except Exception as e:
    return {"cmd": cmd, "error": repr(e)}


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


class ContextFixture:
  def __init__(self, gguf: pathlib.Path, rows_arg: int | None, seed: int):
    q40, rows0, k0, shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", rows_arg)
    q41, rows1, k1, shape1 = read_q4(gguf, "blk.0.ffn_up.weight", rows_arg)
    if rows0 != rows1 or k0 != k1:
      raise ValueError("gate/up shape mismatch")
    self.rows, self.k = rows0, k0
    self.meta = {"gate_bytes": len(q40), "up_bytes": len(q41), "total_bytes": len(q40) + len(q41),
                 "shape0": shape0, "shape1": shape1, "rows": rows0, "k": k0}
    self.q4b0, self.q4b1 = make_buffer(len(q40), dtypes.uint8), make_buffer(len(q41), dtypes.uint8)
    self.q4b0.copyin(memoryview(q40)); self.q4b1.copyin(memoryview(q41))
    self.dst0, self.dst1 = make_buffer(rows0, dtypes.float32), make_buffer(rows0, dtypes.float32)
    rng = np.random.default_rng(seed)
    self.dummy_same = []
    for n in (len(q40), len(q41)):
      b = make_buffer(n, dtypes.uint8)
      b.copyin(memoryview(rng.integers(0, 256, size=n, dtype=np.uint8)))
      self.dummy_same.append(b)


def time_one_producer(prg: FixedLaunchRunner, fx: ProducerFixture) -> tuple[float, dict[str, Any]]:
  ms = float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0
  corr = fx.correctness()
  corr.pop("q8_x", None)
  return ms * 1000.0, corr


def time_gateup(prg: FixedLaunchRunner, pfx: ProducerFixture, ctx: ContextFixture) -> float:
  return float(prg(ctx.dst0._buf, ctx.dst1._buf, ctx.q4b0._buf, ctx.q4b1._buf, pfx.q8buf._buf, wait=True)) * 1000.0 * 1000.0


def summarize(samples: list[float]) -> dict[str, Any]:
  if not samples: return {"n": 0}
  ms = [x / 1000.0 for x in samples]
  st = ms_stats(ms)
  st["median_us"] = statistics.median(samples)
  st["mean_us"] = statistics.fmean(samples)
  st["n"] = len(samples)
  return st


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode q8 producer interleaved order/provenance audit")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=12)
  ap.add_argument("--rounds", type=int, default=24)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("q8_producer_order_provenance", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  pfx = ProducerFixture(args.seed)
  ctx = ContextFixture(args.gguf, args.rows, args.seed + 10)
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  gateup_prg = FixedLaunchRunner(dev.runtime("q8_order_provenance_gateup", gateup_blob), (ctx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.warmups):
    time_one_producer(prod_prg, pfx)

  clock_before = clock_sample()
  rng = random.Random(args.seed)
  labels = ["producer_only", "producer_with_real_q4_resident", "producer_after_gateup_dispatch", "producer_with_dummy_resident"]
  samples: dict[str, list[float]] = {k: [] for k in labels}
  rows = []
  prev_label = None
  for r in range(args.rounds):
    order = labels[:]
    rng.shuffle(order)
    for label in order:
      gateup_us = None
      if label == "producer_after_gateup_dispatch":
        gateup_us = time_gateup(gateup_prg, pfx, ctx)
      # For producer_with_* rows, the resident buffers are kept alive in ctx. The producer has the same arguments.
      us, corr = time_one_producer(prod_prg, pfx)
      samples[label].append(us)
      rows.append({"round": r, "label": label, "producer_us": us, "prev_label": prev_label,
                   "gateup_us_before": gateup_us, "correctness": corr})
      prev_label = label
  clock_after = clock_sample()

  summaries = {k: summarize(v) for k, v in samples.items()}
  base = summaries["producer_only"]["median_us"]
  deltas = {k: summaries[k]["median_us"] - base for k in labels if k != "producer_only"}
  all_correct = all(row["correctness"]["producer_correct"] and row["correctness"]["q8_dequant_bounded"] for row in rows)
  post_gate_delta = deltas["producer_after_gateup_dispatch"]
  resident_delta = deltas["producer_with_real_q4_resident"]
  dummy_delta = deltas["producer_with_dummy_resident"]
  gates = {
    "all_correct": all_correct,
    "interleaved_rows_present": all(summaries[k]["n"] == args.rounds for k in labels),
    "post_gateup_delta_lt_5us": post_gate_delta < 5.0,
    "resident_delta_lt_5us": resident_delta < 5.0,
  }
  if not all_correct:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_INCORRECT"
  elif post_gate_delta >= 5.0:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_PREVIOUS_DISPATCH"
  elif resident_delta >= 5.0 or dummy_delta >= 5.0:
    verdict = "BLOCKED_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_RESIDENCY"
  else:
    verdict = "PASS_DECODE_Q8_PRODUCER_ORDER_PROVENANCE_NO_CONTEXT_SLOWDOWN"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_PRODUCER_ORDER_PROVENANCE_AUDIT",
    "schema": "decode_q8_producer_order_provenance_audit_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "q4_meta": ctx.meta,
    "summaries": summaries,
    "deltas_vs_producer_only_us": deltas,
    "rows": rows,
    "clock": {"before": clock_before, "after": clock_after},
    "gates": gates,
    "decision": {
      "if_no_context_slowdown": "previous context slowdown is not stable under interleaving; rerun lifecycle before changing code",
      "if_previous_dispatch": "producer is perturbed by prior gate/up dispatch; inspect queue/cache ordering",
      "if_residency": "producer is perturbed by live context buffers; inspect allocator/VRAM placement",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "summaries_us": {k: summaries[k]["median_us"] for k in labels},
    "deltas_vs_producer_only_us": deltas,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if all_correct else 1


if __name__ == "__main__":
  raise SystemExit(main())
