#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, random, statistics, subprocess
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.device import Device
from extra.q8_ffn_artifact_import_route import FixedLaunchRunner
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, ms_stats, read_q4
from extra.q8_ffn_handwritten_oracle import q4_ref_rows
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, copyin_array, copyout_array, make_buffer, q8_dequant


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_owned_q8_interleaved_lifecycle_gate_result.json"


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


class Fixture:
  def __init__(self, gguf: pathlib.Path, rows_arg: int | None, seed: int):
    rng = np.random.default_rng(seed)
    self.x = (rng.standard_normal(4096).astype(np.float32) * 0.9).astype(np.float32)
    self.w = (0.7 + rng.random(4096).astype(np.float32) * 0.2).astype(np.float32)
    rinv = np.float32(1.0 / np.sqrt(np.sum(self.x * self.x, dtype=np.float32) / np.float32(4096.0) + np.float32(1.0e-6)))
    self.ref_norm = (self.x * rinv * self.w).astype(np.float32)
    self.xbuf, self.wbuf = make_buffer(4096, dtypes.float32), make_buffer(4096, dtypes.float32)
    self.norm_out, self.q8buf = make_buffer(4096, dtypes.float32), make_buffer(128 * 36, dtypes.uint8)
    copyin_array(self.xbuf, self.x); copyin_array(self.wbuf, self.w)

    self.q40, self.rows, self.k, self.shape0 = read_q4(gguf, "blk.0.ffn_gate.weight", rows_arg)
    self.q41, rows1, k1, self.shape1 = read_q4(gguf, "blk.0.ffn_up.weight", rows_arg)
    if self.rows != rows1 or self.k != k1:
      raise ValueError("gate/up shape mismatch")
    self.q4b0, self.q4b1 = make_buffer(len(self.q40), dtypes.uint8), make_buffer(len(self.q41), dtypes.uint8)
    self.dst0, self.dst1 = make_buffer(self.rows, dtypes.float32), make_buffer(self.rows, dtypes.float32)
    self.q4b0.copyin(memoryview(self.q40)); self.q4b1.copyin(memoryview(self.q41))

  def producer_correctness(self) -> dict[str, Any]:
    got_norm = copyout_array(self.norm_out, np.empty(4096, dtype=np.float32))
    got_q8 = bytearray(128 * 36)
    self.q8buf.copyout(memoryview(got_q8))
    q8_x = q8_dequant(bytes(got_q8), 4096)
    fp_err, q8_err = np.abs(got_norm - self.ref_norm), np.abs(q8_x - self.ref_norm)
    return {"q8_x": q8_x, "producer_fp_max_abs": float(fp_err.max()), "q8_dequant_max_abs": float(q8_err.max()),
            "producer_correct": float(fp_err.max()) <= 1e-5, "q8_dequant_bounded": float(q8_err.max()) <= 0.02}

  def consumer_correctness(self, q8_x: np.ndarray) -> dict[str, Any]:
    ref0 = q4_ref_rows(self.q40, self.rows, self.k, q8_x)
    ref1 = q4_ref_rows(self.q41, self.rows, self.k, q8_x)
    got0 = copyout_array(self.dst0, np.empty(self.rows, dtype=np.float32))
    got1 = copyout_array(self.dst1, np.empty(self.rows, dtype=np.float32))
    err0, err1 = np.abs(got0 - ref0), np.abs(got1 - ref1)
    return {"gate_max_abs": float(err0.max()), "up_max_abs": float(err1.max()),
            "gate_correct": float(err0.max()) <= 2e-3, "up_correct": float(err1.max()) <= 2e-3}


def producer_once(prg: FixedLaunchRunner, fx: Fixture, check: bool = False) -> tuple[float, np.ndarray | None, dict[str, Any] | None]:
  ms = float(prg(fx.norm_out._buf, fx.q8buf._buf, fx.xbuf._buf, fx.wbuf._buf, wait=True)) * 1000.0
  if not check: return ms * 1000.0, None, None
  corr = fx.producer_correctness()
  q8_x = corr.pop("q8_x")
  return ms * 1000.0, q8_x, corr


def gateup_once(prg: FixedLaunchRunner, fx: Fixture, q8_x: np.ndarray | None = None, check: bool = False) -> tuple[float, dict[str, Any] | None]:
  ms = float(prg(fx.dst0._buf, fx.dst1._buf, fx.q4b0._buf, fx.q4b1._buf, fx.q8buf._buf, wait=True)) * 1000.0
  if not check: return ms * 1000.0, None
  if q8_x is None: q8_x = fx.producer_correctness()["q8_x"]
  return ms * 1000.0, fx.consumer_correctness(q8_x)


def summarize(samples: list[float]) -> dict[str, Any]:
  if not samples: return {"n": 0}
  ms = [x / 1000.0 for x in samples]
  st = ms_stats(ms)
  st["median_us"] = statistics.median(samples)
  st["mean_us"] = statistics.fmean(samples)
  st["n"] = len(samples)
  return st


def main() -> int:
  ap = argparse.ArgumentParser(description="Decode owned q8 interleaved lifecycle gate")
  ap.add_argument("--gguf", type=pathlib.Path, default=pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"))
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--rows", type=int, default=12288)
  ap.add_argument("--seed", type=int, default=7)
  ap.add_argument("--warmups", type=int, default=8)
  ap.add_argument("--rounds", type=int, default=24)
  ap.add_argument("--target-us", type=float, default=115.24)
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  args = ap.parse_args()

  dev = Device["AMD"]
  prod_prg = FixedLaunchRunner(dev.runtime("owned_q8_interleaved_lifecycle_producer", dev.compiler.compile(NORM_SOURCE)), (1, 1, 1), (256, 1, 1))
  gateup_blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  fx = Fixture(args.gguf, args.rows, args.seed)
  gateup_prg = FixedLaunchRunner(dev.runtime("owned_q8_interleaved_lifecycle_gateup", gateup_blob), (fx.rows, 2, 1), (32, 4, 1))

  for _ in range(args.warmups):
    producer_once(prod_prg, fx)
    gateup_once(gateup_prg, fx)
  _, q8_x0, producer_corr_before = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_before = gateup_once(gateup_prg, fx, q8_x0, check=True)

  clock_before = clock_sample()
  rng = random.Random(args.seed)
  row_types = ["producer_only", "lifecycle"]
  rows = []
  samples = {"producer_only": [], "lifecycle_producer": [], "lifecycle_consumer": [], "lifecycle_total": []}
  for r in range(args.rounds):
    order = row_types[:]
    rng.shuffle(order)
    for label in order:
      prod_us, _, _ = producer_once(prod_prg, fx)
      if label == "producer_only":
        samples["producer_only"].append(prod_us)
        rows.append({"round": r, "label": label, "producer_us": prod_us})
      else:
        cons_us, _ = gateup_once(gateup_prg, fx)
        total_us = prod_us + cons_us
        samples["lifecycle_producer"].append(prod_us)
        samples["lifecycle_consumer"].append(cons_us)
        samples["lifecycle_total"].append(total_us)
        rows.append({"round": r, "label": label, "producer_us": prod_us, "consumer_us": cons_us,
                     "total_us": total_us})
  clock_after = clock_sample()
  _, q8_x1, producer_corr_after = producer_once(prod_prg, fx, check=True)
  _, consumer_corr_after = gateup_once(gateup_prg, fx, q8_x1, check=True)

  summaries = {k: summarize(v) for k, v in samples.items()}
  gates = {
    "rows_present": all(summaries[k]["n"] == args.rounds for k in summaries),
    "producer_correct": bool(producer_corr_before and producer_corr_before["producer_correct"] and producer_corr_before["q8_dequant_bounded"] and
                             producer_corr_after and producer_corr_after["producer_correct"] and producer_corr_after["q8_dequant_bounded"]),
    "consumer_correct": bool(consumer_corr_before and consumer_corr_before["gate_correct"] and consumer_corr_before["up_correct"] and
                             consumer_corr_after and consumer_corr_after["gate_correct"] and consumer_corr_after["up_correct"]),
    "lifecycle_total_lte_target": summaries["lifecycle_total"]["median_us"] <= args.target_us,
  }
  if not gates["producer_correct"] or not gates["consumer_correct"]:
    verdict = "BLOCKED_DECODE_Q8_INTERLEAVED_LIFECYCLE_INCORRECT"
  elif gates["lifecycle_total_lte_target"]:
    verdict = "PASS_DECODE_Q8_INTERLEAVED_LIFECYCLE_GATE"
  else:
    verdict = "BLOCKED_DECODE_Q8_INTERLEAVED_LIFECYCLE_STILL_SLOW"

  result = {
    "date": "2026-06-20",
    "phase": "DECODE_Q8_INTERLEAVED_LIFECYCLE_GATE",
    "schema": "decode_q8_interleaved_lifecycle_gate_v1",
    "verdict": verdict,
    "gate_pass": verdict.startswith("PASS_"),
    "default_behavior_changed": False,
    "performance_claim": True,
    "commit": git_sha(),
    "target_lifecycle_us": args.target_us,
    "summaries": summaries,
    "correctness": {
      "producer_before": producer_corr_before,
      "consumer_before": consumer_corr_before,
      "producer_after": producer_corr_after,
      "consumer_after": consumer_corr_after,
    },
    "rows": rows,
    "clock": {"before": clock_before, "after": clock_after},
    "gates": gates,
    "decision": {
      "if_pass": "mixed q8 lifecycle clears target under interleaved ordering; promotion policy can reopen",
      "if_still_slow": "lifecycle gap survives interleaving; use producer/consumer split from this run as next blocker",
      "if_incorrect": "fix lifecycle correctness before timing",
    },
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": verdict,
    "producer_only_us": summaries["producer_only"]["median_us"],
    "lifecycle_producer_us": summaries["lifecycle_producer"]["median_us"],
    "lifecycle_consumer_us": summaries["lifecycle_consumer"]["median_us"],
    "lifecycle_total_us": summaries["lifecycle_total"]["median_us"],
    "target_lifecycle_us": args.target_us,
    "gates": gates,
    "out": rel(args.out),
  }, indent=2))
  return 0 if gates["producer_correct"] and gates["consumer_correct"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
