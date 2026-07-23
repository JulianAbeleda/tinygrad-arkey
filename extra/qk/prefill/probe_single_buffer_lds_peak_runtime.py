"""Unpinned peak-track A/B for the unregistered single-buffer LDS candidate."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from extra.qk.prefill.packed_wmma_correctness_canary import build_artifact, candidate_payload, run_canary
from extra.qk.prefill.probe_single_buffer_lds import one_buffer_payload


def _gpu_state() -> dict[str, str | None]:
  root = Path("/sys/class/drm/card0/device")
  return {name: (root / name).read_text() if (root / name).is_file() else None
          for name in ("power_dpm_force_performance_level", "pp_dpm_sclk", "pp_dpm_mclk")}


def _sample(payload: dict, artifact: str) -> dict:
  outcome = run_canary("Q4_K", artifact, timeout_seconds=60.0, base_payload=payload)
  guarded = outcome["guarded"]
  if not outcome["passed"] or not guarded["numerics_passed"]:
    raise RuntimeError(f"guarded candidate failed: {outcome}")
  return {"kernel_ms": guarded["elapsed_seconds"] * 1000.0, "gpu_state": _gpu_state(),
          "canonical_identity": outcome["identity"]["canonical_identity"], "max_abs_error": guarded["max_abs_error"]}


def _summary(samples: list[dict]) -> dict:
  values = np.asarray([row["kernel_ms"] for row in samples], dtype=np.float64)
  return {"count": int(values.size), "median_ms": float(np.median(values)),
          "p10_ms": float(np.percentile(values, 10)), "p90_ms": float(np.percentile(values, 90)),
          "best_ms": float(np.min(values)), "max_abs_error": max(row["max_abs_error"] for row in samples)}


def run(output: str, warmups: int = 3, samples: int = 10) -> dict:
  root = Path(output).parent
  root.mkdir(parents=True, exist_ok=True)
  baseline = candidate_payload("qwen3_8b_q4k_m_gfx1100", "attn_qo")
  variants = (("two_buffer_baseline", baseline), ("one_buffer_probe", one_buffer_payload(baseline)))
  artifact = build_artifact("Q4_K", str(root / "attn-qo-q4k-peak-canary.npz"), (512, 4096, 4096))
  rows = []
  for label, payload in variants:
    warm = [_sample(payload, artifact["path"]) for _ in range(warmups)]
    measured = [_sample(payload, artifact["path"]) for _ in range(samples)]
    rows.append({"label": label, "warmups": warm, "samples": measured, "summary": _summary(measured)})
  report = {"schema": "tinygrad.prefill_lds_single_buffer_peak_track.v1", "mode": "isolated_guarded_candidate_only",
            "clock_pin": False, "clock_pin_detail": "sysfs DPM pin denied; automatic state recorded per sample",
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "warmup_count": warmups, "sample_count": samples,
            "profile": "qwen3_8b_q4k_m_gfx1100", "role": "attn_qo", "quant_format": "Q4_K",
            "artifact": artifact, "variants": rows, "promotion": "not_registered_not_promoted"}
  Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", required=True)
  parser.add_argument("--warmups", type=int, default=3)
  parser.add_argument("--samples", type=int, default=10)
  args = parser.parse_args()
  print(json.dumps(run(args.output, args.warmups, args.samples), sort_keys=True))
