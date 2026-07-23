"""Isolated runtime A/B for the unregistered single-buffer LDS candidate."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from extra.qk.prefill.packed_wmma_correctness_canary import build_artifact, candidate_payload, run_canary
from extra.qk.prefill.probe_single_buffer_lds import one_buffer_payload


def _control_state() -> dict:
  paths = {"power_level": "/sys/class/drm/card0/device/power_dpm_force_performance_level",
           "sclk": "/sys/class/drm/card0/device/pp_dpm_sclk", "mclk": "/sys/class/drm/card0/device/pp_dpm_mclk"}
  return {"cpu_affinity": sorted(os.sched_getaffinity(0)),
          "gpu_dpm": {name: Path(path).read_text() if Path(path).is_file() else None for name, path in paths.items()}}


def run(output: str) -> dict:
  root = Path(output).parent
  root.mkdir(parents=True, exist_ok=True)
  payloads = (("two_buffer_baseline", candidate_payload("qwen3_8b_q4k_m_gfx1100", "attn_qo")),)
  payloads += (("one_buffer_probe", one_buffer_payload(payloads[0][1])),)
  artifact = build_artifact("Q4_K", str(root / "attn-qo-q4k-canary.npz"), (512, 4096, 4096))
  runs = []
  for label, payload in payloads:
    for phase in ("warmup", "warmed_sample"):
      started = time.monotonic_ns()
      outcome = run_canary("Q4_K", str(root / "attn-qo-q4k-canary.npz"), timeout_seconds=60.0, base_payload=payload)
      runs.append({"label": label, "phase": phase, "wall_ms": (time.monotonic_ns()-started)/1e6, "outcome": outcome})
  report = {"schema": "tinygrad.prefill_lds_single_buffer_runtime_probe.v1", "mode": "isolated_guarded_candidate_only",
            "profile": "qwen3_8b_q4k_m_gfx1100", "role": "attn_qo", "quant_format": "Q4_K",
            "controls": _control_state(), "artifact": artifact, "runs": runs,
            "promotion": "not_registered_not_promoted"}
  Path(output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  return report


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--output", required=True)
  args = parser.parse_args()
  print(json.dumps(run(args.output), sort_keys=True))
