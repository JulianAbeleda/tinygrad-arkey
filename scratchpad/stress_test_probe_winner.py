#!/usr/bin/env python3
"""Independent Part 1 coverage check for the campaign's winning geometry, using the
committed precontract_probe_lane.py probe (zero-initialized output buffer, nonzero-count
coverage, bit-identical determinism across rounds). Not part of the campaign scripts --
an independent re-verification."""
import sys, json
sys.path.insert(0, ".")
from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

if __name__ == "__main__":
  cfg = ProbeConfig(
    quant="Q4_K", role="ffn_gate_up", shape=(512, 12288, 4096),
    geometry=(64, 32, 32, 4, 1, 1),  # tm, tn, tk, wm, wn, bc -- campaign's winning geometry
    device="METAL", rounds=3, warmups=1,
  )
  print("config:", cfg.to_json())
  result = run_precontract_probe(cfg, keep_npz=False)
  print(json.dumps(result.to_json(), indent=2))
