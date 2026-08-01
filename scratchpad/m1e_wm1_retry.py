#!/usr/bin/env python3
"""Retry of the wm=1 wave-count config alone: the first attempt (run as part of
scratchpad/m1e_metal_precontract_sweep.py) hit a transient Metal driver disconnect
(XPC_ERROR_CONNECTION_INTERRUPTED) during compilation, not a numeric result. Re-running it in
isolation, as the single GPU-lane holder, with no other GPU work concurrent."""
import sys, time, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

QUANT, ROLE = "Q4_K", "ffn_gate_up"
SHAPE = (512, 12288, 4096)
GEOMETRY = (256, 64, 32, 1, 1, 1)


def main() -> None:
  config = ProbeConfig(QUANT, ROLE, SHAPE, GEOMETRY, device="METAL", rounds=3, warmups=1)
  t0 = time.monotonic()
  result = run_precontract_probe(config)
  print("elapsed", time.monotonic() - t0)
  print(json.dumps(result.to_json(), indent=2, default=str))


if __name__ == "__main__":
  main()
