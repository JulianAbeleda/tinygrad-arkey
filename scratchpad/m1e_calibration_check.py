#!/usr/bin/env python3
"""One-off calibration: confirm the M1e lane reproduces M1c's established Metal result for the
exact same dispatch before trusting it for the Part 2 sweep. Not part of the sweep itself."""
import sys, time, json
sys.path.insert(0, "/Users/julianabeleda/env/tinygrad-arkey-exp")

from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe


def main() -> None:
  config = ProbeConfig("Q4_K", "ffn_gate_up", (512, 12288, 4096), (256, 64, 32, 8, 1, 1),
                        device="METAL", rounds=3, warmups=1)
  t0 = time.monotonic()
  result = run_precontract_probe(config)
  print("elapsed", time.monotonic() - t0)
  print(json.dumps(result.to_json(), indent=2, default=str))


if __name__ == "__main__":
  main()
