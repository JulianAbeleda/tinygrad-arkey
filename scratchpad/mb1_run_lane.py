#!/usr/bin/env python3
"""MB1: run the metal precontract lane for the fixed BUG-B config and print a summary.

Reuses `extra/llm_research/prefill/precontract_probe_lane.py` (the instrument) -- not a new driver.
Config throughout: Q4_K, ffn_gate_up, (512,12288,4096), geometry (256,64,32,8,1,1), METAL, 3 rounds.
"""
import sys, json
sys.path.insert(0, ".")
from extra.llm_research.prefill.precontract_probe_lane import ProbeConfig, run_precontract_probe

def main():
  cfg = ProbeConfig(quant="Q4_K", role="ffn_gate_up", shape=(512, 12288, 4096),
                     geometry=(256, 64, 32, 8, 1, 1), device="METAL", rounds=3)
  res = run_precontract_probe(cfg)
  d = res.to_json()
  print(json.dumps(d, indent=2))

if __name__ == "__main__":
  main()
