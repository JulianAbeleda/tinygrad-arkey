#!/usr/bin/env python3
"""Record missing M3 inputs; this does not execute a GPU or validate a plan."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
  ap=argparse.ArgumentParser()
  ap.add_argument("--export", required=True, type=Path); ap.add_argument("--q4k-fixture", required=True, type=Path)
  ap.add_argument("--q6k-fixture", required=True, type=Path); ap.add_argument("--device", required=True)
  ap.add_argument("--out", type=Path, default=Path(__file__).parent / "m3_blocked_inputs.json")
  args=ap.parse_args()
  missing=[str(p) for p in (args.export,args.q4k_fixture,args.q6k_fixture) if not p.is_file()]
  payload={"schema":"tinygrad.decode_machine_search_blocked_inputs.v1","status":"BLOCKED","device_requested":args.device,"missing_inputs":missing,"missing_executor":"route-bound generic topology-plan executor","missing_primitive_lowerers":["q4k_packed_block_dot","q6k_packed_block_dot","external_reduce"],"reason":"This recorder never executes a GPU. The topology candidates cannot be lowered or route-bound yet, so no numerical or timing result was synthesized."}
  args.out.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":"))+"\n")
  print(json.dumps(payload, sort_keys=True))
if __name__ == "__main__": main()
