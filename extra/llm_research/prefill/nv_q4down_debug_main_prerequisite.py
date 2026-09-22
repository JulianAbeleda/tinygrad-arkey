#!/usr/bin/env python3
"""Capability probe for the C0 Q4-down diagnostic-main prerequisite.

This deliberately does not alter, wrap, or launch the production main.  The
compiler-generated asset is an already compiled ELF with a fixed three-buffer
ABI; this probe records that boundary so C0 cannot accidentally claim that
register-local intermediates are capturable.
"""
from __future__ import annotations
import argparse, json, pathlib

from extra.llm_research.prefill.nv_compiler_q4k_down_pp512_binding import binding_for

DIAGNOSTIC_ID = "nv_q4k_down_debug_main_type12_v1"

def probe(device: str = "NV") -> dict:
  asset = binding_for(device)
  p = asset.main_program
  abi = {
    "program_name": p.arg.name,
    "global_size": list(p.arg.global_size),
    "local_size": list(p.arg.local_size),
    "outs": list(p.arg.outs),
    "ins": list(p.arg.ins),
  }
  return {
    "schema": "tinygrad.nv_q4down_debug_main_prerequisite.v1",
    "status": "BLOCKED",
    "diagnostic_identity": DIAGNOSTIC_ID,
    "production_abi_preserved": True,
    "launchable": False,
    "abi": abi,
    "requested_buffers": {
      "decoded_q4_metadata": False,
      "per_k32_corrected_subtotals": False,
      "pre_epilogue_fp32": False,
      "final_output": True,
    },
    "reason": "compiler path exposes only the precompiled ELF main; decoded metadata and accumulators are register-local and no source/ABI hook exists",
    "next_command": None,
  }

def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default="NV")
  args = ap.parse_args()
  payload = probe(args.device)
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2) + "\n")
  print(json.dumps(payload, indent=2))
  raise SystemExit(2)

if __name__ == "__main__":
  main()
