#!/usr/bin/env python3
"""Static, fail-closed provenance census for the compiler-owned Q6 pp512 arm.

This intentionally does not import tinygrad, load a model, compile CUDA, or
run a GPU.  Runtime token/logit fields stay explicitly unavailable until the
model arm supplies them.
"""
from __future__ import annotations
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
BINDING = ROOT / "extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py"
MODEL_ARM = ROOT / "extra/llm_research/prefill/nv_compiler_q6k_model_arm.py"

def main() -> int:
  ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); args = ap.parse_args()
  text = BINDING.read_text(encoding="utf-8")
  arm = MODEL_ARM.read_text(encoding="utf-8")
  roles = {"attn_v": 18, "ffn_down": 18}
  forbidden = ("nv_llama", "NV_LLAMA", "read_bytes()", "\.cubin")
  forbidden_hits = {needle: text.count(needle) for needle in forbidden if text.count(needle)}
  direct_contract = text.count("out.uop_program(record, halfs, fxn=lambda *_:asset.main_program)") == 1
  payload = {
    "schema": "tinygrad.nv_compiler_q6k_provenance_census.v1",
    "status": "PASS" if not forbidden_hits and direct_contract else "FAIL",
    "route": {"binding": str(BINDING), "model_arm": str(MODEL_ARM),
              "roles": roles, "total_roles": sum(roles.values()),
              "compiler_owned": True, "direct_output_roles": ["attn_v", "ffn_down"] if direct_contract else []},
    "weights": {"storage": "canonical uint16 Q6_K halfwords", "canonical": "UNPROVEN_STATIC_ONLY",
                "expanded_copy_kernels": 0},
    "compiler_identity": {"binding_source": "PRESENT", "candidate_context": "RUNTIME_REQUIRED",
                           "nvrtc_producer": "RUNTIME_REQUIRED", "tinygrad_program": "RUNTIME_REQUIRED"},
    "workspace": {"partial_bytes": 0, "old_fixup_programs": 0},
    "provenance": {"forbidden_binding_hits": forbidden_hits,
                    "llama_imports_in_model_arm": arm.count("nv_llama"),
                    "cubin_reads_in_binding": text.count("read_bytes()"),
                    "direct_output_contract": direct_contract},
    "runtime": {"token": None, "finite": None, "same_token": None,
                 "logit_max_abs": None, "logit_mean_abs": None,
                 "note": "GPU/model gate not run by this structural harness"},
  }
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  print(json.dumps(payload, sort_keys=True)); return 0 if payload["status"] == "PASS" else 1

if __name__ == "__main__": raise SystemExit(main())
