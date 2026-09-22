#!/usr/bin/env python3
"""Fail-closed reconciliation for the native decode epilogue ownership row.

This deliberately does *not* turn llama's overlap-safe Shapley ownership into
an optimization forecast.  It records only construction-compatible native A/B
evidence and makes the distinction machine-checkable.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "tinygrad.nv_decode.native_epilogue_causal_ledger.v1"


def build(nonquant_payload: dict) -> dict:
  native = nonquant_payload["native"]["roles_us"]
  llama = {r["role"]: r["exposed_shapley_us"] for r in nonquant_payload["llama"]["roles"]}
  roles = ("attention_cast", "attention_residual_add", "ffn_activation_cast", "ffn_down_cast",
           "ffn_residual_add", "block_output_contiguous")
  native_us = sum(native[r] for r in roles)
  llama_us = llama["elementwise"]
  # Every item below is a previously measured construction, not an additive
  # decomposition.  `native_credit_us` stays zero unless a native-NV real-token
  # A/B has both a valid construction and an admitted recovery.
  evidence = [
    {"id":"M4-attn-o-residual", "route":"NV", "kind":"native experimental route",
     "result":"NO_GO", "delta_us":69.0,
     "reason":"adds 36 kernels and loses 1.15%; this is a different experimental emitter, not an in-core epilogue"},
    {"id":"P6-C-llama-attn-o", "route":"CUDA", "kind":"llama substitution",
     "result":"NO_GO", "delta_us":21.213,
     "reason":"q8/fp32 adapters plus one added node per role exceed removed adds; not native topology"},
    {"id":"P6-D-q4-down", "route":"CUDA", "kind":"llama substrate vs full epilogue",
     "result":"EPILOGUE_NEUTRAL", "incremental_epilogue_us":0.348,
     "reason":"the 65.8-us diagnostic gain is MMVQ substrate, not residual absorption"},
    {"id":"M5-typed-combine-boundary", "route":"NV", "kind":"typed boundary/copy removal",
     "result":"SMALL_OLD_SESSION_WIN", "removed_nodes":36, "result_percent":0.22,
     "reason":"copy removal is not the current 240.762-us role population and is not a same-session native residual credit"},
    {"id":"KV-store-chain", "route":"NV", "kind":"store-chain construction",
     "result":"WALL_NEUTRAL", "kernel_delta":0,
     "reason":"legacy path was already one fused store kernel/layer (948 -> 948); no topology to recover"},
  ]
  return {"schema":SCHEMA, "status":"CLOSED_NO_ADMISSIBLE_GENERIC_EPILOGUE_AB",
    "ownership": {
      "native_serialized_us":round(native_us, 3), "llama_exposed_shapley_us":round(llama_us, 3),
      "difference_us":round(native_us-llama_us, 3), "native_roles":list(roles),
      "interpretation":"location/ownership only; llama elementwise Shapley is overlap-safe but neither a raw kernel-time comparison nor a recoverable saving"},
    "evidence":evidence,
    "native_credit_us":0.0,
    "conclusion":(
      "The residual/cast/contiguous row is real serialized native work, but existing experiments close every admitted generic boundary: "
      "M4 regresses, llama O substitution regresses, Q4 down's epilogue is neutral, typed-boundary work is an old small win, and KV was already fused. "
      "No GPU A/B is run because it would repeat a closed construction rather than test a new semantic route."),
    "next_blocker":(
      "Create a tinygrad-owned, ordinary-UOp in-core projection epilogue that consumes the native fp32 GEMV output and its residual "
      "without a custom-program boundary, fp16<->fp32 adapter, Q8 quantization, activation recomputation, or changed output-buffer contract. "
      "Only after that construction exists should a native-NV all-role A/B be admitted."),
    "sources":["nv-decode-nonquant-role-partition-20260805.json", "m4-decomposition-measurement-record-20260803.md",
      "nv-decode-parity-p6d-q4-attention-o-family-graph-ab-record-20260804.md",
      "nv-decode-parity-p6d-ffn-down-construction-record-20260804.md",
      "m5-typed-boundary-p0-implementation-record-20260803.md", "decode-kv-store-chain-fusion-scope-20260803.md"]}


def main() -> None:
  root=Path(__file__).resolve().parents[3]
  src=root/"docs/task_workflow/output/nv-decode-nonquant-role-partition-20260805.json"
  out=root/"docs/task_workflow/output/nv-decode-native-epilogue-causal-ledger-20260805.json"
  payload=build(json.loads(src.read_text()))
  out.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")
  print(json.dumps(payload["ownership"], indent=2, sort_keys=True))


if __name__ == "__main__": main()
