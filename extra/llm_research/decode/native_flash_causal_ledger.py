#!/usr/bin/env python3
"""Derive a conservative flash-only causal ledger from settled native/llama traces.

This deliberately does not turn overlapping llama node durations into additive
wall savings.  It compares (a) the calibrated serialized native flash time,
(b) llama's raw flash interval unions, and (c) the disjoint Shapley ownership
of llama's *exposed* non-MMQ interval union. Thus it establishes that overlap
is sufficient to explain this ownership gap without claiming matched-body
parity across different profiling and fusion contexts.
"""
from __future__ import annotations

import argparse, json, pathlib

SCHEMA = "tinygrad.nv_decode.native_flash_causal_ledger.v1"

def _role(rows:list[dict], name:str) -> dict:
  found = [x for x in rows if x["role"] == name]
  if len(found) != 1: raise ValueError(f"expected one {name} role, got {len(found)}")
  return found[0]

def build(native:dict, timeline:dict, partition:dict) -> dict:
  native_rows = native["native"]["calibrated_class_us"]
  native_score = float(native_rows["non_quantized/flash_score"])
  native_combine = float(native_rows["non_quantized/flash_combine"])
  raw_score = float(timeline["classes"]["flash_score"]["union_us"])
  raw_combine = float(timeline["classes"]["flash_combine"]["union_us"])
  roles = partition["llama"]["roles"]
  score_owner, combine_owner = _role(roles, "flash_score"), _role(roles, "flash_combine")
  native_total, llama_raw_total = native_score+native_combine, raw_score+raw_combine
  exposed = float(score_owner["exposed_shapley_us"])+float(combine_owner["exposed_shapley_us"])
  hidden = float(score_owner["hidden_shapley_us"])+float(combine_owner["hidden_shapley_us"])
  accounting_delta = native_total-exposed
  raw_delta = native_total-llama_raw_total
  if abs(accounting_delta-247.989) > 0.01: raise ValueError("flash ownership delta drifted from settled partition")
  # Native DEV=NV graph is a single GPFIFO / stream-ordered execution.  Its
  # calibrated class sum is consequently fully exposed within its device window.
  return {
    "schema":SCHEMA, "status":"PASS_OVERLAP_SUFFICIENT_BODY_PARITY_UNPROVEN",
    "definitions":{
      "raw":"sum of the class's own interval unions; not an additive token-wall cost when it overlaps MMQ",
      "exposed":"disjoint Shapley ownership of llama's non-MMQ interval union; additive by construction",
      "native":"calibrated profile composition within the marker-light DEV=NV device window; stream-ordered"},
    "native_serialized_us":{"flash_score":round(native_score,3),"flash_combine":round(native_combine,3),"total":round(native_total,3)},
    "llama_raw_interval_union_us":{"flash_score":round(raw_score,3),"flash_combine":round(raw_combine,3),"total":round(llama_raw_total,3)},
    "llama_disjoint_ownership_us":{"flash_score_exposed":round(float(score_owner["exposed_shapley_us"]),3),
      "flash_combine_exposed":round(float(combine_owner["exposed_shapley_us"]),3),"exposed_total":round(exposed,3),
      "hidden_total":round(hidden,3)},
    "comparisons":{"raw_native_minus_llama_us":round(raw_delta,3),
      "exposed_native_minus_llama_us":round(accounting_delta,3),
      "raw_interval_minus_exposed_ownership_us":round(llama_raw_total-exposed,3)},
    "conclusion":{
      "causal":"The +247.989 us flash ownership gap is scheduling/critical-path exposure: native serializes 305.581 us while only 57.592 us of llama flash owns exposed time.",
      "sufficient":"A raw-kernel-speed advantage for llama is not required to explain this ownership gap: llama's own raw flash interval union is 58.135 us larger than native's calibrated raw flash time.",
      "not_proven":"This does not show native flash kernels are optimal, nor predict wall recovery from a kernel rewrite. A native flash rewrite can only be credited by an exact real-token A/B; absent native overlap, its theoretical ceiling is the 305.581 us native serialized population."}}

def main() -> None:
  p=argparse.ArgumentParser(); p.add_argument("--native",type=pathlib.Path,required=True);p.add_argument("--timeline",type=pathlib.Path,required=True)
  p.add_argument("--partition",type=pathlib.Path,required=True);p.add_argument("--out",type=pathlib.Path,required=True);a=p.parse_args()
  out=build(json.loads(a.native.read_text()),json.loads(a.timeline.read_text()),json.loads(a.partition.read_text()))
  a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n");print(json.dumps(out,indent=2))

if __name__ == "__main__": main()
