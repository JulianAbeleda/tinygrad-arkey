#!/usr/bin/env python3
"""Attention-only, proof-gated AMD benchmark for shared prefill attention."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, statistics, time, subprocess, sys
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
from tinygrad.uop.ops import SharedAttentionCandidateContext
from extra.qk.shared_attention_promotion import composite_admission_errors
from extra.qk.attention_harness_common import (content_sha as _sha, causal_mask as _mask, amd_sync as _sync,
  load_shared_attention_proof as _proof, make_qkv as _inputs, reference_attention as _baseline_fn,
  synced_time as _time, timing_summary as _summary, candidate_context as _candidate_context, ROUTES)

SCHEMA="tinygrad.shared_attention_benchmark.v1"
DEFAULT_PROOF=Path(__file__).resolve().parents[2]/"docs/artifacts/shared-attention-m10e1-20260723/shared_attention_proof.json"
def _candidate(q,k,v,mask,ctx): return shared_prefill_attention(q,k,v,mask=mask,candidate_context=ctx)
def _baseline(q,k,v,mask,ctx): return _baseline_fn(q,k,v,mask,ctx.hq,ctx.hkv)
def _full_output_numeric_gate(candidate:np.ndarray, baseline:np.ndarray, *, candidate_id:str) -> dict[str,Any]:
  candidate,baseline=np.asarray(candidate,dtype=np.float32),np.asarray(baseline,dtype=np.float32)
  if candidate.shape != baseline.shape: raise RuntimeError(f"full-output shapes differ: candidate={candidate.shape}, baseline={baseline.shape}")
  if candidate.size == 0: raise RuntimeError("full-output numeric gate received an empty output")
  diff=np.abs(candidate-baseline)
  passed=bool(np.isfinite(candidate).all() and np.isfinite(baseline).all() and np.allclose(candidate,baseline,rtol=.03,atol=.006))
  record={"candidate_id":candidate_id,"status":"PASS" if passed else "FAIL","full_output":True,
          "candidate_shape":list(candidate.shape),"baseline_shape":list(baseline.shape),
          "compared_elements":int(candidate.size),"max_abs":float(diff.max()),"rtol":.03,"atol":.006,
          "tolerance_passed":passed}
  if not passed: raise RuntimeError(f"full-output numeric gate failed: max_abs={record['max_abs']}")
  return record
def run_one(proof_path:Path,out:Path,*,profile:str,kv:int,mode:str,samples:int,warmup:int,
            composite_closure:Callable[[Tensor,Tensor,Tensor,Tensor,SharedAttentionCandidateContext],Tensor]|None=None,
            candidate_admission:Mapping[str,Any]|None=None):
  proof=_proof(proof_path); route=next((x for x in ROUTES if x[0]==profile),None)
  if route is None or mode not in {"candidate","baseline"}: raise ValueError("invalid profile or mode")
  profile,strategy,hq,hkv=route; ctx=_candidate_context(profile,strategy,hq,hkv,kv)
  if mode=="candidate":
    errors=composite_admission_errors(candidate_admission,profile=profile,context=kv,strategy=strategy)
    if composite_closure is None: errors.append("missing admitted composite closure")
    if errors: raise ValueError("candidate disabled; proof required: "+"; ".join(errors))
  (q,k,v),raw=_inputs(hq,hkv,512,kv,20260723+hq+kv)
  mask=_mask(ctx.q_tokens,ctx.kv_tokens,ctx.start_pos)
  candidate_calls=0
  def cand():
    nonlocal candidate_calls
    candidate_calls += 1
    return composite_closure(q,k,v,mask,ctx)
  base=lambda:_baseline(q,k,v,mask,ctx)
  numeric=None
  if mode=="candidate":
    cg,bg=cand().numpy().astype(np.float32),base().numpy().astype(np.float32)
    numeric=_full_output_numeric_gate(cg,bg,candidate_id=candidate_admission["candidate_id"])
  # Capture is completed before timing. The sampled closure is JIT replay only.
  replay=TinyJit(cand if mode=="candidate" else base)
  timing=_summary(_time(replay,warmup,samples)); timing["tokens_s"]=512000/timing["median_ms"]
  timing["protocol"]="tinyjit_replay_plus_device_synchronize"
  census=None
  if mode=="candidate":
    row={"candidate_id":candidate_admission["candidate_id"],"profile":profile,"context":kv}
    census={"complete":candidate_calls>0,"expected":[row],"observed":[row] if candidate_calls else [],
            "missing":[] if candidate_calls else [row],"unexpected":[],"invocation_count":candidate_calls}
  artifact={"schema":SCHEMA,"proof_path":str(proof_path),"proof_sha256":_sha(proof_path.read_text()),"config":{"samples":samples,"warmup":warmup,"q_tokens":512,"kv":kv,"profile":profile,"mode":mode,"seed":20260723+hq+kv,"dtype":"float16","device":"AMD"},"hardware":{"platform":platform.platform(),"rocm_visible_devices":os.getenv("ROCR_VISIBLE_DEVICES")},"assumptions":{"flops":"4*Hq*Q*KV*Hd","bytes":"fp16 QKV/output; excludes cache and compiler"},"candidate_context":dict(zip(ctx._fields,ctx)),"candidate_admission":dict(candidate_admission) if mode=="candidate" else None,"candidate_route_census":census,"full_output_numeric":numeric,"timing":timing,"numeric_max_abs":numeric["max_abs"] if numeric else None,"input_sha256":_sha("".join(str(x.shape)+str(x.sum()) for x in raw))}
  out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(artifact,sort_keys=True,indent=2)+"\n"); return artifact
def main():
  p=argparse.ArgumentParser(); p.add_argument("--proof",type=Path,default=DEFAULT_PROOF); p.add_argument("--output",type=Path,required=True); p.add_argument("--samples",type=int,default=10); p.add_argument("--warmup",type=int,default=3); p.add_argument("--profile",choices=tuple(x[0] for x in ROUTES),required=True); p.add_argument("--kv",type=int,required=True); p.add_argument("--mode",choices=("candidate","baseline"),required=True)
  a=p.parse_args();
  if a.samples<10 or a.warmup<1: raise ValueError("require >=10 samples and >=1 warmup")
  if a.mode=="candidate": raise SystemExit("candidate disabled; an admitted composite closure must be supplied programmatically")
  run_one(a.proof,a.output,profile=a.profile,kv=a.kv,mode=a.mode,samples=a.samples,warmup=a.warmup)
if __name__=="__main__": main()
