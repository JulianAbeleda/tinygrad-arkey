#!/usr/bin/env python3
"""Isolated pp512 FP16 K/V split-K occupancy discriminator.

This deliberately does not alter routing.  The control is the installed exact
512x1024x4096 candidate.  Split arms partition K and charge all partial GEMMs
plus the FP32 reduction/cast as one synchronized lifecycle.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, statistics, time
import numpy as np

M, N, K = 512, 1024, 4096


def sync_ms(fn, rounds:int) -> list[float]:
  from tinygrad import Device
  out=[]
  for _ in range(rounds):
    Device["NV"].synchronize(); st=time.perf_counter_ns(); fn().realize(); Device["NV"].synchronize()
    out.append((time.perf_counter_ns()-st)/1e6)
  return out


def stats(xs:list[float]) -> dict:
  return {"n":len(xs), "min_ms":min(xs), "median_ms":statistics.median(xs), "max_ms":max(xs), "samples_ms":xs}


def main() -> None:
  ap=argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--warmups", type=int, default=3)
  ap.add_argument("--out", required=True)
  args=ap.parse_args()
  from tinygrad import Tensor, Device, TinyJit
  from tinygrad.llm.generate import load_model_and_tokenizer
  from tinygrad.llm.prefill_graph_gemm import _attached_candidate_admission, _candidate_warmstart_opts, _contiguous_candidate_operand
  from tinygrad.codegen.opt import postrange as pr

  model,_=load_model_and_tokenizer(args.model, 512, seed=20260617)
  lin=model.blk[0].attn_k
  toks=Tensor([[(i*7)%1000 for i in range(M)]], dtype="int32").contiguous()
  a=model.blk[0].attn_norm(model.token_embd(toks).float()).reshape(M,K).cast("float16").contiguous().realize()
  bt=_contiguous_candidate_operand(lin._pf16_w.cast("float16")).realize()
  admission=_attached_candidate_admission(lin,"attn_kv",(M,N,K))
  if admission is None: raise RuntimeError("exact attn_kv candidate is unavailable")
  key=pr.warmstart_key({M,N},K); target=admission.normalized_payload["workload"]["target"]
  opts=_candidate_warmstart_opts(target["backend"],target["arch"],target["wave_size"])

  arms={}
  with pr.warmstart_candidate_state({key:opts[:1]}, {key:admission.context}):
    @TinyJit
    def control(): return (a @ bt.transpose()).cast("float32").realize()
    for _ in range(args.warmups): control().realize()
    baseline=control().realize(); Device["NV"].synchronize()
    arms["control_32cta"]={"cta_count":32,"k_parts":1,"timing":stats(sync_ms(control,args.rounds))}
    ref=baseline.numpy()

  for parts in (2,4,8):
    step=K//parts
    @TinyJit
    def split(parts=parts,step=step):
      partials=[a[:,p*step:(p+1)*step].contiguous() @ bt[:,p*step:(p+1)*step].contiguous().transpose() for p in range(parts)]
      # Explicit FP32 lifecycle reduction.  The final result is retained in FP32
      # for a clean numerical comparison with the control's FP16 output widened.
      return Tensor.stack(*[x.cast("float32") for x in partials], dim=0).sum(axis=0).realize()
    for _ in range(args.warmups): split().realize()
    got=split().realize(); Device["NV"].synchronize(); arr=got.numpy()
    finite=bool(np.isfinite(arr).all()); diff=np.abs(arr-ref)
    arms[f"splitk{parts}_full_lifecycle"]={
      "aggregate_ctas":32*parts,"max_concurrent_ctas_per_partial_launch":32,"k_parts":parts,"partial_k":step,
      "correctness":{"finite":finite,"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
                     "allclose_atol_0p125_rtol_0p002":bool(np.allclose(arr,ref,atol=.125,rtol=.002)),
                     "sha256":hashlib.sha256(arr.tobytes()).hexdigest()},
      "timing":stats(sync_ms(split,args.rounds)),
    }

  ctl=arms["control_32cta"]["timing"]["min_ms"]
  for arm in arms.values():
    arm["speedup_vs_control_min"]=ctl/arm["timing"]["min_ms"]
    arm["role_tmac_per_s_min"]=(M*N*K/1e12)/(arm["timing"]["min_ms"]/1e3)
  rec={"schema":"tinygrad.nv_prefill_kv_splitk_discriminator.v1","shape":{"m":M,"n":N,"k":K},
       "target":str(Device["NV"]),"model":args.model,"correctness_contract":"finite and allclose(atol=.125,rtol=.002) to installed FP16 candidate",
       "scope":"isolated one K projection; full split partial+reduction lifecycle; no routing edits","arms":arms}
  path=pathlib.Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,indent=2))


if __name__ == "__main__": main()
