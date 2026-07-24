#!/usr/bin/env python3
"""Generate content-addressed shared-attention compiler/numeric artifacts."""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
if __package__ in (None, ""): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.flash_prefill_attention import shared_prefill_attention
from tinygrad.renderer.cstyle import HIPRenderer
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops, SharedAttentionCandidateContext
from extra.qk.shared_attention_capture import SharedAttentionCompilerCapture, build_shared_attention_compiler_capture
from extra.qk.shared_attention_evidence import shared_attention_proof_artifact

_ROUTES = (
  ("8b-overlay-first","qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",32,512,512,0),
  ("8b-overlay-prefix","qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY",32,512,1024,512),
  ("14b-bounded-first","qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",40,512,512,0),
  ("14b-bounded-prefix","qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES",40,512,1024,512),
)

def _canonical_write(path:Path,value) -> None:
  path.write_text(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n")

# TODO(centralize): differs from attention_harness_common.causal_mask (takes a ctx object instead of
# (q_tokens, kv_tokens, start_pos) positional args) — left as-is.
def _mask(ctx:SharedAttentionCandidateContext) -> Tensor:
  return Tensor.full((1,1,ctx.q_tokens,ctx.kv_tokens),float("-inf"),dtype=dtypes.float16,buffer=False).triu(ctx.start_pos+1)

def _schedule(ctx:SharedAttentionCandidateContext):
  q=Tensor.empty(1,ctx.hq,ctx.q_tokens,128,dtype=dtypes.float16,device="AMD")
  k=Tensor.empty(1,ctx.hkv,ctx.kv_tokens,128,dtype=dtypes.float16,device="AMD")
  v=Tensor.empty(1,ctx.hkv,ctx.kv_tokens,128,dtype=dtypes.float16,device="AMD")
  schedule=shared_prefill_attention(q,k,v,mask=_mask(ctx),candidate_context=ctx).schedule_linear()
  calls=[call for call in schedule.src if call.op is Ops.CALL and getattr(call.src[0].arg,"candidate_context",None)==ctx]
  if len(calls)!=1: raise RuntimeError("expected one context-bound compute call")
  ast=calls[0].src[0]
  hip=to_program(ast,HIPRenderer(Target.parse("AMD:HIP:gfx1100")))
  isa=to_program(ast,AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  return schedule,calls[0],hip,isa

def _numeric(ctx:SharedAttentionCandidateContext,seed:int):
  import numpy as np
  rng=np.random.default_rng(seed)
  q=rng.normal(0,.04,(ctx.hq,ctx.q_tokens,128)).astype(np.float16)
  k=rng.normal(0,.04,(ctx.hkv,ctx.kv_tokens,128)).astype(np.float16)
  v=rng.normal(0,.04,(ctx.hkv,ctx.kv_tokens,128)).astype(np.float16)
  tq,tk,tv=(Tensor(x[None],device="AMD") for x in (q,k,v))
  got=shared_prefill_attention(tq,tk,tv,mask=_mask(ctx),candidate_context=ctx).numpy().astype(np.float32)[0]
  ref=np.empty_like(got); group=ctx.hq//ctx.hkv; scale=1/(128**.5)
  keys=np.arange(ctx.kv_tokens)
  for head in range(ctx.hq):
    scores=q[head].astype(np.float32)@k[head//group].astype(np.float32).T*scale
    scores=np.where(keys[None,:] <= ctx.start_pos+np.arange(ctx.q_tokens)[:,None],scores,-np.inf)
    probs=np.exp(scores-scores.max(axis=1,keepdims=True)); probs/=probs.sum(axis=1,keepdims=True)
    ref[head]=probs@v[head//group].astype(np.float32)
  return got,ref

def generate(output_dir:Path,route:str|None=None) -> None:
  output_dir.mkdir(parents=True,exist_ok=True)
  selected=tuple(row for row in _ROUTES if route is None or row[0]==route)
  if not selected: raise ValueError(f"unknown route {route!r}")
  for index,(slug,profile,strategy,hq,qt,kv,start) in enumerate(selected):
    # TODO(centralize): differs from attention_harness_common.candidate_context (explicit qt/start/hkv=8
    # here vs the canonical q_tokens=512 default / kv-q_tokens start_pos formula) — left as-is.
    ctx=SharedAttentionCandidateContext(profile,strategy,qt,kv,start,hq,8,128,True).validate()
    schedule,call,hip,isa=_schedule(ctx); got,ref=_numeric(ctx,20260723+index)
    capture=build_shared_attention_compiler_capture(schedule=schedule,compute_call=call,hip_program=hip,
      amd_isa_program=isa,output=got,reference=ref)
    _canonical_write(output_dir/f"{slug}.json",capture.to_json())
    (output_dir/f"{slug}.hip.cpp").write_text(capture.hip_source)
    (output_dir/f"{slug}.amdisa.s").write_text(capture.amd_isa_text)
  paths=tuple(output_dir/f"{row[0]}.json" for row in _ROUTES)
  if all(path.is_file() for path in paths):
    captures=tuple(SharedAttentionCompilerCapture.from_json(json.loads(path.read_text())) for path in paths)
    _canonical_write(output_dir/"shared_attention_proof.json",shared_attention_proof_artifact(captures))

def main() -> None:
  parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,required=True)
  parser.add_argument("--route",choices=tuple(row[0] for row in _ROUTES))
  args=parser.parse_args(); generate(args.output_dir,args.route)

if __name__ == "__main__": main()
