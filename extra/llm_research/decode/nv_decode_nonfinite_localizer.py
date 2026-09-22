#!/usr/bin/env python3
"""Find the first non-finite native-NV d512 decode boundary (diagnostic only)."""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np


def finite(t) -> dict:
  x = t.realize().numpy()
  return {"finite": bool(np.isfinite(x).all()), "shape": list(x.shape),
          "nan": int(np.isnan(x).sum()), "inf": int(np.isinf(x).sum()),
          "max_abs_finite": float(np.max(np.abs(x[np.isfinite(x)]))) if np.isfinite(x).any() else None}


def prompt(path, depth):
  from tinygrad.llm.gguf import gguf_load_metadata
  from tinygrad.llm.runtime_state import SimpleTokenizer
  kv, _ = gguf_load_metadata(path); tok = SimpleTokenizer.from_gguf_kv(kv)
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  return (ids * (1 + depth//len(ids)))[:depth]


def main():
  ap=argparse.ArgumentParser(); ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--depth",type=int,default=512); ap.add_argument("--max-context",type=int,default=4608); ap.add_argument("--out",required=True)
  a=ap.parse_args()
  from tinygrad import Tensor
  from tinygrad.helpers import Context
  import tinygrad.llm.model as mm
  mm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS=frozenset()
  model,_=mm.Transformer.from_gguf(a.model,a.max_context)
  # Match the native decode authority's setup-only ``--no-fused-prefill``
  # contract. These routes affect prompt construction, not the decode graph.
  object.__setattr__(model.config,"prefill_custom_kernel_attn",False)
  object.__setattr__(model.config,"prefill_tc_attn",False)
  p=Tensor([prompt(a.model,a.depth)],dtype="int32"); temp=Tensor([0.0]); rows=[]
  # Establish exactly the production prefill KV state, then examine the next
  # valid-token decode in eager mode at every block boundary.
  with Context(JIT=0): model(p,0,temp,use_flash=False).realize()
  for b in model.blk: b._use_flash,b._prefill_v2,b._is_prefill,b._ring_freqs,b._ring_full=False,False,False,None,False
  x=model.token_embd(Tensor([[1]],dtype="int32")).float(); rows.append({"boundary":"embedding","index":-1,**finite(x)})
  # Split the first decoder block before its @function wrapper. This keeps the
  # operator ownership visible while retaining the exact block flags/KV state.
  b=model.blk[0]
  rows.append({"boundary":"b0.kv_prefix","index":0,**finite(b.cache_kv[:,:,:,0:a.depth,:])})
  normed=b.attn_norm(x); rows.append({"boundary":"b0.attn_norm","index":0,**finite(normed)})
  q,k,v=b.attn_q(normed),b.attn_k(normed),b.attn_v(normed)
  for name,value in (("b0.q_proj",q),("b0.k_proj",k),("b0.v_proj",v)): rows.append({"boundary":name,"index":0,**finite(value)})
  if all(r["finite"] for r in rows[-3:]):
    # _attention includes Q/K norm, RoPE, store/read, softmax and V combine;
    # its own result distinguishes that entire chain from the FFN tail.
    attn=b._attention(normed,a.depth); rows.append({"boundary":"b0.attention","index":0,**finite(attn)})
    if rows[-1]["finite"]:
      h=x+attn; rows.append({"boundary":"b0.residual","index":0,**finite(h)})
      hn=b.ffn_norm(h); rows.append({"boundary":"b0.ffn_norm","index":0,**finite(hn)})
      gate,up=b.ffn_gate(hn),b.ffn_up(hn)
      rows.append({"boundary":"b0.ffn_gate","index":0,**finite(gate)})
      rows.append({"boundary":"b0.ffn_up","index":0,**finite(up)})
      if rows[-1]["finite"] and rows[-2]["finite"]: rows.append({"boundary":"b0.ffn_down","index":0,**finite(b.ffn_down(gate.silu()*up))})
  for i,b in enumerate(model.blk):
    x=b(x,a.depth); row={"boundary":"block","index":i,**finite(x)}; rows.append(row)
    if not row["finite"]: break
  if rows[-1]["finite"]:
    x=model.output_norm(x); rows.append({"boundary":"output_norm","index":len(model.blk),**finite(x)})
    if rows[-1]["finite"]: rows.append({"boundary":"lm_head","index":len(model.blk),**finite(model.output(x))})
  payload={"schema":"tinygrad.nv_decode_nonfinite_localizer.v1","depth":a.depth,"rows":rows,
           "first_nonfinite":next((r for r in rows if not r["finite"]),None)}
  pathlib.Path(a.out).write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload["first_nonfinite"]))
if __name__=="__main__": main()
