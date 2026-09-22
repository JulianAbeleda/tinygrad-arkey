#!/usr/bin/env python3
"""Deterministic pp512 sequence Q/K norm+RoPE correctness and R20 gate."""
import hashlib, json, os, pathlib, time
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.model import apply_rope
from tinygrad.llm.pp512_qk_norm_rope import PP512QKNormRopeAdmission, pp512_qk_norm_rope_sequence_call

def one(key, heads, seed):
  rng = np.random.default_rng(seed)
  x = (rng.standard_normal((1, heads, 512, 128), dtype=np.float32) * .2).astype(np.float32)
  w = (1.0 + rng.standard_normal(128).astype(np.float32) * .03).astype(np.float16)
  angles = rng.uniform(-3.0, 3.0, (512, 64)).astype(np.float32)
  f = np.concatenate((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)
  xt, wt, ft = Tensor(x, device="NV"), Tensor(w, dtype=dtypes.float16, device="NV"), Tensor(f, device="NV")
  cand = pp512_qk_norm_rope_sequence_call(PP512QKNormRopeAdmission(), key, xt, wt, ft)
  norm = (xt * xt).mean(axis=-1, keepdim=True).add(1e-6).sqrt().reciprocal() * xt * wt.reshape(1,1,1,128)
  ref = apply_rope(norm, ft.reshape(1,512,128))
  cand.realize(); ref.realize(); Device["NV"].synchronize()
  ca, ra = cand.numpy(), ref.numpy()
  bits = ca.view(np.uint32) != ra.view(np.uint32)
  return {"key":key,"shape":list(ca.shape),"mismatch_count":int(bits.sum()),
          "max_abs":float(np.max(np.abs(ca.astype(np.float64)-ra.astype(np.float64)))),
          "candidate_sha256":hashlib.sha256(ca.tobytes()).hexdigest(),
          "reference_sha256":hashlib.sha256(ra.tobytes()).hexdigest()}

def main():
  os.environ.setdefault("DEV", "NV")
  rows=[]
  for key, heads, seed in (("q",32,1201),("k",8,1202)): rows.append(one(key,heads,seed))
  timing={}
  for key, heads, seed in (("q",32,2201),("k",8,2202)):
    rng=np.random.default_rng(seed); x=Tensor(rng.standard_normal((1,heads,512,128),dtype=np.float32),device="NV"); w=Tensor(np.ones(128,dtype=np.float16),device="NV"); f=Tensor(np.tile(np.concatenate((np.ones(64),np.zeros(64))), (512,1)).astype(np.float32),device="NV")
    for _ in range(5): pp512_qk_norm_rope_sequence_call(PP512QKNormRopeAdmission(),key,x,w,f).realize()
    Device["NV"].synchronize(); t=time.perf_counter_ns()
    for _ in range(20): pp512_qk_norm_rope_sequence_call(PP512QKNormRopeAdmission(),key,x,w,f).realize()
    Device["NV"].synchronize(); timing[key]={"r20_us":(time.perf_counter_ns()-t)/20/1000}
  out={"schema":"tinygrad.nv_pp512_qk_norm_rope_sequence_gate.v1","rows":rows,"timing":timing,"exact":all(r["mismatch_count"]==0 for r in rows)}
  print(json.dumps(out,indent=2,sort_keys=True))
if __name__ == "__main__": main()
