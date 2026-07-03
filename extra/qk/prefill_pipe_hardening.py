"""pipe_tm2_tn2 promotion hardening: long-context (>=8192) synced whole-prefill + correctness fingerprint.

Reuses the recovered authority methodology (extra/qk/prefill_whole_synced.py): TinyJit(m.forward) + synced bursts
(K=8, min-of-3), whole-prefill@L = sum of per-chunk JIT'd forward times. Arm is selected by the CALLER's env
(PREFILL_GEMM_PIPELINE / _TM / _TN read at kernel-gen) so run one arm per subprocess. Env knobs:
  PIPE_MAXC   (default 8704)  -- model max_context
  PIPE_SPS    (default "0,512,1024,2048,3584,4096,5120,6144,7168,7680") -- chunk start_pos sample points
  PIPE_LS     (default "512,1024,2048,4096,8192") -- whole-prefill context targets
  PIPE_FINGERPRINT=1 -- also emit an output fingerprint of forward@sp=0 (argmax + sum) for equivalence checking
Prints one @@JSON line.
"""
import os, time, json
from tinygrad import Tensor, Device, TinyJit
os.environ.setdefault("PREFILL_V2","1")
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
from tinygrad.llm.model import PREFILL_GRAPH_GEMM
import bisect

MAXC=int(os.environ.get("PIPE_MAXC","8704"))
SPS=[int(x) for x in os.environ.get("PIPE_SPS","0,512,1024,2048,3584,4096,5120,6144,7168,7680").split(",")]
LS=[int(x) for x in os.environ.get("PIPE_LS","512,1024,2048,4096,8192").split(",")]
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,MAXC,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
temp=Tensor([0.0]); N=512
chunk=Tensor([[(i*7)%1000 for i in range(N)]],dtype="int32").contiguous()

def burst(sp_int, K=8):
  j=TinyJit(m.forward)
  for _ in range(4): j(chunk, sp_int, temp).realize()
  dev.synchronize()
  ts=[]
  for _ in range(3):
    dev.synchronize(); t0=time.perf_counter()
    for _ in range(K): j(chunk, sp_int, temp).realize()
    dev.synchronize(); ts.append((time.perf_counter()-t0)/K*1e3)
  return min(ts)

out={"route":{"graph_gemm":PREFILL_GRAPH_GEMM},
     "pipe":{"on":os.environ.get("PREFILL_GEMM_PIPELINE"),"tm":os.environ.get("PREFILL_GEMM_PIPELINE_TM"),"tn":os.environ.get("PREFILL_GEMM_PIPELINE_TN")},
     "maxc":MAXC,"chunk_ms":{}}
if os.environ.get("PIPE_FINGERPRINT"):
  r=m.forward(chunk,0,temp).realize()
  fp=r.flatten()
  out["fingerprint"]={"argmax":int(fp.argmax().item()),"sum":float(fp.sum().item()),"shape":list(r.shape)}
for sp in SPS:
  try:
    ms=burst(sp); out["chunk_ms"][str(sp)]=round(ms,2)
  except Exception as e:
    out["chunk_ms"][str(sp)]=f"ERR:{type(e).__name__}:{str(e)[:80]}"; break

def whole(L, cm):
  pts=sorted((int(k),v) for k,v in cm.items() if isinstance(v,(int,float)))
  if not pts: return None
  xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
  if L-512 > xs[-1]: return None   # not enough measured points to cover L
  def interp(s):
    if s<=xs[0]: return ys[0]
    if s>=xs[-1]: return ys[-1]
    i=bisect.bisect_right(xs,s)-1; return ys[i]+(ys[i+1]-ys[i])*(s-xs[i])/(xs[i+1]-xs[i])
  return round(L/sum(interp(s) for s in range(0,L,512))*1e3,1)
out["whole_prefill_tok_s"]={str(L):whole(L,out["chunk_ms"]) for L in LS}
print("@@"+json.dumps(out))
