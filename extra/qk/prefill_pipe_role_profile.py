"""H3: per-role effective TFLOPS for one arm (default or pipe), env-selected. PROFILE=1 ProfileRangeEvent on a warmed
chunk forward @sp=0. Bucket by prefill_gen_sched_gemm_512_N_K name; eff_tflops = role_flop / per-call gpu time. Prints @@JSON."""
import os, json
os.environ.setdefault("PREFILL_V2","1"); os.environ["PROFILE"]="1"
from tinygrad import Tensor, Device, TinyJit
import tinygrad.runtime.ops_amd  # noqa: enables AMD profiling capture
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.llm.generate import load_model_and_tokenizer
from extra.qk.harness_contract import DEFAULT_MODEL
ROLE={"prefill_gen_sched_gemm_512_12288_4096":("ffn_gate_up",2*512*12288*4096,69.8),
      "prefill_gen_sched_gemm_512_4096_12288":("ffn_down",2*512*4096*12288,70.9),
      "prefill_gen_sched_gemm_512_4096_4096":("attn_qo",2*512*4096*4096,76.7),
      "prefill_gen_sched_gemm_512_1024_4096":("attn_kv",2*512*1024*4096,51.8)}
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,1024,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
N=512; chunk=Tensor([[(i*7)%1000 for i in range(N)]],dtype="int32").contiguous(); temp=Tensor([0.0])
for _ in range(2): m.forward(chunk,0,temp).realize()  # warm (eager: per-kernel ProfileRangeEvents; JIT batches to 1 opaque range)
dev.synchronize()
Compiled.profile_events=[]
for _ in range(3): m.forward(chunk,0,temp).realize()
dev.synchronize()
agg={}; calls={}
for e in Compiled.profile_events:
  if isinstance(e,ProfileRangeEvent) and e.en is not None:
    nm=getattr(e.name,"name",None) or str(e.name); agg[nm]=agg.get(nm,0.0)+float(e.en-e.st); calls[nm]=calls.get(nm,0)+1
roles={}
for nm,dur in agg.items():
  for pref,(role,flop,blas) in ROLE.items():
    if nm.startswith(pref):
      per_call_us=dur/max(1,calls[nm])/8  # /8 bursts
      # but calls already summed over 8 bursts*layers; use total dur over total calls
      pc=dur/max(1,calls[nm]); tfl=flop/(pc*1e-6)/1e12
      r=roles.setdefault(role,{"flop":flop,"blas":blas,"dur_us_total":0.0,"calls":0})
      r["dur_us_total"]+=dur; r["calls"]+=calls[nm]
for role,r in roles.items():
  pc=r["dur_us_total"]/max(1,r["calls"]); r["eff_tflops"]=round(r["flop"]/(pc*1e-6)/1e12,1); r["pct_blas"]=round(100*r["eff_tflops"]/r["blas"],1); r["per_call_us"]=round(pc,1)
top=sorted(((nm,round(agg[nm],1),calls[nm]) for nm in agg),key=lambda x:-x[1])[:12]
print("@@"+json.dumps({"pipe":os.environ.get("PREFILL_GEMM_PIPELINE"),"roles":roles,"top_kernels":top if not roles else []}))
