"""Synced per-role in-model PREFILL time-tax (attribution-only, PROFILE GPU-busy).

Captures the TinyJit prefill graph UNDER Context(PROFILE=1) (so the HCQ graph carries per-kernel signals), then
aggregates ProfileGraphEvent GPU-busy per role. Parses the graph-GEMM kernel name (prefill_graph_gemm_512_N_K) to
compute per-role achieved TFLOPS. Answers "why does a parity-class GEMM not fully transfer to whole-prefill": the
single-config dependency-free kernel is at parity on its tuned ffn_gate_up shape but WG-starved on small-N roles.

  DEV=AMD PREFILL_V2=1 [PREFILL_TENSILE_GEMM=1] PYTHONPATH=. .venv/bin/python extra/qk_prefill_per_role_time_tax.py

Authority: attribution-only (PROFILE), per HARNESS_GUIDE Measurement-Authority. NOT a promotion number. Concrete
start_pos=0 chunk; whole multi-chunk prefill (symbolic later chunks) is a separate axis.
"""
import os, re, collections, statistics, sys
os.environ.setdefault("PREFILL_V2","1")
from tinygrad import Tensor, Device, Context, TinyJit
from tinygrad.device import Compiled
from extra.llm_generate import load_model_and_tokenizer
from extra.qk_harness_contract import DEFAULT_MODEL
from tinygrad.llm.model import PREFILL_GRAPH_GEMM, PREFILL_TENSILE_GEMM
ANSI=re.compile(r'\x1b\[[0-9;]*m')
dev=Device["AMD"]
m,tok=load_model_and_tokenizer(DEFAULT_MODEL,4608,seed=20260617)
for b in m.blk: b._use_flash,b._prefill_v2=True,True
temp=Tensor([0.0]); N=512
chunk=Tensor([[(i*7)%1000 for i in range(N)]],dtype="int32").contiguous()
samples=[]
with Context(PROFILE=1):
  sp=TinyJit(m.forward)   # CAPTURE under PROFILE so the HCQ graph carries per-kernel profiling signals
  for _ in range(4): sp(chunk,0,temp).realize()
  dev.synchronize(); dev._at_profile_finalize()
  for r in range(5):
    base=len(Compiled.profile_events); sp(chunk,0,temp).realize(); dev.synchronize(); dev._at_profile_finalize()
    a=collections.defaultdict(float)
    for e in Compiled.profile_events[base:]:
      if type(e).__name__!="ProfileGraphEvent": continue
      sigs=[float(s) for s in e.sigs]
      for ent in e.ents: a[ANSI.sub('',str(ent.name))]+=sigs[ent.en_id]-sigs[ent.st_id]
    samples.append(a)
allk=set().union(*[s.keys() for s in samples])
agg={k:statistics.median([s.get(k,0.0) for s in samples]) for k in allk}
total=sum(agg.values())
# bucket
def role_of(name, n, k):
  if n==12288 and k==4096: return "ffn_gate_up"
  if n==4096 and k==12288: return "ffn_down"
  if n==4096 and k==4096: return "qo_proj"
  if n==1024 and k==4096: return "kv_proj"
  if k==4096 and n>50000: return "lm_head"
  return f"gemm_{n}_{k}"
gg=collections.defaultdict(lambda:[0.0,0])   # role -> [us, calls]
nongemm=collections.defaultdict(float)
for name,us in agg.items():
  mobj=re.search(r'prefill_graph_gemm_512_(\d+)_(\d+)', name)
  if mobj:
    n,k=int(mobj.group(1)),int(mobj.group(2)); r=role_of(name,n,k)
    gg[r][0]+=us; gg[r][1]+=1; gg[r].append((n,k))
  else:
    # classify non-graph-gemm
    nl=name.lower()
    if 'wmma' in nl or ('r_' in nl and 'matmul' in nl): nongemm['WMMA_fallback_matmul']+=us
    elif any(t in nl for t in ['flash','attn','softmax']): nongemm['attention']+=us
    elif any(t in nl for t in ['cast','copy','contiguous']): nongemm['copy_cast']+=us
    elif any(t in nl for t in ['norm','rope']): nongemm['norm_rope']+=us
    else: nongemm['other_'+name[:24]]+=us
gemm_total=sum(gg[r][0] for r in gg)+sum(v for kk,v in nongemm.items() if any(t in kk.lower() for t in ["tensile","cijk","gemm"]))
print(f"GEMM_BUCKET_TOTAL_MS {gemm_total/1e3:.2f}")
print(f"=== PREFILL per-role (GRAPH_GEMM={PREFILL_GRAPH_GEMM} TENSILE={PREFILL_TENSILE_GEMM}) total GPU-busy {total/1e3:.2f}ms ===")
print("--- graph-GEMM roles (achieved TFLOPS = 2*512*N*K / time) ---")
for r in sorted(gg, key=lambda x:-gg[x][0]):
  us,calls=gg[r][0],gg[r][1]; shapes=gg[r][2] if len(gg[r])>2 else None
  n,k=shapes if shapes else (0,0)
  tflops=(2*512*n*k*calls)/(us*1e-6)/1e12 if us>0 and n else 0
  print(f"  {r:14} {us/1e3:6.2f}ms ({100*us/total:4.1f}%) calls={calls:2} shape=512x{n}x{k} -> {tflops:5.1f} TFLOPS/call-equiv")
print("--- non-graph-GEMM ---")
for r in sorted(nongemm, key=lambda x:-nongemm[x])[:10]:
  print(f"  {r:24} {nongemm[r]/1e3:6.2f}ms ({100*nongemm[r]/total:4.1f}%)")
