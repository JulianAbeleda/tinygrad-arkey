import dataclasses,time,statistics,json,numpy as np
from tinygrad import Tensor,Device,TinyJit
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for,M,K
base=binding_for('NV',variant='streamk',pair_q8_reuse=False,native_weight_a_x4=True,coalesced_swapped_output=True)
variants={}
for n in ('q4pubprod','q4pub_q8uint2'):
 src=open(f'/tmp/gate_{n}.cu').read();binary=open(f'/tmp/gate_{n}.cubin','rb').read()
 p=base.q_program.replace(src=tuple(u.replace(arg=src) if u.op is Ops.SOURCE else u.replace(arg=binary) if u.op is Ops.BINARY else u for u in base.q_program.src))
 variants[n]=dataclasses.replace(base,q_program=p,records=[],outputs=[],partials=[],partial_ids=[],q_records=[])
from extra.llm_research.prefill.nv_q4_imma_72main_proxy import load_model_and_tokenizer
model,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',4608,seed=20260617);w=model.blk[0].ffn_gate.prefill_packed_weight().contiguous().realize();x=Tensor(np.random.default_rng(20260907).standard_normal((M,K),dtype=np.float32).astype(np.float16),device='NV').realize();wb=w.numpy().copy();xb=x.numpy().copy()
def make(cap):
 def f(x):cap.begin_trace();return cap.project(x,w,model_family='qwen3_8b',role='ffn_gate').contiguous()
 return TinyJit(f)
j={n:make(c) for n,c in variants.items()}
def run(k):o=j[k](x);o.realize();Device['NV'].synchronize();return o
# baseline health first
ref=run('q4pubprod').numpy(); assert np.isfinite(ref).all()
for _ in range(5):
 for k in j:run(k)
s={k:[] for k in j}
for z in range(15):
 order=list(j) if z%2==0 else list(reversed(j))
 for k in order:
  t=time.perf_counter_ns();run(k);s[k].append((time.perf_counter_ns()-t)/1e3)
r={}
for k in j:
 a=run(k).numpy();d=np.abs(a-ref);r[k]={'max_abs':float(d.max()),'mean_abs':float(d.mean()),'allclose':bool(np.allclose(a,ref,rtol=2e-5,atol=2e-3)),'median_us':statistics.median(s[k]),'min_us':min(s[k]),'samples':s[k]}
r['read_only']=bool(np.array_equal(w.numpy(),wb) and np.array_equal(x.numpy(),xb));open('/tmp/gate_half_variants_r15.json','w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
