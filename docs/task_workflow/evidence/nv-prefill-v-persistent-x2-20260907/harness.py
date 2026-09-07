import dataclasses,json,time,statistics,numpy as np
from types import MappingProxyType
from tinygrad import Tensor,Device,TinyJit
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_q4_imma_72main_proxy import load_model_and_tokenizer
from extra.llm_research.prefill.nv_compiler_q4v_serialized_binding import binding_for as q4binding
from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for as q6binding

def patch_program(p,n):
 src=open(f'/tmp/generated_{n}_persist2.cu').read();binary=open(f'/tmp/generated_{n}_persist2.cubin','rb').read()
 info=dataclasses.replace(p.arg,global_size=(16,8,1))
 return p.replace(arg=info,src=tuple(u.replace(arg=src) if u.op is Ops.SOURCE else u.replace(arg=binary) if u.op is Ops.BINARY else u for u in p.src))
q4=q4binding('NV');q4c=dataclasses.replace(q4,main_program=patch_program(q4.main_program,'q4v'))
q6=q6binding('NV');a=q6.roles['attn_v'];ac=dataclasses.replace(a,main_program=patch_program(a.main_program,'q6v'));q6c=dataclasses.replace(q6,roles=MappingProxyType({**q6.roles,'attn_v':ac}))
m,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',512,seed=1)
x=Tensor(np.random.default_rng(7).standard_normal((512,4096)).astype(np.float16),device='NV').realize();xb=x.numpy().copy()
arms={}
for typ,b,bc,layer in [('q4',q4,q4c,4),('q6',q6,q6c,0)]:
 w=m.blk[layer].attn_v.prefill_packed_weight().contiguous().realize();wb=w.numpy().copy()
 def mk(cap,typ,w):
  def f(x):
   c=cap.new_capture();c.begin_trace();return c.project(x,w,model_family='qwen3_8b',role='attn_v',weight_type=('Q4_K' if typ=='q4' else 'Q6_K')).contiguous()
  return TinyJit(f)
 jc,jp=mk(b,typ,w),mk(bc,typ,w)
 def run(j):o=j(x);o.realize();Device['NV'].synchronize();return o
 ref=run(jc).numpy();
 for _ in range(5):run(jc);run(jp)
 ts={'control':[],'persist2':[]}
 for i in range(15):
  for n,j in ((('control',jc),('persist2',jp)) if i%2==0 else (('persist2',jp),('control',jc))):
   t=time.perf_counter_ns();run(j);ts[n].append((time.perf_counter_ns()-t)/1e3)
 got=run(jp).numpy();d=np.abs(got-ref)
 arms[typ]={'correct':bool(np.array_equal(got,ref)),'max_abs':float(d.max()),'read_only':bool(np.array_equal(w.numpy(),wb) and np.array_equal(x.numpy(),xb)),'timing':{k:{'median_us':statistics.median(v),'min_us':min(v),'samples':v} for k,v in ts.items()}}
open('/tmp/v_persist2_r15.json','w').write(json.dumps(arms,indent=2)+'\n');print(json.dumps(arms,indent=2))
