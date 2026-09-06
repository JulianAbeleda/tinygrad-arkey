import numpy as np,time,statistics,pathlib
from tinygrad import Tensor,Device,TinyJit
from extra.llm_research.layout import read_metadata,packed_u16_slice
from extra.llm_research.prefill.nv_compiler_q6k_pp512_binding import binding_for as cb
from extra.llm_research.prefill.nv_qkv_packed_pp512_binding import binding_for as lb
P=pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf');md=read_metadata(P); infos=[i for i in md.infos if i.name.endswith('.attn_v.weight') and i.typ==14]; print('weights',len(infos))
w=[packed_u16_slice(P,md,i,device='NV').contiguous().realize() for i in infos]; z=np.random.default_rng(3).standard_normal((512,4096)).astype(np.float16); x=Tensor(z,device='NV').realize(); ca=cb('NV'); la=lb('NV').new_capture()
def fn(a):
 c=ca.new_capture();c.begin_trace();return tuple(c.project(a,q,model_family='qwen3_8b',role='attn_v') for q in w)
def gn(a):
 la.begin_trace();return tuple(la.project_q6_v(a,q) for q in w)
cj,lj=TinyJit(fn),TinyJit(gn)
def run(j,a):o=j(a);Tensor.realize(*o);Device['NV'].synchronize();return np.stack([v.numpy().copy() for v in o])
for _ in range(3):run(cj,x);run(lj,x)
cs=[];ls=[]
for i in range(31):
 for n,j in ([('c',cj),('l',lj)] if i%2==0 else [('l',lj),('c',cj)]):
  t=time.perf_counter_ns();run(j,x);(cs if n=='c' else ls).append((time.perf_counter_ns()-t)/1e6)
yc,yl=run(cj,x),run(lj,x); xb=Tensor(z*.7,device='NV').realize();yc2,yl2=run(cj,xb),run(lj,xb); print({'candidate_ms':statistics.median(cs),'llama_ms':statistics.median(ls),'max_abs':float(np.max(abs(yc-yl))),'allclose':bool(np.allclose(yc,yl,rtol=.02,atol=.5)),'second_close':bool(np.allclose(yc2,yl2,rtol=.02,atol=.5)),'distinct':float(np.max(abs(yc2-yc)))>0})
