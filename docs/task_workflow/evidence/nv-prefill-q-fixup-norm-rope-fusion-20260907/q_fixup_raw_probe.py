import dataclasses,json,time,statistics,numpy as np
from tinygrad import Tensor,Device,TinyJit,dtypes
from tinygrad.llm.model import precompute_freqs_cis
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program
from extra.llm_research.prefill.nv_compiler_q4k_qo_binding import binding_for
from extra.llm_research.prefill.nv_q4_imma_72main_proxy import load_model_and_tokenizer
SRC=r'''extern "C" __global__ void q_fused(float *out,const float *partials,const int *map,const int *active,const float *nw,const float *fr,int M,int N) {
  int tile=active[blockIdx.x],z=blockIdx.y*4096+threadIdx.x,s0=map[3*tile],s1=map[3*tile+1],s2=map[3*tile+2];
  int nb=(tile%(N/128))*128,mb=(tile/(N/128))*128; if(s0<0)return;
  for(;z<(blockIdx.y+1)*4096;z+=128) { int r=z/128,c=z%128; out[(mb+r)*N+nb+c]=partials[s0*16384+z]+(s1>=0?partials[s1*16384+z]:0)+(s2>=0?partials[s2*16384+z]:0); }
}'''
m,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',512,seed=1); w=m.blk[0].attn_q.prefill_packed_weight().contiguous().realize(); x=Tensor(np.random.default_rng(99).standard_normal((512,4096)).astype(np.float16),device='NV').realize(); nw=m.blk[0].attn_q_norm.weight.contiguous().realize(); fr=precompute_freqs_cis(128,512,theta=m.config.rope_theta,device='NV').contiguous().realize()
sk=dataclasses.replace(binding_for('NV',variant='streamk'),population=36,roles=('attn_q',)).new_capture()
# Capture one production main and its raw workspaces.
raw=sk.project(x,w,model_family='qwen3_8b',role='attn_q').realize(); Device['NV'].synchronize(); partial=sk.partials[-1]
bin=NVRTCCompiler(Device['NV'].arch,ptx=False).compile(SRC)
p=native_nv_program('q_fused',bin,global_size=(len(sk.active.numpy()),4,1),local_size=(128,1,1),globals=(0,1,2,3,4,5),outs=(0,),ins=(1,2,3,4,5),vals=(512,4096))
def fused():
 out=Tensor.empty(512*4096,dtype=dtypes.float32,device='NV'); out,_,_,_,_,_=out.uop_program(partial,sk.slots,sk.active,nw,fr,fxn=lambda *_:p); return out.reshape(512,4096).contiguous()
def baseline(): return raw.contiguous()
jb,jf=TinyJit(baseline),TinyJit(fused)
def run(j): o=j();o.realize();Device['NV'].synchronize();return o
ref=raw.numpy(); got=run(jf).numpy(); diff=np.abs(ref.astype(np.float32)-got.astype(np.float32)); print({'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'exact':bool(np.array_equal(ref,got))}); raise SystemExit
for _ in range(5):run(jb);run(jf)
s={'baseline':[],'fused':[]}
for i in range(15):
 for k,j in ((('baseline',jb),('fused',jf)) if i%2==0 else (('fused',jf),('baseline',jb))):
  t=time.perf_counter_ns();run(j);s[k].append((time.perf_counter_ns()-t)/1e3)
r={'max_abs':float(diff.max()),'mean_abs':float(diff.mean()),'allclose':bool(np.allclose(ref,got,rtol=.02,atol=.01)),'timing':{k:{'median_us':statistics.median(v),'min_us':min(v),'samples':v} for k,v in s.items()}}
open('/tmp/q_fixup_norm_rope_probe.json','w').write(json.dumps(r,indent=2)+'\n');print(r)
