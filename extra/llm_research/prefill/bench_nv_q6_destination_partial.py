import hashlib,json,pathlib,statistics
import numpy as np
from tinygrad import Device,Tensor,dtypes
from tinygrad.runtime.ops_nv import NVProgram
from tinygrad.uop.ops import UOp
from extra.llm_research.layout import GGML_Q6_K,packed_u16_slice,read_metadata
from extra.llm_research.prefill.bench_nv_q6_oracle_broad_cta import _record as broad_record
from extra.llm_research.prefill.bench_nv_q6_oracle_reduction_policy import LAUNCH_SHARED_BYTES,_buf,_compile_ast
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record as wide_record
from extra.llm_research.prefill.nv_q6_oracle_reduction_policy import M,N,K,K256,OWNERS,TILES,TILES_M,COLS,TILE_ELEMS,build_packed_one_body_ast,build_reduction_schedule,emit_ordered_fixup
from extra.llm_research.prefill.nv_q6_sass_census import analyze_cubin
ROOT=pathlib.Path('docs/task_workflow/evidence/nv-q6-destination-partial-full-gate19-20260901');ART=ROOT/'artifacts';ART.mkdir(parents=True,exist_ok=True)
def stats(x):return {'samples_us':x,'min_us':min(x),'median_us':statistics.median(x),'max_us':max(x)}
def fix_source():return '''#include <cuda_runtime.h>
extern "C" __global__ __launch_bounds__(128) void nv_q6_destination_major_fixup(float *out,const float *partials,const int *slots,const int *counts) {
 const int tile=blockIdx.x,slice=blockIdx.y,lane=threadIdx.x,count=counts[tile];
 const int s0=slots[tile*3],s1=slots[tile*3+1],s2=slots[tile*3+2];
 #pragma unroll 1
 for(int iteration=0;iteration<32;iteration++) { const int wr=lane,mc=slice*32+iteration,z=mc*128+wr; float acc=0.0f;
  if(count>0) acc=__fadd_rn(acc,partials[s0*16384+z]); if(count>1) acc=__fadd_rn(acc,partials[s1*16384+z]); if(count>2) acc=__fadd_rn(acc,partials[s2*16384+z]);
  const int mt=tile%4,nt=tile/4; out[(mt*128+mc)*4096+nt*128+wr]=acc; }
}'''
def compile_fix():
 d=ART/'candidate_fixup';d.mkdir(exist_ok=True);src=fix_source();(d/'candidate_fixup.cu').write_text(src);b=Device['NV'].compiler.compile(src);p=d/'candidate_fixup.cubin';p.write_bytes(b);c=analyze_cubin(p,d/'sass','nv_q6_destination_major_fixup')['summary'];return b,c,hashlib.sha256(b).hexdigest()
anchor_ast=build_packed_one_body_ast();candidate_ast=build_packed_one_body_ast(partial_output_layout='destination_major')
an,ab,aa=_compile_ast(anchor_ast,'anchor_main',ART);cn,cb,ca=_compile_ast(candidate_ast,'candidate_main',ART)
p=lambda n,t,i:UOp.placeholder((n,),t,i)
fast=emit_ordered_fixup(p(M*N,dtypes.float32,0),p(2*OWNERS*TILE_ELEMS,dtypes.float32,1),p(TILES*3,dtypes.int32,2),p(TILES,dtypes.int32,3))
fn,fb,fa=_compile_ast(fast,'anchor_fixup',ART);cfb,cfs,cfsha=compile_fix()
schedule=build_reduction_schedule();slot_map,counts,_=schedule.arrays();slots=Tensor(slot_map.reshape(-1),device='NV').contiguous().realize();cts=Tensor(counts,device='NV').contiguous().realize()
meta=read_metadata(pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'));info=next(x for x in meta.infos if x.name=='blk.0.ffn_down.weight');assert info.typ==GGML_Q6_K
halfs=packed_u16_slice(pathlib.Path('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'),meta,info,device='NV').contiguous().realize();_,q,scales=wide_record(M,K);broad=[]
for mt in range(TILES_M):
 for epoch in range(K256):broad.append(broad_record(np.ascontiguousarray(q[mt*COLS:(mt+1)*COLS,epoch*256:(epoch+1)*256].T),np.ascontiguousarray(scales[mt*COLS:(mt+1)*COLS,epoch*8:(epoch+1)*8].T)))
q8=Tensor(np.concatenate(broad).reshape(-1),device='NV').contiguous().realize()
programs={'anchor_main':NVProgram(Device['NV'],an,ab,shared_mem=LAUNCH_SHARED_BYTES),'candidate_main':NVProgram(Device['NV'],cn,cb,shared_mem=LAUNCH_SHARED_BYTES),'anchor_fixup':NVProgram(Device['NV'],fn,fb),'candidate_fixup':NVProgram(Device['NV'],'nv_q6_destination_major_fixup',cfb)}
partials={'anchor':Tensor.full((2*OWNERS*TILE_ELEMS),float('nan'),device='NV').contiguous().realize(),'candidate':Tensor.full((2*OWNERS*TILE_ELEMS),float('nan'),device='NV').contiguous().realize()};outs={'anchor':Tensor.full((M,N),float('nan'),device='NV').contiguous().realize(),'candidate':Tensor.full((M,N),float('nan'),device='NV').contiguous().realize()}
def main(arm):return programs[arm+'_main'](_buf(partials[arm]),_buf(halfs),_buf(q8),global_size=(OWNERS,1,1),local_size=(256,1,1),wait=True,timeout=120000)*1e6
def fix(arm):
 p=programs[arm+'_fixup'];return p(_buf(outs[arm]),_buf(partials[arm]),_buf(slots),_buf(cts),global_size=((TILES,1,1) if arm=='anchor' else (TILES,4,1)),local_size=((256,1,1) if arm=='anchor' else (128,1,1)),wait=True)*1e6
main('anchor');main('candidate');fix('anchor');fix('candidate');ra=partials['anchor'].numpy().reshape(2*OWNERS,128,128);rc=partials['candidate'].numpy().reshape(2*OWNERS,128,128)
active=sorted(schedule.active_slots);unused=sorted(set(range(2*OWNERS))-set(active));partial_exact=bool(np.array_equal(ra[active].view(np.uint32),rc[active].transpose(0,2,1).copy().view(np.uint32)));unused_nan=bool(np.isnan(rc[unused]).all());oa,oc=outs['anchor'].numpy(),outs['candidate'].numpy();output_exact=bool(np.array_equal(oa.view(np.uint32),oc.view(np.uint32)))
for i in range(3):
 for arm in (('anchor','candidate') if i%2==0 else ('candidate','anchor')):main(arm);fix(arm)
s={a:{'main':[],'fixup':[],'total':[]} for a in ('anchor','candidate')};orders=[]
for i in range(31):
 order=('anchor','candidate') if i%2==0 else ('candidate','anchor');orders.append(order)
 for arm in order:
  m=main(arm);f=fix(arm);s[arm]['main'].append(m);s[arm]['fixup'].append(f);s[arm]['total'].append(m+f)
paired={};
for k in ('main','fixup','total'):
 d=[c-a for a,c in zip(s['anchor'][k],s['candidate'][k])];med=statistics.median(d);mad=statistics.median(abs(x-med) for x in d);paired[k]={'median_us':med,'mad_us':mad,'wins':sum(x<0 for x in d),'samples_us':d}
res={'schema':'tinygrad.nv_q6_destination_partial_full_gate.v1','correctness':{'active_partials_permutation_bit_exact':partial_exact,'unused_candidate_nan':unused_nan,'final_output_bit_exact':output_exact,'finite':bool(np.isfinite(oc).all())},'compiler':{'anchor_main':aa,'candidate_main':ca,'anchor_fixup':fa,'candidate_fixup':{'cubin_sha256':cfsha,'sass':cfs}},'timing':{a:{k:stats(v) for k,v in x.items()} for a,x in s.items()},'paired':paired,'orders':orders,'promotion_gate':partial_exact and unused_nan and output_exact and bool(np.isfinite(oc).all()) and paired['total']['median_us']<=-3 and paired['total']['wins']>=24}
ROOT.mkdir(exist_ok=True);(ROOT/'result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps({'correctness':res['correctness'],'paired':paired,'medians':{a:{k:statistics.median(v) for k,v in x.items()} for a,x in s.items()},'promotion_gate':res['promotion_gate'],'hashes':{'anchor_main':aa['cubin_sha256'],'candidate_main':ca['cubin_sha256'],'candidate_fixup':cfsha}},sort_keys=True))
