import dataclasses,json,numpy as np
from tinygrad import Tensor,Device,TinyJit
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for,M,K
from extra.llm_research.prefill.nv_q4_imma_72main_proxy import load_model_and_tokenizer
base=binding_for('NV',variant='streamk',pair_q8_reuse=False,native_weight_a_x4=True,coalesced_swapped_output=True)
def arm(n):
 src=open('/tmp/gate_'+n+'.cu').read();bin=open('/tmp/gate_'+n+'.cubin','rb').read();p=base.q_program.replace(src=tuple(u.replace(arg=src) if u.op is Ops.SOURCE else u.replace(arg=bin) if u.op is Ops.BINARY else u for u in base.q_program.src));return dataclasses.replace(base,q_program=p,records=[],outputs=[],partials=[],partial_ids=[],q_records=[])
a,b=arm('q4pubprod'),arm('q4pub_q8uint2');m,_=load_model_and_tokenizer('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf',512,seed=1);w=m.blk[0].ffn_gate.prefill_packed_weight().contiguous().realize();x=Tensor(np.random.default_rng(2).standard_normal((M,K)).astype(np.float16),device='NV').realize()
def f(c):
 def go(x):c.begin_trace();return tuple(c.project(x,w,model_family='qwen3_8b',role='ffn_gate') for _ in range(8))
 return TinyJit(go)
ja,jb=f(a),f(b)
def run(j):o=j(x);[z.realize() for z in o];Device['NV'].synchronize();return [z.numpy().copy() for z in o]
ref=run(ja);rows=[]
for cyc in range(10):
 ctl,cand=run(ja),run(jb);rows.append({'cycle':cyc,'control':all(np.array_equal(x,y) for x,y in zip(ref,ctl)),'candidate':all(np.array_equal(x,y) for x,y in zip(ref,cand)),'max_abs':max(float(np.abs(x-y).max()) for x,y in zip(ref,cand))})
r={'all_control':all(x['control'] for x in rows),'all_candidate':all(x['candidate'] for x in rows),'rows':rows};open('/tmp/gate_q8uint2_multicall.json','w').write(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
