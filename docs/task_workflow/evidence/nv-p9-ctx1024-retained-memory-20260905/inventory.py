import json,time
from tinygrad import Device
from tinygrad.uop.ops import Ops,buffers
from tinygrad.engine.realize import get_call_arg_uops, graph_cache
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.generate import load_model_and_tokenizer
from extra.llm_research.decode.decode_runtime_overhead import _make_prompt,_prefill,_decode_jits,_nv_gpu_state
MODEL='/home/ubuntu/storage/models/Qwen3-8B-Q4_K_M.gguf'; DEPTH=1024; MAXC=1536

def inventory(model):
 out={}
 def visit(name,x):
  if hasattr(x,'cnt') and hasattr(x,'captured'):
   if x.captured is None: out[name]={'cnt':x.cnt,'captured':False}; return
   cap=x.captured; allocs={}; shadows={}
   for call in cap.linear.toposort():
    if call.op is not Ops.CALL: continue
    for u in get_call_arg_uops(call):
     u=u.base
     if (b:=buffers.get(u)) is None: continue
     base=b._base if b._base is not None else b
     allocs[id(base)]={'bytes':base.nbytes,'device':base.device,'initialized':base.is_initialized()}
   for slot,u in cap._written_input_shadows.items():
    if (b:=buffers.get(u.base)) is None: continue
    base=b._base if b._base is not None else b
    shadows[slot]={'bytes':base.nbytes,'device':base.device,'initialized':base.is_initialized()}
   out[name]={'cnt':x.cnt,'captured':True,'linear_calls':len(cap.linear.src),'unique_base_allocations':len(allocs),
              'unique_base_bytes':sum(v['bytes'] for v in allocs.values()),'written_input_shadows':shadows,
              'written_input_shadow_bytes':sum(v['bytes'] for v in shadows.values()),'allocation_ids':allocs}
  elif isinstance(x,(tuple,list)):
   for i,y in enumerate(x): visit(f'{name}[{i}]',y)
  elif isinstance(x,dict):
   for k,y in x.items(): visit(f'{name}[{k!r}]',y)
 for name,x in vars(model).items():
  if 'jit' in name: visit(name,x)
 return out

def snap(model,stage):
 inv=inventory(model); unions={}
 for row in inv.values(): unions.update(row.get('allocation_ids',{}))
 for row in inv.values(): row.pop('allocation_ids',None)
 return {'stage':stage,'gpu_state':_nv_gpu_state(),'global_mem_used':GlobalCounters.mem_used_per_device[Device.DEFAULT],
         'graph_cache_entries':len(graph_cache),'jit_union_allocations':len(unions),'jit_union_bytes':sum(v['bytes'] for v in unions.values()),'jits':inv}

dev=Device[Device.DEFAULT]; model,tok=load_model_and_tokenizer(MODEL,MAXC,seed=20260617)
base=(tok.prefix() if hasattr(tok,'prefix') else [])+tok.encode('the quick brown fox jumps. '*800)
res={'schema':'nv-p9-jit-retained-memory.v1','depth':DEPTH,'max_context':MAXC,'snapshots':[snap(model,'after_load')]}
model.reset_generation_state(); gen,first=_prefill(model,_make_prompt(base,DEPTH),32,4)
res['snapshots'].append(snap(model,'after_prefill_first_yield'))
for _ in range(3): next(gen)
dev.synchronize(); res['snapshots'].append(snap(model,'after_three_decode_warm_tokens')); gen.close()
open('/tmp/nv-p9-memory-owner-ctx1024.json','w').write(json.dumps(res,indent=2,sort_keys=True)+'\n')
print(json.dumps({s['stage']:(s['global_mem_used'],s['jit_union_bytes'],s['graph_cache_entries']) for s in res['snapshots']}))
