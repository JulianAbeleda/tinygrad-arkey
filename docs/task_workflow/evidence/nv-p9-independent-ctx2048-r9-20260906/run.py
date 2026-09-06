import gc,json,time
from tinygrad import Device
from tinygrad.helpers import GlobalCounters
from tinygrad.llm.generate import load_model_and_tokenizer
from extra.llm_research.decode.decode_runtime_overhead import _make_prompt,_prefill,_nv_gpu_state
M='/home/ubuntu/storage/models/Qwen3-8B-Q4_K_M.gguf'; model,tok=load_model_and_tokenizer(M,2560,seed=20260617)
base=(tok.prefix() if hasattr(tok,'prefix') else [])+tok.encode('the quick brown fox jumps. '*800); prompt=_make_prompt(base,2048)
out={'schema':'nv-p9-patched-two-request.v1','commit':'ce714b3df','rows':[]}
for i in range(2):
 row={'request':i+1,'before':_nv_gpu_state(),'global_before':GlobalCounters.mem_used_per_device[Device.DEFAULT],'started':time.time()}
 try:
  model.reset_generation_state(); gen,first=_prefill(model,prompt,32,4); row.update(status='success',first=first); gen.close(); del gen
 except Exception as e: row.update(status='error',error_type=type(e).__name__,message=str(e))
 gc.collect(); Device[Device.DEFAULT].synchronize(); row.update(after=_nv_gpu_state(),global_after=GlobalCounters.mem_used_per_device[Device.DEFAULT],elapsed_s=time.time()-row['started']); out['rows'].append(row)
 if row['status']!='success': break
open('/tmp/nv-p9-two-request-ctx2048-r9.json','w').write(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps(out))
