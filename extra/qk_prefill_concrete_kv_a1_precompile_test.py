import os, json, time
from tinygrad import Tensor, Device
from extra.llm_generate import load_model_and_tokenizer
dev=Device["AMD"]
ck=os.environ.get("PREFILL_CONCRETE_KV","0")
t_load0=time.perf_counter()
m, tok = load_model_and_tokenizer("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 1536, seed=0)
load_s=time.perf_counter()-t_load0
pfx=(tok.prefix() if hasattr(tok,"prefix") else [])
A=(pfx+tok.encode("Alpha "+("the quick brown fox jumps over a lazy dog near rivers and hills "*200)))[:1024]
dev.synchronize(); t0=time.perf_counter()
tok0=next(m.generate(list(A), chunk_size=32, temperature=0.0)); dev.synchronize()
first_gen_s=time.perf_counter()-t0
print("@@A1@@"+json.dumps({"ck":ck,"load_s":round(load_s,1),"FIRST_gen_prefill_s":round(first_gen_s,2),"tok0":tok0}))
