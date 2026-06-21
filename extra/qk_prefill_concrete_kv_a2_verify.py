import os, json, time
from tinygrad import Tensor, Device
from extra.llm_generate import load_model_and_tokenizer
dev=Device["AMD"]
ck=os.environ.get("PREFILL_CONCRETE_KV","0")
m, tok = load_model_and_tokenizer("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf", 1536, seed=0)
pfx=(tok.prefix() if hasattr(tok,"prefix") else [])
# ensure >=1024 ACTUAL tokens (truncate token list, not text). Divergent first word -> early start_pos.
A=(pfx+tok.encode("Alpha "+("the quick brown fox jumps over a lazy dog near rivers and hills "*200)))[:1024]
B=(pfx+tok.encode("Zeta "+("a distant galaxy spins beyond bright stars while comets streak void "*200)))[:1024]
T=type(m); orig=T.__call__; log=[]
def tr(self,tokens,start_pos,*a,**k):
    sh=tokens.shape[1]; log.append((type(start_pos).__name__, sh if isinstance(sh,int) else "sym"))
    return orig(self,tokens,start_pos,*a,**k)
T.__call__=tr
def prefill_wall(ids):
    log.clear(); dev.synchronize(); t0=time.perf_counter()
    next(m.generate(list(ids), chunk_size=32, temperature=0.0)); dev.synchronize()
    return (time.perf_counter()-t0)*1e3, list(log)
cold_ms, cold_log = prefill_wall(A)
warm_ms, warm_log = prefill_wall(B)
print("@@V@@"+json.dumps({"ck":ck,"lenA":len(A),"cold_ms":round(cold_ms,1),"warm_ms":round(warm_ms,1),
  "cold_calls":cold_log,"warm_calls":warm_log}))
