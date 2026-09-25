"""vLLM prefill profile: one 10k-token prompt (and a group of 8), max_tokens=1, cudaProfilerStart around the call."""
import os, json, random, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model="/home/ubuntu/storage/models/NVIDIA-Nemotron-3-Nano-4B-BF16", dtype="bfloat16", trust_remote_code=True,
          mamba_ssm_cache_dtype="float32", max_num_seqs=256, max_model_len=14400, gpu_memory_utilization=0.85, seed=0,
          enable_prefix_caching=False)
rng = random.Random(11)
sp = SamplingParams(n=1, temperature=1.0, max_tokens=1, logprobs=1)
def go(P):
    p = TokensPrompt(prompt_token_ids=[rng.randrange(1000, 131072) for _ in range(P)])
    t = time.perf_counter(); llm.generate([p], sp, use_tqdm=False); torch.cuda.synchronize(); return time.perf_counter() - t
go(10000); go(10000)
torch.cuda.cudart().cudaProfilerStart(); t = go(10000); torch.cuda.synchronize(); torch.cuda.cudart().cudaProfilerStop()
print(json.dumps({"P": 10000, "wall_profiled_s": t}), flush=True)
