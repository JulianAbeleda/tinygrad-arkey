"""Steady-decode profile target: in-process engine, cudaProfilerStart around one generate() call.
Run under: nsys profile --capture-range=cudaProfilerApi --cuda-graph-trace=node ...
"""
import os, sys, json, random, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

BS = [tuple(int(y) for y in x.split(":")) for x in sys.argv[1].split(",")]  # B:P
N = int(sys.argv[2]) if len(sys.argv) > 2 else 400
util = float(os.environ.get("GPU_UTIL", "0.6"))
llm = LLM(model="/home/ubuntu/storage/models/NVIDIA-Nemotron-3-Nano-4B-BF16", dtype="bfloat16",
          trust_remote_code=True, mamba_ssm_cache_dtype=os.environ.get("MAMBA_DTYPE", "float32"),
          max_num_seqs=256, max_model_len=14400, gpu_memory_utilization=util, seed=0,
          **json.loads(os.environ.get("EXTRA", "{}")))
rng = random.Random(7)
def go(B, P, max_tokens):
    n = 1 if B == 1 else 8
    prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 131072) for _ in range(P)]) for _ in range(max(1, B // 8))]
    sp = SamplingParams(n=n, temperature=1.0, top_k=0, top_p=1.0, logprobs=1, max_tokens=max_tokens, ignore_eos=True)
    t = time.perf_counter(); llm.generate(prompts, sp, use_tqdm=False); torch.cuda.synchronize()
    return time.perf_counter() - t
for B, P in BS:
    go(B, P, 64)
    t_unprof = go(B, P, N)
    torch.cuda.cudart().cudaProfilerStart()
    t = go(B, P, N)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    print(json.dumps({"B": B, "P": P, "N": N, "wall_unprofiled_s": t_unprof, "wall_profiled_s": t}), flush=True)
