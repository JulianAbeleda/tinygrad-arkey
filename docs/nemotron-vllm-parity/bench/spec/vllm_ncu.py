"""vLLM target for ncu: cudaProfilerStart/Stop around one decode (or prefill) generate() call.
usage: vllm_ncu.py decode B P N | prefill P
Run under: ncu --profile-from-start off --kernel-name regex:... --launch-skip K --launch-count C ...
"""
import os, sys, random, time, json
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
import torch
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

mode = sys.argv[1]
llm = LLM(model="/home/ubuntu/storage/models/NVIDIA-Nemotron-3-Nano-4B-BF16", dtype="bfloat16",
          trust_remote_code=True, mamba_ssm_cache_dtype="float32", max_num_seqs=256, max_model_len=14400,
          gpu_memory_utilization=0.85, seed=0)
rng = random.Random(11)
if mode == "decode":
  B, P, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
  prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 131072) for _ in range(P)]) for _ in range(max(1, B // 8))]
  sp = SamplingParams(n=1 if B == 1 else 8, temperature=1.0, top_k=0, top_p=1.0, logprobs=1, max_tokens=N, ignore_eos=True)
else:
  P = int(sys.argv[2])
  prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 131072) for _ in range(P)])]
  sp = SamplingParams(n=1, temperature=1.0, top_k=0, top_p=1.0, logprobs=1, max_tokens=1, ignore_eos=True)
llm.generate(prompts, sp, use_tqdm=False)  # warm (a different random prompt below avoids the prefix cache)
prompts = [TokensPrompt(prompt_token_ids=[rng.randrange(1000, 131072) for _ in range(len(p["prompt_token_ids"]))]) for p in prompts]
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
llm.generate(prompts, sp, use_tqdm=False)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("done", flush=True)
