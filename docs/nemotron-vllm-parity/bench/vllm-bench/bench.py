"""vLLM oracle benchmark for Nemotron 3 Nano 4B BF16 (RL rollout workload).

Workload: groups of n=8, temperature 1.0, top_k off, top_p 1, logprobs=1.
B = total concurrent sequences = (#prompts) * 8  (B=1 -> one prompt, n=1).
For each (P, B): T1 = wall for max_tokens=1 (prefill phase), TN = wall for max_tokens=N.
decode tok/s = B*(N-1)/(TN-T1); step latency (ITL) = (TN-T1)/(N-1); e2e tok/s = B*N/TN.
Prefix caching is left at vLLM's default; distinct prompts per run (random token ids, fresh seed)
so nothing is reused across runs.
"""
import argparse, json, os, random, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/home/ubuntu/storage/models/NVIDIA-Nemotron-3-Nano-4B-BF16")
ap.add_argument("--prompt-lens", default="200,2000,10000")
ap.add_argument("--bs", default="1,8,32,64,128,256")
ap.add_argument("--max-tokens", type=int, default=4096)
ap.add_argument("--mamba-dtype", default="float32")
ap.add_argument("--max-num-seqs", type=int, default=256)
ap.add_argument("--max-model-len", type=int, default=14400)
ap.add_argument("--gpu-util", type=float, default=0.0)  # 0 -> derive from free memory
ap.add_argument("--natural", action="store_true", help="natural-EOS variant with real text prompts")
ap.add_argument("--prefill-lens", default="", help="measure single-prompt prefill (max_tokens=1) at these lengths first")
ap.add_argument("--out", required=True)
ap.add_argument("--extra", default="{}", help="json of extra LLM kwargs")
args = ap.parse_args()

def free_frac():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"]).decode().split(",")
    return float(out[0]) / float(out[1])

util = args.gpu_util or round(min(0.85, free_frac() - 0.08), 3)
print(f"gpu_memory_utilization={util}", flush=True)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import vllm, torch

llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
          mamba_ssm_cache_dtype=args.mamba_dtype, max_num_seqs=args.max_num_seqs,
          max_model_len=args.max_model_len, gpu_memory_utilization=util, seed=0,
          **json.loads(args.extra))
tok = llm.get_tokenizer()
vocab = 131072
rng = random.Random(1234)

def rand_prompt(P):
    # avoid special ids (<1000) so the prompt is ordinary text-like tokens
    return TokensPrompt(prompt_token_ids=[rng.randrange(1000, vocab) for _ in range(P)])

readme = open(os.path.join(args.model, "README.md")).read()
def text_prompt(P, salt):
    # real-text prompt ending in a question; chat template so EOS is natural
    q = f"(variant {salt}) Summarize the key facts in the document above, then solve: what is {salt} * 37 + 11? Think step by step."
    body_ids = tok(readme, add_special_tokens=False).input_ids
    head = tok(tok.apply_chat_template([{"role": "user", "content": "DOC"}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    budget = max(0, P - len(head) - len(tok(q).input_ids) - 8)
    off = (salt * 997) % max(1, len(body_ids) - budget) if len(body_ids) > budget else 0
    doc_ids = (body_ids * (budget // max(1, len(body_ids)) + 2))[off:off + budget]
    doc = tok.decode(doc_ids)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": doc + "\n\n" + q}], tokenize=False, add_generation_prompt=True), add_special_tokens=False).input_ids
    return TokensPrompt(prompt_token_ids=ids)

def run(P, B, max_tokens, ignore_eos=True, natural=False):
    n = 1 if B == 1 else 8
    nprompts = max(1, B // 8) if B > 1 else 1
    sp = SamplingParams(n=n, temperature=1.0, top_k=0, top_p=1.0, logprobs=1, max_tokens=max_tokens,
                        ignore_eos=ignore_eos)
    prompts = [text_prompt(P, rng.randrange(1 << 20)) if natural else rand_prompt(P) for _ in range(nprompts)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    lens = [len(c.token_ids) for o in outs for c in o.outputs]
    has_lp = all(c.logprobs is not None and len(c.logprobs) == len(c.token_ids) for o in outs for c in o.outputs)
    return dt, lens, has_lp

results = {"vllm": vllm.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
           "gpu_util": util, "mamba_dtype": args.mamba_dtype, "extra": args.extra, "rows": []}

# warmup (compile + graphs already captured at init; this warms sampler / logprob paths)
run(200, 8, 64)

def save():
    json.dump(results, open(args.out, "w"), indent=1)

if args.prefill_lens:
    for P in [int(x) for x in args.prefill_lens.split(",")]:
        ts = []
        for _ in range(3):
            dt, _, _ = run(P, 1, 1)
            ts.append(dt)
        row = {"kind": "prefill", "P": P, "t1_s": ts}
        print(json.dumps(row), flush=True); results["rows"].append(row); save()
    # also: a group of 8 (n=8) prefill, to see whether the prompt is computed once or 8x
    for P in [int(x) for x in args.prefill_lens.split(",")]:
        dt, _, _ = run(P, 8, 1)
        row = {"kind": "prefill_group8", "P": P, "t1_s": dt}
        print(json.dumps(row), flush=True); results["rows"].append(row); save()

for P in [int(x) for x in args.prompt_lens.split(",") if x]:
    for B in [int(x) for x in args.bs.split(",")]:
        if P + args.max_tokens > args.max_model_len:
            continue
        t1, _, _ = run(P, B, 1, natural=args.natural)
        if args.natural:
            tn, lens, has_lp = run(P, B, args.max_tokens, ignore_eos=False, natural=True)
            row = {"kind": "natural", "P": P, "B": B, "t1_s": t1, "tN_s": tn, "gen_tokens": sum(lens),
                   "mean_len": sum(lens) / len(lens), "max_len": max(lens), "e2e_tok_s": sum(lens) / tn, "logprobs_ok": has_lp}
        else:
            tn, lens, has_lp = run(P, B, args.max_tokens)
            N = args.max_tokens
            assert all(l == N for l in lens), set(lens)
            row = {"kind": "fixed", "P": P, "B": B, "N": N, "t1_s": t1, "tN_s": tn,
                   "decode_tok_s": B * (N - 1) / (tn - t1), "itl_ms": 1000 * (tn - t1) / (N - 1),
                   "e2e_tok_s": B * N / tn, "logprobs_ok": has_lp}
        print(json.dumps(row), flush=True)
        results["rows"].append(row); save()
