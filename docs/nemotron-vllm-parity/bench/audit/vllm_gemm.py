"""vLLM-side GEMM microbench: exactly what vLLM 0.30 runs for an unquantized bf16 linear on CUDA
(model_executor/layers/utils.py default_unquantized_gemm = torch.nn.functional.linear, bf16 in / bf16 out).

usage: vllm_gemm.py OUT.json            (timing: CUPTI kernel durations via torch.profiler)
       vllm_gemm.py --ncu ROLES MS      (one call per shape inside cudaProfilerStart/Stop, for ncu --profile-from-start off)

L2 discipline: the RTX 5090 has a 96 MB L2 and several weights fit in it, while the real model streams 42 layers of
distinct weights. Every measurement rotates over R distinct weight copies whose total is >= 384 MB.
"""
import sys, json, collections, tempfile, os
import torch
import torch.nn.functional as F

# vLLM's logical ops for Nemotron-3-Nano-4B (N, K). vLLM fuses q/k/v into one QKVParallelLinear (5120+1024+1024).
ROLES = {"ssm_in": (17504, 3136), "ssm_out": (3136, 7680), "qkv": (7168, 3136), "attn_q": (5120, 3136),
         "attn_kv": (1024, 3136), "attn_o": (3136, 5120), "ffn_up": (12544, 3136), "ffn_down": (3136, 12544),
         "output": (131072, 3136)}
MS = [8, 16, 32, 64, 128, 512, 1024, 2048, 4096, 8192]
ITERS = 20

def weights(n, k):
  r = max(2, -(-384 * 2**20 // (n * k * 2)))
  return [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(r)]

def ncu_one(role, m):
  n, k = ROLES[role]
  ws = weights(n, k); x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
  for w in ws: F.linear(x, w)
  torch.cuda.synchronize()
  torch.cuda.nvtx.range_push(f"{role}/{m}")
  torch.cuda.cudart().cudaProfilerStart()
  F.linear(x, ws[0])
  torch.cuda.synchronize()
  torch.cuda.cudart().cudaProfilerStop()
  torch.cuda.nvtx.range_pop()
  del ws; torch.cuda.empty_cache()

def bench(role, m):
  n, k = ROLES[role]
  ws = weights(n, k); x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
  for _ in range(3):
    for w in ws: F.linear(x, w)
  torch.cuda.synchronize()
  calls = ITERS * len(ws)
  from torch.profiler import profile, ProfilerActivity
  with profile(activities=[ProfilerActivity.CUDA]) as prof:
    for _ in range(ITERS):
      for w in ws: F.linear(x, w)
    torch.cuda.synchronize()
  fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
  prof.export_chrome_trace(path)
  ev = [e for e in json.load(open(path))["traceEvents"] if e.get("cat") == "kernel"]
  os.unlink(path)
  kern = collections.OrderedDict()
  for e in ev:
    a = e.get("args", {})
    key = e["name"]
    d = kern.setdefault(key, {"name": key, "count": 0, "us": 0.0, "grid": a.get("grid"), "block": a.get("block"),
                              "regs": a.get("registers per thread"), "smem": a.get("shared memory")})
    d["count"] += 1; d["us"] += e["dur"]
  ks = list(kern.values())
  for d in ks: d["us_per_call"] = d["us"] / calls; d["per_call"] = d["count"] / calls
  # wall with CUDA events over back-to-back launches (includes launch gaps; reference only)
  s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  s.record()
  for _ in range(ITERS):
    for w in ws: F.linear(x, w)
  e.record(); torch.cuda.synchronize()
  return {"role": role, "m": m, "n": n, "k": k, "copies": len(ws), "kernels": ks,
          "gpu_us": sum(d["us_per_call"] for d in ks), "event_us": s.elapsed_time(e) * 1e3 / calls}

if __name__ == "__main__":
  if sys.argv[1] == "--ncu":
    # vllm_gemm.py --ncu ROLES MS : one profiled call per (role, M), in this order
    for role in sys.argv[2].split(","):
      for m in (int(v) for v in sys.argv[3].split(",")): ncu_one(role, m)
    sys.exit(0)
  roles = sys.argv[2].split(",") if len(sys.argv) > 2 else list(ROLES)
  out = []
  for role in roles:
    for m in MS:
      r = bench(role, m); out.append(r)
      n, k = r["n"], r["k"]
      print(f"{role:8s} M={m:5d} {r['gpu_us']:9.1f}us {2*m*n*k/r['gpu_us']/1e6:7.1f}TF  "
            + " | ".join(f"{d['name'][:90]} g{d['grid']} x{d['per_call']:.0f}" for d in r["kernels"]), flush=True)
      torch.cuda.empty_cache()
  json.dump(out, open(sys.argv[1], "w"), indent=1)
