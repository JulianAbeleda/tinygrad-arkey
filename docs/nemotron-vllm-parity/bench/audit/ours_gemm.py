"""BoltBeam-side GEMM microbench: each Nemotron-H projection through the production _Linear path.

usage: DEV=NV PROFILE=1 HCQ_GRAPH_PROFILE_JSON=prof.jsonl python3 ours_gemm.py OUT.json [roles] [ms] [mode]
  mode prod  : _Linear.__call__ exactly as the model calls it (rows<=16 fp32 matvec; 17..128 routed hi/lo;
               prefill rows inside route_scratch(), as NemotronHPrefill runs its pieces)
  mode bf16  : route_dense_bf16 on a bf16 activation (no hi/lo), the plain-bf16 prefill option
Per-kernel device durations come from the HCQ graph profile (one graph = R distinct weight copies, >= 384 MB total,
so no weight is served from the 96 MB L2 as in the real 42-layer stream).
"""
import sys, os, json, math, pathlib, collections, re, time
from tinygrad import Tensor, dtypes, Device, TinyJit
from tinygrad.llm.nemotron_h import _Linear
from tinygrad.llm.dense_candidate_gemm import bind_projection, route_scratch, route_dense_bf16

ROLES = {"ssm_in": (17504, 3136), "ssm_out": (3136, 7680), "attn_q": (5120, 3136), "attn_kv": (1024, 3136),
         "attn_o": (3136, 5120), "ffn_up": (12544, 3136), "ffn_down": (3136, 12544), "output": (131072, 3136)}
# Prefill runs in pieces of <= 1024 rows (NemotronHPrefill piece=1024, the production setting of 4aafc5e7c), so a
# prefill of M rows is M/1024 calls at rows=1024; rows > 1024 in one call is not a production shape.
MS = [8, 16, 32, 64, 128, 512, 1024]
PJ = os.environ.get("HCQ_GRAPH_PROFILE_JSON", "")
def nlines(): return sum(1 for _ in open(PJ)) if os.path.exists(PJ) else 0
def strip(nm): return re.sub(r"_[0-9a-f]{64}$", "", re.sub(r"\x1b\[[0-9;]*m", "", nm))

def run(role, m, mode, iters=10):
  Tensor.manual_seed(0)
  n, k = ROLES[role]
  r = max(2, -(-384 * 2**20 // (n * k * 2)))
  lins = []
  for i in range(r):
    lin = _Linear(k, n, bias=False)
    lin.weight = (Tensor.randn(n, k, dtype=dtypes.float) * 0.02).cast(dtypes.bfloat16).contiguous().realize()
    assert bind_projection(lin, role) is not None
    lins.append(lin)
  # a distinct activation per copy: production splits/sums a different x per layer (no cross-copy CSE)
  xs = [Tensor.randn(m, k).contiguous().realize() for _ in range(r)]
  if mode == "bf16": xs = [x.cast(dtypes.bfloat16).contiguous().realize() for x in xs]
  def body(*xs):
    if mode == "bf16":
      return [route_dense_bf16(x, l._candidate.padded, role, n).contiguous() for l, x in zip(lins, xs)]
    if m > 128:
      with route_scratch(): return [l(x).contiguous() for l, x in zip(lins, xs)]
    return [l(x).contiguous() for l, x in zip(lins, xs)]
  jit = TinyJit(lambda *xs: [o.realize() for o in body(*xs)])
  for _ in range(3): jit(*xs)
  Device.default.synchronize()
  l0 = nlines(); t0 = time.perf_counter()
  for _ in range(iters): jit(*xs)
  Device.default.synchronize()
  wall = (time.perf_counter() - t0) / iters / r
  l1 = nlines()
  rows = [json.loads(l) for l in open(PJ)][l0:l1]
  agg = collections.OrderedDict()
  for g in rows:
    for e in g["entries"]:
      nm = strip(e["name"]); md = e.get("metadata") or {}
      d = agg.setdefault(nm, {"name": nm, "count": 0, "ns": 0, "gemm": "canonical_identity" in md,
                              "cid": md.get("canonical_identity", "")[:16]})
      d["count"] += 1; d["ns"] += float(e["duration"]) * 1e3
  calls = iters * r  # a jit call may span several HCQ graphs
  ks = [{**d, "per_call": d["count"] / calls, "us_per_call": d["ns"] / 1e3 / calls} for d in agg.values()]
  return {"role": role, "m": m, "n": n, "k": k, "mode": mode, "copies": r, "graphs": len(rows), "kernels": ks,
          "gpu_us": sum(d["us_per_call"] for d in ks), "gemm_us": sum(d["us_per_call"] for d in ks if d["gemm"]),
          "wall_us": wall * 1e6}

if __name__ == "__main__":
  out_path = sys.argv[1]
  roles = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else list(ROLES)
  ms = [int(v) for v in sys.argv[3].split(",")] if len(sys.argv) > 3 and sys.argv[3] else MS
  mode = sys.argv[4] if len(sys.argv) > 4 else "prod"
  out = json.load(open(out_path)) if os.path.exists(out_path) else []
  for role in roles:
    for m in ms:
      if role == "output" and m > 128: continue  # prefill samples one row per sequence; no large-M lm_head
      try: res = run(role, m, mode)
      except Exception as e:
        print(f"{role} M={m} {mode} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True); continue
      out = [o for o in out if (o["role"], o["m"], o["mode"]) != (role, m, mode)] + [res]
      n, k = res["n"], res["k"]
      print(f"{role:8s} M={m:5d} {mode} total {res['gpu_us']:9.1f}us ({2*m*n*k/res['gpu_us']/1e6:6.1f} useful TF) "
            f"gemm {res['gemm_us']:9.1f}us wall {res['wall_us']:8.1f}us | "
            + " | ".join(f"{d['name'][:40]}{'*' if d['gemm'] else ''} x{d['per_call']:.0f} {d['us_per_call']:.1f}" for d in res["kernels"]), flush=True)
      json.dump(out, open(out_path, "w"), indent=1)
      import gc; gc.collect()
      if hasattr(alloc := Device[Device.DEFAULT].allocator, "free_cache"): alloc.free_cache()
