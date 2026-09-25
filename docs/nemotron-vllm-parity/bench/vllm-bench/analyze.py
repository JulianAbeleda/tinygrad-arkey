"""Per-step kernel table from an nsys sqlite export (steady decode window).
usage: analyze.py trace.sqlite out.txt
Step = one decode iteration. Boundaries come from CUDA graph launches (kernels with graphId);
we take the middle 60% of the captured GPU timeline as the steady window.
"""
import sqlite3, sys, collections, re

db = sqlite3.connect(sys.argv[1]); out = open(sys.argv[2], "w")
def P(*a):
    print(*a); print(*a, file=out)

strs = dict(db.execute("select id, value from StringIds"))
cols = [r[1] for r in db.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
has_graph = "graphId" in cols
q = "select start, end, shortName, demangledName, correlationId, gridX,gridY,gridZ, blockX,blockY,blockZ, registersPerThread, staticSharedMemory+dynamicSharedMemory"
q += ", graphId, graphNodeId" if has_graph else ", 0, 0"
K = [dict(zip(["s","e","short","dem","corr","gx","gy","gz","bx","by","bz","regs","smem","gid","gexec"], r))
     for r in db.execute(q + " from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
for k in K:
    k["short"] = strs.get(k["short"], str(k["short"])); k["dem"] = strs.get(k["dem"], str(k["dem"]))
t0, t1 = K[0]["s"], K[-1]["e"]
P(f"kernels captured: {len(K)}  span {(t1-t0)/1e6:.1f} ms  graphId column: {has_graph}")

# graph launches from runtime API
rt = list(db.execute("select r.start, r.correlationId, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where s.value like 'cudaGraphLaunch%' order by r.start"))
P(f"cudaGraphLaunch calls: {len(rt)}")
graph_corr = {c for _, c, _ in rt}
in_graph = [k for k in K if k["corr"] in graph_corr]
P(f"kernels inside graph launches: {len(in_graph)} / {len(K)}")

# steady window: middle 60%
ws, we = t0 + 0.2 * (t1 - t0), t0 + 0.8 * (t1 - t0)
W = [k for k in K if ws <= k["s"] < we]
# step count: graph launches whose first kernel lies in window
first_by_corr = {}
for k in in_graph:
    first_by_corr.setdefault(k["corr"], k["s"])
launch_starts = sorted(s for s in first_by_corr.values() if ws <= s < we)
# kernels per graph launch -> identify distinct graph kinds
per_launch = collections.Counter(k["corr"] for k in in_graph)
sizes = collections.Counter(per_launch.values())
P(f"kernels per graph launch (size:count): {dict(sizes.most_common(8))}")

# steps = count of the most frequent 'once per step' anchor: use the lm_head-like kernel = the kernel with
# the largest single duration among those whose count equals the number of big graph launches. Simpler: graph launches
# per step is inferred from the largest-size graph (one per step).
big = max(sizes, key=lambda s: s * sizes[s]) if sizes else None
big_corrs = sorted((first_by_corr[c], c) for c, n in per_launch.items() if n == big)
steps_in_w = [s for s, _ in big_corrs if ws <= s < we]
nsteps = len(steps_in_w)
if nsteps < 2:
    P("WARNING: could not find graph steps; falling back to window/unique counts"); nsteps = 1
step_wall = (steps_in_w[-1] - steps_in_w[0]) / (nsteps - 1) / 1e3 if nsteps > 1 else float("nan")
Wk = [k for k in K if steps_in_w[0] <= k["s"] < steps_in_w[-1]] if nsteps > 1 else W
ns = max(1, nsteps - 1)
busy = sum(k["e"] - k["s"] for k in Wk) / ns / 1e3
P(f"steady steps: {ns}  GPU step period: {step_wall/1e3:.3f} ms  kernel-busy per step: {busy/1e3:.3f} ms  ({100*busy/step_wall:.1f}% busy)")
P(f"kernels per step: {len(Wk)/ns:.1f}  (in-graph: {sum(1 for k in Wk if k['corr'] in graph_corr)/ns:.1f})")

agg = collections.defaultdict(lambda: [0, 0, None])
def classify(n):
    l = n.lower()
    if "selective_scan_update" in l or "selective_state_update" in l or "ssu" in l.split("_") or "replayssm" in l or "_state_update" in l: return "ssm_state_update"
    if "causal_conv1d" in l or "conv1d" in l: return "mamba_conv"
    if "chunk_scan" in l or "chunk_state" in l or "state_passing" in l or "bmm_chunk" in l or "chunk_cumsum" in l: return "mamba_ssd_prefill"
    if "layer_norm_fwd" in l or ("gated" in l and "norm" in l) or ("silu" in l and "rsqrt" in l): return "gated_rmsnorm(+silu gate)"
    if "mamba_align" in l or "postprocess_mamba" in l: return "mamba_align_bookkeeping"
    if "pow_relu" in l: return "relu2"
    if "kernel2<cutlass" in l or "wmma_tensorop" in l: return "GEMM"
    if "rms" in l or "norm" in l: return "norm/fused"
    if any(x in l for x in ["flash", "fmha", "attn", "attention", "paged", "decode_kernel", "batchdecode", "batchprefill"]): return "attention"
    if any(x in l for x in ["gemm", "gemv", "cutlass", "sm80_xmma", "sm90", "sm100", "sm120", "cublas", "matmul", "ampere", "xmma", "splitk", "nvjet"]): return "GEMM"
    if any(x in l for x in ["softmax", "sample", "mbtopk", "_ranks_kernel", "topk", "topk", "top_k", "argmax", "exponential", "gumbel", "logprob", "sort", "radix", "scatter_gather", "gather", "multinomial", "cumsum", "max_", "reduce"]): return "sampling/reduce"
    if "triton" in l: return "triton-other"
    return "other(elementwise/copy)"

def key(k):
    return k["short"] + (f" grid{k['gx']}x{k['gy']}x{k['gz']}" if classify(k["short"] + " " + k["dem"][:200]) == "GEMM" else "")
for k in Wk:
    a = agg[key(k)]; a[0] += 1; a[1] += k["e"] - k["s"]; a[2] = k
tot = sum(v[1] for v in agg.values())
P(f"\n{'kernel':84s} {'class':26s} {'/step':>6s} {'us/step':>9s} {'share':>6s}  grid/block/regs/smem")
cls = collections.defaultdict(lambda: [0, 0])
for name, (c, t, k) in sorted(agg.items(), key=lambda x: -x[1][1]):
    cl = classify(name + " " + k["dem"][:200])
    cls[cl][0] += c; cls[cl][1] += t
    P(f"{name[:84]:84s} {cl:26s} {c/ns:6.1f} {t/ns/1e3:9.1f} {100*t/tot:5.1f}%  ({k['gx']},{k['gy']},{k['gz']})/({k['bx']},{k['by']},{k['bz']}) r{k['regs']} s{k['smem']}")
P(f"\n{'class':26s} {'/step':>6s} {'us/step':>9s} {'share':>6s}")
for cl, (c, t) in sorted(cls.items(), key=lambda x: -x[1][1]):
    P(f"{cl:26s} {c/ns:6.1f} {t/ns/1e3:9.1f} {100*t/tot:5.1f}%")
P("\nfull demangled names of top 25:")
for name, (c, t, k) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]:
    P(f"- {k['dem'][:400]}")
