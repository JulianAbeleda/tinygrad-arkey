"""Profile our Nemotron-H batch sampler decode step, kernel by kernel.
usage: PROFILE=1 HCQ_GRAPH_PROFILE_JSON=out.jsonl python3 ours_prof.py B P CAP OUT_META.json
Writes OUT_META.json: ordered kernel lists (name, metadata, gsz, lsz) of the step graph and the flush graph,
the profile-JSON line range of a measured window of NSTEPS chained steps, and wall per step.
"""
import sys, os, time, json, re, pathlib
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.nemotron_h import load
from tinygrad.llm.nemotron_h_sampler import NemotronHBatchSampler
from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
from tinygrad.uop.ops import Ops

B, L, CAP, OUT = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
NSTEPS = 64
PJ = os.environ.get("HCQ_GRAPH_PROFILE_JSON", "")
def nlines(): return sum(1 for _ in open(PJ)) if PJ and os.path.exists(PJ) else 0

model, meta = load('/home/ubuntu/storage/models/NVIDIA-Nemotron3-Nano-4B-BF16.gguf', max_context=CAP + L)
s = NemotronHBatchSampler(model, B, CAP, prefix_capacity=L)
prompt = [(i * 7919 + 13) % 30000 + 100 for i in range(L)]
pf = NemotronHPrefill(model, capacity=L)
hidden = s.prime(prompt, prefill=pf)
token, chosen = s.first(hidden, 1.0)
rate = Tensor([1.0]).realize()
s._record(token, chosen, 0)
# warm: 64 + 40 steps so the measured window [104, 168) stays in the 128..255 suffix bucket? bucket(index) = smallest pow2 > index
tok = token
idx = 0
WARM = 131
for _ in range(WARM):
  tok, _ = s.step(tok, idx, rate); idx += 1
Device.default.synchronize()
tok, _ = s.step(tok, idx, rate); idx += 1
Device.default.synchronize()
l0 = nlines()
t0 = time.perf_counter()
for _ in range(NSTEPS):
  tok, _ = s.step(tok, idx, rate); idx += 1
Device.default.synchronize()
wall = (time.perf_counter() - t0) / NSTEPS
tok, _ = s.step(tok, idx, rate); idx += 1
Device.default.synchronize()
l1 = nlines()

def kernels(jit):
  out = []
  for call in jit.captured.linear.toposort():
    if call.op is not Ops.CALL: continue
    ast = call.src[0]
    name = getattr(ast.arg, "name", None) or str(ast.op)
    md = [str(m) for m in (call.arg.metadata or ())] if hasattr(call.arg, "metadata") else []
    gsz = getattr(ast.arg, "global_size", None); lsz = getattr(ast.arg, "local_size", None)
    out.append({"name": str(name), "md": md, "gsz": str(gsz), "lsz": str(lsz)})
  return out

meta_out = {"B": B, "L": L, "CAP": CAP, "nsteps": NSTEPS, "wall_ms": wall * 1e3, "lines": [l0, l1],
            "buckets": sorted(s.graphs.keys())}
try:
  meta_out["step"] = {str(k): kernels(g) for k, g in s.graphs.items()}
  meta_out["flush"] = kernels(s.flush)
except Exception as e:
  meta_out["err"] = repr(e)
pathlib.Path(OUT).write_text(json.dumps(meta_out))
print(f"B={B} L={L} wall/step={wall*1e3:.2f} ms lines={l0}..{l1} buckets={meta_out['buckets']}", flush=True)
