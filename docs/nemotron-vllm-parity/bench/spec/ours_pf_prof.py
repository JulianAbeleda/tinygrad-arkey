"""Profile our Nemotron-H 10k prefill per graph and kernel.
usage: PROFILE=1 HCQ_GRAPH_PROFILE_JSON=out.jsonl python3 ours_pf_prof.py L PIECE MAMBA PREC OUT_META.json
"""
import sys, os, time, json, pathlib
from tinygrad import Device
from tinygrad.llm.nemotron_h import load
from tinygrad.llm.nemotron_h_prefill import NemotronHPrefill
from tinygrad.uop.ops import Ops

L, PIECE, MAMBA, PREC, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
PJ = os.environ.get("HCQ_GRAPH_PROFILE_JSON", "")
def nlines(): return sum(1 for _ in open(PJ)) if PJ and os.path.exists(PJ) else 0
model, meta = load('/home/ubuntu/storage/models/NVIDIA-Nemotron3-Nano-4B-BF16.gguf', max_context=L + 1024)
p = NemotronHPrefill(model, capacity=L, piece=PIECE, mamba=MAMBA, ssd_precision=PREC)
prompt = [(i * 7919 + 13) % 30000 + 100 for i in range(L)]
walls = []
marks = []
for rep in range(3):
  Device.default.synchronize(); marks.append(nlines())
  t = time.perf_counter(); p(prompt); Device.default.synchronize(); walls.append(time.perf_counter() - t)
  print(f"rep {rep} {walls[-1]:.3f}s", flush=True)
marks.append(nlines())

def kernels(jit):
  out = []
  cap = getattr(jit, "captured", None)
  if cap is None: return out
  for call in cap.linear.toposort():
    if call.op is not Ops.CALL: continue
    ast = call.src[0]
    name = getattr(ast.arg, "name", None) or str(ast.op)
    md = [str(m) for m in (call.arg.metadata or ())] if hasattr(call.arg, "metadata") else []
    out.append({"name": str(name), "md": md})
  return out
graphs = {repr(k): kernels(g) for k, g in {**p.block_graphs, **{("attn",) + tuple(k): g for k, g in p.graphs.items()}}.items()}
pathlib.Path(OUT).write_text(json.dumps({"L": L, "piece": PIECE, "mamba": MAMBA, "prec": PREC, "walls": walls,
                                          "marks": marks, "graphs": graphs, "pieces": p.pieces(L)}))
print(f"RESULT L={L} piece={PIECE} mamba={MAMBA} prec={PREC} warm={walls[1]:.3f}/{walls[2]:.3f}s marks={marks}", flush=True)
