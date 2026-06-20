"""P0 — kernel-identity + warmstart-firing gate. Read-only introspection.
Builds the WMMA prefill jit (joff) under a chosen SETUP, measures tok/s to label FAST vs STUCK, then dumps per-kernel
launch dims / lib md5 / occupancy + the process-wide postrange._warmstart_stats. Goal: is the stuck kernel a
DIFFERENT (non-TC) compiled binary than the fast one (=> schedule cause), or identical (=> hardware state)?

SETUP=stuck : boost_probe-like (build joff only, measure)         -> usually ~1438 tok/s
SETUP=fast  : ab_measure-like (build joff + Tensile jon, interleave) -> usually ~2674 tok/s
Run: DEV=AMD PREFILL_V2=1 PREFILL_TENSILE_GEMM=1 WARMSTART_DUMP=1 SETUP=fast PYTHONPATH=. python3 extra/qk_prefill_kernel_identity.py
"""
import os, time, json, hashlib, statistics, pathlib
import tinygrad.llm.model as Mod
import tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, TinyJit, Device
from tinygrad.uop.ops import Ops
from tinygrad.engine.realize import get_runtime
from tinygrad.llm.model import Transformer, PREFILL_UBATCH
import extra.qk_tensile_inmodel as TI

ART = pathlib.Path("bench/qk-prefill-boost"); ART.mkdir(parents=True, exist_ok=True)
SETUP = os.environ.get("SETUP", "stuck")
MODEL = '/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf'
Tensor.manual_seed(0)
model, _ = Transformer.from_gguf(MODEL, 2048)
N = PREFILL_UBATCH; TI.install(Device['AMD'])
t = Tensor([5,6,7,8,9,10]*200+[0]*(2048-1200), dtype='int32').reshape(1,2048); chunk = t[:,0:N].contiguous(); temp = Tensor([0.0])
for b in model.blk: b._use_flash, b._prefill_v2 = False, True
saved = pr._WARMSTART_OPTS
def build(flag):
  Mod.PREFILL_TENSILE_GEMM = flag
  return TinyJit(model.forward)
def run(j):
  pr._WARMSTART_OPTS = model._pf16_warmstart
  try: j(chunk,0,temp).realize(); Device['AMD'].synchronize()
  finally: pr._WARMSTART_OPTS = saved

joff = build(False); run(joff)
if SETUP == "fast":
  jon = build(True); run(jon)
  for _ in range(10): run(joff); run(jon)
else:
  for _ in range(10): run(joff)
# measure joff tok/s
ts = []
for _ in range(25): a=time.perf_counter(); run(joff); ts.append(time.perf_counter()-a)
tokps = round(N/statistics.median(ts))
state = "FAST" if tokps>2400 else ("STUCK" if tokps<1800 else "MID")

def decode_vgpr(rsrc1):  # RDNA wave32: GRANULATED_WORKITEM_VGPR_COUNT bits[5:0], vgpr=(g+1)*8
  try: return ((rsrc1 & 0x3F)+1)*8
  except Exception: return None

# dump every PROGRAM kernel in joff's captured graph
kernels = []
seen = set()
for u in joff.captured.linear.toposort():
  if u.op is not Ops.PROGRAM: continue
  pi = u.arg
  libh = hashlib.md5(u.src[4].arg).hexdigest()[:12] if len(u.src) > 4 and u.src[4].op is Ops.BINARY else None
  key = (pi.name, libh)
  if key in seen: continue
  seen.add(key)
  occ = {}
  try:
    rt = get_runtime('AMD', u)
    occ = {"lds": getattr(rt,'group_segment_size',None), "scratch": getattr(rt,'private_segment_size',None),
           "wave32": getattr(rt,'wave32',None), "vgpr_approx": decode_vgpr(getattr(rt,'rsrc1',0))}
  except Exception as e:
    occ = {"err": str(e)[:60]}
  kernels.append({"name": pi.name, "global": list(pi.global_size), "local": list(pi.local_size) if pi.local_size else None,
                  "lib_md5": libh, **occ})

# identify the FFN matmul kernels (big out/in: gate/up 12288x4096, down 4096x12288); heuristic = name has the dims or large global
ffn = [k for k in kernels if any(str(d) in k["name"] for d in (12288, 4096)) and ('r_' in k["name"] or 'E_' in k["name"] or 'wmma' in k["name"].lower() or 'matmul' in k["name"].lower())]

out = {"setup": SETUP, "tokps": tokps, "state": state,
       "n_kernels": len(kernels),
       "graph_libhash": hashlib.md5("".join(sorted(k["lib_md5"] or "" for k in kernels)).encode()).hexdigest()[:12],
       "warmstart_stats": {k:v for k,v in pr._warmstart_stats.items()},
       "ffn_matmul_kernels": ffn,
       "all_kernels": kernels}
(ART/f"p0_kernel_identity_{SETUP}.json").write_text(json.dumps(out, indent=2))
print(f"SETUP={SETUP} tokps={tokps} state={state} n_kernels={len(kernels)} graph_libhash={out['graph_libhash']}")
print(f"WARMSTART_STATS match={pr._warmstart_stats.get('match')} apply={pr._warmstart_stats.get('apply')} error={pr._warmstart_stats.get('error')}")
for e in pr._warmstart_stats.get('errs', [])[:3]: print("  ERR:", e)
print(f"FFN matmul kernels ({len(ffn)}):")
for k in ffn[:8]: print(f"  {k['name'][:48]:48s} libmd5={k['lib_md5']} lds={k.get('lds')} vgpr~{k.get('vgpr_approx')} global={k['global']} local={k['local']}")
