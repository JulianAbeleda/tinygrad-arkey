"""Measure the 4 prefill primitives (LDS staging / occupancy / pipeline-stalls / VALU overhead) via native PMC,
for the isolated gateup GEMM (out=12288, in=4096, T=512): tinygrad WMMA vs rocBLAS Tensile.
Counters come from PMC_COUNTERS env (≤~8/pass). Picks the dominant matmul kernel (max GRBM_GUI_ACTIVE) per side.

Run (one pass): DEV=AMD PREFILL_V2=1 PMC=1 PROFILE=1 PMC_COUNTERS=... PYTHONPATH=. python3 extra/qk_prefill_primitive_pmc.py
The bash orchestrator runs it several times with different PMC_COUNTERS sets and merges.
"""
import os, json, struct, pathlib
import tinygrad.llm.model as Mod, tinygrad.codegen.opt.postrange as pr
from tinygrad import Tensor, Device, dtypes
from tinygrad.device import Compiled
from tinygrad.llm.model import Transformer, PREFILL_UBATCH, _pf16
import extra.qk_tensile_inmodel as TI
from extra.qk_pmc_capture import decode_pmc

ART = pathlib.Path("bench/qk-prefill-boost"); ART.mkdir(parents=True, exist_ok=True)
Tensor.manual_seed(0)
model, _ = Transformer.from_gguf('/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf', 2048)
TI.install(Device['AMD'])
gate = model.blk[0].ffn_gate                      # in=4096 out=12288
x = Tensor.randn(1, PREFILL_UBATCH, 4096, dtype=dtypes.float16).contiguous().realize()
saved = pr._WARMSTART_OPTS

def wmma_fn():
  pr._WARMSTART_OPTS = model._pf16_warmstart
  try: _pf16(gate, x).realize(); Device['AMD'].synchronize()
  finally: pr._WARMSTART_OPTS = saved
def tensile_fn():
  Mod.PREFILL_TENSILE_GEMM = True
  TI.route_pf16(gate, x.reshape(PREFILL_UBATCH, 4096)).realize(); Device['AMD'].synchronize()

def cap_full(fn, label, warmup=3):
  for _ in range(warmup): fn()
  base = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
  fn(); Device['AMD'].synchronize(); Device['AMD']._at_profile_finalize()
  evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][base:]
  rows = [{"label": label, "kern": ev.kern, **decode_pmc(ev)} for ev in evs]
  return rows

def dominant(rows):  # the matmul = max GRBM_GUI_ACTIVE (fallback SQ_BUSY_CYCLES)
  if not rows: return None
  k = "GRBM_GUI_ACTIVE" if "GRBM_GUI_ACTIVE" in rows[0] else ("SQ_BUSY_CYCLES" if "SQ_BUSY_CYCLES" in rows[0] else None)
  return max(rows, key=lambda r: r.get(k, 0)) if k else rows[-1]

counters = os.environ.get("PMC_COUNTERS", "")
wr = cap_full(wmma_fn, "wmma")
tr = cap_full(tensile_fn, "tensile")
dw, dt = dominant(wr), dominant(tr)
out = {"counters": counters,
       "wmma_dominant": dw, "tensile_dominant": dt,
       "wmma_all": [{k: r[k] for k in r if k not in ('label',)} for r in wr],
       "tensile_all": [{k: r[k] for k in r if k not in ('label',)} for r in tr]}
# append to a per-pass file
fn = ART / f"primitive_pmc_pass_{abs(hash(counters))%100000}.json"
fn.write_text(json.dumps(out, indent=2))
print(f"COUNTERS={counters}")
print(f"  WMMA dominant kern={dw.get('kern') if dw else None}: " + "  ".join(f"{k}={v}" for k,v in (dw or {}).items() if k not in ('label','kern')))
print(f"  TENS dominant kern={dt.get('kern') if dt else None}: " + "  ".join(f"{k}={v}" for k,v in (dt or {}).items() if k not in ('label','kern')))
print(f"  (wmma kernels captured={len(wr)} tensile kernels captured={len(tr)})")
