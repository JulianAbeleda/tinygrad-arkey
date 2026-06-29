"""AMD native ISA backend — Inc 3 acceptance gate (opt-in DEV=AMD:ISA, default unchanged).

Proves multi-dimensional indexing: 2D/3D kernels whose index uses workgroup ids gidx0/1/2 and workitem ids
lidx0/1/2 run correct on gfx1100. Contiguous elementwise collapses to 1D (gidx0/lidx0); transposed/permuted
patterns force genuine multi-dim ids, which is what this gate exercises.

ABI added this increment:
  - workgroup id.{x,y,z} (gidx0/1/2) -> system SGPRs s2/s3/s4 (after the 2 user SGPRs = kernarg ptr s0:1);
    descriptor ENABLE_SGPR_WORKGROUP_ID_{X,Y,Z} set by elf.py scanning the sink for gidx* SPECIALs.
  - workitem id.{x,y,z} (lidx0/1/2) are PACKED into v0 (x=bits[9:0], y=[19:10], z=[29:20]); extracted with
    v_bfe_u32. descriptor ENABLE_VGPR_WORKITEM_ID = max lidx dim (set by elf.py scanning for lidx* SPECIALs).
  - dims > 2 fail loudly (gfx1100 only has x/y/z).

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_inc3_gate.py
Writes: bench/amd-isa-backend-inc3/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
import numpy as np
import tinygrad.codegen as CG
from tinygrad import Tensor, Device

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_inc3_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-inc3", "latest.json")

# observe the SPECIAL id dims each kernel actually uses
_specials: list = []
_OrigCtx = CG.IselContext
class _Ctx(_OrigCtx):
  def __init__(self, sink):
    _specials.append(sorted({str(u.arg) for u in sink.toposort() if u.op.name == "SPECIAL"}))
    super().__init__(sink)
CG.IselContext = _Ctx

def _verdict_for(exc: Exception) -> str:
  m = str(exc)
  if isinstance(exc, NotImplementedError): return "AMD_ISA_INC3_BLOCKED_SPECIAL_ID_ABI"
  if isinstance(exc, KeyError): return "AMD_ISA_INC3_BLOCKED_INDEX_ISEL"
  if "descriptor" in m.lower(): return "AMD_ISA_INC3_BLOCKED_DESCRIPTOR_ENABLE"
  return "AMD_ISA_INC3_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Inc 3: multi-dim (gidx0/1/2, lidx0/1/2) indexing correctness"}
  try:
    ren = type(Device["AMD"].renderer).__name__
    rec["selected_renderer"] = ren
    if ren != "AMDISARenderer":
      rec["verdict"] = "AMD_ISA_INC3_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"selected {ren}, not AMDISARenderer"
      return rec
    rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"

    rng = np.random.default_rng(20260629)
    cases, results, max_dim_seen = [], {}, 0
    def add(name, fn, exp):
      cases.append((name, fn, exp))
    # contiguous 2D/3D (collapse to 1D gidx0/lidx0)
    for shp in [(64, 64), (256, 256), (16, 16, 16)]:
      a = rng.standard_normal(shp).astype(np.float32); b = rng.standard_normal(shp).astype(np.float32)
      add(f"contig{shp}", (lambda a=a, b=b: Tensor(a, device="AMD") + Tensor(b, device="AMD")), a + b)
    # transposed / permuted -> genuine multi-dim ids
    a2 = rng.standard_normal((64, 64)).astype(np.float32); b2 = rng.standard_normal((64, 64)).astype(np.float32)
    add("transpose2d", (lambda: Tensor(a2, device="AMD") + Tensor(b2, device="AMD").T), a2 + b2.T)
    m = rng.standard_normal((48, 80)).astype(np.float32)
    add("transpose48x80", (lambda: Tensor(m, device="AMD").T + 1.0), m.T + 1.0)
    x3 = rng.standard_normal((24, 24, 24)).astype(np.float32)
    add("permute3d", (lambda: Tensor(x3, device="AMD").permute(2, 1, 0) * 2.0), x3.transpose(2, 1, 0) * 2.0)

    for name, fn, exp in cases:
      _specials.clear()
      got = fn().numpy()
      sp = _specials[-1] if _specials else []
      max_dim_seen = max([max_dim_seen] + [int(s[-1]) for s in sp])
      results[name] = {"ok": bool(np.allclose(got, exp, atol=1e-4)), "specials": sp}

    rec["per_case"] = results
    rec["max_id_dim_exercised"] = max_dim_seen
    rec["multidim_exercised"] = any(int(s[-1]) > 0 for r in results.values() for s in r["specials"])
    all_ok = all(r["ok"] for r in results.values())
    rec["correctness"] = {"result": "PASS" if all_ok else "FAIL", "per_case_ok": {k: r["ok"] for k, r in results.items()}}
    if all_ok and rec["multidim_exercised"]:
      rec["verdict"] = "AMD_ISA_INC3_PASS_MULTIDIM_INDEXING"
    elif not all_ok:
      rec["verdict"] = "AMD_ISA_INC3_BLOCKED_INDEX_ISEL"; rec["blocker"] = f"mismatch: {rec['correctness']}"
    else:
      rec["verdict"] = "AMD_ISA_INC3_BLOCKED_INDEX_ISEL"; rec["blocker"] = "no multi-dim id path exercised"
  except Exception as e:
    rec["verdict"] = _verdict_for(e); rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred"] = "reductions/RANGE-END; GEMV; real b128/consecutive-VGPR alloc; v_dot2/LDS/barriers; scheduler/waitcnt"
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nINC3", "PASS" if rec["verdict"] == "AMD_ISA_INC3_PASS_MULTIDIM_INDEXING" else f"-> {rec['verdict']}")
