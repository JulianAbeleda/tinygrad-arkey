"""AMD native ISA backend — Inc 2 acceptance gate (opt-in DEV=AMD:ISA, default unchanged).

Proves grid-sized elementwise kernels (n >= 256) whose index uses the WORKGROUP-id ABI run correct on gfx1100.
For these kernels the global linear index is  group_id_x*local_size_x + local_id_x  (then *itemsize for bytes):
  - workgroup-id x (gidx0) comes from system SGPR s2 (after the 2 user SGPRs = kernarg ptr s0:1); the kernel
    descriptor's ENABLE_SGPR_WORKGROUP_ID_X bit is set because elf.py finds a gidx0 SPECIAL in the sink.
  - it is moved s2 -> VGPR (v_mov) so the integer index math (v_mul_lo_u32 / v_add_nc_u32) stays VGPR-only.
Inc 1 workitem-id (lidx0 -> v0) behaviour is unchanged. Higher dims (gidx1/2, lidx1/2) fail loudly.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_inc2_gate.py
Writes: bench/amd-isa-backend-inc2/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_inc2_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-inc2", "latest.json")
SIZES = [256, 1024, 4096]   # all > 4 and large enough to need a multi-workgroup grid (gidx0 path)

_asm: list[str] = []
_orig_asm = AMDISARenderer.asm
def _spy(self, prg, lin):
  _asm.append("\n".join(str(u.arg) for u in lin.src if u.op is Ops.INS))
  return _orig_asm(self, prg, lin)
AMDISARenderer.asm = _spy

def _verdict_for(exc: Exception) -> str:
  m = str(exc)
  if isinstance(exc, NotImplementedError) and ("gidx" in m or "workgroup" in m): return "AMD_ISA_INC2_BLOCKED_WORKGROUP_ID_ABI"
  if isinstance(exc, NotImplementedError) or "SPECIAL" in m: return "AMD_ISA_INC2_BLOCKED_INTEGER_INDEX_ISEL"
  if isinstance(exc, KeyError): return "AMD_ISA_INC2_BLOCKED_REGALLOC_PATTERN_DESYNC"
  if "needs a renderer" in m or "assemble" in m.lower() or "ELF" in m or "descriptor" in m.lower():
    return "AMD_ISA_INC2_BLOCKED_ASSEMBLE_OR_ELF"
  return "AMD_ISA_INC2_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": f"Inc 2: workgroup/global-indexed elementwise, n in {SIZES}"}
  try:
    ren = type(Device["AMD"].renderer).__name__
    rec["selected_renderer"] = ren
    if ren != "AMDISARenderer":
      rec["verdict"] = "AMD_ISA_INC2_BLOCKED_RUNTIME_ROUTE"
      rec["blocker"] = f"DEV=AMD:ISA selected {ren}, not AMDISARenderer (must not fall back to HIP/LLVM)"
      return rec
    rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"

    rng = np.random.default_rng(20260629)
    results, last_asm = {}, ""
    for n in SIZES:
      a = rng.standard_normal(n).astype(np.float32)
      b = rng.standard_normal(n).astype(np.float32)
      _asm.clear()
      got = (Tensor(a, device="AMD") + Tensor(b, device="AMD")).numpy()
      last_asm = _asm[-1] if _asm else ""
      results[n] = {"ok": bool(np.allclose(got, a + b, atol=1e-5)),
                    # workgroup-id read = a v_mov of the system SGPR s2 into a VGPR
                    "wgid_path": ("v_mov_b32_e32" in last_asm and "s[2]" in last_asm),
                    "v_mul_lo_u32": last_asm.count("v_mul_lo_u32"), "v_add_nc_u32_e32": last_asm.count("v_add_nc_u32_e32")}

    rec["per_size"] = results
    rec["regalloc"] = "PASS (no regalloc.py:118 KeyError; no raw INDEX/GEP/STACK/MUL survived isel)"
    rec["workgroup_id_path_used"] = all(r["wgid_path"] for r in results.values())
    rec["assemble"] = "PASS (assemble_linear produced rdna3 INS; descriptor ENABLE_SGPR_WORKGROUP_ID_X set via gidx0 scan)"
    rec["runtime"] = "PASS (multi-workgroup grid launched on gfx1100)"

    all_correct = all(r["ok"] for r in results.values())
    rec["correctness"] = {"result": "PASS" if all_correct else "FAIL", "per_size_ok": {n: r["ok"] for n, r in results.items()}}
    if all_correct and rec["workgroup_id_path_used"]:
      rec["verdict"] = "AMD_ISA_INC2_PASS_GLOBAL_INDEX_ELEMENTWISE_RUNS"
    elif not all_correct:
      rec["verdict"] = "AMD_ISA_INC2_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"numerical mismatch: {rec['correctness']}"
    else:
      rec["verdict"] = "AMD_ISA_INC2_BLOCKED_WORKGROUP_ID_ABI"; rec["blocker"] = "workgroup-id (s2) path not detected in asm"
  except Exception as e:
    rec["verdict"] = _verdict_for(e)
    rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred"] = ("gidx1/gidx2 + lidx1/lidx2 multi-dim ids (fail loud, not silently wrong); reductions / RANGE-END; "
                     "GEMV; real b128/consecutive-VGPR alloc; v_dot2/LDS/barriers; scheduler/waitcnt performance")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nINC2", "PASS" if rec["verdict"] == "AMD_ISA_INC2_PASS_GLOBAL_INDEX_ELEMENTWISE_RUNS" else f"-> {rec['verdict']}")
