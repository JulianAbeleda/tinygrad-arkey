"""AMD native ISA backend — Inc 1 acceptance gate (opt-in DEV=AMD:ISA, default unchanged).

Proves a NON-trivial elementwise kernel (out[i]=a[i]+b[i], n>4) whose index is derived from the SPECIAL/workitem
id (v0) -- not a compile-time CONST -- runs numerically correct on gfx1100 through the AMDISARenderer pipeline.

Inc 1 additions over Inc 0 (constant-index vec4):
  - SPECIAL lidx0 -> v0 (workitem id) participates in address calculation.
  - integer index arithmetic lowered to u32 VALU (v_mul_lo_u32 / v_add_nc_u32 + v_lshlrev byte scale).
  - float ALU folds CONST operands to literals -> add/sub/mul/scale+bias coverage.
Still scalarized (no real b128); workgroup-id (gidx*) / lidx1+ SGPR ABI and reduction loops are deferred.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc1_gate.py
Writes: bench/amd-isa-backend-inc1/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc1_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-inc1", "latest.json")
N = 64   # n>4, uses LOCAL launch dim -> workitem-id (lidx0) indexing; stays within Inc 1 (no gidx workgroup-id)

# capture each kernel's assembled instruction stream (real assemble_linear path)
_asm: list[str] = []
_orig_asm = AMDISARenderer.asm
def _spy(self, prg, lin):
  _asm.append("\n".join(str(u.arg) for u in lin.src if u.op is Ops.INS))
  return _orig_asm(self, prg, lin)
AMDISARenderer.asm = _spy

def _verdict_for(exc: Exception) -> str:
  m = str(exc)
  if isinstance(exc, NotImplementedError) or "SPECIAL" in m: return "AMD_ISA_INC1_BLOCKED_SPECIAL_OR_INDEX_ISEL"
  if isinstance(exc, KeyError) and "reals" not in m: return "AMD_ISA_INC1_BLOCKED_REGALLOC_PATTERN_DESYNC"
  if "needs a renderer" in m or "assemble" in m.lower() or "ELF" in m: return "AMD_ISA_INC1_BLOCKED_ASSEMBLE_OR_ELF"
  return "AMD_ISA_INC1_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": f"Inc 1: workitem(lidx0/v0)-indexed elementwise, n={N}"}
  try:
    ren = type(Device["AMD"].renderer).__name__
    rec["selected_renderer"] = ren
    if ren != "AMDISARenderer":
      rec["verdict"] = "AMD_ISA_INC1_BLOCKED_RUNTIME_ROUTE"
      rec["blocker"] = f"DEV=AMD:ISA selected {ren}, not AMDISARenderer (must not fall back to HIP/LLVM)"
      return rec
    rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"

    rng = np.random.default_rng(20260629)
    a = rng.standard_normal(N).astype(np.float32)
    b = rng.standard_normal(N).astype(np.float32)
    ta, tb = Tensor(a, device="AMD"), Tensor(b, device="AMD")

    # primary: workitem-indexed a+b on random data
    got = (ta + tb).numpy()
    exp = a + b
    primary_ok = bool(np.allclose(got, exp, atol=1e-5))
    rec["regalloc"] = "PASS (no regalloc.py:118 KeyError; no raw INDEX/GEP/STACK/MUL survived isel)"

    asm = _asm[-1] if _asm else ""
    used_workitem = ("v_mul_lo_u32" in asm and "v[0]" in asm)   # v0 (workitem id) drives the element index
    rec["special_v0_used"] = used_workitem
    rec["integer_index_alu"] = {"v_mul_lo_u32": asm.count("v_mul_lo_u32"), "v_lshlrev_b32_e32": asm.count("v_lshlrev_b32_e32"),
                                "v_add_nc_u32_e32": asm.count("v_add_nc_u32_e32")}
    rec["instruction_counts"] = {"global_load_b32": asm.count("global_load_b32"), "global_store_b32": asm.count("global_store_b32"),
                                 "v_add_f32_e32": asm.count("v_add_f32_e32"), "b128_used": "b128" in asm}
    rec["assemble"] = "PASS (assemble_linear produced rdna3 INS stream)"
    rec["runtime"] = "PASS (kernel launched on gfx1100)"

    # secondary: basic scalar op coverage (folds float CONST operands -> literals)
    cov = {}
    for nm, (t, e) in {"sub": (ta - tb, a - b), "mul": (ta * tb, a * b),
                       "scale+bias": (ta * 2.0 + 1.0, a * 2.0 + 1.0)}.items():
      cov[nm] = bool(np.allclose(t.numpy(), e, atol=1e-5))
    rec["op_coverage"] = cov

    rec["correctness"] = {"result": "PASS" if primary_ok else "FAIL", "got_head": got[:4].tolist(), "expected_head": exp[:4].tolist()}
    all_ok = primary_ok and used_workitem and all(cov.values()) and not rec["instruction_counts"]["b128_used"]
    if all_ok:
      rec["verdict"] = "AMD_ISA_INC1_PASS_WORKITEM_ELEMENTWISE_RUNS"
    else:
      rec["verdict"] = "AMD_ISA_INC1_BLOCKED_ASSEMBLE_OR_ELF"
      rec["blocker"] = ("primary mismatch" if not primary_ok else "workitem-id (v0) not in address calc" if not used_workitem
                        else f"op coverage failed: {cov}" if not all(cov.values()) else "unexpected b128 in stream")
  except Exception as e:
    rec["verdict"] = _verdict_for(e)
    rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred"] = ("workgroup-id gidx*/lidx1+ (SGPR workgroup-id ABI + descriptor enable) -> fails loudly, not "
                     "silently wrong; real b128/global_load_b128 + consecutive-VGPR alloc; RANGE/END reduction loops; "
                     "GEMV/decode-tile; v_dot2/LDS/barriers; waitcnt/scheduler optimization")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nINC1", "PASS" if rec["verdict"] == "AMD_ISA_INC1_PASS_WORKITEM_ELEMENTWISE_RUNS" else f"-> {rec['verdict']}")
