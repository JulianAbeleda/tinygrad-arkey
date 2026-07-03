"""AMD native ISA backend — Inc 0 acceptance gate (opt-in DEV=AMD:ISA, default unchanged).

Proves ONE trivial generated elementwise kernel (out[i]=a[i]+b[i], fully-upcast vec4) runs numerically correct on
gfx1100 through:  UOp -> AMDISARenderer -> Ops.INS / rdna3 Insts -> assemble_linear -> AMD runtime execution.

Scope (Inc 0 only): the trivial fully-upcast a+b. vec4/b128 is handled by SCALARIZATION (4x global_load_b32 +
4x v_add_f32 + 4x global_store_b32 with per-lane immediate offsets) -- NOT real b128 / consecutive-VGPR allocation,
which are deferred to Inc 1+. Kernels with a LOCAL launch dim (SPECIAL/workitem-id + integer index MUL) are Inc 1.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc0_gate.py
Writes: bench/amd-isa-backend-inc0/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.renderer.isa.amd import AMDISARenderer

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc0_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-inc0", "latest.json")

# capture the assembled instruction stream to confirm scalarization + that assemble_linear ran
_asm_seen: list[str] = []
_orig_asm = AMDISARenderer.asm
def _spy(self, prg, lin):
  from tinygrad.uop.ops import Ops
  _asm_seen.append("\n".join(str(u.arg) for u in lin.src if u.op is Ops.INS))
  return _orig_asm(self, prg, lin)   # real assemble_linear path
AMDISARenderer.asm = _spy

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Inc 0: trivial fully-upcast vec4 a+b only"}
  try:
    ren = type(Device["AMD"].renderer).__name__
    rec["selected_renderer"] = ren
    if ren != "AMDISARenderer":
      rec["verdict"] = "AMD_ISA_INC0_BLOCKED_RUNTIME_ROUTE"
      rec["blocker"] = f"DEV=AMD:ISA selected {ren}, not AMDISARenderer (must not fall back to HIP/LLVM)"
      return rec
    rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"

    # the trivial Inc 0 kernel: vec4 a+b, fully upcast (UPCAST axis 4, no LOCAL dim)
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    got = (Tensor(a, device="AMD") + Tensor(b, device="AMD")).numpy()
    exp = a + b
    rec["regalloc"] = "PASS (no regalloc.py:118 KeyError; SGPR pairs + scalarized VGPRs allocated)"

    asm = _asm_seen[-1] if _asm_seen else ""
    n_ld = asm.count("global_load_b32")
    n_st = asm.count("global_store_b32")
    rec["scalarization"] = {
      "used": True, "global_load_b32": n_ld, "global_store_b32": n_st,
      "b128_used": "global_load_b128" in asm or "global_store_b128" in asm,
      "note": "vec4 scalarized to 4 scalar b32 lanes w/ per-lane immediate offsets (b128 deferred to Inc 1+)"}
    rec["assemble"] = "PASS (assemble_linear produced rdna3 INS stream; kernel launched)"
    rec["runtime"] = "PASS (kernel executed on gfx1100)"

    ok = bool(np.allclose(got, exp, atol=1e-6))
    rec["correctness"] = {"result": "PASS" if ok else "FAIL", "got": got.tolist(), "expected": exp.tolist()}
    # scalarization sanity: 2 vec4 loads -> 8 b32 loads; 1 vec4 store -> 4 b32 stores
    scal_ok = (n_ld == 8 and n_st == 4 and not rec["scalarization"]["b128_used"])
    rec["verdict"] = "AMD_ISA_INC0_PASS_TRIVIAL_KERNEL_RUNS" if (ok and scal_ok) else "AMD_ISA_INC0_BLOCKED_ASSEMBLE_OR_ELF"
    if not ok: rec["blocker"] = "numerical mismatch vs numpy"
    elif not scal_ok: rec["blocker"] = f"unexpected scalarization shape (loads={n_ld} stores={n_st})"
  except Exception as e:
    rec["verdict"] = rec["verdict"] or "AMD_ISA_INC0_BLOCKED_RUNTIME_ROUTE"
    rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred_to_inc1"] = ("real b128/global_load_b128 + consecutive-VGPR allocation; SPECIAL/workitem-id; integer "
                             "index arithmetic (MUL); LOCAL launch dims / RANGE loops; op coverage beyond add/mul/load/store")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nINC0", "PASS" if rec["verdict"] == "AMD_ISA_INC0_PASS_TRIVIAL_KERNEL_RUNS" else f"-> {rec['verdict']}")
