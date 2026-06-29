"""AMD native ISA backend — Phase B gate: correctness-first RANGE/END reductions (opt-in DEV=AMD:ISA).

Proves simple sum reductions run correct on gfx1100 through AMDISARenderer -> rdna3 Insts -> assemble_linear, using:
  - RANGE/END counted-loop lowering: a uniform SGPR loop counter (s_mov init, s_cmp_lt_i32, s_cbranch_scc0 exit,
    s_add_i32 increment, s_branch backedge).
  - a label-resolution pass in AMDISARenderer.asm() that bakes PC-relative simm16 dword offsets for forward/backward
    branches (NOT hand-baked in instruction constructors); out-of-range offsets fail loudly.
  - an LDS-backed reduction accumulator (Ops.DEFINE_REG -> a fixed LDS slot, ds_store_b32/ds_load_b32 with conservative
    s_waitcnt drains). The accumulator is plain memory, so the per-iteration read-modify-write needs no SSA/regalloc
    accumulator coalescing. elf.py sizes the kernel group segment from DEFINE_REG. The RMW ordering is preserved by
    threading the AFTER chain through the LDS ops.

Scope: single-thread reductions (NOOPT, local_size=1) -- the minimal shape tinygrad emits. Multi-thread cross-lane
group reductions (OptOps.GROUPTOP: DEFINE_LOCAL + s_barrier + cross-lane) are out of scope for Phase B.

Run:  DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_b_reduction_gate.py
Writes: bench/amd-isa-backend-phase-b/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
os.environ.setdefault("NOOPT", "1")   # minimal single-thread RANGE/END reduction (no GROUPTOP cross-lane / barriers)
import numpy as np
from tinygrad import Tensor, Device
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops

CMD = "DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_b_reduction_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-phase-b", "latest.json")

# capture the compiled binary + the resolved instruction stream of the last kernel
_bin, _asm = [], []
_orig = AMDISARenderer.asm
def _spy(self, prg, lin):
  _asm.append([str(u.arg) for u in self._resolve_labels(list(lin.src)) if u.op is Ops.INS])
  b = _orig(self, prg, lin); _bin.append(b); return b
AMDISARenderer.asm = _spy

def _classify(tb: str, exc: Exception) -> str:
  if "branch offset" in str(exc) or "simm16" in tb: return "AMD_ISA_PHASE_B_BLOCKED_BRANCH_LABEL_RESOLUTION"
  if "regalloc.py" in tb: return "AMD_ISA_PHASE_B_BLOCKED_REGALLOC_LIVENESS"
  if "RANGE" in tb or "END" in tb: return "AMD_ISA_PHASE_B_BLOCKED_RANGE_END_ISEL"
  if "group_segment" in tb or "lds" in tb.lower(): return "AMD_ISA_PHASE_B_BLOCKED_LDS_DESCRIPTOR"
  if "assemble" in tb.lower() or "ELF" in tb: return "AMD_ISA_PHASE_B_BLOCKED_ASSEMBLE_OR_ELF"
  return "AMD_ISA_PHASE_B_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Phase B: correctness-first single-thread sum reductions"}
  ren = type(Device["AMD"].renderer).__name__
  rec["selected_renderer"] = ren
  if ren != "AMDISARenderer":
    rec["verdict"] = "AMD_ISA_PHASE_B_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"selected {ren}, not AMDISARenderer"; return rec
  rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"
  rec["accumulator_backing"] = "lds"
  rec["reduction_shape"] = ("DEFINE_REG(acc) -> STORE(acc,0); RANGE -> {LOAD(acc); LOAD(buf,i); ADD; STORE(acc)} -> END;"
                            " LOAD(acc) -> STORE(out)  (Ops.DEFINE_REG accumulator in LDS, counted RANGE/END loop)")
  try:
    rng = np.random.default_rng(20260629)
    results = {}
    # simple 1D sum reductions, including non-power-of-two sizes
    for n in [8, 16, 31, 64, 100, 127, 256]:
      a = rng.standard_normal(n).astype(np.float32)
      got = float(Tensor(a, device="AMD").sum().numpy())
      results[f"sum_{n}"] = {"ok": bool(np.isclose(got, float(a.sum()), rtol=1e-4, atol=1e-3)), "got": got, "exp": float(a.sum())}
    # simple row reduction (single thread per row: gidx0 + RANGE, no cross-lane)
    m = rng.standard_normal((4, 16)).astype(np.float32)
    rgot = Tensor(m, device="AMD").sum(axis=1).numpy()
    results["rowsum_4x16"] = {"ok": bool(np.allclose(rgot, m.sum(axis=1), rtol=1e-4, atol=1e-3))}
    rec["correctness"] = results

    # repeated-run stability (catch nondeterministic LDS/ordering bugs)
    a = rng.standard_normal(127).astype(np.float32)
    trials = [float(Tensor(a, device="AMD").sum().numpy()) for _ in range(8)]
    rec["repeated_run_stability"] = {"stable": all(t == trials[0] for t in trials), "trials": trials[:3] + ["..."]}

    asm = _asm[-1] if _asm else []
    rec["range_end_lowered"] = not any("RANGE" in s or "END" in s for s in asm)   # no RANGE/END survive to assembly
    rec["branch_label_resolution"] = "pass" if any("s_branch" in s or "s_cbranch" in s for s in asm) and \
                                                not any("label" in s or "branch'" in s for s in asm) else "fail"
    from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf
    rec["group_segment_size"] = group_segment_fixed_size_from_elf(_bin[-1]) if _bin else None

    all_ok = all(v["ok"] for v in results.values()) and rec["repeated_run_stability"]["stable"]
    if all_ok and rec["range_end_lowered"] and rec["branch_label_resolution"] == "pass":
      rec["verdict"] = "AMD_ISA_PHASE_B_PASS_SUM_REDUCTION"
    else:
      rec["verdict"] = "AMD_ISA_PHASE_B_BLOCKED_RUNTIME_ROUTE"
      rec["blocker"] = f"correctness/stability/lowering check failed: {results} stable={rec['repeated_run_stability']['stable']}"
  except Exception as e:
    tb = traceback.format_exc()
    rec["verdict"] = _classify(tb, e); rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["first_failure_site"] = next((l.strip() for l in reversed(tb.splitlines()) if "amd.py" in l or "regalloc.py" in l or "elf.py" in l), tb.splitlines()[-1].strip())
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred"] = ("multi-thread cross-lane group reduction (OptOps.GROUPTOP: DEFINE_LOCAL + s_barrier + cross-lane "
                     "via ds_bpermute); max reduction; non-conservative waitcnt scheduling")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nPHASE_B", "PASS" if rec["verdict"] == "AMD_ISA_PHASE_B_PASS_SUM_REDUCTION" else f"-> {rec['verdict']}")
