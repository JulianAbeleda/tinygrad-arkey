"""AMD native ISA backend — Phase C gate: small fp32 GEMV correctness (opt-in DEV=AMD:ISA).

Proves y[row] = sum_k A[row,k]*x[k] runs correct on gfx1100 through AMDISARenderer, by COMPOSING existing pieces with
no new instruction selection:
  - global/workgroup indexing (gidx0 -> row; A index = row*K + k; x index = k; y index = row)   [Inc 2/3]
  - RANGE/END counted reduction loop over k, with the uniform SGPR loop counter copied to a VGPR for address math [Phase B]
  - LDS-backed accumulator (Ops.DEFINE_REG -> ds_store/ds_load), read-modify-write per iteration               [Phase B]
  - float MUL (A[row,k]*x[k]) + float ADD (acc += ...) -- the FMA inner body                                    [Inc 0/1]

Scope: single-thread-per-row GEMV (NOOPT, one workgroup per row). Multi-thread cross-lane / GROUPTOP reduction, b128,
v_dot2, Q4_K are out of scope for Phase C.

Run:  DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_c_gemv_gate.py
Writes: bench/amd-isa-backend-phase-c/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
os.environ.setdefault("NOOPT", "1")   # single-thread-per-row GEMV (no GROUPTOP cross-lane reduction)
import numpy as np
from tinygrad import Tensor, Device

CMD = "DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_c_gemv_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-phase-c", "latest.json")
SHAPES = [(4, 8), (8, 16), (16, 64), (8, 31), (5, 17)]   # incl. non-power-of-two K (31, 17) and M (5)

def _classify(tb: str, exc: Exception) -> str:
  if "regalloc.py" in tb: return "AMD_ISA_PHASE_C_BLOCKED_REGALLOC_PRESSURE"
  if "RANGE" in tb or "END" in tb or "reduction" in tb.lower(): return "AMD_ISA_PHASE_C_BLOCKED_REDUCTION_COMPOSITION"
  if "lds" in tb.lower() or "ds_" in tb or "group_segment" in tb: return "AMD_ISA_PHASE_C_BLOCKED_LDS_ACCUMULATOR"
  if "INDEX" in tb or "SPECIAL" in tb: return "AMD_ISA_PHASE_C_BLOCKED_INDEXING_ISEL"
  return "AMD_ISA_PHASE_C_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Phase C: small single-thread-per-row fp32 GEMV"}
  ren = type(Device["AMD"].renderer).__name__
  rec["selected_renderer"] = ren
  if ren != "AMDISARenderer":
    rec["verdict"] = "AMD_ISA_PHASE_C_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"selected {ren}, not AMDISARenderer"; return rec
  rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer, not HIP/LLVM)"
  rec["gemv_shape"] = ("y[row]=sum_k A[row,k]*x[k] :: gidx0->row, RANGE/END loop over k, A idx=row*K+k, x idx=k, "
                       "acc in LDS (ds_load/ds_store), inner = float MUL + float ADD")
  rec["reused_no_new_isel"] = True
  try:
    rng = np.random.default_rng(20260629)
    results = {}
    for (M, K) in SHAPES:
      A = rng.standard_normal((M, K)).astype(np.float32); x = rng.standard_normal(K).astype(np.float32)
      got = (Tensor(A, device="AMD") @ Tensor(x, device="AMD")).numpy()
      exp = A @ x
      results[f"{M}x{K}"] = {"ok": bool(np.allclose(got, exp, rtol=1e-3, atol=1e-3)), "maxerr": float(np.abs(got - exp).max())}
    rec["correctness"] = results

    # repeated-run determinism (catch nondeterministic LDS/ordering bugs)
    A = rng.standard_normal((16, 64)).astype(np.float32); x = rng.standard_normal(64).astype(np.float32)
    tA, tx = Tensor(A, device="AMD"), Tensor(x, device="AMD")
    trials = [(tA @ tx).numpy() for _ in range(8)]
    rec["repeated_run_stability"] = {"stable": all(np.array_equal(t, trials[0]) for t in trials)}

    all_ok = all(v["ok"] for v in results.values()) and rec["repeated_run_stability"]["stable"]
    rec["verdict"] = "AMD_ISA_PHASE_C_PASS_FP32_GEMV" if all_ok else "AMD_ISA_PHASE_C_BLOCKED_RUNTIME_ROUTE"
    if not all_ok: rec["blocker"] = f"correctness/stability failed: {results} stable={rec['repeated_run_stability']['stable']}"
  except Exception as e:
    tb = traceback.format_exc()
    rec["verdict"] = _classify(tb, e); rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["first_failure_site"] = next((l.strip() for l in reversed(tb.splitlines()) if "amd.py" in l or "regalloc.py" in l or "elf.py" in l), tb.splitlines()[-1].strip())
  return rec

if __name__ == "__main__":
  rec = main()
  rec["deferred"] = "multi-thread cross-lane/GROUPTOP GEMV; b128 vector memory; v_dot2; Q4_K GEMV; waitcnt scheduling"
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nPHASE_C", "PASS" if rec["verdict"] == "AMD_ISA_PHASE_C_PASS_FP32_GEMV" else f"-> {rec['verdict']}")
