"""AMD native ISA backend — Inc 4 gate (Phase B: RANGE/END reductions). opt-in DEV=AMD:ISA.

STATUS: BLOCKED. This gate documents, precisely and reproducibly, why correctness-first reductions do not yet
lower through AMDISARenderer, and emits the first hard-blocker verdict.

What tinygrad emits for a reduction (NOOPT sum, the minimal shape):
  DEFINE_REG(acc)  -> STORE(INDEX(acc,0), 0.0)            # zero the accumulator
  RANGE(reduce) -> AFTER -> LOAD(INDEX(acc,0)) ; LOAD(INDEX(buf, RANGE)) ; ADD ; STORE(INDEX(acc,0), ADD) -> END
  LOAD(INDEX(acc,0)) -> STORE(out, acc)                  # write the result

Three missing subsystems block this (all confirmed):
  1. RANGE/END control flow is not lowered. The framework regalloc builds loop live-ranges from RANGE.reg
     (regalloc.py:30 `ranges.append(u.reg)`); AMDISARenderer never tags RANGE with a loop-counter register, so
     live-range construction dies at regalloc.py:29 `KeyError: None`. Lowering needs counter init + loop label +
     compare + s_cbranch (and END -> increment + s_branch back + out-label).
  2. No branch/label resolution. assemble_linear (renderer/amd/elf.py) only concatenates inst.to_bytes(); there is
     NO pass that resolves labels to PC-relative dword offsets and patches s_branch/s_cbranch simm16 fields.
  3. The reduction accumulator (Ops.DEFINE_REG) needs a MUTABLE per-thread location (written every iteration). The
     framework models this via the stack: DEFINE_REG->DEFINE_LOCAL->stack slot, and spill/fill/stack_pointer
     (regalloc.py:86-88,119,122-123). AMDISARenderer.stack_pointer/spill/fill all raise NotImplementedError, and AMD
     has no CPU-style stack -- it needs the scratch/private-segment ABI (flat_scratch SGPRs + descriptor scratch
     fields + scratch_load/store + runtime scratch alloc). A register-resident accumulator is incompatible with the
     single-def SSA regalloc (the accumulator is written once per iteration). LDS-backed accumulation is Phase F.

Verdict: AMD_ISA_INC4_BLOCKED_RANGE_END_LOWERING (first failure), with the accumulator-storage and
branch-resolution gaps documented as the deeper required work.

Run:  DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc4_gate.py
Writes: bench/amd-isa-backend-inc4/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
os.environ.setdefault("NOOPT", "1")   # avoid GROUPTOP (LDS+barrier) so we hit the minimal RANGE/END reduction shape
import numpy as np
from tinygrad import Tensor, Device

CMD = "DEV=AMD:ISA NOOPT=1 PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/inc4_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-inc4", "latest.json")

def _classify(tb: str, exc: Exception) -> str:
  if "regalloc.py" in tb and ("ranges.append" in tb or "lr[rng]" in tb or "KeyError" in tb): return "AMD_ISA_INC4_BLOCKED_RANGE_END_LOWERING"
  if "stack_pointer" in tb or "spill" in tb or "fill" in tb or "no spills" in str(exc): return "AMD_ISA_INC4_BLOCKED_REGALLOC_LIVE_RANGE"
  if "branch" in tb.lower() or "label" in tb.lower(): return "AMD_ISA_INC4_BLOCKED_BRANCH_OR_LABEL_ISEL"
  return "AMD_ISA_INC4_BLOCKED_RUNTIME_ROUTE"

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Inc 4 / Phase B: correctness-first RANGE/END reductions"}
  ren = type(Device["AMD"].renderer).__name__
  rec["selected_renderer"] = ren
  if ren != "AMDISARenderer":
    rec["verdict"] = "AMD_ISA_INC4_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"selected {ren}, not AMDISARenderer"; return rec
  rec["no_hidden_fallback"] = "PASS (selected AMDISARenderer)"
  try:
    a = np.arange(64, dtype=np.float32)
    got = Tensor(a, device="AMD").sum().numpy()
    if np.allclose(got, a.sum()):
      rec["verdict"] = "AMD_ISA_INC4_PASS_REDUCTIONS"; rec["correctness"] = {"sum64": "PASS", "got": float(got)}
    else:
      rec["verdict"] = "AMD_ISA_INC4_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"sum mismatch got={got} exp={a.sum()}"
  except Exception as e:
    tb = traceback.format_exc()
    rec["verdict"] = _classify(tb, e)
    rec["blocker"] = f"{type(e).__name__}: {e}"
    rec["first_failure_site"] = next((l.strip() for l in reversed(tb.splitlines()) if "regalloc.py" in l or "amd.py" in l), tb.splitlines()[-1].strip())
    rec["missing_subsystems"] = [
      "RANGE/END control-flow lowering (counter reg + loop label + s_cmp + s_cbranch + s_branch back); "
      "AMDISARenderer never tags RANGE so regalloc live-range build dies at regalloc.py:29 KeyError: None",
      "branch/label PC-offset resolution pass before assemble_linear (renderer/amd/elf.py concatenates bytes only)",
      "mutable reduction accumulator storage: framework routes DEFINE_REG->stack (stack_pointer/spill/fill all "
      "NotImplementedError in AMDISARenderer); AMD needs the scratch/private-segment ABI, or LDS accumulation (Phase F)",
    ]
  return rec

if __name__ == "__main__":
  rec = main()
  rec["next_minimal_action"] = ("implement RANGE/END isel+lowering (uniform SGPR counter, s_cmp_lt_i32, s_cbranch_scc0, "
                                "s_branch) + a label-resolution pass in AMDISARenderer.asm() that bakes simm16 dword "
                                "offsets; then pick an accumulator backing: easiest correct path is LDS (Phase F) or a "
                                "scratch/private-segment ABI for stack_pointer/spill/fill")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nINC4", "PASS" if rec["verdict"] == "AMD_ISA_INC4_PASS_REDUCTIONS" else f"-> {rec['verdict']}")
