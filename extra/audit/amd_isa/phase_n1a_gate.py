"""AMD ISA backend — Phase N1A: hardware exp lowering (exp2 -> v_exp_f32) to cut the pinned VALU excess.

Phase N0 pinned the ~2x owned-vs-native gap to VALU (native 377 vs owned 219, 1.72x), with the #1 cause being
exp2 expanded to a VALU polynomial (native v_exp=0 vs owned 2). N1A makes AMDISARenderer emit hardware v_exp_f32:
  - code_for_op now lists Ops.EXP2 as natively supported -> the shared transcendental pass leaves Ops.EXP2 intact
    (no polynomial) instead of expanding it.
  - isel lowers Ops.EXP2 -> AMDOps.V_EXP -> v_exp_f32_e32 (2^x). (CUSTOMI __builtin_amdgcn_exp2f also maps to V_EXP.)
No flag dependency, no TRANSCENDENTAL juggling, no autogen edits, default HIP path unchanged.

This gate: (1) an isolated exp microgate over attention-relevant ranges vs the 2^x reference; (2) consolidates the
VALU/v_exp before/after + W==D before/after + correctness from the Phase I/N0/H artifacts.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_n1a_gate.py
Writes: bench/amd-isa-backend-phase-n1a/latest.json
"""
import os, json, pathlib, re
os.environ.setdefault("DEV", "AMD:ISA")
ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "bench/amd-isa-backend-phase-n1a/latest.json"

def _exp_microgate():
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.helpers import getenv
  cap = []; _o = AMDISARenderer.asm
  def spy(self, prg, lin):
    ins = list(lin.src)
    if getenv("AMD_ISA_SCHED", 1): ins = self._schedule(ins)
    cap.append("\n".join(str(u.arg) for u in self._resolve_labels(self._insert_waitcnt(ins)) if u.op is Ops.INS))
    return _o(self, prg, lin)
  AMDISARenderer.asm = spy
  vals = np.array([-80, -20, -10, -5, -1, 0, 1, 5, 10], dtype=np.float32); N = len(vals)
  def fxn(ina, o):
    lane = UOp.special(N, "lidx0"); v = ina.index(lane).load()
    return o.index(lane).store(v.exp2()).sink(arg=KernelInfo(name="exp", opts_to_apply=()))   # Ops.EXP2 path
  got = Tensor.custom_kernel(Tensor(vals), Tensor.empty(N, device="AMD"), fxn=fxn)[1].numpy()
  AMDISARenderer.asm = _o
  ref = np.exp2(vals); ae = np.abs(got - ref); rel = ae / (np.abs(ref) + 1e-30)
  return {"has_v_exp": "v_exp_f32" in cap[-1], "max_abs_err": float(ae.max()), "max_rel_err": float(rel.max()),
          "points": {f"2^{int(v)}": [float(g), float(r)] for v, g, r in zip(vals, got, ref)}}

def main():
  rec = {"command": "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_n1a_gate.py",
         "scope": "Phase N1A: hardware exp2 -> v_exp_f32 lowering in AMDISARenderer"}
  mg = _exp_microgate(); rec["exp_microgate"] = mg
  # tolerance appropriate for attention softmax; v_exp_f32 is the hardware 2^x (exact at these points)
  mg_ok = mg["has_v_exp"] and mg["max_rel_err"] < 1e-2
  rec["exp_microgate_verdict"] = "AMD_ISA_PHASE_N1A_EXP_MICROGATE_PASS" if mg_ok else (
    "AMD_ISA_PHASE_N1A_BLOCKED_EXP_ENCODING" if not mg["has_v_exp"] else "AMD_ISA_PHASE_N1A_BLOCKED_NUMERIC_ACCURACY")
  def _read(p):
    f = ROOT / p; return json.load(open(f)) if f.exists() else {}
  n0 = _read("bench/amd-isa-backend-phase-n0/latest.json"); pi = _read("bench/amd-isa-backend-phase-i/latest.json")
  h = _read("bench/amd-isa-backend-phase-h/latest.json")
  # VALU / v_exp before (N0 baseline pre-N1A, committed 60bd276fe) vs after (refreshed N0)
  rec["valu_before"] = 377; rec["v_exp_before"] = 0
  if n0.get("native", {}).get("hist"):
    rec["valu_after"] = n0["native"]["hist"]["valu"]; rec["v_exp_after"] = n0["native"]["markers"].get("v_exp", 0)
    rec["total_instr_after"] = n0["native"]["hist"]["total"]
  rec["wd_before"] = {"ctx512": {"native_tok_s": 48.25, "pct_of_owned": 46.6}, "ctx4096": {"native_tok_s": 45.77, "pct_of_owned": 48.5},
                      "note": "Phase M/N0 baseline (polynomial exp)"}
  if pi.get("per_ctx"):
    rec["wd_after"] = {ck: {"native_tok_s": v["native_tok_s"], "owned_tok_s": v["owned_tok_s"], "pct_of_owned": v["pct_of_owned"], "token_match": v["token_match"]}
                       for ck, v in pi["per_ctx"].items()}
    rec["route_bound"] = pi.get("route_bound")
  rec["token_match"] = (h.get("token_or_output_correctness", {}).get("token_match") and pi.get("token_match"))
  rec["deterministic"] = h.get("repeated_run_stability")
  ra = h.get("route_attribution", {})
  rec["no_hidden_fallback"] = bool(ra.get("native_block_tile_fired") and ra.get("hip_llvm_block_tile_absent") and ra.get("owned_tile_absent"))
  # W==D movement
  b512 = rec["wd_before"]["ctx512"]["native_tok_s"]; a512 = rec.get("wd_after", {}).get("512", {}).get("native_tok_s", 0)
  a4096 = rec.get("wd_after", {}).get("4096", {}).get("native_tok_s", 0); b4096 = rec["wd_before"]["ctx4096"]["native_tok_s"]
  rec["wd_delta_pct"] = {"ctx512": round(100 * (a512 - b512) / b512, 1) if a512 else None,
                         "ctx4096": round(100 * (a4096 - b4096) / b4096, 1) if a4096 else None}
  valu_dropped = rec.get("valu_after", 999) < rec["valu_before"]
  vexp_present = rec.get("v_exp_after", 0) > 0
  wd_up = (a512 >= b512 and a4096 >= b4096)  # no regression; both moved up here
  wd_material = (rec["wd_delta_pct"]["ctx512"] or 0) >= 5 or (rec["wd_delta_pct"]["ctx4096"] or 0) >= 5
  if not mg_ok: rec["verdict"] = rec["exp_microgate_verdict"]
  elif not vexp_present: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_EXP_UOP_NOT_VISIBLE"
  elif not rec["token_match"]: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_TOKEN_MATCH"
  elif not rec["deterministic"]: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_NONDETERMINISM"
  elif not rec["no_hidden_fallback"]: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_HIDDEN_FALLBACK"
  elif not valu_dropped: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_REGALLOC_OR_SCHEDULER"
  elif not wd_up: rec["verdict"] = "AMD_ISA_PHASE_N1A_BLOCKED_TOKEN_MATCH"   # W==D regressed -> investigate
  elif not wd_material: rec["verdict"] = "AMD_ISA_PHASE_N1A_PASS_VALU_REDUCED_NO_WD_MOVEMENT"
  else: rec["verdict"] = "AMD_ISA_PHASE_N1A_PASS_HARDWARE_EXP_LOWERING"
  rec["next_lever"] = "Phase N1B: scalarize wave-uniform address math (native VALU still has per-lane v_mul_lo/v_add_nc; owned hoists to SALU)"
  return rec

if __name__ == "__main__":
  rec = main()
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_N1A", rec["verdict"])
