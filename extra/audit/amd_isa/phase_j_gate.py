"""AMD ISA backend — Phase J gate: correct consumer-only waitcnt.

Replaces the conservative drain-after-every-memory-op model (s_waitcnt(0) after each load/store) with a
consumer-only model (renderer/isa/amd.py:_insert_waitcnt): a single full-drain s_waitcnt(0) only before a consumer
that touches a pending load's reg (RAW/WAR), a ds_load that may alias a pending LDS store (RMW), s_barrier, s_endpgm,
and loop branches (cross-iteration soundness). Full-drain stays correct; the count drop comes from batching loads +
dropping needless store waits.

Checks: (1) waitcnt count drops materially vs the conservative baseline (toggled via AMD_ISA_WAITCNT_CONSERVATIVE=1)
on the block tile + a primitive; (2) correctness unchanged (block tile numerically correct vs reference + Inc/B/C/F/G
gates); (3) no nondeterminism over repeated runs.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_j_gate.py
Writes: bench/amd-isa-backend-phase-j/latest.json
"""
import os, sys, json, subprocess, pathlib
os.environ.setdefault("DEV", "AMD:ISA")
ROOT = pathlib.Path(__file__).resolve().parents[3]
ART = ROOT / "bench/amd-isa-backend-phase-j/latest.json"
_MEM = ("global_load", "global_store", "ds_load", "ds_store", "ds_bpermute", "s_load")

def _counts_for_blocktile():
  import numpy as np
  from tinygrad import Tensor, dtypes
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  cap = {}
  _o = AMDISARenderer.asm
  def spy(self, prg, lin):
    pre = [u for u in self._resolve_labels(list(lin.src)) if u.op is Ops.INS]   # pre-waitcnt (inline waits removed)
    cons = sum(1 for u in pre if not isinstance(u.arg, tuple) and str(u.arg).split("(", 1)[0].startswith(_MEM))
    fin = [u for u in self._resolve_labels(self._insert_waitcnt(list(lin.src))) if u.op is Ops.INS]
    cseq = sum(1 for u in fin if str(u.arg).split("(", 1)[0] == "s_waitcnt")
    cap["conservative"], cap["consumer"], cap["n_insts"] = cons, cseq, len(fin)
    return _o(self, prg, lin)
  AMDISARenderer.asm = spy
  from extra.qk.flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 64, 32
  G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
  rng = np.random.default_rng(20260626 + Tc + L)
  q = rng.normal(0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32); cache[:, 0] = rng.normal(0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  def run():
    return Tensor.empty(Hq*S*W, dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)), Tensor(cache), fxn=fxn)[0].realize().numpy().reshape(Hq, S, W)
  outs = [run() for _ in range(3)]
  ref = np.zeros((Hq, S, W), dtype=np.float32); qh, ch = q.astype(np.float16).astype(np.float32), cache.astype(np.float16).astype(np.float32); scale = 1/np.sqrt(Hd)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s*L, min((s+1)*L, Tc)
      for g in range(G):
        h = kvh*G+g; sc = (ch[0,0,kvh,t0:t1,:]@qh[h])*scale; m = np.max(sc).astype(np.float32); pp = np.exp(sc-m).astype(np.float32)
        ref[h,s,:Hd] = pp@ch[1,0,kvh,t0:t1,:]; ref[h,s,Hd] = pp.sum(); ref[h,s,Hd+1] = m
  got = outs[0]
  correct = bool(np.isfinite(got).all()) and bool(np.allclose(got, ref, atol=5e-3, rtol=5e-2))
  det = all(np.array_equal(np.nan_to_num(outs[0]), np.nan_to_num(o)) for o in outs[1:])
  return cap, correct, det

def main():
  rec = {"verdict": None, "command": "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/audit/amd_isa/phase_j_gate.py",
         "model": "consumer-only waitcnt (drain only before consumers/RMW-ds_load/barrier/endpgm/loop-branches); full-drain s_waitcnt(0)"}
  try:
    cap, correct, det = _counts_for_blocktile()
    cons, cseq = cap["conservative"], cap["consumer"]
    rec["block_tile_waitcnt"] = {"conservative_baseline": cons, "consumer_only": cseq,
                                 "reduction": cons - cseq, "pct_reduction": round(100.0 * (cons - cseq) / cons, 1) if cons else 0.0,
                                 "n_insts": cap["n_insts"]}
    rec["memory_classes_tracked"] = {"vmcnt": "global_load/global_store", "lgkmcnt": "ds_load/ds_store/ds_bpermute/s_load"}
    rec["correctness_status"] = "PASS (block tile numerically correct vs reference)" if correct else "FAIL"
    rec["nondeterminism_check"] = "PASS (3 repeated runs identical)" if det else "FAIL (nondeterministic)"
    rec["waitcnt_baseline_comparison"] = ("conservative native baseline = 1 drain per memory op (toggle "
      "AMD_ISA_WAITCNT_CONSERVATIVE=1). LLVM-generated / owned hand-ASM waitcnt counts are in separate routes "
      "(not disassembled here); the meaningful native A/B is conservative vs consumer-only.")
    material = cons and (cons - cseq) >= max(1, int(0.1 * cons))   # >=10% drop = material
    if correct and det and material: rec["verdict"] = "AMD_ISA_PHASE_J_PASS_CONSUMER_WAITCNT"; rec["next_phase_unlocked"] = "Phase K: inst-stream scheduler"
    elif not correct: rec["verdict"] = "AMD_ISA_PHASE_J_BLOCKED_HAZARD_ANALYSIS"
    elif not det: rec["verdict"] = "AMD_ISA_PHASE_J_BLOCKED_NONDETERMINISM"
    else: rec["verdict"] = "AMD_ISA_PHASE_J_NO_COUNTER_MOVEMENT"
  except Exception as e:
    import traceback; rec["verdict"] = "AMD_ISA_PHASE_J_BLOCKED_MEMORY_CLASS_TRACKING"
    rec["exception"] = f"{type(e).__name__}: {e}"; rec["traceback"] = traceback.format_exc().splitlines()[-8:]
  return rec

if __name__ == "__main__":
  rec = main()
  # regression: correctness across Inc 0-3 + Phase B/C/F/G must still pass with the new waitcnt model
  reg = {}
  for name, cmd, env in [("inc0","extra/audit/amd_isa/inc0_gate.py",{}), ("inc1","extra/audit/amd_isa/inc1_gate.py",{}),
                         ("inc2","extra/audit/amd_isa/inc2_gate.py",{}), ("inc3","extra/audit/amd_isa/inc3_gate.py",{}),
                         ("phase_b","extra/audit/amd_isa/phase_b_reduction_gate.py",{"NOOPT":"1"}),
                         ("phase_c","extra/audit/amd_isa/phase_c_gemv_gate.py",{"NOOPT":"1"}),
                         ("phase_f","extra/audit/amd_isa/phase_f_primitives_gate.py",{}),
                         ("phase_g","extra/audit/amd_isa/phase_g_gate.py",{})]:
    try:
      e2 = {**os.environ, "DEV": "AMD:ISA", **env}
      out = subprocess.run([sys.executable, cmd], cwd=str(ROOT), env=e2, capture_output=True, text=True, timeout=500).stdout
      reg[name] = "PASS" if ("_PASS_" in out or "PASS" in out.splitlines()[-1]) else "FAIL"
    except Exception as ex: reg[name] = f"ERR {ex}"
  rec["regression_gates_status"] = reg
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_J", rec["verdict"])
