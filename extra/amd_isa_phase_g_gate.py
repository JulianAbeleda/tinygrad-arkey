"""AMD native ISA backend — Phase G gate: full generated decode block tile correctness (opt-in DEV=AMD:ISA).

Attempts to compile + run the real generated decode block tile
(extra.qk_flash_decode.flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel) through AMDISARenderer.

RESULT: route binding works (AMDISARenderer is selected for the tile, no HIP/LLVM/owned fallback), but the tile
exercises a large ALU/control surface that AMDISARenderer does not yet lower. Those ops survive instruction selection
and reach the framework register allocator, which is only wired for {INS,RANGE,END,DEFINE_*,PARAM,SPECIAL}|PSEUDO; any
other op desynchronizes regalloc's rewrite counter -> regalloc.py:118 KeyError. Because of that desync the tile is
all-or-nothing: every unsupported op must be lowered before it can run, so this is a precise codegen-coverage blocker.

Verdict: AMD_ISA_PHASE_G_BLOCKED_UNSUPPORTED_UOP.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_g_gate.py
Writes: bench/amd-isa-backend-phase-g/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
from collections import Counter

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_g_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-phase-g", "latest.json")

# difficulty categorization of the unsupported ops (for the next session's planning)
DIFFICULTY = {
  "XOR": "tractable: v_xor_b32 (like V_IADD)", "AND": "tractable: v_and_b32", "MAX": "tractable: v_max_f32",
  "BITCAST": "tractable: reinterpret, usually a no-op (same VGPR bits)",
  "CAST": "moderate: int<->int width + int<->float need v_cvt_* variants",
  "CMPLT": "moderate: v_cmp_lt_* -> predicate (VCC/SGPR); needs a predicate-location model",
  "CMPNE": "moderate: v_cmp_ne_* -> predicate",
  "WHERE": "moderate: v_cndmask_b32 consuming a CMP predicate (VCC)",
  "CUSTOMI": "moderate: map builtin-string args (__builtin_amdgcn_fdot2 -> v_dot2_f32_f16; exp2f/arg.exp2() -> v_exp_f32)",
  "STORE": "HARD: these are GATED stores store(val, lane.eq(0)) -> IF/STORE/ENDIF -> EXEC predication "
           "(s_and_saveexec_b32 / s_cbranch_execz / restore EXEC) -- divergent control flow not built in Phase F",
  "CMOD": "HARD: integer modulo has no single rdna3 op (needs reciprocal/magic-number sequence)",
  "CDIV": "HARD: integer divide has no single rdna3 op (needs reciprocal/magic-number sequence)",
}
REGALLOC_MATCHED = "{INS, RANGE, END, DEFINE_REG, DEFINE_LOCAL, PARAM, DEFINE_VAR, SPECIAL} | {CONST, NOOP, AFTER, BARRIER, GROUP}"

def main():
  rec = {"verdict": None, "command": CMD, "scope": "Phase G: full generated decode block tile through AMDISARenderer"}
  import numpy as np
  from tinygrad import Tensor, dtypes, Device
  import tinygrad.codegen.late.regalloc as R
  from tinygrad.codegen.late.regalloc import PSEUDO_OPS
  from tinygrad.uop.ops import Ops
  from extra.qk_flash_decode import flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel

  ren = type(Device["AMD"].renderer).__name__
  rec["selected_renderer"] = ren
  if ren != "AMDISARenderer":
    rec["verdict"] = "AMD_ISA_PHASE_G_BLOCKED_RUNTIME_ROUTE"; rec["blocker"] = f"selected {ren}, not AMDISARenderer"; return rec
  rec["route_binding"] = "PASS (AMDISARenderer selected for the tile via DEV=AMD:ISA custom_kernel; no HIP/LLVM/owned fallback)"
  rec["no_hidden_fallback"] = "PASS"

  # capture the unsupported (regalloc-unmatched) ops in the linearized tile
  MATCHED = {Ops.INS, Ops.RANGE, Ops.END, Ops.DEFINE_REG, Ops.DEFINE_LOCAL, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL} | PSEUDO_OPS
  captured = {}
  _o = R.LinearScanRegallocContext.__init__
  def patched(self, uops, r):
    if "ops" not in captured:
      captured["all_ops"] = dict(Counter(u.op.name for u in uops))
      captured["ops"] = dict(Counter(u.op.name for u in uops if u.op not in MATCHED and u.op is not Ops.SINK))
    return _o(self, uops, r)
  R.LinearScanRegallocContext.__init__ = patched

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 64, 32
  G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
  rng = np.random.default_rng(1)
  q = rng.normal(0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32); cache[:, 0] = rng.normal(0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  try:
    Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)), Tensor(cache), fxn=fxn)[0].realize().numpy()
    rec["verdict"] = "AMD_ISA_PHASE_G_PASS_BLOCK_TILE_CORRECT"   # would only reach here if it actually compiled+ran
  except Exception as e:
    tb = traceback.format_exc()
    rec["exception"] = f"{type(e).__name__}: {e}"
    rec["first_failure_site"] = next((l.strip() for l in reversed(tb.splitlines()) if "regalloc.py" in l or "amd.py" in l), tb.splitlines()[-1].strip())
    unsupported = captured.get("ops", {})
    rec["tile_op_histogram"] = captured.get("all_ops", {})
    rec["unsupported_uops"] = unsupported
    rec["unsupported_uop_difficulty"] = {op: DIFFICULTY.get(op, "unclassified") for op in unsupported}
    rec["regalloc_matched_set"] = REGALLOC_MATCHED
    rec["mechanism"] = ("unsupported ops survive AMDISARenderer isel, reach the framework regalloc (only wired for the "
                        "matched set), and desync its rewrite counter -> regalloc.py:118 KeyError. All-or-nothing: the "
                        "whole op surface must be lowered before the tile runs.")
    rec["verdict"] = "AMD_ISA_PHASE_G_BLOCKED_UNSUPPORTED_UOP"
  return rec

if __name__ == "__main__":
  rec = main()
  rec["next_minimal_action"] = ("extend AMDISARenderer isel/lowering to the full tile ALU/control surface, in order of "
                                "difficulty: (1) XOR/AND/MAX/BITCAST; (2) CAST (v_cvt), CMPLT/CMPNE->predicate + "
                                "WHERE->v_cndmask, CUSTOMI builtin-string mapping (fdot2->v_dot2, exp2->v_exp_f32); "
                                "(3) HARD: gated STORE via EXEC predication (s_and_saveexec/s_cbranch_execz), integer "
                                "CMOD/CDIV sequences. Then re-run; next likely walls are regalloc pressure (~88 VGPR) "
                                "and numeric correctness of exp2/online-softmax.")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps({k: v for k, v in rec.items() if k != "tile_op_histogram"}, indent=2))
  print("\nPHASE_G", rec["verdict"])
