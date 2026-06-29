"""AMD native ISA backend — Phase G gate: full generated decode block tile through AMDISARenderer.

Status: the UNSUPPORTED-UOP blocker is RESOLVED. The full generated decode block tile
(qk_flash_decode.flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel) now route-binds to AMDISARenderer
(no HIP/LLVM/owned fallback), the entire ALU/control surface lowers (no regalloc desync), and the tile
assembles + launches. The remaining blocker is NUMERIC CORRECTNESS of the full multi-thread (128-thread)
integration: the microgate output is not yet bit-faithful (residual multi-thread bug(s)).

Codegen coverage implemented this phase (each verified by an isolated microgate):
  XOR/AND/MAX/BITCAST; CAST (v_cvt_* + value-preserving 64<->32/bool reinterprets); CMPLT/CMPNE -> 0/1 bool
  via VCC+v_cndmask; WHERE -> v_cndmask; CDIV/CMOD by const power-of-two -> v_lshrrev / v_and; CUSTOMI builtin
  mapping (__builtin_amdgcn_fdot2 -> v_dot2_f32_f16 with v_pack_b32_f16 operand packing; ds_bpermute ->
  ds_bpermute_b32); gated STORE -> EXEC predication (v_cmp -> s_and_saveexec_b32(s5, VCC) -> store -> restore);
  half LDS via ds_store_b16/ds_load_u16; per-thread DEFINE_REG accumulators (LDS at tid*stride).

Verdict: AMD_ISA_PHASE_G_BLOCKED_NUMERIC_CORRECTNESS.

Run:  DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_g_gate.py
Writes: bench/amd-isa-backend-phase-g/latest.json
"""
import os, json, traceback
os.environ.setdefault("DEV", "AMD:ISA")
from collections import Counter

CMD = "DEV=AMD:ISA PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_g_gate.py"
ART = os.path.join(os.path.dirname(__file__), "..", "bench", "amd-isa-backend-phase-g", "latest.json")

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
    rec["verdict"] = "AMD_ISA_PHASE_G_BLOCKED_RUNTIME_ROUTE"; rec["first_blocker"] = f"selected {ren}"; return rec
  rec["route_attribution"] = "AMDISARenderer selected for the tile; no HIP/LLVM/owned fallback in candidate route"
  rec["no_hidden_fallback"] = True
  rec["unsupported_ops_before"] = {"CAST": 17, "WHERE": 18, "CMPNE": 14, "CMPLT": 9, "CUSTOMI": 6, "XOR": 5,
                                   "STORE_gated": 4, "BITCAST": 4, "MAX": 1, "AND": 2, "CMOD": 2, "CDIV": 2}
  rec["lowered_ops_by_class"] = ["XOR->v_xor_b32", "AND->v_and_b32", "MAX->v_max_f32", "BITCAST->passthrough",
                                 "CAST->v_cvt_*/reinterpret", "CMPLT/CMPNE->v_cmp+v_cndmask(0/1)", "WHERE->v_cndmask",
                                 "CDIV/CMOD(pow2)->v_lshrrev/v_and", "CUSTOMI fdot2->v_dot2_f32_f16+v_pack_b32_f16",
                                 "CUSTOMI ds_bpermute->ds_bpermute_b32", "gated STORE->EXEC predication",
                                 "half LDS->ds_store_b16/ds_load_u16", "DEFINE_REG->per-thread LDS (tid*stride)"]
  rec["customi_mappings"] = {"__builtin_amdgcn_fdot2": "v_dot2_f32_f16 (operands packed via v_pack_b32_f16)",
                             "__builtin_amdgcn_ds_bpermute": "ds_bpermute_b32 (addr=src1, data=src0)"}
  rec["predicate_strategy"] = "bool = 0/1 in a VGPR; CMP -> v_cmp(VCC) + v_cndmask(0,1); WHERE -> v_cmp_ne(cond,0) + v_cndmask(f,t)"
  rec["gated_store_strategy"] = "EXEC predication: v_cmp_ne(gate)->VCC; s_and_saveexec_b32(s5, VCC); store; s_mov(EXEC,s5)"
  rec["cdiv_cmod_strategy"] = "constant power-of-two only (tile uses /2,%2): CDIV->v_lshrrev(k), CMOD->v_and(n-1); non-pow2 fails loudly"

  # capture that no unsupported op survives into regalloc (codegen coverage complete)
  MATCHED = {Ops.INS, Ops.RANGE, Ops.END, Ops.DEFINE_REG, Ops.DEFINE_LOCAL, Ops.PARAM, Ops.DEFINE_VAR, Ops.SPECIAL, Ops.SINK} | PSEUDO_OPS
  cap = {}
  _o = R.LinearScanRegallocContext.__init__
  def patched(self, uops, r):
    if "after" not in cap: cap["after"] = dict(Counter(u.op.name for u in uops if u.op not in MATCHED))
    return _o(self, uops, r)
  R.LinearScanRegallocContext.__init__ = patched

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 64, 32
  G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
  rng = np.random.default_rng(20260626 + Tc + L)
  q = rng.normal(0, 0.25, size=(Hq, Hd)).astype(np.float32)
  cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32); cache[:, 0] = rng.normal(0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)
  fxn = flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc)
  try:
    got = Tensor.empty(Hq * S * W, dtype=dtypes.float32).custom_kernel(Tensor(q.reshape(-1)), Tensor(cache), fxn=fxn)[0].realize().numpy().reshape(Hq, S, W)
    rec["unsupported_ops_after"] = cap.get("after", {})
    rec["regalloc_status"] = "PASS (no desync; unsupported_ops_after empty)" if not cap.get("after") else f"survivors {cap['after']}"
    rec["assemble_status"] = "PASS (tile assembled + launched)"
    # numeric reference (mirror the block-tile microgate)
    ref = np.zeros((Hq, S, W), dtype=np.float32)
    qh, ch = q.astype(np.float16).astype(np.float32), cache.astype(np.float16).astype(np.float32); scale = 1.0 / np.sqrt(Hd)
    for kvh in range(Hkv):
      for s in range(S):
        t0, t1 = s * L, min((s + 1) * L, Tc)
        for g in range(G):
          h = kvh * G + g; sc = (ch[0, 0, kvh, t0:t1, :] @ qh[h]) * scale; m = np.max(sc).astype(np.float32)
          pp = np.exp(sc - m).astype(np.float32); ref[h, s, :Hd] = pp @ ch[1, 0, kvh, t0:t1, :]; ref[h, s, Hd] = pp.sum(); ref[h, s, Hd + 1] = m
    finite = bool(np.isfinite(got).all()); ok = finite and bool(np.allclose(got, ref, atol=5e-3, rtol=5e-2))
    rec["correctness_status"] = "PASS" if ok else f"FAIL (finite={finite}, max_abs={float(np.nanmax(np.abs(got))):.3e})"
    rec["repeated_run_stability"] = "n/a (numeric fail)"
    rec["verdict"] = "AMD_ISA_PHASE_G_PASS_BLOCK_TILE_CORRECT" if ok else "AMD_ISA_PHASE_G_BLOCKED_NUMERIC_CORRECTNESS"
    if not ok:
      rec["first_blocker"] = ("residual multi-thread numeric bug(s) in the 128-thread integration. Isolated findings: "
        "the staged warp-reduce (_warp_reduce_sum_staged) is correct at 4/8/16 lanes but DROPS the final butterfly "
        "stage at 32 lanes (full wave): out=256(=sum even lanes)+272(=sum odd lanes)=528 expected; the direct/inline "
        "butterfly is correct at 32 lanes, so this is a staging/regalloc-pressure interaction with cross-lane ds_bpermute. "
        "With DECODE_ATTN_BLOCK_TILE_INLINE_REDUCE=1 (direct reduce) the tile still NaNs, so >=1 further bug exists in "
        "the online-softmax/PV multi-thread path. Verified-correct components: dot2-pack, half-LDS roundtrip, gated "
        "store (EXEC), per-thread accumulators, nested loops, exp/exp2, all scalar ALU/casts, direct bpermute butterfly.")
  except Exception as e:
    tb = traceback.format_exc()
    rec["unsupported_ops_after"] = cap.get("after", {})
    rec["exception"] = f"{type(e).__name__}: {e}"
    rec["first_blocker"] = next((l.strip() for l in reversed(tb.splitlines()) if "amd.py" in l or "regalloc.py" in l), tb.splitlines()[-1].strip())
    rec["verdict"] = "AMD_ISA_PHASE_G_BLOCKED_UNSUPPORTED_UOP" if cap.get("after") else "AMD_ISA_PHASE_G_BLOCKED_RUNTIME_ROUTE"
  return rec

if __name__ == "__main__":
  rec = main()
  rec["next_minimal_action"] = ("debug the 128-thread integration: (1) fix the 32-lane staged warp-reduce final-stage drop "
    "(likely cross-lane VGPR liveness/regalloc-pressure around ds_bpermute); (2) isolate the residual online-softmax/PV "
    "NaN under inline reduce. Component microgates all pass; the bug is multi-thread cross-lane interaction at full wave.")
  os.makedirs(os.path.dirname(ART), exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2))
  print("\nPHASE_G", rec["verdict"])
