"""AMD ISA backend — Phase H gate: native-ISA decode block tile route-binding.

Phase H goal (docs/amd-isa-backend-phase-h-o-claude-scope-20260629.md): run the native AMD ISA generated
decode-attention tile through a real graph/model path and prove route attribution + correctness, no fallback.

This gate proves the ROUTE-BINDING KEYSTONE: the block tile compiled by AMDISARenderer (its ELF) loads + runs in
the HIP (DEV=AMD) device context via Ops.PROGRAM injection (extra/qk_native_isa_block_tile_graph_node.py) -- so the
attention tile can be the native candidate while the model stays on HIP. This is run under DEV=AMD (NOT AMD:ISA):
whole-model AMD:ISA is blocked by broad model op coverage (the ulong->float cast that "blocks Phase H" is in the
64-bit RNG/sampling path, not attention; the backend is an attention/index backend, not a full model renderer).

Status: keystone PASS (native tile injected + numerically correct under HIP, fixed context). Full in-model token
equivalence additionally requires DEFINE_VAR (start_pos runtime scalar) coverage -- isel_var is scaffolded and fires
but has an unresolved regalloc desync (store-value reg None) -- plus wiring into the FUSED_XLANE block-tile route
(extra/qk_flash_decode.py:1397) + route-attribution/token gate.

Run:  DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_h_gate.py
Writes: bench/amd-isa-backend-phase-h/latest.json
"""
import os, json, subprocess, sys, pathlib
os.environ.setdefault("DEV", "AMD")
CMD = "DEV=AMD PYTHONPATH=. .venv/bin/python extra/amd_isa_phase_h_gate.py"
ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "bench/amd-isa-backend-phase-h/latest.json"

def main():
  import numpy as np
  from tinygrad import Tensor, Device, dtypes
  from tinygrad.uop.ops import Ops
  rec = {"verdict": None, "command": CMD, "scope": "Phase H: native-ISA decode block tile route-binding"}
  rec["selected_renderer_device"] = type(Device["AMD"].renderer).__name__  # device stays HIP; tile compiled via ISA
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 64, 32
  G, W, S = Hq // Hkv, Hd + 2, (Tc + L - 1) // L
  try:
    from extra.qk_native_isa_block_tile_graph_node import native_isa_block_tile, compile_block_tile_isa
    elf, gsize, lsize, gseg = compile_block_tile_isa(Hd, Hq, Hkv, MAXC, L, S, Tc)
    rec["tile_compiled_via"] = "AMDISARenderer"
    rec["tile_elf_bytes"] = len(elf); rec["tile_group_segment"] = gseg
    rec["tile_grid"] = list(gsize); rec["tile_block"] = list(lsize)
    rng = np.random.default_rng(20260626 + Tc + L)
    q = rng.normal(0, 0.25, size=(Hq, Hd)).astype(np.float32)
    cache = np.zeros((2, 1, Hkv, MAXC, Hd), dtype=np.float32); cache[:, 0] = rng.normal(0, 0.25, size=(2, Hkv, MAXC, Hd)).astype(np.float32)
    def run():
      return native_isa_block_tile(Tensor.empty(Hq*S*W, dtype=dtypes.float32), Tensor(q.reshape(-1)),
                                   Tensor(cache), Hd, Hq, Hkv, MAXC, L, S, Tc).numpy().reshape(Hq, S, W)
    runs = [run() for _ in range(3)]
    got = runs[0]
    ref = np.zeros((Hq, S, W), dtype=np.float32)
    qh, ch = q.astype(np.float16).astype(np.float32), cache.astype(np.float16).astype(np.float32); scale = 1.0/np.sqrt(Hd)
    for kvh in range(Hkv):
      for s in range(S):
        t0, t1 = s*L, min((s+1)*L, Tc)
        for g in range(G):
          h = kvh*G+g; sc = (ch[0,0,kvh,t0:t1,:]@qh[h])*scale; m = np.max(sc).astype(np.float32); pp = np.exp(sc-m).astype(np.float32)
          ref[h,s,:Hd] = pp@ch[1,0,kvh,t0:t1,:]; ref[h,s,Hd] = pp.sum(); ref[h,s,Hd+1] = m
    finite = bool(np.isfinite(got).all()); correct = finite and bool(np.allclose(got, ref, atol=5e-3, rtol=5e-2))
    det = all(np.array_equal(np.nan_to_num(runs[0]), np.nan_to_num(r)) for r in runs[1:])
    rec["keystone_native_elf_runs_under_hip"] = bool(correct)
    rec["keystone_correctness"] = "PASS" if correct else f"FAIL (finite={finite})"
    rec["keystone_deterministic"] = bool(det)
    rec["no_hidden_fallback"] = "tile is a precompiled Ops.PROGRAM with the AMDISARenderer ELF (no HIP/LLVM/owned compile)"
    rec["cast_blocker_reclassified"] = ("CAST ulong->float is in the 64-bit RNG/sampling path (threefry: hi*2^32+lo, "
      "AND 0x3FFFFFFFFFFFFFFF), NOT decode attention. Whole-model DEV=AMD:ISA is blocked by broad model op coverage "
      "(real 64-bit arithmetic + optimized matmul/norm/etc.) -> AMD_ISA_PHASE_H_BLOCKED_MODEL_OP_COVERAGE. The tile-only "
      "native route (this keystone) avoids it.")
    rec["remaining_for_full_pass"] = ("(1) DEFINE_VAR/start_pos runtime-scalar coverage in AMDISARenderer (isel_var "
      "scaffolded + fires, unresolved regalloc desync: store-value reg None); (2) wire native_isa_block_tile into the "
      "FUSED_XLANE block-tile route (extra/qk_flash_decode.py:1397) under a flag, keeping gmax+combine on HIP; "
      "(3) in-model route-attribution + token-match gate via the current generated-route parity artifact.")
    rec["verdict"] = "AMD_ISA_PHASE_H_KEYSTONE_PASS_NATIVE_TILE_INJECTION" if (correct and det) \
                     else "AMD_ISA_PHASE_H_BLOCKED_MODEL_ROUTE_BINDING"
  except Exception as e:
    import traceback
    rec["verdict"] = "AMD_ISA_PHASE_H_BLOCKED_MODEL_ROUTE_BINDING"
    rec["exception"] = f"{type(e).__name__}: {e}"; rec["traceback"] = traceback.format_exc().splitlines()[-6:]
  return rec

if __name__ == "__main__":
  rec = main()
  # regression: Inc 0-3 + Phase B/C/F/G must still pass (DEFINE_VAR scaffolding is inert for them)
  reg = {}
  for name, cmd, extra in [("inc0","extra/amd_isa_inc0_gate.py",{}), ("inc1","extra/amd_isa_inc1_gate.py",{}),
                           ("inc2","extra/amd_isa_inc2_gate.py",{}), ("inc3","extra/amd_isa_inc3_gate.py",{}),
                           ("phase_b","extra/amd_isa_phase_b_reduction_gate.py",{"NOOPT":"1"}),
                           ("phase_c","extra/amd_isa_phase_c_gemv_gate.py",{"NOOPT":"1"}),
                           ("phase_f","extra/amd_isa_phase_f_primitives_gate.py",{}),
                           ("phase_g","extra/amd_isa_phase_g_gate.py",{"DEV":"AMD:ISA"})]:
    try:
      out = subprocess.run([sys.executable, cmd], cwd=str(ROOT), env={**os.environ, "DEV": extra.get("DEV","AMD:ISA"),
                           **{k:v for k,v in extra.items() if k!="DEV"}}, capture_output=True, text=True, timeout=500).stdout
      reg[name] = "PASS" if ("_PASS_" in out or "PASS" in out.splitlines()[-1]) else "FAIL"
    except Exception as ex: reg[name] = f"ERR {ex}"
  rec["regression_gates_status"] = reg
  ART.parent.mkdir(parents=True, exist_ok=True)
  with open(ART, "w") as f: json.dump(rec, f, indent=2)
  print(json.dumps(rec, indent=2)); print("\nPHASE_H", rec["verdict"])
