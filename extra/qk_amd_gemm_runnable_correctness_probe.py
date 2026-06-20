#!/usr/bin/env python3
# AMD GEMM runnable + correctness probe (REAL GPU launch, numeric gate, NO timing, NO perf claim, NO BEAM).
#
# Moves the selected ffn_gate/up structure from "structurally emitted loop" (placeholder addresses, not
# runnable) to "runnable candidate with a correctness gate." It drives the proven LDS-staged, 4-wave
# cooperative, 128x128 macro-tile WMMA GEMM (`build_gemm_lds` in extra/gemm/rdna3_wmma_matmul.py) — the same
# global_load -> LDS store -> barrier -> LDS read -> WMMA structure the emission probe emitted, but with a
# REAL fixed-shape address model — launches it on the AMD GPU, and validates the output against A @ B.
#
# This replaces the STRUCTURAL_EMISSION_ONLY placeholder addressing with runnable tiled addressing and proves
# the candidate actually computes A@B. It does NOT time and makes NO performance claim: if correctness fails,
# it emits the precise blocked/fail verdict. The candidate is single-buffer LDS (full barrier); the
# double-buffered A0/B0/A1/B1 (PGR1) variant is a perf overlap (RDNA3-refuted), out of scope for this gate.
from __future__ import annotations

import importlib.util, json, pathlib, traceback
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
EMISSION = "bench/amd-broad-backend-roadmap/amd_gemm_emission_result.json"
LOWERING = "bench/amd-broad-backend-roadmap/amd_gemm_lowering_plan_result.json"
CONTRACT = "bench/qk-tensile-extraction/ffn_gate_up_contract.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"

# (M, N, K). Bring-up first (fast sanity of the mapping), then the authority shape.
SHAPES = [
  {"name": "bring_up", "M": 128, "N": 128, "K": 256, "note": "reduced shape; same 128x128 macro-tile mapping as authority"},
  {"name": "authority", "M": 512, "N": 12288, "K": 4096, "note": "selected ffn_gate/up fixed shape"},
]
REL_RMSE_PASS = 0.02   # fp16 input / fp32 accumulate; K up to 4096 typically yields ~1e-3

# ---- documented mapping (read out of build_gemm_lds), required by the task ----
ADDRESS_MODEL = {
  "macro_tile": "BM=BN=128, BK=16; grid=(N//128, M//128, 1); workgroup=128 threads = 4 wave32 as 2x2",
  "per_thread_coop_load": "thread tid stages global row (gy*128+tid) of the 128x16 A-slice and (gx*128+tid) "
                          "of the 128x16 Bt-slice; ds_store to LDS at tid*32 (A region) / tid*32+LDS_B (B region)",
  "A_global_addr": "vA_glob = (gy*128 + tid)*K*2 + k_block*16*2  (A is M x K row-major, fp16)",
  "B_global_addr": "vB_glob = (gx*128 + tid)*K*2 + k_block*16*2  (Bt is N x K row-major = B transposed)",
  "C_global_addr": "row=gy*128 + wave_m*64 + mi*16 + (i*2+parity); col=gx*128 + wave_n*64 + ni*16 + (lane&15); "
                   "store fp16 at (row*N + col)*2",
}
FRAGMENT_MAPPING = {
  "wave_layout": "wave=tid>>5 (0..3); wave_m=wave>>1, wave_n=wave&1 -> 2x2 wave grid; each wave owns a 64x64 "
                 "sub-tile = WM x WN = 4 x 4 = 16 WMMA tiles",
  "lds_regions": "LDS_A=0 (128*16*2=4096 B), LDS_B=4096 (4096 B); single-buffer total 8192 B",
  "ds_store_offsets": "A-slice rows at tid*32 within LDS_A; Bt-slice rows at tid*32 within LDS_B",
  "ds_load_fragments": "vAfrag=wave_m*2048+(tid&15)*32, tile mi at +mi*512; vBfrag=LDS_B+wave_n*2048+(tid&15)*32, "
                       "tile ni at +ni*512 (each fragment = 2 x ds_load_b128)",
  "wmma_feed": "v_wmma(acc[mi*WN+ni], src0=FA[mi], src1=FB[ni], src2=acc[mi*WN+ni]) for mi,ni in 0..3 (16 WMMA)",
  "accumulator_to_output": "16 accumulator fragments ACCb+(mi*WN+ni)*8 map to the 64x64 wave sub-tile via the "
                           "C_global_addr formula above",
}


def read_json(rel: str) -> dict[str, Any]:
  path = ROOT / rel
  if not path.exists(): raise FileNotFoundError(f"required artifact missing: {rel}")
  return json.loads(path.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def emit(result: dict[str, Any]) -> int:
  write_json("amd_gemm_runnable_correctness_result.json", result)
  print(json.dumps({k: result[k] for k in ("verdict", "gate_pass") if k in result}, indent=2))
  if "exact_blocker" in result: print("exact_blocker:", result["exact_blocker"])
  for s in result.get("shapes", []):
    print(f"  {s['name']:9} M={s['M']} N={s['N']} K={s['K']}: "
          f"{s.get('status')}" + (f" rel_rmse={s['rel_rmse']:.6f} max_abs={s['max_abs']:.4f}" if "rel_rmse" in s else ""))
  return 0 if result.get("gate_pass") else 1


def load_ref():
  spec = importlib.util.spec_from_file_location("rdna3_wmma_matmul_ref", REF_SRC)
  mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod)
  return mod


def base(verdict: str) -> dict[str, Any]:
  return {
    "date": "2026-06-20", "phase": "AMD_GEMM_RUNNABLE_CORRECTNESS", "schema": "amd_gemm_runnable_correctness_v1",
    "role": "ffn_gate/up", "verdict": verdict, "gate_pass": verdict == "PASS_GEMM_RUNNABLE_CORRECTNESS",
    "default_behavior_changed": False, "performance_claim": False, "timed": False,
    "candidate": "build_gemm_lds (LDS-staged 4-wave cooperative 128x128 macro-tile WMMA GEMM; alpha=1, beta=0)",
    "candidate_source": "extra/gemm/rdna3_wmma_matmul.py:build_gemm_lds",
    "lds_buffering": "single-buffer (full barrier); 8192 B used (vs 25088 B authority double-buffered/padded)",
    "address_model": ADDRESS_MODEL, "fragment_mapping": FRAGMENT_MAPPING,
    "rel_rmse_pass_threshold": REL_RMSE_PASS,
    "input_artifacts": [EMISSION, LOWERING, CONTRACT, "extra/gemm/rdna3_wmma_matmul.py"],
  }


def main() -> int:
  # gate inputs
  try:
    emi = read_json(EMISSION); low = read_json(LOWERING); read_json(CONTRACT)
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_LAUNCH_OR_KERNARG"), "exact_blocker": f"missing input artifact: {ex!r}"})
  if emi.get("verdict") != "PASS_GEMM_STRUCTURAL_EMISSION" or low.get("verdict") != "PASS_GEMM_LOWERING_PLAN_READY":
    return emit({**base("BLOCKED_GEMM_LAUNCH_OR_KERNARG"),
                 "exact_blocker": f"upstream gate not passed: emission={emi.get('verdict')} lowering={low.get('verdict')}"})

  try:
    import numpy as np
    from tinygrad import Tensor, Device
    from tinygrad.dtype import dtypes
    from tinygrad.engine.realize import run_linear
    ref = load_ref()
  except Exception as ex:
    return emit({**base("BLOCKED_GEMM_LAUNCH_OR_KERNARG"),
                 "exact_blocker": f"import/env failure (need AMD device + reference module): {ex!r}",
                 "trace": traceback.format_exc().splitlines()[-4:]})

  shape_results: list[dict[str, Any]] = []
  blocker: tuple[str, str] | None = None
  for sh in SHAPES:
    M, N, K = sh["M"], sh["N"], sh["K"]
    rec = {"name": sh["name"], "M": M, "N": N, "K": K, "note": sh["note"]}
    # 1) build the runnable candidate (real tiled address model)
    try:
      insts = ref.build_gemm_lds(M, N, K)
    except Exception as ex:
      rec["status"] = "BLOCKED_ADDRESS_MODEL"; rec["error"] = repr(ex)
      shape_results.append(rec); blocker = ("BLOCKED_GEMM_ADDRESS_MODEL", f"{sh['name']}: build_gemm_lds failed: {ex!r}"); break
    rec["instruction_count"] = len(insts)
    # 2) launch on GPU + download output
    try:
      rng = np.random.default_rng(1)
      a_np = (rng.standard_normal((M, K)) * 0.1).astype(np.float16)
      bt_np = (rng.standard_normal((N, K)) * 0.1).astype(np.float16)   # Bt = B transposed (N x K)
      c = Tensor.empty(M, N, dtype=dtypes.half); Tensor.realize(c)
      linear, out = ref._run_insts_lds(insts, Tensor(a_np), Tensor(bt_np), c, M, N, K, "qk_correctness", 8192)
      run_linear(linear)
      c_np = out.float().numpy()
    except Exception as ex:
      rec["status"] = "BLOCKED_LAUNCH"; rec["error"] = repr(ex)
      shape_results.append(rec); blocker = ("BLOCKED_GEMM_LAUNCH_OR_KERNARG", f"{sh['name']}: launch/download failed: {ex!r}"); break
    # 3) numeric correctness vs A @ B  (B = bt.T)
    refmat = a_np.astype(np.float32) @ bt_np.astype(np.float32).T
    diff = c_np.astype(np.float32) - refmat
    rec["max_abs"] = float(np.max(np.abs(diff)))
    rec["rmse"] = float(np.sqrt(np.mean(diff ** 2)))
    rec["rel_rmse"] = float(rec["rmse"] / (np.sqrt(np.mean(refmat ** 2)) + 1e-9))
    rec["status"] = "PASS" if rec["rel_rmse"] < REL_RMSE_PASS else "FAIL_NUMERIC"
    shape_results.append(rec)

  if blocker is not None:
    return emit({**base(blocker[0]), "exact_blocker": blocker[1], "shapes": shape_results})
  failed = [r for r in shape_results if r["status"] != "PASS"]
  if failed:
    return emit({**base("FAIL_GEMM_NUMERIC_CORRECTNESS"),
                 "exact_blocker": f"numeric mismatch at shapes {[r['name'] for r in failed]} "
                                  f"(rel_rmse {[round(r.get('rel_rmse', -1), 6) for r in failed]} >= {REL_RMSE_PASS}); "
                                  "next: inspect fragment/output mapping or accumulation order for the failing shape",
                 "shapes": shape_results})

  result = {
    **base("PASS_GEMM_RUNNABLE_CORRECTNESS"), "shapes": shape_results, "runnable": True,
    "addressing_mode": "RUNNABLE_FIXED_SHAPE",
    "validated": "LDS-staged 4-wave cooperative 128x128 macro-tile WMMA GEMM computes A@B correctly at the "
                 "authority shape (M=512,N=12288,K=4096) and a bring-up shape; real tiled addressing, fragment "
                 "mapping, and accumulator->output mapping all confirmed numerically.",
    "explicitly_not_claimed": ["performance / TFLOPS", "double-buffered A0/B0/A1/B1 overlap (single-buffer here; "
                               "PGR1 LDS double-buffer is RDNA3-refuted)", "bit-exact Tensile layout"],
    "next_action": "Correctness passed -> the next and only remaining gate is TIMING under the PTM-1 "
                   "interleaved one-clock harness (authority shape, ratio vs the tinygrad authority row). "
                   "No BEAM/search.",
  }
  return emit(result)


if __name__ == "__main__":
  raise SystemExit(main())
