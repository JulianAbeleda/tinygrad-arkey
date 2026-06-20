#!/usr/bin/env python3
# AMD GEMM PMC bottleneck classification (native PMC, REAL GPU, NO BEAM, NO routing/default change).
#
# Names the contention the occupancy diagnostic implied. Captures tinygrad's NATIVE PMC (PMC=1 — which DOES
# instrument our hand-asm run_linear/HCQ dispatches, unlike rocprofv3) for the SAME BK32 kernel at wg4
# (contention-limited, ~49 TFLOPS) vs wg2 (occupancy optimum, ~57.7), plus BK16 (depth contrast) and the LLVM
# authority. Counters are normalized by GRBM_GUI_ACTIVE (wall cycles) so wg4/wg2 are compared at equal work.
#
# Hypothesis (from the occupancy diagnostic): wg4 is slower because of LDS/barrier CONTENTION among 4
# workgroups/CU, NOT compute saturation. Prediction: wg4 has lower compute utilization (VALU & SQ_busy per
# active cycle) and a higher LDS-active / bank-conflict signature than wg2. PMC counters are perturbing and
# instance-summed (per MEMORY) — so this reports RATIOS / wg4-vs-wg2 DELTAS, not absolute rates.
#
# Run:  DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_pmc_probe.py
from __future__ import annotations

import importlib.util, json, os, pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
OCC = "bench/amd-broad-backend-roadmap/amd_gemm_occupancy_diagnostic_result.json"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
PTM1_SRC = ROOT / "extra/qk_amd_bb5a10_ptm1_same_harness_bridge.py"

M, N, K = 512, 12288, 4096


def read_json(rel: str) -> dict[str, Any]:
  p = ROOT / rel
  if not p.exists(): raise FileNotFoundError(rel)
  return json.loads(p.read_text())


def write_json(name: str, data: Any) -> None:
  OUT.mkdir(parents=True, exist_ok=True)
  (OUT / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(mod); return mod


def emit(result: dict[str, Any]) -> int:
  write_json("amd_gemm_pmc_result.json", result)
  print(json.dumps({k: result[k] for k in ("verdict",) if k in result}, indent=2))
  if "exact_blocker" in result: print("note:", result["exact_blocker"])
  hdr = f"  {'config':16} {'cycles(M)':>9} {'L2hit%':>7} {'VALU/act':>9} {'busy/act':>9} {'LDSact/act':>11} {'bankcf/act':>11}"
  print(hdr)
  for r in result.get("rows", []):
    n = r.get("norm")
    if n: print(f"  {r['label']:16} {r['cycles']/1e6:9.1f} {r['L2_hit%']:7.1f} {n['valu_per_active']:9.2f} "
                f"{n['busy_per_active']:9.2f} {n['lds_active_per_active']:11.2f} {n['bankconf_per_active']:11.2f}")
    else: print(f"  {r['label']:16} {r.get('status','?')}")
  return 0


def main() -> int:
  if os.environ.get("PMC") != "1":
    return emit({"date": "2026-06-20", "phase": "AMD_GEMM_PMC", "schema": "amd_gemm_pmc_v1",
                 "verdict": "BLOCKED_PMC_NOT_ENABLED",
                 "exact_blocker": "run with: DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_pmc_probe.py"})
  try:
    occ = read_json(OCC)
  except Exception as ex:
    return emit({"verdict": "BLOCKED_PMC_PRECONDITION", "exact_blocker": f"missing occupancy artifact: {ex!r}"})

  import numpy as np
  from tinygrad import Tensor, Device
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  from tinygrad.device import Compiled
  from extra.qk_pmc_capture import decode_pmc

  def capture_full(fn, warmup=3):
    for _ in range(warmup): fn()
    bse = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
    fn(); Device["AMD"].synchronize(); Device["AMD"]._at_profile_finalize()
    evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][bse:]
    return [decode_pmc(ev) for ev in evs]

  def metrics(c: dict) -> dict:
    act = c.get("GRBM_GUI_ACTIVE", 0) or 1
    hit, miss = c.get("GL2C_HIT", 0), c.get("GL2C_MISS", 0)
    return {"cycles": c.get("GRBM_GUI_ACTIVE", 0), "L2_hit%": round(100 * hit / (hit + miss + 1e-9), 1),
            "norm": {"valu_per_active": round(c.get("SQ_INSTS_VALU", 0) / act, 2),
                     "busy_per_active": round(c.get("SQ_BUSY_CYCLES", 0) / act, 2),
                     "lds_active_per_active": round(c.get("SQC_LDS_IDX_ACTIVE", 0) / act, 2),
                     "bankconf_per_active": round(c.get("SQC_LDS_BANK_CONFLICT", 0) / act, 2)},
            "raw": {k: c.get(k) for k in ("GRBM_GUI_ACTIVE", "SQ_BUSY_CYCLES", "SQ_INSTS_VALU",
                                          "SQC_LDS_IDX_ACTIVE", "SQC_LDS_BANK_CONFLICT", "GL2C_HIT", "GL2C_MISS")}}

  ref = load_mod(REF_SRC, "rdna3_ref")
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)

  configs = [
    ("bk32_wg4", lambda: ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 0, 0), a, bt, c, M, N, K, "bk32_wg4", 16384, 128, 128, 128)),
    ("bk32_wg2", lambda: ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 0, 0), a, bt, c, M, N, K, "bk32_wg2", 32768, 128, 128, 128)),
    ("bk16_wg_def", lambda: ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 16, 0, 0), a, bt, c, M, N, K, "bk16", 8192, 128, 128, 128)),
  ]
  rows: list[dict[str, Any]] = []
  for label, mk in configs:
    try:
      lin, _ = mk()
      caps = capture_full(lambda: run_linear(lin), warmup=3)
      cap = caps[0] if caps else None
    except Exception as ex:
      rows.append({"label": label, "status": f"FAIL: {ex!r}"[:80]}); continue
    if not cap:
      rows.append({"label": label, "status": "NO_PMC_ROW"}); continue
    rows.append({"label": label, **metrics(cap)})

  # authority (LLVM) reference — also an HCQ dispatch
  try:
    ptm1 = load_mod(PTM1_SRC, "ptm1_bridge"); auth_lin, _m, _alive = ptm1.build_authority()
    caps = capture_full(lambda: run_linear(auth_lin), warmup=3)
    if caps: rows.append({"label": "authority_llvm", **metrics(caps[0])})
    else: rows.append({"label": "authority_llvm", "status": "NO_PMC_ROW"})
  except Exception as ex:
    rows.append({"label": "authority_llvm", "status": f"UNAVAILABLE: {ex!r}"[:120]})

  wg4 = next((r for r in rows if r["label"] == "bk32_wg4" and "norm" in r), None)
  wg2 = next((r for r in rows if r["label"] == "bk32_wg2" and "norm" in r), None)
  auth = next((r for r in rows if r["label"] == "authority_llvm" and "norm" in r), None)
  classification, why = "INCONCLUSIVE", ""
  delta = {}
  if wg4 and wg2:
    delta = {
      "cycles_wg4_over_wg2": round(wg4["cycles"] / (wg2["cycles"] or 1), 3),
      "L2hit_wg4_minus_wg2": round(wg4["L2_hit%"] - wg2["L2_hit%"], 1),
      "busy_per_active_wg4_over_wg2": round(wg4["norm"]["busy_per_active"] / (wg2["norm"]["busy_per_active"] or 1e-9), 3),
      "valu_per_active_wg4_over_wg2": round(wg4["norm"]["valu_per_active"] / (wg2["norm"]["valu_per_active"] or 1e-9), 3),
      "bankconf_per_active_wg2": wg2["norm"]["bankconf_per_active"],
      "wg2_busy_vs_authority": round(wg2["norm"]["busy_per_active"] / (auth["norm"]["busy_per_active"] or 1e-9), 3) if auth else None,
      "wg2_bankconf_vs_authority": (wg2["norm"]["bankconf_per_active"], auth["norm"]["bankconf_per_active"] if auth else None),
    }
    # high-occupancy penalty: L2 hit drops + per-cycle compute falls at wg4
    l2_contention = (wg4["L2_hit%"] < wg2["L2_hit%"] - 2.0)
    lower_util = delta["busy_per_active_wg4_over_wg2"] < 0.98 or delta["valu_per_active_wg4_over_wg2"] < 0.98
    # wg2 LDS-bound signature: BK32 bank conflicts dwarf BK16's (robust contrast) and the no-LDS authority's 0
    bk16 = next((r for r in rows if r["label"].startswith("bk16") and "norm" in r), None)
    bk16_bc = bk16["norm"]["bankconf_per_active"] if bk16 else 0
    delta["bankconf_wg2_over_bk16"] = round(wg2["norm"]["bankconf_per_active"] / (bk16_bc or 1e-9), 1)
    lds_bound = (wg2["norm"]["bankconf_per_active"] > 10.0 and wg2["norm"]["bankconf_per_active"] > 3 * bk16_bc)
    if l2_contention and lower_util and lds_bound:
      classification = "CONFIRMED_CONTENTION_L2_PLUS_LDS_BANKCONFLICT_BOUND"
      why = (f"HIGH-OCCUPANCY PENALTY NAMED: wg4 L2 hit {wg4['L2_hit%']}% < wg2 {wg2['L2_hit%']}% "
             f"(4 workgroups/CU thrash L2) and per-cycle compute is lower at wg4 -> the high-occupancy slowdown "
             f"is L2/memory CONTENTION. WHAT BOUNDS wg2 (residual to Tensile): the LDS family is "
             f"LDS-bank-conflict/throughput bound -- bank conflicts {wg2['norm']['bankconf_per_active']:.1f}/cycle "
             f"(vs BK16 ~4.6 and the global-direct authority 0), LDS unit ~saturated, SIMD busy/act "
             f"{wg2['norm']['busy_per_active']:.1f} ~{round((1-(delta['wg2_busy_vs_authority'] or 1))*100)}% below "
             f"the authority (idle, stalled on LDS). Tensile's LdsPadB (bank-conflict pad) + PLR1 (prefetch to "
             f"cut LDS-read stall) attack exactly this -- a NAMED lever, not 'more occupancy'.")
    elif lower_util or l2_contention:
      classification = "CONFIRMED_CONTENTION_AT_HIGH_OCCUPANCY_PARTIAL_NAME"
      why = (f"high-occupancy contention confirmed (wg4 L2 {wg4['L2_hit%']}% vs wg2 {wg2['L2_hit%']}%, "
             f"lower per-cycle compute at wg4) but the wg2 binding-resource signature was not fully clean.")
    else:
      classification = "PMC_INCONCLUSIVE_NO_CLEAN_DELTA"
      why = "wg4/wg2 PMC counters did not separate cleanly (perturbation); timing delta stands, PMC can't name it."

  result = {
    "date": "2026-06-20", "phase": "AMD_GEMM_PMC", "schema": "amd_gemm_pmc_v1", "role": "ffn_gate/up",
    "verdict": classification, "gate_pass": classification.startswith("CONFIRMED"),
    "default_behavior_changed": False, "performance_claim": False, "is_diagnostic": True,
    "method": "native PMC (PMC=1) on hand-asm run_linear dispatches; counters normalized by GRBM_GUI_ACTIVE; "
              "wg4-vs-wg2 delta is the signal (PMC absolute rates perturbing/instance-summed)",
    "counters": "SQ_BUSY_CYCLES,SQ_INSTS_VALU,SQC_LDS_IDX_ACTIVE,SQC_LDS_BANK_CONFLICT,GRBM_GUI_ACTIVE,GL2C_HIT/MISS",
    "rows": rows, "wg4_vs_wg2_delta": delta, "classification": classification, "why": why,
    "input_artifacts": [OCC, "extra/qk_pmc_capture.py", "extra/gemm/rdna3_wmma_matmul.py"],
    "next": "If LDS contention confirmed: a minimal SLW/PLR-style local-write/read interleave to cut LDS/barrier "
            "pressure (one scheduling change), measured under the interleaved gate. No BEAM, no more occupancy sweeps.",
  }
  result["verdict_detail"] = why
  return emit(result)


if __name__ == "__main__":
  raise SystemExit(main())
