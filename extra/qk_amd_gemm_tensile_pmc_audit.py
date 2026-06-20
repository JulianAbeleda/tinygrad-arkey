#!/usr/bin/env python3
# HARD AUDIT: PMC ours vs the Tensile .co at the counter level to NAME the residual gap (it's not prefetch).
#
# Both our kernel (build_gemm_lds2 BK32+PAD16+PLRA=1) and the vendored Tensile .co are HCQ dispatches, so
# tinygrad native PMC (PMC=1) captures BOTH. GRBM_GUI_ACTIVE = GPU active cycles per launch = launch-mechanism-
# INDEPENDENT "time" (removes the wall-clock launch-context artifact). The cycle ratio is the real gap; the
# other counters (L2 hit -> WGM8 locality; VALU/inst count -> scheduling/instruction efficiency; LDS-active/
# bankconf -> LDS; SQ_WAVES -> occupancy) decompose WHY.
#
# Run:  DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_tensile_pmc_audit.py
#       (optionally PMC_COUNTERS=SQ_BUSY_CYCLES,SQ_INSTS_VALU,GRBM_GUI_ACTIVE,GL2C_HIT,GL2C_MISS,SQ_WAVES,...)
from __future__ import annotations

import importlib.util, json, os, pathlib, struct
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
REF_SRC = ROOT / "extra/gemm/rdna3_wmma_matmul.py"
HCQ_SRC = ROOT / "extra/qk_tensile_hcq_launch.py"
M, N, K = 512, 12288, 4096
FLOP = 2 * M * N * K


def load_mod(path, name):
  spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def main() -> int:
  if os.environ.get("PMC") != "1":
    print("run with: DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_tensile_pmc_audit.py"); return 1
  import numpy as np
  from tinygrad import Tensor, Device
  from tinygrad.dtype import dtypes
  from tinygrad.engine.realize import run_linear
  from tinygrad.device import Compiled
  from extra.qk_pmc_capture import decode_pmc
  dev = Device["AMD"]; ref = load_mod(REF_SRC, "rdna3_ref"); hcq = load_mod(HCQ_SRC, "hcq_launch")

  def cap(fn, warmup=4):
    for _ in range(warmup): fn()
    bse = len([e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"])
    fn(); dev.synchronize(); dev._at_profile_finalize()
    evs = [e for e in Compiled.profile_events if type(e).__name__ == "ProfilePMCEvent"][bse:]
    return decode_pmc(evs[0]) if evs else None

  def metrics(c: dict) -> dict:
    act = c.get("GRBM_GUI_ACTIVE", 0) or 1
    hit, miss = c.get("GL2C_HIT", 0), c.get("GL2C_MISS", 0)
    return {"cycles": c.get("GRBM_GUI_ACTIVE", 0), "L2_hit%": round(100 * hit / (hit + miss + 1e-9), 1),
            "valu_total": c.get("SQ_INSTS_VALU", 0), "salu_total": c.get("SQ_INSTS_SALU", 0),
            "busy_per_active": round(c.get("SQ_BUSY_CYCLES", 0) / act, 2),
            "valu_per_active": round(c.get("SQ_INSTS_VALU", 0) / act, 3),
            "lds_active_per_active": round(c.get("SQC_LDS_IDX_ACTIVE", 0) / act, 2),
            "bankconf_per_active": round(c.get("SQC_LDS_BANK_CONFLICT", 0) / act, 2),
            "waves": c.get("SQ_WAVES", 0), "raw": {k: c.get(k) for k in sorted(c)}}

  # ours
  rng = np.random.default_rng(1)
  a = Tensor((rng.standard_normal((M, K)) * 0.1).astype(np.float16), device="AMD")
  bt = Tensor((rng.standard_normal((N, K)) * 0.1).astype(np.float16), device="AMD")
  c = Tensor.empty(M, N, dtype=dtypes.half, device="AMD"); Tensor.realize(a, bt, c)
  ours_lin, _ = ref._run_insts_lds(ref.build_gemm_lds2(M, N, K, 2, 2, 4, 4, 32, 16, 0, 1), a, bt, c, M, N, K, "ours", 32768, 128, 128, 128)
  ours = metrics(cap(lambda: run_linear(ours_lin)))

  # tensile .co
  cap_k = json.load(open("/tmp/kernarg.json")); raw = bytearray(cap_k["kernarg_bytes"])
  sym = json.load(open(ROOT / "bench/qk-tensile-extraction/selection.json"))["selected"]["rocblas"]["kernel_symbol"]
  A_t = Tensor.randn(K, M, dtype=dtypes.half).contiguous().realize(); B_t = Tensor.randn(N, K, dtype=dtypes.half).contiguous().realize(); C_t = Tensor.zeros(N, M, dtype=dtypes.half).contiguous().realize()
  dev.synchronize(); va = lambda t: t.uop.buffer._buf.va_addr
  struct.pack_into("<Q", raw, 16, va(C_t)); struct.pack_into("<Q", raw, 24, va(C_t)); struct.pack_into("<Q", raw, 32, va(A_t)); struct.pack_into("<Q", raw, 40, va(B_t))
  elf = hcq.unbundle(); kd = hcq.kd_offset(elf, sym); tprg = hcq.NamedAMDProgram(dev, "tensile", elf, kd, bytes(raw))
  tensile = metrics(cap(lambda: tprg(global_size=(4, 96, 1), local_size=(128, 1, 1), wait=False)))

  cyc_ratio = ours["cycles"] / (tensile["cycles"] or 1)   # >1 => ours slower (more GPU cycles)
  decomp = {
    "cycles_ours_over_tensile": round(cyc_ratio, 3),
    "gap_pct": round((cyc_ratio - 1) * 100, 1),
    "L2_hit_delta_ours_minus_tensile": round(ours["L2_hit%"] - tensile["L2_hit%"], 1),
    "valu_total_ours_over_tensile": round(ours["valu_total"] / (tensile["valu_total"] or 1), 3),
    "busy_per_active_ours_over_tensile": round(ours["busy_per_active"] / (tensile["busy_per_active"] or 1e-9), 3),
    "lds_active_ours_over_tensile": round(ours["lds_active_per_active"] / (tensile["lds_active_per_active"] or 1e-9), 3),
    "bankconf_ours_over_tensile": round(ours["bankconf_per_active"] / (tensile["bankconf_per_active"] or 1e-9), 3),
    "waves_ours_over_tensile": round((ours["waves"] or 0) / (tensile["waves"] or 1), 3) if (ours["waves"] or tensile["waves"]) else None,
  }
  # name the dominant cause
  causes = []
  if decomp["L2_hit_delta_ours_minus_tensile"] < -3: causes.append(f"L2 locality: ours L2 hit {ours['L2_hit%']}% < Tensile {tensile['L2_hit%']}% (Tensile WGM8 workgroup remap)")
  if decomp["valu_total_ours_over_tensile"] > 1.05: causes.append(f"more VALU instructions: ours {decomp['valu_total_ours_over_tensile']}x Tensile (address/overhead per FLOP)")
  if decomp["lds_active_ours_over_tensile"] > 1.05 or decomp["bankconf_ours_over_tensile"] > 1.2: causes.append("LDS pressure: ours more LDS-active/bank-conflicts per cycle")
  if decomp.get("waves_ours_over_tensile") and decomp["waves_ours_over_tensile"] < 0.9: causes.append(f"occupancy: ours fewer waves ({decomp['waves_ours_over_tensile']}x Tensile)")
  if decomp["busy_per_active_ours_over_tensile"] < 0.95: causes.append("lower SIMD utilization (more stall) per cycle")

  result = {"date": "2026-06-20", "phase": "AMD_GEMM_TENSILE_PMC_AUDIT", "schema": "amd_gemm_tensile_pmc_audit_v1",
            "role": "ffn_gate/up", "default_behavior_changed": False, "performance_claim": False, "is_audit": True,
            "counters_env": os.environ.get("PMC_COUNTERS", "default"),
            "ours": ours, "tensile": tensile, "decomposition": decomp,
            "residual_causes": causes or ["no single counter separates cleanly at this counter set -- add SQ_WAVES/occupancy or instruction-mix counters"],
            "note": "GRBM cycles are launch-independent (removes wall-clock launch-context artifact); cycle ratio is the real gap."}
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "amd_gemm_tensile_pmc_audit_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({"cycles_ours": ours["cycles"], "cycles_tensile": tensile["cycles"], "decomposition": decomp,
                    "ours_L2%": ours["L2_hit%"], "tensile_L2%": tensile["L2_hit%"],
                    "ours_valu": ours["valu_total"], "tensile_valu": tensile["valu_total"],
                    "residual_causes": result["residual_causes"]}, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
