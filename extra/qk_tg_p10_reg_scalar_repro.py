#!/usr/bin/env python3
"""TG-P10.1: minimal generated-UOp repro for the REG scalar / reduction-accumulator lowering blocker.

Emits a deterministic tinygrad.reg_scalar_lowering.v1 artifact with four cases (one passing control + three failure
modes) so BoltBeam can mechanically classify the blocker as EMITTER_BLOCKED. All cases are GENERATED UOp only (no
HIP/ASM), on the same synthetic per-split partials the attention combine consumes.

  shipped_per_d_combine_compiles     control: the shipped single-reduce per-d combine compiles + is numerically ok
  shared_weight_combine_compile_fails: the LDS/warp weight-sharing combine -> invalid_reg_vector_store at compile
  fused_gmax_combine_compile_fails   : the inline-gmax (two-reduce) combine -> invalid_reg_vector_store at compile
  reg_store_devec_compiles_nan       : same shared-weight combine under REG_STORE_DEVEC=1 -> compiles but NaN output

REG_STORE_DEVEC is a compile-time getenv (memoized), so the DEVEC case runs in a fresh subprocess.

Run: DEV=AMD PYTHONPATH=. python3 extra/qk_tg_p10_reg_scalar_repro.py
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys

os.environ.setdefault("DEV", "AMD")

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/tg-p10-reg-scalar-combine-lowering"
Hq, Hd, S = 32, 128, 36


def _synth():
  import numpy as np
  from tinygrad import Tensor
  W = Hd + 2
  rng = np.random.RandomState(1)
  pn = rng.normal(0, 1, (Hq * S * W)).astype(np.float32)
  p3 = pn.reshape(Hq, S, W)
  p3[:, :, Hd + 1] = rng.normal(-2, 1, (Hq, S)); p3[:, :, Hd] = np.abs(p3[:, :, Hd]) + 0.1
  pn = p3.reshape(-1)
  ref_m = p3[:, :, Hd + 1]; ref_l = p3[:, :, Hd]; ref_pv = p3[:, :, :Hd]
  gm = ref_m.max(1, keepdims=True); w = np.exp(ref_m - gm)
  ref = ((w[:, :, None] * ref_pv).sum(1) / (w * ref_l).sum(1)[:, None]).astype(np.float32)
  return Tensor(pn, device="AMD").realize(), ref


def _run_case(build):
  """Returns (compile_ok, runtime_ok, numeric_ok, observed_accum, err). build() -> np result array."""
  import numpy as np
  try:
    out = build()
  except Exception as e:
    msg = str(e)[:300]
    obs = "vectorized_make_float4" if "not assignable" in msg or "make_float4" in msg else "unknown"
    return False, False, False, obs, f"{type(e).__name__}: {msg}"
  _, ref = _synth()
  nan = bool(np.isnan(out).any())
  numeric = (not nan) and float(np.abs(out - ref).max() / (np.abs(ref).max() + 1e-6)) < 1e-3
  return True, (not nan), numeric, "scalar" if numeric else ("vectorized_make_float4" if nan else "unknown"), (None if numeric else ("nan_output" if nan else "numeric_mismatch"))


def _build_shipped():
  from tinygrad import Tensor, dtypes
  from extra.qk_flash_decode import flash_state_gmax_kernel, flash_state_combine_kernel
  pout, _ = _synth()
  gm = Tensor.empty(Hq, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_state_gmax_kernel(Hd, Hq, S, stride=S))[0]
  return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, gm, fxn=flash_state_combine_kernel(Hd, Hq, S, stride=S))[0].realize().numpy().reshape(Hq, Hd)


def _build_shared_weight():
  from tinygrad import Tensor, dtypes
  from extra.qk_live_split_geometry import flash_fused_gmax_combine_kernel
  pout, _ = _synth()
  return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_fused_gmax_combine_kernel(Hd, Hq, S, stride=S))[0].realize().numpy().reshape(Hq, Hd)


def _build_inline_gmax():
  from tinygrad import Tensor, dtypes
  from extra.qk_live_split_geometry import flash_inline_gm_combine_kernel
  pout, _ = _synth()
  return Tensor.empty(Hq * Hd, dtype=dtypes.float32, device="AMD").custom_kernel(pout, fxn=flash_inline_gm_combine_kernel(Hd, Hq, S, stride=S))[0].realize().numpy().reshape(Hq, Hd)


def _case(case_id, compile_ok, runtime_ok, numeric_ok, error_class, error_excerpt, observed, uses_devec):
  return {"case_id": case_id, "generated_uop_only": True, "uses_external_kernel": False,
          "compile_ok": compile_ok, "runtime_ok": runtime_ok, "numeric_ok": numeric_ok,
          "error_class": error_class, "error_excerpt": (error_excerpt or "")[:200],
          "reg_accumulator_expected": "scalar", "reg_accumulator_observed": observed, "uses_reg_store_devec": uses_devec}


def measure_one():
  # in-process cases (no REG_STORE_DEVEC)
  co, ro, no, obs, err = _run_case(_build_shipped)
  shipped = _case("shipped_per_d_combine_compiles", co, ro, no, "ok" if no else "numeric_mismatch", err, obs, False)
  co, ro, no, obs, err = _run_case(_build_shared_weight)
  shared = _case("shared_weight_combine_compile_fails", co, ro, no, "invalid_reg_vector_store" if not co else "ok", err, obs, False)
  co, ro, no, obs, err = _run_case(_build_inline_gmax)
  fused = _case("fused_gmax_combine_compile_fails", co, ro, no, "invalid_reg_vector_store" if not co else "ok", err, obs, False)
  print("@@RESULT@@" + json.dumps([shipped, shared, fused]))


def measure_devec():
  # REG_STORE_DEVEC=1 case (fresh process)
  co, ro, no, obs, err = _run_case(_build_shared_weight)
  ec = "ok" if no else ("nan_output" if (co and not ro) else ("invalid_reg_vector_store" if not co else "numeric_mismatch"))
  print("@@RESULT@@" + json.dumps(_case("reg_store_devec_compiles_nan", co, ro, no, ec, err, obs, True)))


def _spawn(fn_name, extra_env):
  env = dict(os.environ); env.update(extra_env); env["QK_P10_STAGE"] = fn_name
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__))], env=env, capture_output=True, text=True, cwd=str(ROOT))
  import re
  m = re.search(r"@@RESULT@@(.*)", p.stdout)
  if not m:
    sys.stderr.write(p.stdout[-1500:] + "\n" + p.stderr[-1500:]); raise SystemExit(2)
  return json.loads(m.group(1))


def main():
  stage = os.environ.get("QK_P10_STAGE")
  if stage == "measure_one": measure_one(); return 0
  if stage == "measure_devec": measure_devec(); return 0
  OUT.mkdir(parents=True, exist_ok=True)
  cases = _spawn("measure_one", {}) + [_spawn("measure_devec", {"REG_STORE_DEVEC": "1"})]
  # a valid repro: exactly one passing control + at least one compile-fail + the DEVEC-NaN case
  control_ok = any(c["case_id"].startswith("shipped") and c["numeric_ok"] for c in cases)
  fails = [c for c in cases if not c["compile_ok"]]
  devec_nan = any(c["uses_reg_store_devec"] and c["compile_ok"] and not c["numeric_ok"] for c in cases)
  verdict = ("TG_P10_1_PASS_REG_REPRO_PINNED" if control_ok and fails and devec_nan
             else "TG_P10_1_BLOCKED_REPRO_NOT_MINIMAL")
  art = {"schema": "tinygrad.reg_scalar_lowering.v1", "candidate_id": "decode_attention_split_preserving_lse_combine",
         "model_id": "qwen3-8b-q4_k_m", "target_id": "amd_gfx1100", "verdict": verdict,
         "geometry": {"Hq": Hq, "Hkv": 8, "Hd": Hd, "S": S}, "cases": cases,
         "control_passes": control_ok, "n_compile_fail": len(fails), "reg_store_devec_nan": devec_nan}
  json.dump(art, open(OUT / "reg_scalar_lowering.json", "w"), indent=2)
  print(verdict, "control_ok=", control_ok, "fails=", [c["case_id"] for c in fails], "devec_nan=", devec_nan)
  return 0 if verdict == "TG_P10_1_PASS_REG_REPRO_PINNED" else 1


if __name__ == "__main__":
  raise SystemExit(main())
