#!/usr/bin/env python3
"""Minimal cache-identity indexing/coalescing gate for generated decode tiles.

This isolates the current fused-xlane route blocker below the full model:

  raw cache_kv shape [2,1,Hkv,MAXC,Hd] + lane-sharded contiguous d/e axis

Verdicts distinguish:
- plain 5D cache indexing works,
- dynamic t indexing composes with a per-lane d-shard reduction,
- explicit UPCAST into LDS triggers a verifier/compiler wall.

Run:
  DEV=AMD JIT=1 PYTHONPATH=. python3 extra/qk_decode_cache_identity_index_gate.py
"""
from __future__ import annotations

import json, os, pathlib, time, traceback
from typing import Any

import numpy as np
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import AddrSpace, AxisType, KernelInfo, Ops, UOp

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-cache-identity-index"
LANES, Hkv, MAXC, Hd, R = 32, 8, 16, 128, 4
TOL = 1e-6


def _fki(name: str) -> KernelInfo: return KernelInfo(name=name, opts_to_apply=())


def _static_v_kernel(axis_type: AxisType | None):
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    dd = UOp.range(R, 0, axis_type=axis_type) if axis_type is not None else UOp.range(R, 0)
    d = lane * R + dd
    st = out[d].store(cache[1, 0, 0, 3, d].cast(dtypes.float32)).end(dd)
    return st.sink(arg=_fki(f"cache5_static_v_{'upcast' if axis_type is AxisType.UPCAST else 'scalar'}"))
  return kernel


def _dynamic_v_sum_kernel(axis_type: AxisType | None):
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    acc = UOp.placeholder((R,), dtypes.float32, 20, addrspace=AddrSpace.REG)
    dd0 = UOp.range(R, 0, axis_type=axis_type) if axis_type is not None else UOp.range(R, 0)
    init = acc[dd0].store(0.0).end(dd0)
    acc = acc.after(init)
    j = UOp.range(5, 1, axis_type=AxisType.REDUCE)
    dd = UOp.range(R, 2, axis_type=axis_type) if axis_type is not None else UOp.range(R, 2)
    d = lane * R + dd
    upd = acc[dd].store(acc.after(j)[dd] + cache[1, 0, 0, j, d].cast(dtypes.float32)).end(dd).end(j)
    dd2 = UOp.range(R, 3, axis_type=axis_type) if axis_type is not None else UOp.range(R, 3)
    d2 = lane * R + dd2
    return out[d2].store(acc.after(upd)[dd2]).end(dd2).sink(arg=_fki(f"cache5_dynamic_v_sum_{'upcast' if axis_type is AxisType.UPCAST else 'scalar'}"))
  return kernel


def _ptr_vec_v_kernel():
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    base = (((1 * 1 + 0) * Hkv + 0) * MAXC + 3) * Hd + lane * R
    ptrs = UOp(Ops.PTRCAT, cache.dtype.base.ptr(size=cache.max_numel(), addrspace=cache.addrspace).vec(R),
               tuple(cache.flatten().index(base + i, ptr=True) for i in range(R)))
    vals = ptrs.load(dtype=dtypes.float32.vec(R))
    outp = out.flatten().index(lane * R, ptr=True).cast(dtypes.float32.vec(R).ptr(size=Hd))
    return outp.store(vals).sink(arg=_fki("cache5_ptr_vec_v_load"))
  return kernel


def _k_upcast_lds_kernel():
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    klds = UOp.placeholder((Hd,), dtypes.half, 30, addrspace=AddrSpace.LOCAL)
    rk = UOp.range(R, 0, axis_type=AxisType.UPCAST)
    e = lane * R + rk
    stage = klds[e].store(cache[0, 0, 0, 3, e].cast(dtypes.half)).end(rk)
    bar = UOp.barrier(UOp.group(stage))
    dd = UOp.range(R, 1)
    d = lane * R + dd
    return out[d].store(klds.after(bar)[d].cast(dtypes.float32)).end(dd).sink(arg=_fki("cache5_k_upcast_lds"))
  return kernel


def _k_upcast_lds_dynamic_masked_kernel():
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    klds = UOp.placeholder((Hd,), dtypes.half, 31, addrspace=AddrSpace.LOCAL)
    j = UOp.range(5, 0, axis_type=AxisType.REDUCE)
    t = j + 3
    in_r = j < 4
    t_safe = in_r.where(t, t.const_like(0))
    rk = UOp.range(R, 1, axis_type=AxisType.UPCAST)
    e = lane * R + rk
    stage = klds[e].store(cache[0, 0, 0, t_safe, e].cast(dtypes.half), in_r).end(rk)
    bar = UOp.barrier(UOp.group(stage))
    dd = UOp.range(R, 2)
    d = lane * R + dd
    return out[d].store(klds.after(bar)[d].cast(dtypes.float32)).end(dd).end(j).sink(arg=_fki("cache5_k_upcast_lds_dynamic_masked"))
  return kernel


def _k_upcast_lds_dynamic_unguarded_kernel():
  def kernel(out: UOp, cache: UOp) -> UOp:
    lane = UOp.special(LANES, "lidx0")
    klds = UOp.placeholder((Hd,), dtypes.half, 32, addrspace=AddrSpace.LOCAL)
    j = UOp.range(5, 0, axis_type=AxisType.REDUCE)
    t = j + 3
    in_r = j < 4
    t_safe = in_r.where(t, t.const_like(0))
    rk = UOp.range(R, 1, axis_type=AxisType.UPCAST)
    e = lane * R + rk
    stage = klds[e].store(cache[0, 0, 0, t_safe, e].cast(dtypes.half)).end(rk)
    bar = UOp.barrier(UOp.group(stage))
    dd = UOp.range(R, 2)
    d = lane * R + dd
    return out[d].store(klds.after(bar)[d].cast(dtypes.float32)).end(dd).end(j).sink(arg=_fki("cache5_k_upcast_lds_dynamic_unguarded"))
  return kernel


def _inputs() -> np.ndarray:
  rng = np.random.default_rng(20260626)
  return rng.normal(0.0, 0.25, size=(2, 1, Hkv, MAXC, Hd)).astype(np.float32)


def _run(name: str, fxn, ref: np.ndarray, cache: np.ndarray) -> dict[str, Any]:
  try:
    got = Tensor.empty(Hd, dtype=dtypes.float32).custom_kernel(Tensor(cache), fxn=fxn)[0].realize().numpy()
  except Exception as e:  # noqa: BLE001
    return {"name": name, "pass": False, "failure_class": "compile_or_verify", "exception_type": type(e).__name__,
            "exception": str(e)[:1200], "traceback_tail": traceback.format_exc()[-4000:]}
  err = float(np.max(np.abs(got - ref)))
  return {"name": name, "pass": bool(np.isfinite(err) and err <= TOL), "max_abs_error": err,
          "got_sample": got[:8].tolist(), "ref_sample": ref[:8].tolist()}


def build() -> dict[str, Any]:
  cache = _inputs()
  rows = [
    _run("static_v_scalar_5d", _static_v_kernel(None), cache[1, 0, 0, 3, :], cache),
    _run("static_v_upcast_5d", _static_v_kernel(AxisType.UPCAST), cache[1, 0, 0, 3, :], cache),
    _run("ptr_vec_v_5d", _ptr_vec_v_kernel(), cache[1, 0, 0, 3, :], cache),
    _run("dynamic_v_sum_scalar_5d", _dynamic_v_sum_kernel(None), cache[1, 0, 0, 0:5, :].sum(axis=0), cache),
    _run("dynamic_v_sum_upcast_5d", _dynamic_v_sum_kernel(AxisType.UPCAST), cache[1, 0, 0, 0:5, :].sum(axis=0), cache),
    _run("k_upcast_lds_5d", _k_upcast_lds_kernel(), cache[0, 0, 0, 3, :].astype(np.float16).astype(np.float32), cache),
    _run("k_upcast_lds_dynamic_unguarded_5d", _k_upcast_lds_dynamic_unguarded_kernel(), cache[0, 0, 0, 0, :].astype(np.float16).astype(np.float32), cache),
    _run("k_upcast_lds_dynamic_masked_5d", _k_upcast_lds_dynamic_masked_kernel(), cache[0, 0, 0, 6, :].astype(np.float16).astype(np.float32), cache),
  ]
  by_name = {r["name"]: r for r in rows}
  if getenv_on := bool(os.environ.get("REG_STORE_DEVEC")):
    if not by_name["static_v_scalar_5d"]["pass"]: verdict = "CACHE_5D_STATIC_INDEX_FAIL"
    elif not by_name["static_v_upcast_5d"]["pass"]: verdict = "CACHE_5D_STATIC_UPCAST_FAIL"
    elif not by_name["dynamic_v_sum_scalar_5d"]["pass"]: verdict = "CACHE_5D_DYNAMIC_INDEX_FAIL"
    elif not by_name["dynamic_v_sum_upcast_5d"]["pass"]: verdict = "CACHE_5D_REG_STORE_DEVEC_FAIL"
    elif not by_name["k_upcast_lds_5d"]["pass"]: verdict = "CACHE_5D_K_UPCAST_LDS_FAIL"
    elif not by_name["k_upcast_lds_dynamic_unguarded_5d"]["pass"]: verdict = "CACHE_5D_K_DYNAMIC_UNGUARDED_UPCAST_LDS_FAIL"
    else: verdict = "CACHE_5D_REG_STORE_DEVEC_PASS"
  elif not by_name["static_v_scalar_5d"]["pass"]: verdict = "CACHE_5D_STATIC_INDEX_FAIL"
  elif not by_name["static_v_upcast_5d"]["pass"]: verdict = "CACHE_5D_STATIC_UPCAST_FAIL"
  elif not by_name["ptr_vec_v_5d"]["pass"]: verdict = "CACHE_5D_PTR_VEC_LOAD_FAIL"
  elif not by_name["dynamic_v_sum_scalar_5d"]["pass"]: verdict = "CACHE_5D_DYNAMIC_INDEX_FAIL"
  elif not by_name["dynamic_v_sum_upcast_5d"]["pass"]: verdict = "CACHE_5D_DYNAMIC_UPCAST_FAIL"
  elif not by_name["k_upcast_lds_5d"]["pass"]: verdict = "CACHE_5D_K_UPCAST_LDS_FAIL"
  elif not by_name["k_upcast_lds_dynamic_masked_5d"]["pass"]: verdict = "CACHE_5D_K_DYNAMIC_MASKED_UPCAST_LDS_FAIL"
  else: verdict = "CACHE_5D_INDEX_AND_UPCAST_PASS"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "shape": {"cache": [2, 1, Hkv, MAXC, Hd], "lanes": LANES, "R": R, "tolerance": TOL}, "reg_store_devec": getenv_on,
          "rows": rows,
          "decision": "If any UPCAST row fails, keep attention layout fixed and repair the coalescing/lowering path before Phase 2 block tiling."}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"cache-identity-index-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] in {"CACHE_5D_INDEX_AND_UPCAST_PASS", "CACHE_5D_REG_STORE_DEVEC_PASS"} else 1


if __name__ == "__main__":
  raise SystemExit(main())
