#!/usr/bin/env python3
"""A3 in-model Tensile prefill route (research-only, PREFILL_TENSILE_GEMM=1). Install-once, robust routing.

Mechanism (avoids A2's get_runtime-hook quirk):
- patch `dev.runtime` (the program factory get_runtime calls, device.py:292): route function_names starting
  "tensile_" to the role's TensileRunner; else the real AMDProgram.
- patch `AMDComputeQueue.exec`: force TensileRunner's own Tensile launch dims.
- TensileRunner.fill_kernargs binds the 4 pointer VAs rebindably (A1) so JIT-rebound activation VAs work.

route_pf16(lin, x, role): the [feature,T] Tensile matmul for one eligible linear. x[T,in] -> x^T[in,T] (A),
w=lin._pf16_w natural [out,in] (B, no transpose), C[out,T] (=out^T), returned transposed to [T,out].

ELIGIBLE (TPE-5, T=512): "gateup" (in=4096,out=12288), "down" (in=12288,out=4096).
"""
from __future__ import annotations
import json, functools
from tinygrad import Tensor, Device, dtypes, UOp
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.runtime.ops_amd as ops_amd
from extra.qk_tensile_runtime import TensileRunner
from extra.qk_tensile_hcq_launch import unbundle

_CAPS = "bench/qk-tensile-extraction/kernarg_all.jsonl"
_ROLE_CAP = {"gateup": "ffn_gate_up", "down": "ffn_down"}
_installed = {"done": False, "runners": {}}

def trivial_fxn(role:str):
  def fxn(c:UOp, a:UOp, b:UOp) -> UOp:
    nb = (c.shape[0]*c.shape[1])//128
    r = UOp.range(nb, 0, AxisType.GLOBAL); l = UOp.range(128, 1, AxisType.LOCAL)
    val = (a.reshape(-1)[0] + b.reshape(-1)[0]).cast(c.dtype)
    return c.reshape(nb, 128)[r, l].store(val).end(l, r).sink(arg=KernelInfo(name=f"tensile_{role}"))
  return fxn

def install(dev=None):
  if _installed["done"]: return _installed["runners"]
  dev = dev or Device[Device.DEFAULT]
  caps = {json.loads(l)["role"]: json.loads(l) for l in open(_CAPS)}
  elf = unbundle()
  runners = {role: TensileRunner(dev, _ROLE_CAP[role], caps[_ROLE_CAP[role]], elf) for role in _ROLE_CAP}
  # 1) route the program factory: tensile_<role> function_name -> that role's TensileRunner
  orig_runtime = dev.runtime
  def routed_runtime(function_name, lib, *aux, **kw):
    for role, runner in runners.items():
      if f"tensile_{role}" in function_name: return runner
    return orig_runtime(function_name, lib, *aux, **kw)
  dev.runtime = routed_runtime
  # 2) force Tensile dims at queue exec
  if not getattr(ops_amd.AMDComputeQueue, "_tensile_patched", False):
    _orig_exec = ops_amd.AMDComputeQueue.exec
    def _patched(self, prg, args_state, gs, ls):
      if isinstance(prg, TensileRunner): gs, ls = prg.tensile_global, prg.tensile_local
      return _orig_exec(self, prg, args_state, gs, ls)
    ops_amd.AMDComputeQueue.exec = _patched; ops_amd.AMDComputeQueue._tensile_patched = True
  _installed["done"] = True; _installed["runners"] = runners
  return runners

# eligible (in, out) at T=512 -> role
ELIGIBLE = {(4096, 12288): "gateup", (12288, 4096): "down"}

def route_pf16(lin, x:Tensor):
  """Tensile route for a PREFILL_V2 _pf16 call. Returns out[T,out] or None if not eligible."""
  w = getattr(lin, "_pf16_w", None)
  if w is None or getattr(lin, "bias", None) is not None or len(w.shape) != 2: return None
  out_f, in_f = w.shape                                # _pf16_w is [out, in]
  T = x.shape[0] if x.ndim == 2 else (x.shape[0]*x.shape[1])
  if not isinstance(T, int) or T != 512 or (in_f, out_f) not in ELIGIBLE: return None
  if not _installed["done"]: return None   # must be install()ed eagerly at model setup (outside the prefill trace)
  role = ELIGIBLE[(in_f, out_f)]
  x2 = x.reshape(T, in_f).cast(dtypes.float16)
  x_t = x2.transpose().contiguous()                                   # [in, T] (A)
  C = Tensor.zeros(out_f, T, dtype=dtypes.float16).contiguous()       # [out, T] (=out^T)
  out_t = C.custom_kernel(x_t, w, fxn=trivial_fxn(role))[0]           # routed -> TensileRunner
  return out_t.transpose().reshape(*x.shape[:-1], out_f)              # [T, out]
