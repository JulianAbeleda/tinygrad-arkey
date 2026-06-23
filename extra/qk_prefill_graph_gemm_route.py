from __future__ import annotations

from functools import lru_cache

from tinygrad import Tensor, dtypes, getenv
from tinygrad.device import Device
from tinygrad.engine.realize import Estimates
from tinygrad.helpers import colored
from tinygrad.dtype import AddrSpace
from tinygrad.uop.ops import KernelInfo, Ops, UOp
from extra.gemm import rdna3_wmma_matmul as ref


@lru_cache(maxsize=None)
def _kernel(out_f: int, in_f: int):
  m, n, k = 512, out_f, in_f
  waves_m, waves_n, wm, wn, bk, pad, dbuf, plra = 2, 2, 4, 4, 32, 16, 0, 1
  if out_f <= 1024:  # small-N roles (kv_proj) are WG-starved at BN=128 -> halve BN to 2x the workgroups
    waves_n, wn = 1, 4
  bm, bn, threads = waves_m * wm * 16, waves_n * wn * 16, waves_m * waves_n * 32
  if m % bm or n % bn or k % bk: return None
  insts = ref.build_gemm_lds2(m, n, k, waves_m, waves_n, wm, wn, bk, pad, dbuf, PLRA=plra)
  lds_bytes = max((bk * 2 + pad) * (bm + bn) * (2 if dbuf else 1), 65536 // 8)
  name = f"prefill_graph_gemm_{m}_{n}_{k}"
  return insts, lds_bytes, bm, bn, threads, name


def route_pf16_graph_gemm(lin, x: Tensor) -> Tensor | None:
  # NOTE: the gfx1100 arch restriction for default-on lives in model.PREFILL_GRAPH_GEMM (computed once at import);
  # it is NOT checked here because Device[...] access is disallowed during JIT capture (ALLOW_DEVICE_USAGE). The
  # T==512 / tile-divisible / bias / role guards below restrict to the validated dense prefill shapes; everything
  # else silently falls back to the normal PREFILL_V2 matmul.
  roles = str(getenv("PREFILL_GRAPH_GEMM_ROLES", ""))
  if roles:
    role = getattr(lin, "_prefill_graph_role", None)
    if role is None or role not in {r.strip() for r in roles.split(",") if r.strip()}: return None
  w = getattr(lin, "_pf16_w", None)
  b = getattr(lin, "bias", None)
  if w is None or b is not None or x.ndim < 2: return None
  if not isinstance(x.shape[-2], int) or not isinstance(x.shape[-1], int): return None
  if x.shape[-2] != 512: return None
  out_f, in_f = w.shape
  if in_f != x.shape[-1]: return None
  built = _kernel(out_f, in_f)
  if built is None: return None
  insts, lds_bytes, bm, bn, threads, name = built
  a = x.reshape(512, in_f).cast(dtypes.float16).contiguous()
  bt = w.cast(dtypes.float16).contiguous()
  c = Tensor.empty(512, out_f, dtype=dtypes.half, device=x.device).contiguous()
  grid = (out_f // bn, 512 // bm, 1)
  def asm_kernel(A, Bt, C):
    lds = UOp(Ops.DEFINE_LOCAL, dtypes.uint8.ptr(size=lds_bytes, addrspace=AddrSpace.LOCAL), (), "lds")
    g = [UOp.special(grid[0], "gidx0"), UOp.special(grid[1], "gidx1")]
    sink = UOp.sink(A.base, Bt.base, C.base, lds, *g, UOp.special(threads, "lidx0"),
                    arg=KernelInfo(name=colored(name, "cyan"),
                                   estimates=Estimates(ops=512*out_f*in_f*2, mem=(512*in_f+out_f*in_f+512*out_f)*2)))
    return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg=Device.DEFAULT),
                                 UOp(Ops.LINEAR, src=tuple([UOp(Ops.INS, arg=i) for i in insts]))))
  out = Tensor.custom_kernel(a, bt, c, fxn=asm_kernel)[2]
  return out.reshape(*x.shape[:-1], out_f)
