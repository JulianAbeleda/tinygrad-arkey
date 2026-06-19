from __future__ import annotations

from functools import cache

from tinygrad import Tensor, UOp, dtypes
from tinygrad.device import Device
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.engine.realize as R
import tinygrad.runtime.ops_amd as ops_amd
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, hip_norm_source

DIM, HIDDEN, Q8_BYTES = 4096, 12288, (4096 // 32) * 36
PROD_THREADS = 1024
Q4_WORDS = HIDDEN * (DIM // 256) * 144 // 4

class Q8ArtifactRunner:
  def __init__(self, prg, global_size:tuple[int, int, int], local_size:tuple[int, int, int]):
    self.prg, self.q8_global, self.q8_local = prg, global_size, local_size
  def __getattr__(self, name): return getattr(self.prg, name)
  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=None):
    return self.prg(*bufs, global_size=self.q8_global, local_size=self.q8_local, vals=vals, wait=wait, timeout=timeout)

_orig_exec = ops_amd.AMDComputeQueue.exec
def _patched_exec(self, prg, args_state, global_size, local_size):
  if isinstance(prg, Q8ArtifactRunner): global_size, local_size = prg.q8_global, prg.q8_local
  return _orig_exec(self, prg, args_state, global_size, local_size)
ops_amd.AMDComputeQueue.exec = _patched_exec

def producer_stub(norm_out:UOp, q8:UOp, x:UOp, w:UOp) -> UOp:
  l0 = UOp.range(PROD_THREADS, 0, AxisType.LOCAL)
  norm, q8f, xf, wf = norm_out.flatten(), q8.flatten(), x.flatten(), w.flatten()
  st0 = norm[l0].store(xf[l0] + wf[l0])
  st1 = q8f[l0].store((xf[l0] + wf[l0]).cast(q8f.dtype))
  return UOp.group(st0, st1).end(l0).sink(arg=KernelInfo(name="q8_rmsnorm_side_inject", opts_to_apply=()))

def gateup_stub(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  g0 = UOp.range(HIDDEN, 0, AxisType.GLOBAL)
  g1 = UOp.range(2, 1, AxisType.GLOBAL)
  l0 = UOp.range(32, 2, AxisType.LOCAL)
  l1 = UOp.range(4, 3, AxisType.LOCAL)
  gf, uf = gate.flatten(), up.flatten()
  v = gate_words.flatten()[0].cast(dtypes.float32) + up_words.flatten()[0].cast(dtypes.float32) + q8.flatten()[0].cast(dtypes.float32) + \
      l0.cast(dtypes.float32) + l1.cast(dtypes.float32)
  st0 = gf[g0].store(v)
  st1 = uf[g0].store(v + g1.cast(dtypes.float32) * 0.0)
  return UOp.group(st0, st1).end(l1, l0, g1, g0).sink(arg=KernelInfo(name="q8_mmvq_gateup_inject", opts_to_apply=()))

@cache
def install_q8_ffn_artifacts() -> None:
  dev = Device["AMD"]
  prod_prg = Q8ArtifactRunner(dev.runtime("q8_rmsnorm_side_model_artifact", compile_hipcc_linked(hip_norm_source(PROD_THREADS), "gfx1100")),
                              (1,1,1), (PROD_THREADS,1,1))
  gateup_prg = Q8ArtifactRunner(dev.runtime("q8_mmvq_gateup_model_artifact", compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, "gfx1100")),
                                (HIDDEN,2,1), (32,4,1))
  keys: dict[str, tuple[bytes, str]] = {}
  orig = R.get_runtime
  def hook(device, ast, cache=True):
    if ast.arg.name in {"q8_rmsnorm_side_inject", "q8_mmvq_gateup_inject"}: keys[ast.arg.name] = (ast.key, device)
    return orig(device, ast, cache)
  R.get_runtime = hook
  try:
    x = Tensor.ones(DIM, dtype=dtypes.float32, device="AMD").contiguous().realize()
    w = Tensor.ones(DIM, dtype=dtypes.float32, device="AMD").contiguous().realize()
    norm = Tensor.empty(DIM, dtype=dtypes.float32, device="AMD").contiguous()
    q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
    norm.custom_kernel(q8, x, w, fxn=producer_stub)[:2][0].realize()
    gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
    up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
    gw = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    uw = Tensor.empty(Q4_WORDS, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    gate.custom_kernel(up, gw, uw, q8.realize(), fxn=gateup_stub)[:2][0].realize()
  finally:
    R.get_runtime = orig
  if set(keys) != {"q8_rmsnorm_side_inject", "q8_mmvq_gateup_inject"}:
    raise RuntimeError(f"q8 FFN install failed to capture expected PROGRAM keys: {sorted(keys)}")
  R.runtime_cache[keys["q8_rmsnorm_side_inject"]] = prod_prg
  R.runtime_cache[keys["q8_mmvq_gateup_inject"]] = gateup_prg

def _q4_words(linear, device:str) -> Tensor:
  words = linear.q4k_storage.words.to(device)
  if linear.q4k_storage.mode == "q4_ondemand": words = words.contiguous()
  return words.realize()

def route_q8_ffn(block, x:Tensor) -> Tensor|None:
  if x.shape != (1, 1, DIM) or getattr(block, "config", None) is None or block.config.hidden_dim != HIDDEN: return None
  if not all(hasattr(getattr(block, n, None), "q4k_storage") for n in ("ffn_gate", "ffn_up")): return None
  install_q8_ffn_artifacts()
  device = x.device
  norm_w = block.ffn_norm.weight.cast(dtypes.float32).to(device).contiguous()
  x_vec = x.reshape(DIM).cast(dtypes.float32).contiguous()
  norm = Tensor.empty(DIM, dtype=dtypes.float32, device=device).contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device=device).contiguous()
  norm, q8, *_ = norm.custom_kernel(q8, x_vec, norm_w, fxn=producer_stub)
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device=device).contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device=device).contiguous()
  gate, up, *_ = gate.custom_kernel(up, _q4_words(block.ffn_gate, device), _q4_words(block.ffn_up, device), q8, fxn=gateup_stub)
  gate, up = gate.reshape(1, 1, HIDDEN), up.reshape(1, 1, HIDDEN)
  return block.ffn_down(gate.silu().contiguous() * up)
