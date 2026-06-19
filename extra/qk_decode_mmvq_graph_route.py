from __future__ import annotations

import json, pathlib, struct
from functools import cache

from tinygrad import Tensor, UOp, dtypes
from tinygrad.device import Device
from tinygrad.helpers import PROFILE, unwrap
from tinygrad.runtime.support.hcq import HCQArgsState, hcq_profile
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.engine.realize as R
import tinygrad.runtime.ops_amd as ops_amd

from extra.qk_decode_mmvq_p3_q4_correctness import OBJ, OUT, RawKernargAMDProgram, kd_offset
from extra.qk_decode_mmvq_p5_lifecycle_probe import Q8_BYTES, Q8_QUANT_SOURCE

DIM = 4096


class FixedLaunchRunner:
  def __init__(self, prg, global_size: tuple[int, int, int], local_size: tuple[int, int, int]):
    self.prg, self.qk_global, self.qk_local = prg, global_size, local_size

  def __getattr__(self, name):
    return getattr(self.prg, name)

  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=None):
    kernargs = self.fill_kernargs(bufs, vals)
    q = unwrap(self.prg.dev.hw_compute_queue_t)().wait(self.prg.dev.timeline_signal, self.prg.dev.timeline_value - 1).memory_barrier()
    self.prg.dev.prof_exec_counter += 1
    with hcq_profile(self.prg.dev, queue=q, desc=self.prg.name, enabled=wait or PROFILE) as (sig_st, sig_en):
      q.exec(self, kernargs, self.qk_global, self.qk_local)
    q.signal(self.prg.dev.timeline_signal, self.prg.dev.next_timeline()).submit(self.prg.dev)
    if wait:
      self.prg.dev.synchronize(timeout=timeout)
    return (float(sig_en.timestamp - sig_st.timestamp) / 1e6) if wait else None


class ImportedQ4MMVQRunner(FixedLaunchRunner):
  def __init__(self, prg: RawKernargAMDProgram, raw_template: bytes, global_size: tuple[int, int, int],
               local_size: tuple[int, int, int]):
    super().__init__(prg, global_size, local_size)
    self.raw_template = bytes(raw_template)
    rows = global_size[0]
    self.q4_nbytes = rows * (DIM // 256) * 144
    self.q8_nbytes = Q8_BYTES
    self.out_nbytes = rows * 4

  @staticmethod
  def _nbytes(buf) -> int:
    if hasattr(buf, "size"):
      return int(buf.size)
    if hasattr(buf, "nbytes"):
      return int(buf.nbytes)
    raise RuntimeError(f"cannot infer buffer size for {buf!r}")

  def _classify_bufs(self, bufs):
    by_size = {self._nbytes(b): b for b in bufs}
    try:
      return by_size[self.out_nbytes], by_size[self.q4_nbytes], by_size[self.q8_nbytes]
    except KeyError:
      # Fallback for older buffer objects where size is unavailable or not bytes. Keep the original P7a assumption,
      # but surface the sizes in the exception path for diagnosis if it faults again.
      if len(bufs) >= 3:
        return bufs[0], bufs[1], bufs[2]
      raise RuntimeError(f"cannot classify imported MMVQ buffers: sizes={[self._nbytes(b) for b in bufs]}")

  def fill_kernargs(self, bufs, vals=(), kernargs=None):
    # Llama kernarg order: q4, q8, ids/null, ..., dst. ProgramInfo buffer order is not part of the contract, so
    # identify the three buffers by allocation size instead of assuming stub source order.
    raw = bytearray(self.raw_template)
    out, q4, q8 = self._classify_bufs(bufs)
    struct.pack_into("<Q", raw, 16, 0)
    ab = kernargs or self.prg.dev.kernargs_buf.offset(offset=self.prg.dev.kernargs_offset_allocator.alloc(self.prg.kernargs_alloc_size, 8),
                                                      size=self.prg.kernargs_alloc_size)
    ab.cpu_view().view(size=len(raw), fmt="B")[:] = raw
    st = HCQArgsState(ab, self.prg, tuple(bufs), vals=tuple(vals))
    st.bind_sints_to_buf(q4.va_addr, buf=ab, fmt="Q", offset=0)
    st.bind_sints_to_buf(q8.va_addr, buf=ab, fmt="Q", offset=8)
    st.bind_sints_to_buf(out.va_addr, buf=ab, fmt="Q", offset=56)
    return st


_orig_exec = ops_amd.AMDComputeQueue.exec


def _patched_exec(self, prg, args_state, global_size, local_size):
  if isinstance(prg, FixedLaunchRunner):
    global_size, local_size = prg.qk_global, prg.qk_local
  return _orig_exec(self, prg, args_state, global_size, local_size)


ops_amd.AMDComputeQueue.exec = _patched_exec


def q8_quant_stub(q8: UOp, x: UOp) -> UOp:
  l0 = UOp.range(128, 0, AxisType.LOCAL)
  q8f, xf = q8.flatten(), x.flatten()
  # Placeholder only. Runtime is swapped to q8_quantize_4096.
  st0 = q8f[l0].store(xf[l0].cast(q8f.dtype))
  return st0.end(l0).sink(arg=KernelInfo(name="llama_q8_quant_4096_inject", opts_to_apply=()))


def q4_mmvq_stub(out: UOp, q4: UOp, q8: UOp) -> UOp:
  rows = out.shape[0]
  g0 = UOp.range(rows, 0, AxisType.GLOBAL)
  l0 = UOp.range(32, 1, AxisType.LOCAL)
  of, q4f, q8f = out.flatten(), q4.flatten(), q8.flatten()
  v = q4f[0].cast(dtypes.float32) + q8f[0].cast(dtypes.float32) + l0.cast(dtypes.float32) * 0.0
  st0 = of[g0].store(v)
  return st0.end(l0, g0).sink(arg=KernelInfo(name=f"llama_q4_mmvq_{rows}_inject", opts_to_apply=()))


@cache
def install_imported_q4_mmvq(rows: int) -> dict:
  if Device.DEFAULT != "AMD":
    raise RuntimeError(f"imported Q4 MMVQ route requires DEV=AMD, got {Device.DEFAULT!r}")
  dev = Device["AMD"]
  cap = json.loads((OUT / "p2_kernarg_capture.json").read_text())["selected"]["q4_attn_q_or_o"]
  q8_prg = FixedLaunchRunner(dev.runtime("llama_q8_quant_4096_graph", dev.compiler.compile(Q8_QUANT_SOURCE)),
                             (1, 1, 1), (128, 1, 1))
  mmvq_raw = bytearray(cap["kernarg_bytes"])
  # The row count is represented in the launch geometry for this no-fusion Q4 template; the captured scalar fields are
  # K/stride constants for the 4096-wide activation.
  mmvq_prg = RawKernargAMDProgram(dev, f"llama_q4_mmvq_{rows}_graph", OBJ.read_bytes(), kd_offset(OBJ.read_bytes(), cap["kernel_symbol"]),
                                  bytes(mmvq_raw))
  mmvq_runner = ImportedQ4MMVQRunner(mmvq_prg, bytes(mmvq_raw), (rows, 1, 1), tuple(cap["local"]))

  keys: dict[str, tuple[bytes, str]] = {}
  orig = R.get_runtime

  def hook(device, ast, cache=True):
    if ast.arg.name in {"llama_q8_quant_4096_inject", f"llama_q4_mmvq_{rows}_inject"}:
      keys[ast.arg.name] = (ast.key, device)
    return orig(device, ast, cache)

  R.get_runtime = hook
  try:
    x = Tensor.ones(DIM, dtype=dtypes.float32, device="AMD").contiguous().realize()
    q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
    q8.custom_kernel(x, fxn=q8_quant_stub)[0].realize()
    out = Tensor.empty(rows, dtype=dtypes.float32, device="AMD").contiguous()
    q4 = Tensor.empty(rows * (DIM // 256) * 144 // 4, dtype=dtypes.uint32, device="AMD").contiguous().realize()
    out.custom_kernel(q4, q8.realize(), fxn=q4_mmvq_stub)[0].realize()
  finally:
    R.get_runtime = orig

  q8_key = "llama_q8_quant_4096_inject"
  mmvq_key = f"llama_q4_mmvq_{rows}_inject"
  if q8_key not in keys or mmvq_key not in keys:
    raise RuntimeError(f"failed to capture imported MMVQ graph keys: {sorted(keys)}")
  R.runtime_cache[keys[q8_key]] = q8_prg
  R.runtime_cache[keys[mmvq_key]] = mmvq_runner
  return {
    "rows": rows,
    "q8_key_installed": True,
    "mmvq_key_installed": True,
    "q8_launch": {"global": list(q8_prg.qk_global), "local": list(q8_prg.qk_local)},
    "mmvq_launch": {"global": list(mmvq_runner.qk_global), "local": list(mmvq_runner.qk_local)},
    "kernel_symbol": cap["kernel_symbol"],
  }


def q4_words(linear, device: str) -> Tensor:
  words = linear.q4k_storage.words.to(device)
  if linear.q4k_storage.mode == "q4_ondemand":
    words = words.contiguous()
  return words.realize()


def route_imported_q4_mmvq(linear, x: Tensor, q8_side: Tensor | None = None, out_side: Tensor | None = None) -> Tensor | None:
  if x.shape != (1, 1, DIM) or not hasattr(linear, "q4k_storage") or linear.out_features % 32 != 0:
    return None
  rows = linear.out_features
  install_imported_q4_mmvq(rows)
  device = x.device
  x_vec = x.reshape(DIM).cast(dtypes.float32).contiguous()
  q8 = (q8_side if q8_side is not None else Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device=device)).contiguous()
  q8 = q8.custom_kernel(x_vec, fxn=q8_quant_stub)[0]
  out = (out_side if out_side is not None else Tensor.empty(rows, dtype=dtypes.float32, device=device)).contiguous()
  out = out.custom_kernel(q4_words(linear, device), q8, fxn=q4_mmvq_stub)[0]
  return out.reshape(1, 1, rows)
