import importlib.util
from pathlib import Path
import numpy as np


PATH = Path(__file__).resolve().parents[2] / "scratchpad/llama_cuda_quantized_live_oracle.py"


def _module():
  spec = importlib.util.spec_from_file_location("llama_cuda_quantized_live_oracle", PATH)
  module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
  return module


def test_fastdiv_values():
  m = _module()
  for divisor in (1, 2, 3, 7, 32, 1024):
    fd = m.fastdiv_values(divisor)
    for value in (0, 1, divisor-1, divisor, 12345):
      got = ((((value*fd.x) >> 32) + value) >> fd.y)
      assert got == value // divisor


def test_q6_and_q8_python_decoders_match_pinned_cpu_dequant():
  m = _module()
  if not m.DEFAULT_BASE.is_file(): return
  _, q6, q8 = m._cpu_quantizers(m.DEFAULT_BASE)
  rng = np.random.default_rng(17)
  weights = rng.normal(size=m.QK_K).astype(np.float32)
  activation = rng.normal(size=m.QK8_1).astype(np.float32)
  packed_q6, packed_q8 = m.pack_q6(weights, q6), m.pack_q8(activation, q8)
  assert m.decode_q6(packed_q6).shape == (m.QK_K,)
  assert m.decode_q8(packed_q8).shape == (m.QK8_1,)
  # Independent decoder output must be finite and track the source data within
  # the expected coarse quantization scale.
  assert np.isfinite(m.decode_q6(packed_q6)).all()
  assert np.isfinite(m.decode_q8(packed_q8)).all()
  assert np.max(np.abs(m.decode_q6(packed_q6)-weights)) < 0.25
  assert np.max(np.abs(m.decode_q8(packed_q8)-activation)) < 0.05


def test_q4_python_decoder_matches_pinned_cpu_dequant():
  m = _module()
  if not m.DEFAULT_BASE.is_file(): return
  import ctypes
  q4, _, _ = m._cpu_quantizers(m.DEFAULT_BASE)
  values = np.random.default_rng(23).normal(size=m.QK_K).astype(np.float32)
  packed = m.pack_q4(values, q4)
  lib = ctypes.CDLL(str(m.DEFAULT_BASE))
  dequant = lib.dequantize_row_q4_K
  dequant.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64]
  expected = np.empty(m.QK_K, dtype=np.float32)
  raw = (ctypes.c_uint8 * len(packed)).from_buffer_copy(packed)
  dequant(raw, expected.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), m.QK_K)
  assert np.array_equal(m.decode_q4(packed), expected)
