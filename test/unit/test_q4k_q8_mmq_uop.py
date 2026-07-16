import os
import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import full_rewrite_to_sink, to_program
from tinygrad.codegen.opt import OptOps
from tinygrad.device import Device
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import AxisType, KernelInfo, Ops, UOp

from extra.qk.amdgpu_metadata import parse_amdgpu_metadata
from extra.qk.q4k_q8_mmq_uop import (describe_q4k_q8_mmq_uop, describe_q4k_q8_mmq_wmma,
  describe_q4k_q8_mmq_sum_original_fp_wmma, describe_q4k_q8_mmq_role_sized_wmma, describe_q4k_q8_mmq_wide_wmma,
  emit_q4k_q8_mmq_uop, emit_q4k_q8_mmq_wmma, emit_q4k_q8_mmq_sum_original_fp_wmma,
  emit_q4k_q8_mmq_role_sized_wmma, emit_q4k_q8_mmq_wide_wmma,
  LLAMA_Q4K_Q8_1_DS4_SOURCE_ANCHORS)
from extra.qk.q4k_q8_mmq_uop_validation import _fixture, independent_packed_byte_reference
from extra.qk.mmq_q4k_q8_reference import (Q81MMQDS4Activation, Q81MMQDS4ActivationSpec, Q8_1_MMQ_DS4_LAYOUT,
  describe_q4k_q8_1_mmq_tile, q4k_q8_1_mmq_ds4_tile_reference, q8_1_mmq_ds4_quantize_reference)


def _sink(m=2, n=35, k=256):
  spec = describe_q4k_q8_mmq_uop(m, n, k)
  return emit_q4k_q8_mmq_uop(spec)(UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n * (k//256) * 36,), dtypes.uint32, 1), UOp.placeholder((m*k,), dtypes.int8, 2),
    UOp.placeholder((m*(k//32),), dtypes.float32, 3))


def test_direct_uop_owner_structure():
  sink = _sink()
  nodes = sink.toposort()
  assert sink.op is Ops.SINK and isinstance(sink.arg, KernelInfo)
  specials = {u.arg:u for u in nodes if u.op is Ops.SPECIAL}
  assert set(specials) == {"gidx0", "gidx1", "lidx0"}
  assert specials["gidx0"].src[0].arg == 2 and specials["gidx1"].src[0].arg == 2
  ranges = [u for u in nodes if u.op is Ops.RANGE]
  assert len(ranges) == 1 and ranges[0].src[0].arg == 256 and ranges[0].arg[1] is AxisType.REDUCE
  assert not any(u.op is Ops.UNROLL for u in nodes)
  stores = [u for u in nodes if u.op is Ops.STORE]
  assert len(stores) == 1 and any(u.op is Ops.SPECIAL for u in stores[0].src[0].toposort())
  # uint32 payload/metadata and int8 activation are both indexed, not expanded.
  indexed_bases = {u.src[0].dtype.base for u in nodes if u.op is Ops.INDEX}
  assert dtypes.uint32 in indexed_bases and dtypes.int8 in indexed_bases


def _packed_constant_q4(n:int, q:int) -> np.ndarray:
  # d=1, dmin=0; all eight six-bit scales are one and mins are zero.
  raw = np.zeros((n, 144), dtype=np.uint8)
  raw[:, :2] = np.frombuffer(np.float16(1).tobytes(), dtype=np.uint8)
  raw[:, 4:8] = 1
  raw[:, 12:16] = 1
  raw[:, 16:] = np.uint8((q & 15) | ((q & 15) << 4))
  return raw.reshape(-1).view(np.uint32)


def _packed_q4_block(d:float, dmin:float, scales:list[int], mins:list[int], quants:list[int]) -> np.ndarray:
  """Independent byte packer for one Q4_K block (not shared with the emitter)."""
  assert len(scales) == len(mins) == len(quants) == 8
  raw = np.zeros(144, dtype=np.uint8)
  raw[:4] = np.frombuffer(np.array([d, dmin], dtype=np.float16).tobytes(), dtype=np.uint8)
  u = raw[4:16]
  for g in range(4):
    u[g] = (scales[g] & 0x3f) | ((scales[g+4] >> 4) << 6)
    u[4+g] = (mins[g] & 0x3f) | ((mins[g+4] >> 4) << 6)
    u[8+g] = (scales[g+4] & 0xf) | ((mins[g+4] & 0xf) << 4)
  payload = raw[16:].reshape(4, 32)
  for g, q in enumerate(quants): payload[g//2] |= np.uint8((q & 0xf) << (4 * (g % 2)))
  return raw.view(np.uint32)


def test_direct_uop_python_correctness():
  m, n, k, q = 2, 3, 256, 3
  rng = np.random.default_rng(7)
  xq = rng.integers(-8, 9, size=(m, k), dtype=np.int8)
  scales = rng.uniform(0.01, 0.2, size=(m, k//32)).astype(np.float32)
  expected = np.repeat((q * (xq.reshape(m, k//32, 32).astype(np.float32) * scales[:, :, None])).sum((1, 2))[:, None], n, 1)
  spec = describe_q4k_q8_mmq_uop(m, n, k)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device="PYTHON").custom_kernel(
    Tensor(_packed_constant_q4(n, q), device="PYTHON"), Tensor(xq.reshape(-1), device="PYTHON"),
    Tensor(scales.reshape(-1), device="PYTHON"), fxn=emit_q4k_q8_mmq_uop(spec))[0].numpy()
  np.testing.assert_allclose(out, expected, rtol=2e-5, atol=2e-5)


def test_upper_group_metadata_and_min_correction_python():
  # Distinct upper metadata catches grp-vs-h indexing mistakes.  Groups 4 and
  # 7 also carry different q payloads and both have nonzero minimum correction.
  m, n, k = 1, 1, 256
  d, dmin = 0.5, 0.25
  scales = [2, 5, 9, 13, 17, 34, 51, 60]
  mins = [3, 6, 10, 14, 19, 36, 53, 62]
  quants = [0, 0, 0, 0, 6, 0, 0, 11]
  xq = np.arange(-16, 16, dtype=np.int8).reshape(1, 32).repeat(8, axis=0).reshape(1, k)
  xscale = np.array([[0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.17, 0.19]], dtype=np.float32)
  expected = np.float32(0)
  for g in range(8):
    q8 = xq[0, g*32:(g+1)*32].astype(np.float32)
    expected += xscale[0, g] * (d * scales[g] * quants[g] * q8.sum() - dmin * mins[g] * q8.sum())
  spec = describe_q4k_q8_mmq_uop(m, n, k)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device="PYTHON").custom_kernel(
    Tensor(_packed_q4_block(d, dmin, scales, mins, quants), device="PYTHON"),
    Tensor(xq.reshape(-1), device="PYTHON"), Tensor(xscale.reshape(-1), device="PYTHON"),
    fxn=emit_q4k_q8_mmq_uop(spec))[0].numpy()
  np.testing.assert_allclose(out, [[expected]], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("device", ["PYTHON", pytest.param("AMD", marks=pytest.mark.skipif(
  not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable"))])
def test_scalar_random_full_packed_byte_differential(device):
  m, n, k = 2, 3, 256
  words, xq, xscale = _fixture(m, n, k)
  reference = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  spec = describe_q4k_q8_mmq_uop(m, n, k)
  got = Tensor.empty(m, n, dtype=dtypes.float32, device=device).custom_kernel(
    Tensor(words, device=device), Tensor(xq.reshape(-1), device=device),
    Tensor(xscale.reshape(-1), device=device), fxn=emit_q4k_q8_mmq_uop(spec))[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-4)


def test_spec_rejects_unaligned_k():
  with pytest.raises(ValueError, match="multiple of 256"): describe_q4k_q8_mmq_uop(1, 1, 32)


def _wmma_sink(m=16, n=16, k=256):
  spec = describe_q4k_q8_mmq_wmma(m=m, n=n, k=k)
  return emit_q4k_q8_mmq_wmma(spec)(UOp.placeholder((m, n), dtypes.float32, 10),
    UOp.placeholder((n*(k//256)*36,), dtypes.uint32, 11), UOp.placeholder((m*k,), dtypes.int8, 12),
    UOp.placeholder((m*(k//32),), dtypes.float32, 13))


def test_wmma_candidate_is_exact_and_generic():
  with pytest.raises(ValueError, match="positive M/N multiples of 16"): describe_q4k_q8_mmq_wmma(m=8)
  sink = _wmma_sink()
  assert sink.arg.name.endswith("16x16x256")
  nodes = sink.toposort()
  assert not any(u.op in (Ops.WMMA, Ops.SHAPED_WMMA) for u in nodes)
  assert not any(u.op is Ops.SPECIAL for u in nodes)
  assert {(u.src[0].arg, u.arg[1]) for u in nodes if u.op is Ops.RANGE}.issuperset(
    {(16, AxisType.LOOP), (8, AxisType.REDUCE), (32, AxisType.REDUCE)})
  assert len([u for u in nodes if u.op is Ops.REDUCE]) == 3  # dot, q8 sum, group correction
  assert any(u.op is Ops.MUL and u.dtype is dtypes.int8 and u.src[0].dtype is dtypes.int8 and
             u.src[1].dtype is dtypes.int8 for u in nodes)
  assert len([u for u in nodes if u.op is Ops.STORE]) == 1
  tc = sink.arg.opts_to_apply[0]
  assert tc.op is OptOps.TC and tc.axis == 0 and tc.arg == (-1, 2, 1)


def test_wmma_candidate_amd_isa():
  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  lowered = full_rewrite_to_sink(_wmma_sink(), renderer, optimize=True)
  assert len([u for u in lowered.toposort() if u.op is Ops.WMMA]) == 1
  assert {u.arg for u in lowered.toposort() if u.op is Ops.SPECIAL} == {"lidx0"}
  assert any(u.op is Ops.RANGE and u.src[0].op is Ops.CONST and u.src[0].arg == 2 for u in lowered.toposort())
  program = to_program(_wmma_sink(), renderer)
  lines = [str(u.arg) for linear in program.src if linear.op is Ops.LINEAR for u in linear.src if not isinstance(u.arg, tuple)]
  wmma = [line for line in lines if line.startswith("v_wmma_i32_16x16x16_iu8")]
  assert len(wmma) == 1 and wmma[0].endswith(", 3)")


def test_wmma_32x32x512_one_program_grid_and_signed_isa():
  sink = _wmma_sink(32, 32, 512)
  assert sink.arg.name.endswith("32x32x512")
  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  lowered = full_rewrite_to_sink(sink, renderer, optimize=True)
  specials = {u.arg:u.src[0].arg for u in lowered.toposort() if u.op is Ops.SPECIAL}
  assert specials == {"gidx0":2, "lidx0":32, "gidx1":2}
  assert len([u for u in lowered.toposort() if u.op is Ops.WMMA]) == 1
  program = to_program(sink, renderer)
  assert len([u for u in program.src if u.op is Ops.LINEAR]) == 1
  lines = [str(u.arg) for linear in program.src if linear.op is Ops.LINEAR for u in linear.src if not isinstance(u.arg, tuple)]
  wmma = [line for line in lines if line.startswith("v_wmma_i32_16x16x16_iu8")]
  assert len(wmma) == 1 and wmma[0].endswith(", 3)")


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_wmma_candidate_amd_correctness():
  rng = np.random.default_rng(19)
  d, dmin = 0.5, 0.25
  scales = [2, 5, 9, 13, 17, 34, 51, 60]
  mins = [3, 6, 10, 14, 19, 36, 53, 62]
  quants = [1, 3, 5, 7, 6, 8, 10, 11]
  words = np.tile(_packed_q4_block(d, dmin, scales, mins, quants), 16)
  xq = rng.integers(-16, 17, size=(16, 256), dtype=np.int8)
  xscale = rng.uniform(0.01, 0.2, size=(16, 8)).astype(np.float32)
  expected = np.zeros((16, 16), dtype=np.float32)
  for m in range(16):
    total = np.float32(0)
    for g in range(8):
      vals = xq[m, g*32:(g+1)*32].astype(np.int32)
      dot, qsum = np.int32(quants[g]) * vals.sum(dtype=np.int32), vals.sum(dtype=np.int32)
      total += xscale[m, g] * (d * scales[g] * dot - dmin * mins[g] * qsum)
    expected[m, :] = total
  spec = describe_q4k_q8_mmq_wmma()
  out = Tensor.empty(16, 16, dtype=dtypes.float32, device="AMD").custom_kernel(
    Tensor(words, device="AMD"), Tensor(xq.reshape(-1), device="AMD"),
    Tensor(xscale.reshape(-1), device="AMD"), fxn=emit_q4k_q8_mmq_wmma(spec))[0].numpy()
  np.testing.assert_allclose(out, expected, rtol=3e-4, atol=3e-3)


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_wmma_32x32x512_random_packed_byte_amd_correctness():
  m, n, k = 32, 32, 512
  words, xq, xscale = _fixture(m, n, k)
  reference = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  spec = describe_q4k_q8_mmq_wmma(m=m, n=n, k=k)
  out = Tensor.empty(m, n, dtype=dtypes.float32, device="AMD").custom_kernel(
    Tensor(words, device="AMD"), Tensor(xq.reshape(-1), device="AMD"),
    Tensor(xscale.reshape(-1), device="AMD"), fxn=emit_q4k_q8_mmq_wmma(spec))[0].numpy()
  np.testing.assert_allclose(out, reference, rtol=3e-4, atol=3e-3)


def _wide_sink(k=256):
  spec = describe_q4k_q8_mmq_wide_wmma(k=k)
  return emit_q4k_q8_mmq_wide_wmma(spec)(UOp.placeholder((16, 32), dtypes.float32, 20),
    UOp.placeholder((32*(k//256)*36,), dtypes.uint32, 21), UOp.placeholder((16*k,), dtypes.int8, 22),
    UOp.placeholder((16*(k//32),), dtypes.float32, 23))


def test_wide_candidate_contract_options_and_structural_sharing():
  for kwargs in ({"m":32}, {"n":64}, {"k":128}):
    with pytest.raises(ValueError, match="exact M=16, N=32"): describe_q4k_q8_mmq_wide_wmma(**kwargs)
  sink = _wide_sink()
  assert sink.arg.name == "q4k_q8_mmq_uop_wide_wmma_16x32x256"
  assert [(o.op, o.axis, o.arg) for o in sink.arg.opts_to_apply] == [
    (OptOps.TC, 0, (-1, 2, 1)), (OptOps.UPCAST, 0, 2)]
  nodes = sink.toposort()
  assert not any(u.op in (Ops.WMMA, Ops.SHAPED_WMMA) for u in nodes)
  dot = next(u for u in nodes if u.op is Ops.REDUCE and u.dtype is dtypes.int32 and
             u.src[0].op is Ops.CAST and u.src[0].src[0].op is Ops.MUL)
  mul = dot.src[0].src[0]
  q8 = next(x for x in mul.src if x.op is Ops.INDEX and x.dtype is dtypes.int8)
  qsums = [u for u in nodes if u.op is Ops.REDUCE and u.dtype is dtypes.int32 and
           u.src[0].op is Ops.CAST and u.src[0].src[0] is q8]
  assert len(qsums) == 1
  # The symbolic xscale load is independent of N and shared by both output subtiles.
  xscale_indexes = [u for u in nodes if u.op is Ops.INDEX and any(
    x.op is Ops.PARAM and x.arg.slot == 23 for x in u.toposort())]
  assert len(xscale_indexes) == 1
  assert not any(u.op is Ops.RANGE and u.arg[0] == 1 for u in xscale_indexes[0].toposort())


def test_wide_candidate_one_workgroup_resources_and_signed_wmma():
  sink = _wide_sink()
  program = to_program(sink, Device["AMD"].renderer)
  assert tuple(program.arg.global_size) == (1, 1, 1) and tuple(program.arg.local_size) == (32, 1, 1)
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  wmmas = [u for u in linear.src if u.op is Ops.WMMA]
  assert len(wmmas) == 2 and all("signed_char_int" in u.arg[0] for u in wmmas)
  binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
  resources = parse_amdgpu_metadata(binary)
  assert 145 <= resources["vgpr"] <= 155
  assert resources["scratch_bytes"] == resources["vgpr_spills"] == resources["sgpr_spills"] == 0


def test_wide_candidate_random_packed_byte_python_oracle():
  m, n, k = 16, 32, 256
  words, _, xscale = _fixture(m, n, k)
  # Generic Python executes char MUL literally, while WMMA widens products.
  # Keep random q8 in the exact non-overflowing range for this backend oracle.
  xq = np.random.default_rng(23).integers(-8, 9, size=(m, k), dtype=np.int8)
  reference = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  # Python has no TC capability. Execute the exact authored arithmetic graph
  # with only backend scheduling options removed; AMD below executes wide mode.
  callback = emit_q4k_q8_mmq_wide_wmma(describe_q4k_q8_mmq_wide_wmma())
  def python_callback(*args): return callback(*args).replace(arg=KernelInfo(name="wide_python_oracle", opts_to_apply=()))
  got = Tensor.empty(m, n, dtype=dtypes.float32, device="PYTHON").custom_kernel(
    Tensor(words, device="PYTHON"), Tensor(xq.reshape(-1), device="PYTHON"),
    Tensor(xscale.reshape(-1), device="PYTHON"), fxn=python_callback)[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-4)


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_wide_candidate_random_packed_byte_amd_correctness():
  m, n, k = 16, 32, 256
  words, xq, xscale = _fixture(m, n, k)
  reference = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  spec = describe_q4k_q8_mmq_wide_wmma()
  got = Tensor.empty(m, n, dtype=dtypes.float32, device="AMD").custom_kernel(
    Tensor(words, device="AMD"), Tensor(xq.reshape(-1), device="AMD"),
    Tensor(xscale.reshape(-1), device="AMD"), fxn=emit_q4k_q8_mmq_wide_wmma(spec))[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-3)


def _sum_sink(m=16, n=16, k=256):
  spec = describe_q4k_q8_mmq_sum_original_fp_wmma(m, n, k)
  return emit_q4k_q8_mmq_sum_original_fp_wmma(spec)(UOp.placeholder((m, n), dtypes.float32, 30),
    UOp.placeholder((n*(k//256)*36,), dtypes.uint32, 31), UOp.placeholder((m*k,), dtypes.int8, 32),
    UOp.placeholder((m*(k//32),), dtypes.float32, 33), UOp.placeholder((m*(k//32),), dtypes.float32, 34))


def _role_sized_sink(m=16, n=16, k=256):
  spec = describe_q4k_q8_mmq_role_sized_wmma(m, n, k)
  return emit_q4k_q8_mmq_role_sized_wmma(spec)(UOp.placeholder((m, n), dtypes.float32, 0),
    UOp.placeholder((n*(k//256)*36,), dtypes.uint32, 1), UOp.placeholder((k//128*m*128,), dtypes.int8, 2),
    UOp.placeholder((k//128*m*4,), dtypes.float32, 3), UOp.placeholder((k//128*m*4,), dtypes.float32, 4))


def _reference_with_original_fp_sum(words, xq, xscale, supplied, *, m, n, k):
  """Independent reference derived from packed bytes plus supplied-sum delta."""
  out = independent_packed_byte_reference(words, xq, xscale, m=m, n=n, k=k)
  derived = xscale.reshape(m, k//32) * xq.reshape(m, k//32, 32).astype(np.float32).sum(axis=2)
  delta = supplied.reshape(m, k//32).astype(np.float32) - derived
  raw = np.asarray(words, dtype=np.uint32).astype("<u4", copy=False).view(np.uint8).reshape(n, k//256, 144)
  for col in range(n):
    for block in range(k//256):
      dmin = np.frombuffer(raw[col, block, 2:4].tobytes(), dtype="<f2").astype(np.float32)[0]
      meta = raw[col, block, 4:16]
      for g in range(8):
        if g < 4: mn = int(meta[4+g] & 63)
        else:
          h = g - 4
          mn = int((meta[8+h] >> 4) | ((meta[4+h] >> 6) << 4))
        out[:, col] -= dmin * mn * delta[:, block*8+g]
  return out


def test_sum_original_fp_contract_anchors_and_no_qsum_reduction():
  with pytest.raises(ValueError, match="positive M/N multiples of 16"): describe_q4k_q8_mmq_sum_original_fp_wmma(8, 16, 256)
  for semantics in ("", "sum_dequant_q8", "sum_int8"):
    with pytest.raises(ValueError, match="original-fp32 group sum"): describe_q4k_q8_mmq_sum_original_fp_wmma(
      16, 16, 256, sum_semantics=semantics)
  sink, default = _sum_sink(), _wmma_sink()
  spec = describe_q4k_q8_mmq_sum_original_fp_wmma(16, 16, 256)
  assert "llama_ds4_y_original_fp_sum" in sink.arg.name
  assert spec.source_anchors == LLAMA_Q4K_Q8_1_DS4_SOURCE_ANCHORS
  assert any("quantize_mmq_q8_1" in anchor and "make_half2(d, sum)" in anchor for anchor in spec.source_anchors)
  assert any("vec_dot_q4_K_q8_1_impl_mmq" in anchor for anchor in spec.source_anchors)
  nodes, default_nodes = sink.toposort(), default.toposort()
  assert not any(u.op in (Ops.WMMA, Ops.SHAPED_WMMA) for u in nodes)
  assert len([u for u in nodes if u.op is Ops.REDUCE]) == 2
  assert len([u for u in default_nodes if u.op is Ops.REDUCE]) == 3
  q8 = next(u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.int8)
  q8_reduces = [u for u in nodes if u.op is Ops.REDUCE and q8 in u.toposort()]
  default_q8 = next(u for u in default_nodes if u.op is Ops.INDEX and u.dtype is dtypes.int8)
  default_q8_reduces = [u for u in default_nodes if u.op is Ops.REDUCE and default_q8 in u.toposort()]
  assert len(q8_reduces) == 2 and len(default_q8_reduces) == 3
  assert len([u for u in nodes if u.op is Ops.INDEX and u.dtype is dtypes.int8]) == 1


def test_sum_original_fp_semantic_split_python():
  m = n = 16; k = 256
  words, _, xscale = _fixture(m, n, k)
  xq = np.random.default_rng(41).integers(-8, 9, size=(m, k), dtype=np.int8)
  derived = xscale * xq.reshape(m, 8, 32).astype(np.float32).sum(axis=2)
  supplied = derived + np.linspace(-0.75, 0.5, m*8, dtype=np.float32).reshape(m, 8)
  reference = _reference_with_original_fp_sum(words, xq, xscale, supplied, m=m, n=n, k=k)
  callback = emit_q4k_q8_mmq_sum_original_fp_wmma(describe_q4k_q8_mmq_sum_original_fp_wmma(m, n, k))
  def python_callback(*args): return callback(*args).replace(arg=KernelInfo(name="sum_original_fp_python", opts_to_apply=()))
  got = Tensor.empty(m, n, dtype=dtypes.float32, device="PYTHON").custom_kernel(
    Tensor(words, device="PYTHON"), Tensor(xq.reshape(-1), device="PYTHON"), Tensor(xscale.reshape(-1), device="PYTHON"),
    Tensor(supplied.reshape(-1), device="PYTHON"), fxn=python_callback)[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-4)


def test_role_sized_exact_physical_ds4_layout_python():
  m = n = 16; k = 512
  words, _, _ = _fixture(m, n, k)
  rng = np.random.default_rng(59)
  values = rng.integers(-8, 9, size=(k//128, m, 128), dtype=np.int8)
  scales = rng.uniform(0.01, 0.2, size=(k//128, m, 4)).astype(np.float32)
  # Deliberately differ from a dequantized-int8 sum to pin original-fp semantics.
  sums = (values.reshape(k//128, m, 4, 32).astype(np.float32).sum(3) * scales +
          rng.uniform(-0.5, 0.5, size=(k//128, m, 4))).astype(np.float32)
  ds4 = Q81MMQDS4Activation(values, scales, sums, Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  oracle = describe_q4k_q8_1_mmq_tile(role="physical_ds4", m=m, n=n, k=k, m_tile=m, n_tile=n,
    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words.view(np.uint8), ds4, oracle)
  callback = emit_q4k_q8_mmq_role_sized_wmma(describe_q4k_q8_mmq_role_sized_wmma(m, n, k))
  def python_callback(*args): return callback(*args).replace(arg=KernelInfo(name="role_sized_python", opts_to_apply=()))
  got = Tensor.empty(m, n, dtype=dtypes.float32, device="PYTHON").custom_kernel(
    Tensor(words, device="PYTHON"), Tensor(values.reshape(-1), device="PYTHON"),
    Tensor(scales.reshape(-1), device="PYTHON"), Tensor(sums.reshape(-1), device="PYTHON"), fxn=python_callback)[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-4)


def test_role_sized_amd_isa_program_abi_and_signed_wmma():
  sink = _role_sized_sink(32, 32, 512)
  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  program = full_rewrite_to_sink(sink, renderer, optimize=True)
  assert len([u for u in sink.toposort() if u.op is Ops.STORE]) == 1
  assert {u.arg.slot:u.dtype.base for u in sink.toposort() if u.op is Ops.PARAM} == {
    0:dtypes.float32, 1:dtypes.uint32, 2:dtypes.int8, 3:dtypes.float32, 4:dtypes.float32}
  assert program.op is Ops.SINK and program.arg.name == sink.arg.name
  wmmas = [u for u in program.toposort() if u.op is Ops.WMMA]
  assert len(wmmas) == 1 and "signed_char_int" in wmmas[0].arg[0]


def test_role_sized_real_role_grid_is_structural_only():
  sink = _role_sized_sink(512, 1024, 5120)
  renderer = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  lowered = full_rewrite_to_sink(sink, renderer, optimize=True)
  assert {u.arg:u.src[0].arg for u in lowered.toposort() if u.op is Ops.SPECIAL} == {"gidx0":64, "lidx0":32, "gidx1":32}
  assert len([u for u in sink.toposort() if u.op is Ops.STORE]) == 1


def test_role_sized_rejects_tails_and_wrong_layout():
  for shape in ((15, 16, 256), (16, 17, 256), (16, 16, 384)):
    with pytest.raises(ValueError, match="no tails"): describe_q4k_q8_mmq_role_sized_wmma(*shape)
  with pytest.raises(ValueError, match="q8_1_mmq_ds4_transposed_blocks"):
    describe_q4k_q8_mmq_role_sized_wmma(16, 16, 256, activation_layout="q8_1_row_major_mk_scales_per_32")


def test_sum_original_fp_one_program_signed_wmma():
  sink = _sum_sink()
  program = to_program(sink, Device["AMD"].renderer)
  assert program.arg.function_name == sink.arg.name
  assert len([u for u in program.src if u.op is Ops.LINEAR]) == 1
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  wmmas = [u for u in linear.src if u.op is Ops.WMMA]
  assert len(wmmas) == 1 and "signed_char_int" in wmmas[0].arg[0]


def test_llama_ds4_original_fp_sum_differs_from_dequantized_q8_sum():
  source = np.random.default_rng(47).standard_normal((16, 256), dtype=np.float32)
  values, scales, original_sums = q8_1_mmq_ds4_quantize_reference(source)
  dequant_sums = values.reshape(2, 16, 4, 32).astype(np.float32).sum(axis=3) * scales
  assert np.max(np.abs(original_sums - dequant_sums)) > 1e-4


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_sum_original_fp_llama_ds4_random_packed_byte_amd_correctness():
  m = n = 16; k = 256
  words, _, _ = _fixture(m, n, k)
  source = np.random.default_rng(53).standard_normal((m, k), dtype=np.float32)
  values, scales, sums = q8_1_mmq_ds4_quantize_reference(source)
  xq = values.transpose(1, 0, 2).reshape(m, k)
  xscale = scales.transpose(1, 0, 2).reshape(m, k//32)
  supplied = sums.transpose(1, 0, 2).reshape(m, k//32)
  ds4 = Q81MMQDS4Activation(values, scales, sums, Q81MMQDS4ActivationSpec(m=m, k=k, m_tile=m))
  oracle_spec = describe_q4k_q8_1_mmq_tile(role="uop_sum_original_fp_oracle", m=m, n=n, k=k,
    m_tile=m, n_tile=n, activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  reference = q4k_q8_1_mmq_ds4_tile_reference(words.view(np.uint8), ds4, oracle_spec)
  spec = describe_q4k_q8_mmq_sum_original_fp_wmma(m, n, k)
  got = Tensor.empty(m, n, dtype=dtypes.float32, device="AMD").custom_kernel(
    Tensor(words, device="AMD"), Tensor(xq.reshape(-1), device="AMD"), Tensor(xscale.reshape(-1), device="AMD"),
    Tensor(supplied.reshape(-1), device="AMD"), fxn=emit_q4k_q8_mmq_sum_original_fp_wmma(spec))[0].numpy()
  np.testing.assert_allclose(got, reference, rtol=3e-4, atol=3e-3)
