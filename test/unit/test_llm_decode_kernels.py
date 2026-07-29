import hashlib
from pathlib import Path

import pytest

from tinygrad import dtypes
from tinygrad.llm import decode_kernels
from tinygrad.uop.ops import UOp


def _q4_args(rows:int, k:int):
  return (UOp.placeholder((rows,), dtypes.float32, 0),
          UOp.placeholder((rows * (k//256) * 36,), dtypes.uint32, 1),
          UOp.placeholder((k,), dtypes.float16, 2))


def _q6_args(spec):
  return (UOp.placeholder((spec.rows, spec.partial_axis_extent), dtypes.float32, 0),
          UOp.placeholder((spec.rows * spec.k_blocks * 105,), dtypes.uint16, 1),
          UOp.placeholder((spec.k,), dtypes.float16, 2))


def test_production_decode_kernels_have_no_research_dependency():
  source = Path(decode_kernels.__file__).read_text()
  assert "extra.llm_research" not in source
  assert "from extra" not in source


@pytest.mark.parametrize("rows,k,digest", [
  (32, 1024, "38824cc99a243d1341ceb6bacafb932695da49aadbbdb209c4d82820ecafef6b"),
  (4096, 4096, "5fdb3e94c16b764a3cddef363b8fa8ee950f9bdc6991bd15cdf0a8e397b3847e"),
  (17408, 5120, "18d537cc1957495049fb02d9c792eccc0754ba0f393113bc828266b60c15ce84"),
])
def test_q4_g3_production_lowering_preserves_promoted_uop_identity(rows, k, digest):
  args = _q4_args(rows, k)
  promoted = decode_kernels.q4k_g3_lanemap_gemv_kernel(rows, k)(*args)
  assert hashlib.sha256(repr(promoted.key).encode()).hexdigest() == digest
  assert promoted.arg.name == f"q4k_g3_lanemap_gemv_{rows}_{k}"


@pytest.mark.parametrize("rows,k,parts,use_coop,family,digest", [
  (16, 256, 1, True, "q6k_coop", "a30e1686b4731061bdb504eb4481d6fb8d01fe1aa4048484dab83702a16b4a34"),
  (15, 512, 3, False, "q6k_partial", "7bdf86fc211ddaf5ea7495d51c3a7b18b8874e66e099a031174b7fba9a7c3881"),
])
def test_q6_production_spec_and_lowering_preserve_promoted_identity(rows, k, parts, use_coop, family, digest):
  kwargs = {"parts": parts, "row_tile": 4, "use_coop": use_coop, "opts": ()}
  promoted_spec = decode_kernels.q6k_spec_for_role(rows, k, **kwargs)
  assert promoted_spec.route_family == family
  assert promoted_spec.to_json()["target"] == "amd_gfx1100"
  args = _q6_args(promoted_spec)
  emitted = decode_kernels.emit_q6k_gemv_kernel(promoted_spec)(*args)
  assert hashlib.sha256(repr(emitted.key).encode()).hexdigest() == digest


def test_q6_route_family_guards_and_vocab_reducer_identity():
  with pytest.raises(ValueError, match="multiple of 256"):
    decode_kernels.emit_q6k_gemv_kernel(decode_kernels.q6k_spec_for_role(16, 128))
  with pytest.raises(ValueError, match="rows.*row_tile"):
    decode_kernels.emit_q6k_gemv_kernel(decode_kernels.q6k_spec_for_role(15, 256))

  small = decode_kernels.q6k_spec_for_role(16, 256)
  assert not decode_kernels.q6k_vocab_scalar_reduce_eligible(small)
  with pytest.raises(ValueError, match="not admitted"):
    decode_kernels.emit_q6k_vocab_scalar_reduce_kernel(small)

  vocab = decode_kernels.q6k_spec_for_role(131072, 256)
  assert decode_kernels.q6k_vocab_scalar_reduce_eligible(vocab)
  out = UOp.placeholder((vocab.rows,), dtypes.float32, 0)
  partials = UOp.placeholder((vocab.rows, vocab.partial_axis_extent), dtypes.float32, 1)
  emitted = decode_kernels.emit_q6k_vocab_scalar_reduce_kernel(vocab)(out, partials)
  assert emitted.arg.name == "q6k_vocab_scalar_reduce_131072_256"


def test_q6_partial_fallback_keeps_parts_and_opts():
  marker = object()
  spec = decode_kernels.q6k_spec_for_role(15, 512, parts=3, use_coop=False, opts=(marker,))
  assert spec.route_family == "q6k_partial" and spec.partial_axis_extent == 3 and spec.opts == (marker,)
  assert spec.kernel_name == "q6k_gen_partial_15_512_3"
