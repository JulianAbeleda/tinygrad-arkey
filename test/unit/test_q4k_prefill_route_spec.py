import pytest

from tinygrad.codegen.opt import OptOps
from tinygrad import dtypes
from tinygrad.uop.ops import UOp

from extra.qk.layout import Q4K_WORDS_PER_BLOCK


def _direct_out_placeholders(spec):
  word_elems = spec.rows * spec.k_blocks * Q4K_WORDS_PER_BLOCK
  words = UOp.placeholder((word_elems,), dtypes.uint32, 0)
  x = UOp.placeholder((spec.tokens * spec.k,), dtypes.float16, 1)
  out = UOp.placeholder((spec.tokens, spec.rows), dtypes.float32, 2)
  return out, words, x


def _partials_placeholders(spec):
  word_elems = spec.rows * spec.k_blocks * Q4K_WORDS_PER_BLOCK
  words = UOp.placeholder((word_elems,), dtypes.uint32, 0)
  x = UOp.placeholder((spec.tokens * spec.k,), dtypes.float16, 1)
  partials = UOp.placeholder((spec.rows, spec.tokens, spec.parts), dtypes.float32, 2)
  return partials, words, x


def test_describe_q4k_packed_prefill_direct_out_defaults():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  spec = describe_q4k_packed_prefill(17408, 5120, 512)
  assert spec.output_layout == "direct_out"
  assert spec.parts == 1
  assert spec.kernel_name == "q4k_gen_prefill_direct_out_17408_5120_512"


def test_describe_q4k_packed_prefill_partials_layout():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  spec = describe_q4k_packed_prefill(17408, 5120, 512, parts=4, output_layout="partials")
  assert spec.parts == 4
  assert spec.kernel_name == "q4k_gen_prefill_partials_17408_5120_512_4"


def test_describe_q4k_packed_prefill_rejects_bad_layout():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  with pytest.raises(ValueError, match="unsupported output_layout"):
    describe_q4k_packed_prefill(128, 256, 32, output_layout="mismatch")


def test_describe_q4k_packed_prefill_rejects_direct_out_parts():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  with pytest.raises(ValueError, match="direct_out output_layout requires parts==1"):
    describe_q4k_packed_prefill(128, 256, 32, parts=2, output_layout="direct_out")
  with pytest.raises(ValueError, match="reduce_out output_layout requires parts==1"):
    describe_q4k_packed_prefill(128, 256, 32, parts=2, output_layout="reduce_out")


def test_describe_q4k_packed_prefill_rejects_bad_k():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  with pytest.raises(ValueError, match="must be a multiple of 256"):
    describe_q4k_packed_prefill(128, 128, 32, output_layout="direct_out")


def test_describe_q4k_packed_prefill_parses_opts():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill
  spec = describe_q4k_packed_prefill(128, 256, 32, output_layout="direct_out",
                                     opts=("UPCAST:1:4", "LOCAL:0:16"))
  assert [(x.op, x.axis, x.arg) for x in spec.opts] == [(OptOps.UPCAST, 1, 4), (OptOps.LOCAL, 0, 16)]


def test_emit_q4k_packed_prefill_kernel_direct_out():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
  spec = describe_q4k_packed_prefill(4, 256, 8, output_layout="direct_out", opts=("UPCAST:1:4",))
  out, words, x = _direct_out_placeholders(spec)
  uops = emit_q4k_packed_prefill_kernel(spec)(out, words, x)
  assert uops.arg.name == spec.kernel_name
  assert uops.arg.opts_to_apply == spec.opts


def test_emit_q4k_packed_prefill_kernel_partials():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
  spec = describe_q4k_packed_prefill(4, 256, 8, parts=2, output_layout="partials")
  partials, words, x = _partials_placeholders(spec)
  uops = emit_q4k_packed_prefill_kernel(spec)(partials, words, x)
  assert uops.arg.name == spec.kernel_name


def test_emit_q4k_packed_prefill_kernel_reduce_out():
  from extra.qk.q4k_prefill_route_spec import describe_q4k_packed_prefill, emit_q4k_packed_prefill_kernel
  spec = describe_q4k_packed_prefill(4, 256, 8, output_layout="reduce_out")
  out, words, x = _direct_out_placeholders(spec)
  uops = emit_q4k_packed_prefill_kernel(spec)(out, words, x)
  assert uops.arg.name == spec.kernel_name
