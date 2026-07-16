import pytest
from tinygrad import Tensor, dtypes

from extra.qk.q4k_fused_mmq import emit_fused_q4k_mmq_tile, fused_q4k_mmq_admitted
from extra.qk.q4k_fused_mmq_contract import FUSED_Q4K_MMQ_CONTRACT, QWEN3_14B_FUSED_ROLE_SHAPES, FusedQ4KMMQTileSpec
from extra.qk.prefill_int8_wmma_spec import build_fused_q4k_mmq_dynamic_owner
from tinygrad.uop.ops import Ops


def test_fused_q4k_contract_is_bounded_and_typed():
  spec = FusedQ4KMMQTileSpec()
  assert FUSED_Q4K_MMQ_CONTRACT.endswith("v2")
  assert spec.words_shape == (576,) and spec.xq_shape == (16, 256) and spec.xscales_shape == (16, 8)
  assert spec.compiler_geometry().lds_windows[0].role == "A"


def test_fused_q4k_admission_is_fail_closed_for_larger_tiles():
  with pytest.raises(NotImplementedError): FusedQ4KMMQTileSpec(m=32).validate()
  with pytest.raises(NotImplementedError): FusedQ4KMMQTileSpec(extension="large_tile").validate()
  assert not fused_q4k_mmq_admitted(compile_evidence=True)
  assert not fused_q4k_mmq_admitted(correctness_evidence=True)
  assert fused_q4k_mmq_admitted(compile_evidence=True, correctness_evidence=True)
  assert not fused_q4k_mmq_admitted(compile_evidence=True, correctness_evidence=True,
                                    spec=FusedQ4KMMQTileSpec(m=512, n=1024, k=4096))


@pytest.mark.parametrize("m,n,k", QWEN3_14B_FUSED_ROLE_SHAPES)
def test_fused_q4k_admits_only_real_14b_role_shapes_with_bounded_raw(m, n, k):
  spec = FusedQ4KMMQTileSpec(m=m, n=n, k=k)
  spec.validate()
  assert spec.live_raw_elems == 16 * 16 * 8
  assert spec.words_shape == (n * (k // 256) * 36,)


def test_fused_q4k_zero_tile_correctness_on_tensor_pipeline():
  spec = FusedQ4KMMQTileSpec()
  out = emit_fused_q4k_mmq_tile(Tensor.zeros(spec.words_shape, dtype=dtypes.uint32),
    Tensor.zeros(spec.xq_shape, dtype=dtypes.int8), Tensor.zeros(spec.xscales_shape, dtype=dtypes.float32)).realize()
  assert out.shape == (16, 16) and (out.numpy() == 0).all()


def test_dynamic_fused_q4_owner_is_one_two_tile_graph():
  spec = FusedQ4KMMQTileSpec()
  # Empty buffers keep this a compile/ownership proof; no host tile loop and
  # no device-specific route or backend mutation are involved.
  graph = build_fused_q4k_mmq_dynamic_owner(
    Tensor.empty(2 * spec.words_shape[0], dtype=dtypes.uint32),
    Tensor.empty(2 * spec.xq_shape[0] * spec.xq_shape[1], dtype=dtypes.int8),
    Tensor.empty(2 * spec.xscales_shape[0] * spec.xscales_shape[1], dtype=dtypes.float32),
    Tensor.empty(2 * spec.m * spec.n, dtype=dtypes.float32))
  nodes = graph.toposort()
  assert any(u.op is Ops.RANGE and u.arg[0] == 9600 for u in nodes)
  assert any(u.op is Ops.STORE for u in nodes)
  assert any(u.op is Ops.INDEX for u in nodes)
