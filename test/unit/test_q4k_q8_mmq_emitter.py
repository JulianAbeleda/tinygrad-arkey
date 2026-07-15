import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops

from extra.qk.q4k_q8_mmq_emitter import Q4KQ8MMQPrefillSpec, emit_q4k_q8_mmq_prefill


def test_mmq_spec_requires_explicit_geometry_and_serializes_it():
  spec = Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=16, n=32, k=256, tile_m=8, tile_n=16, tile_k=128)
  spec.validate()
  assert spec.to_json()["mmq"]["tile_m"] == 8
  with pytest.raises(ValueError, match="tile_k violates"):
    Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=16, n=32, k=256, tile_m=8, tile_n=16, tile_k=100).validate()


def test_mmq_emitter_builds_graph_without_dispatch():
  spec = Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=8, n=16, k=256, tile_m=8, tile_n=16, tile_k=256)
  words = Tensor.zeros((spec.n * spec.k // 256 * 36,), dtype=dtypes.uint32)
  xq = Tensor.zeros((spec.m, spec.k), dtype=dtypes.int8)
  scales = Tensor.ones((spec.m, spec.k // 32), dtype=dtypes.float32)
  out = emit_q4k_q8_mmq_prefill(words, xq, scales, spec)
  assert out.shape == (spec.m, spec.n)


def test_mmq_emitter_rejects_unowned_tails_and_bad_storage_geometry():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=16, n=32, k=512, tile_m=8, tile_n=16, tile_k=256)
  with pytest.raises(ValueError, match="divisible"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(32 * 2 * 36, dtype=dtypes.uint32),
                            Tensor.empty((16, 512), dtype=dtypes.int8),
                            Tensor.empty((16, 16), dtype=dtypes.float32),
                            spec.__class__(**{**spec.__dict__, "m": 17}))
  with pytest.raises(ValueError, match="words shape"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(36, dtype=dtypes.uint32),
                            Tensor.empty((16, 512), dtype=dtypes.int8),
                            Tensor.empty((16, 16), dtype=dtypes.float32), spec)


def test_mmq_emitter_compile_regression_uses_flat_abi_roots():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=8, n=16, k=256, tile_m=8, tile_n=16, tile_k=256)
  out = emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                                Tensor.empty((8, 256), dtype=dtypes.int8),
                                Tensor.empty((8, 8), dtype=dtypes.float32), spec)
  sink = out.uop.sink().replace(arg=KernelInfo(name="q4k_q8_mmq_flat_root_regression"))
  # The graph must be fully lowered from flat roots; a leaked Tensor view
  # appears as an INDEX UOp before the renderer's linearization pass.
  assert not any(u.op is Ops.INDEX for u in sink.toposort())


def test_tiled_mmq_scale_producer_uses_scalar_flat_pointer_load():
  from extra.qk.mmq_regression import reject_vector_pointer_bases
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=16, n=16, k=256, tile_m=16, tile_n=16, tile_k=256)
  out = emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                                Tensor.empty((16, 256), dtype=dtypes.int8),
                                Tensor.empty((16, 8), dtype=dtypes.float32), spec)
  reject_vector_pointer_bases(out.uop.sink())
