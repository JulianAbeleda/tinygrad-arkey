import pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import KernelInfo, Ops

from extra.qk.q4k_q8_mmq_emitter import MMQEmitterCandidate, Q4KQ8MMQPrefillSpec, emit_q4k_q8_mmq_prefill


def _candidate(spec, *, lifecycle="group", wmma=(16, 16, 16), **overrides):
  values = dict(spec=spec, wmma_m=wmma[0], wmma_n=wmma[1], wmma_k=wmma[2], lifecycle=lifecycle,
                output_layout=spec.output_layout, activation_layout=spec.activation_layout,
                tile_x_layout=spec.tile_x_layout, tile_y_layout=spec.tile_y_layout,
                staging_strategy=spec.staging_strategy, writeback_strategy=spec.writeback_strategy)
  values.update(overrides)
  return MMQEmitterCandidate(**values)

def _declared(spec, **kw):
  return _candidate(spec, wmma=(min(spec.m, 16), min(spec.n, 16), 16), **kw)


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
  out = emit_q4k_q8_mmq_prefill(words, xq, scales, _declared(spec))
  assert out.shape == (spec.m, spec.n)


def test_mmq_emitter_has_no_implicit_descriptor_fallback():
  spec = Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=8, n=16, k=256, tile_m=8, tile_n=16, tile_k=256)
  with pytest.raises(TypeError, match="requires an MMQEmitterCandidate"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                            Tensor.empty((8, 256), dtype=dtypes.int8),
                            Tensor.empty((8, 8), dtype=dtypes.float32), spec)

def test_mmq_emitter_rejects_raw_prefill_spec():
  spec = Q4KQ8MMQPrefillSpec(workload="test", profile="test", role="test", quant_format="Q4_K", activation_format="Q8_1", weight_layout="q4k", output_layout="tokens_rows", m=8, n=16, k=256, tile_m=8, tile_n=16, tile_k=256)
  with pytest.raises(TypeError, match="MMQEmitterCandidate"):
    emit_q4k_q8_mmq_prefill(Tensor.zeros((16 * 36,), dtype=dtypes.uint32),
                            Tensor.zeros((8, 256), dtype=dtypes.int8),
                            Tensor.ones((8, 8), dtype=dtypes.float32), spec)


def test_mmq_emitter_consumes_declared_lifecycle_and_geometry():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=16, n=16, k=256, tile_m=16, tile_n=16, tile_k=256)
  candidate = _candidate(spec, lifecycle="tiled")
  out = emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                                Tensor.empty((16, 256), dtype=dtypes.int8),
                                Tensor.empty((16, 8), dtype=dtypes.float32), candidate)
  assert out.shape == (16, 16)


def test_mmq_emitter_rejects_unlowered_candidate_mapping():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=16, n=16, k=256, tile_m=16, tile_n=16, tile_k=256)
  with pytest.raises(ValueError, match="layouts"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                            Tensor.empty((16, 256), dtype=dtypes.int8),
                            Tensor.empty((16, 8), dtype=dtypes.float32),
                            _candidate(spec, tile_x_layout="unsupported"))


def test_mmq_emitter_rejects_unowned_tails_and_bad_storage_geometry():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=16, n=32, k=512, tile_m=8, tile_n=16, tile_k=256)
  with pytest.raises(ValueError, match="divisible"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(32 * 2 * 36, dtype=dtypes.uint32),
                            Tensor.empty((16, 512), dtype=dtypes.int8),
                            Tensor.empty((16, 16), dtype=dtypes.float32),
                            _declared(spec.__class__(**{**spec.__dict__, "m": 17})))
  with pytest.raises(ValueError, match="words shape"):
    emit_q4k_q8_mmq_prefill(Tensor.empty(36, dtype=dtypes.uint32),
                            Tensor.empty((16, 512), dtype=dtypes.int8),
                            Tensor.empty((16, 16), dtype=dtypes.float32), _declared(spec))


def test_mmq_emitter_compile_regression_uses_flat_abi_roots():
  spec = Q4KQ8MMQPrefillSpec("test", "test", "test", "Q4_K", "Q8_1", "q4k", "tokens_rows",
                             m=8, n=16, k=256, tile_m=8, tile_n=16, tile_k=256)
  out = emit_q4k_q8_mmq_prefill(Tensor.empty(16 * 36, dtype=dtypes.uint32),
                                Tensor.empty((8, 256), dtype=dtypes.int8),
                                Tensor.empty((8, 8), dtype=dtypes.float32), _declared(spec))
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
                                Tensor.empty((16, 8), dtype=dtypes.float32), _declared(spec))
  reject_vector_pointer_bases(out.uop.sink())
