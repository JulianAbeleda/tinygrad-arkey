import os

import numpy as np
import pytest

from extra.qk.q4k_q8_activation_producer import AMD_NATIVE_VGPR_WAVE_REDUCE
from extra.qk.prefill import q4k_q8_five_buffer_pipeline as pipe
from extra.qk.prefill.q4k_q8_five_buffer_artifact import build_q4k_q8_five_buffer_artifact
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import AMD_ISA_TARGET
from extra.qk.runtime_specs import derive_q4k_q8_1_five_buffer_candidate
from test.unit.test_q4k_q8_five_buffer_compile_adapter import _payload
from tinygrad.uop.ops import Ops
from tinygrad.codegen import full_rewrite_to_sink
from tinygrad.codegen import to_program_cache
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad import Tensor, dtypes


def test_static_pipeline_builds_two_programs_with_shared_context():
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload())
  mmq_sink, admission = pipe.build_q4k_q8_five_buffer_sink(entry.payload, entry.canonical_identity)
  producer_sink, spec = pipe.build_physical_ds4_q8_producer(admission)
  assert producer_sink.arg.candidate_context is admission.context
  assert producer_sink.arg.candidate_context.canonical_identity == entry.canonical_identity
  assert (spec.m, spec.k) == (256, 4096)
  assert spec.wave_reduce_lowering == AMD_NATIVE_VGPR_WAVE_REDUCE
  assert {u.arg.slot for u in mmq_sink.toposort() if u.op is Ops.PARAM} == {0, 1, 2, 3, 4}


def test_static_pipeline_recompile_accepts_equal_cached_candidate_context():
  to_program_cache.clear()
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((16, 16, 256)))
  first = pipe.compile_q4k_q8_five_buffer_pipeline(entry.payload, entry.canonical_identity)
  second = pipe.compile_q4k_q8_five_buffer_pipeline(entry.payload, entry.canonical_identity)
  assert first.admission.context == second.admission.context
  assert first.admission.context is not second.admission.context
  assert second.producer.src[0].arg.candidate_context == second.admission.context
  assert second.mmq.src[0].arg.candidate_context == second.admission.context


def test_m256_producer_rangeifies_without_vector_weakint_cast():
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload())
  _, admission = pipe.build_q4k_q8_five_buffer_sink(entry.payload, entry.canonical_identity)
  sink, _ = pipe.build_physical_ds4_q8_producer(admission)
  lowered = full_rewrite_to_sink(sink, AMDISARenderer(Target.parse(AMD_ISA_TARGET)), optimize=True)
  assert any(u.op is Ops.STORE for u in lowered.toposort())


def test_cpu_execution_graph_is_two_programs_with_flat_physical_edges():
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((16, 16, 256)))
  graph = pipe.build_q4k_q8_five_buffer_execution(entry.payload, entry.canonical_identity,
    Tensor.empty(16 * 36, dtype=dtypes.uint32), Tensor.empty(16 * 256, dtype=dtypes.float32))
  assert graph.values.shape == (16 * 256,)
  assert graph.scales.shape == graph.sums.shape == (2 * 16 * 4,)
  sinks = [u for u in graph.output.schedule_linear().toposort() if u.op is Ops.SINK]
  assert len(sinks) == 2
  assert all(sink.arg.candidate_context is graph.candidate_context for sink in sinks)
  params = [{u.arg.slot: tuple(u.shape) for u in sink.toposort() if u.op is Ops.PARAM} for sink in sinks]
  assert any(row.get(2) == (16 * 256,) and row.get(3) == (2 * 16 * 4,) and
             row.get(4) == (2 * 16 * 4,) for row in params)


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_two_program_pipeline_16x16x256_amd_matches_independent_artifact_oracle():
  m = n = 16; k = 256
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((m, n, k)))
  artifact = build_q4k_q8_five_buffer_artifact(m, n, k, seed=23)
  source = np.zeros((m, k), dtype=np.float32)
  positions = np.asarray(artifact.metadata["selected_positions"], dtype=np.int64)
  coefficients = np.asarray(artifact.metadata["coefficients_fp32"], dtype=np.float32)
  source[np.arange(m), positions] = coefficients
  graph = pipe.build_q4k_q8_five_buffer_execution(entry.payload, entry.canonical_identity,
    Tensor(artifact.q4_packed_words, device="AMD"), Tensor(source.reshape(-1), device="AMD"))
  np.testing.assert_allclose(graph.output.numpy(), artifact.reference, rtol=3e-4, atol=3e-3)
