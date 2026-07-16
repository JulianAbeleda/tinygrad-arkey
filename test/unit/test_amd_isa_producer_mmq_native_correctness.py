import os
import hashlib

import numpy as np
import pytest

from extra.qk.prefill.q4k_q8_five_buffer_artifact import build_q4k_q8_five_buffer_artifact
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import admitted_buffer_descriptors
from extra.qk.prefill.q4k_q8_five_buffer_pipeline import compile_q4k_q8_five_buffer_pipeline
from extra.qk.runtime_specs import derive_q4k_q8_1_five_buffer_candidate
from test.unit.test_q4k_q8_five_buffer_compile_adapter import _payload
from tinygrad.device import Buffer
from tinygrad.runtime.bridge import prepare_executable
from tinygrad.uop.ops import Ops


@pytest.mark.skipif(not os.path.exists("/dev/kfd"), reason="AMD KFD is unavailable")
def test_native_amd_isa_physical_ds4_producer_to_mmq_matches_cpu_oracle():
  """Dispatch the two static AMD:ISA programs, never the portable lazy Tensor graph."""
  m = n = 16
  k = 256
  entry = derive_q4k_q8_1_five_buffer_candidate(_payload((m, n, k), role="native_correctness"))
  artifact = build_q4k_q8_five_buffer_artifact(m, n, k, seed=31)

  activation = np.zeros((m, k), dtype=np.float32)
  positions = np.asarray(artifact.metadata["selected_positions"], dtype=np.int64)
  coefficients = np.asarray(artifact.metadata["coefficients_fp32"], dtype=np.float32)
  activation[np.arange(m), positions] = coefficients

  pipeline = compile_q4k_q8_five_buffer_pipeline(entry.payload, entry.canonical_identity)
  assert all(program.src[1].op is Ops.DEVICE and program.src[1].arg == "AMD" for program in (pipeline.producer, pipeline.mmq))
  descriptors = {row.name: row for row in admitted_buffer_descriptors(pipeline.admission)}

  def executable(program):
    binary = next(u.arg for u in program.src if u.op is Ops.BINARY)
    return prepare_executable(program, {"passed": True, "binary_sha256": hashlib.sha256(binary).hexdigest()})

  buffers = {name: Buffer("AMD", int(np.prod(row.flat_shape)), row.dtype, preallocate=True)
             for name, row in descriptors.items()}
  activation_buffer = Buffer("AMD", activation.size, descriptors["output"].dtype, preallocate=True)
  buffers["q4_packed_words"].copyin(memoryview(artifact.q4_packed_words))
  activation_buffer.copyin(memoryview(np.ascontiguousarray(activation.reshape(-1))))
  producer, mmq = executable(pipeline.producer), executable(pipeline.mmq)
  try:
    producer.dispatch(*(buffer.get_buf("AMD") for buffer in
      (buffers["q8_ds4_values"], buffers["q8_scales"], buffers["q8_weighted_sums"], activation_buffer)))
    mmq.dispatch(*(buffer.get_buf("AMD") for buffer in
      (buffers["output"], buffers["q4_packed_words"], buffers["q8_ds4_values"],
       buffers["q8_scales"], buffers["q8_weighted_sums"])))
    got = buffers["output"].numpy().reshape(m, n)
  finally:
    producer.close()
    mmq.close()
    activation_buffer.deallocate()
    for buffer in buffers.values(): buffer.deallocate()
  np.testing.assert_allclose(got, artifact.reference, rtol=3e-4, atol=3e-3)
