"""Static coordinator for the non-fitting Q4_K/Q8_1 physical-DS4 path.

This module is deliberately a coordinator, not a route selector: callers provide
one already selected candidate payload and identity.  Admission is performed once
and its context is attached to both the activation producer and MMQ PROGRAM.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Any

from extra.qk.q4k_q8_activation_producer import (AMD_NATIVE_VGPR_WAVE_REDUCE, PORTABLE_STAGED_WAVE_REDUCE,
  PhysicalDS4Q8ActivationSpec, emit_physical_ds4_q8_1_kernel)
from extra.qk.prefill.q4k_q8_five_buffer_compile_adapter import (
  AMD_ISA_TARGET, admitted_buffer_descriptors, build_q4k_q8_five_buffer_sink,
)
from extra.qk.runtime_specs import full_kernel_workload


@dataclass(frozen=True)
class Q4KQ8FiveBufferPipeline:
  """The two static programs and their single admitted candidate context."""
  producer: Any
  mmq: Any
  admission: Any

  @property
  def candidate_context(self):
    return self.admission.context


@dataclass(frozen=True)
class Q4KQ8FiveBufferExecution:
  """Lazy two-program graph.  The three producer outputs are the MMQ inputs."""
  output: Any
  values: Any
  scales: Any
  sums: Any
  admission: Any

  @property
  def candidate_context(self):
    return self.admission.context


def _compile_sink(sink, *, target: str, compile_environment: dict[str, Any]):
  from tinygrad.codegen import to_program
  from tinygrad.helpers import getenv, Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  old = {key: os.environ.get(key) for key in compile_environment}
  try:
    os.environ.update({key: str(value) for key, value in compile_environment.items()})
    getenv.cache_clear()
    program = to_program(sink, AMDISARenderer(Target.parse(target)))
  finally:
    for key, value in old.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
  programs = [u for u in program.toposort() if u.op.name == "PROGRAM"]
  if programs != [program]: raise RuntimeError("pipeline stage did not lower to exactly one PROGRAM")
  return program


def build_physical_ds4_q8_producer(admission):
  """Build the producer sink, using only the admitted workload M/K geometry."""
  workload = full_kernel_workload(admission.normalized_payload)
  m, _, k = workload.shape
  spec = PhysicalDS4Q8ActivationSpec(m, k, wave_reduce_lowering=AMD_NATIVE_VGPR_WAVE_REDUCE)
  spec.validate()
  from tinygrad import dtypes
  from tinygrad.uop.ops import KernelInfo, UOp
  sink = emit_physical_ds4_q8_1_kernel(spec)(
    UOp.placeholder((m * k,), dtypes.int8, 0),
    UOp.placeholder((spec.waves,), dtypes.float32, 1),
    UOp.placeholder((spec.waves,), dtypes.float32, 2),
    # Keep the logical producer contract [M,K], then flatten at the emitter
    # boundary.  The physical producer indexes one linear activation stream;
    # leaving the rank-2 UOp there makes AMD rangeify vectorize row constants
    # (and type_verify rejects the resulting weakint.vec casts for large M).
    UOp.placeholder((m, k), dtypes.float32, 3).reshape(-1),
  )
  return sink.replace(arg=KernelInfo(name=sink.arg.name, opts_to_apply=sink.arg.opts_to_apply,
                                      candidate_context=admission.context)), spec


def build_q4k_q8_five_buffer_execution(payload: dict[str, Any], canonical_identity: str,
                                        q4_packed_words, source) -> Q4KQ8FiveBufferExecution:
  """Build, but do not realize, the admitted producer -> MMQ graph.

  Admission is deliberately performed once.  Shapes and dtypes come only from
  that admission; no inventory scan, model name, or implicit route is consulted.
  """
  mmq_sink, admission = build_q4k_q8_five_buffer_sink(payload, canonical_identity)
  producer_sink, spec = build_physical_ds4_q8_producer(admission)
  descriptors = {row.name: row for row in admitted_buffer_descriptors(admission)}
  from tinygrad import Tensor, dtypes
  expected = descriptors["q4_packed_words"]
  if tuple(q4_packed_words.shape) != expected.flat_shape or q4_packed_words.dtype != expected.dtype:
    raise ValueError(f"q4_packed_words must be flat {expected.storage_dtype}{expected.flat_shape}")
  if tuple(source.shape) != (spec.m * spec.k,) or source.dtype != dtypes.float32:
    raise ValueError(f"source must be flat float32{(spec.m * spec.k,)}")
  if q4_packed_words.device != source.device: raise ValueError("pipeline inputs must share one device")

  runtime_spec = replace(spec, wave_reduce_lowering=PORTABLE_STAGED_WAVE_REDUCE)
  def producer(values, scales, sums, activation):
    sink = emit_physical_ds4_q8_1_kernel(runtime_spec)(values, scales, sums, activation)
    return sink.replace(arg=producer_sink.arg)

  value_desc, scale_desc, sum_desc = (descriptors[name] for name in
    ("q8_ds4_values", "q8_scales", "q8_weighted_sums"))
  values = Tensor.empty(value_desc.flat_shape, dtype=value_desc.dtype, device=source.device)
  scales = Tensor.empty(scale_desc.flat_shape, dtype=scale_desc.dtype, device=source.device)
  sums = Tensor.empty(sum_desc.flat_shape, dtype=sum_desc.dtype, device=source.device)
  values, scales, sums = values.custom_kernel(scales, sums, source, fxn=producer)[:3]

  # Do not reshape these edges: flat physical values/scales/sums are the ABI.
  output_desc = descriptors["output"]
  output = Tensor.empty(output_desc.flat_shape, dtype=output_desc.dtype, device=source.device)
  def mmq(out, words, flat_values, flat_scales, flat_sums):
    from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_role_sized_wmma, emit_q4k_q8_mmq_role_sized_wmma
    mmq_spec = describe_q4k_q8_mmq_role_sized_wmma(spec.m, output_desc.logical_shape[1], spec.k)
    sink = emit_q4k_q8_mmq_role_sized_wmma(mmq_spec)(out, words, flat_values, flat_scales, flat_sums)
    return sink.replace(arg=mmq_sink.arg)
  output = output.custom_kernel(q4_packed_words, values, scales, sums, fxn=mmq)[0]
  return Q4KQ8FiveBufferExecution(output, values, scales, sums, admission)


def compile_q4k_q8_five_buffer_pipeline(payload: dict[str, Any], canonical_identity: str,
                                        *, target: str = AMD_ISA_TARGET) -> Q4KQ8FiveBufferPipeline:
  """Compile producer and downstream MMQ statically; never allocate or dispatch."""
  if target != AMD_ISA_TARGET: raise ValueError(f"compile target drift: expected {AMD_ISA_TARGET}")
  mmq_sink, admission = build_q4k_q8_five_buffer_sink(payload, canonical_identity)
  producer_sink, _ = build_physical_ds4_q8_producer(admission)
  environment = admission.normalized_payload["schedule"]["compile_environment"]
  producer = _compile_sink(producer_sink, target=target, compile_environment=environment)
  mmq = _compile_sink(mmq_sink, target=target, compile_environment=environment)
  for program in (producer, mmq):
    # Lowering caches preserve the equivalent context object from the first compile.  Semantic identity is the frozen
    # descriptor value/canonical digest, not Python object identity across cache reconstruction.
    if getattr(program.src[0].arg, "candidate_context", None) != admission.context:
      raise RuntimeError("PROGRAM candidate context drift")
  if tuple(mmq.arg.globals) != (0, 1, 2, 3, 4):
    raise RuntimeError("MMQ five-buffer ABI drift")
  return Q4KQ8FiveBufferPipeline(producer, mmq, admission)


__all__ = ["Q4KQ8FiveBufferPipeline", "Q4KQ8FiveBufferExecution", "build_physical_ds4_q8_producer",
           "build_q4k_q8_five_buffer_execution", "compile_q4k_q8_five_buffer_pipeline"]
