"""Compile-only bridge from an admitted Q4_K/Q8_1 five-buffer payload to its UOp emitter."""
from __future__ import annotations

from math import prod
import os
from typing import Any, Mapping

from extra.qk.runtime_specs import (Q4K_Q8_1_FIVE_BUFFER_ABI, admit_full_kernel_candidate,
  full_kernel_candidate_capability, full_kernel_workload)

AMD_ISA_TARGET = "AMD:ISA:gfx1100"


def _immutable_json(value:Any) -> bool:
  return (isinstance(value, Mapping) and type(value) is not dict and
          all(isinstance(k, str) and _immutable_json(v) for k, v in value.items())) or \
         (isinstance(value, tuple) and all(_immutable_json(v) for v in value)) or \
         isinstance(value, (str, int, float, bool, type(None)))


def _plain_json(value:Any):
  if isinstance(value, Mapping): return {key:_plain_json(item) for key,item in value.items()}
  if isinstance(value, tuple): return [_plain_json(item) for item in value]
  return value


def admit_q4k_q8_five_buffer_compile(payload:dict[str, Any], canonical_identity:str):
  """Admit only the canonical, exact, aligned five-buffer workload."""
  workload = full_kernel_workload(payload)
  admission = admit_full_kernel_candidate(payload, canonical_identity, profile=workload.profile, role=workload.role,
    shape=workload.shape, target=workload.target, capability=full_kernel_candidate_capability(payload))
  if canonical_identity != admission.canonical_identity:
    raise ValueError("canonical identity drift: legacy aliases are not compile authority")
  plan = admission.operand_plan
  if plan is None or not _immutable_json(plan) or plan.get("family") != Q4K_Q8_1_FIVE_BUFFER_ABI:
    raise ValueError("immutable Q4_K/Q8_1 five-buffer operand_plan is required")
  if _plain_json(plan) != admission.normalized_payload.get("kernel_abi"):
    raise ValueError("admitted operand_plan differs from canonical kernel_abi")
  return admission


def _extent(expr:tuple, workload:Mapping[str,int], geometry:Mapping[str,int]) -> int:
  if not isinstance(expr, tuple) or not expr: raise ValueError("ABI axis extent must be a non-empty immutable expression")
  if expr[0] in ("workload", "block_geometry") and len(expr) == 2:
    source = workload if expr[0] == "workload" else geometry
    value = source.get(expr[1])
  elif expr[0] == "quotient" and len(expr) == 3:
    numerator, denominator = _extent(expr[1], workload, geometry), _extent(expr[2], workload, geometry)
    if denominator <= 0 or numerator % denominator: raise ValueError("ABI quotient extent is not exact")
    value = numerator // denominator
  else: raise ValueError(f"unsupported ABI axis extent expression {expr!r}")
  if not isinstance(value, int) or isinstance(value, bool) or value <= 0: raise ValueError("ABI axis extent must resolve to a positive int")
  return value


def _ordered_buffers(admission, dtypes):
  plan, workload = admission.operand_plan, full_kernel_workload(admission.normalized_payload)
  dimensions = dict(zip(("m", "n", "k"), workload.shape))
  rows = []
  for name, descriptor in plan["buffers"].items():
    axes, expressions = descriptor["logical_axes"], descriptor["axis_extents"]
    if len(axes) != len(expressions) or len(set(axes)) != len(axes): raise ValueError(f"ABI buffer {name!r} has invalid logical axes")
    logical_shape = tuple(_extent(expr, dimensions, plan["block_geometry"]) for expr in expressions)
    shape = logical_shape if descriptor["access"] == "logical" else (prod(logical_shape),)
    dtype = getattr(dtypes, descriptor["storage_dtype"], None)
    if dtype is None: raise ValueError(f"ABI buffer {name!r} has unknown storage dtype")
    rows.append((descriptor["abi_slot"], descriptor["direction"], dtype, shape))
  rows.sort(key=lambda row:row[0])
  if [row[0] for row in rows] != list(range(len(rows))): raise ValueError("ABI slots must be unique and contiguous from zero")
  return tuple(rows)


def build_q4k_q8_five_buffer_sink(payload:dict[str, Any], canonical_identity:str):
  """Construct the sole physical-DS4 emitter sink without allocating or compiling."""
  admission = admit_q4k_q8_five_buffer_compile(payload, canonical_identity)
  m, n, k = full_kernel_workload(admission.normalized_payload).shape
  from tinygrad import dtypes
  from tinygrad.uop.ops import KernelInfo, UOp
  from extra.qk.q4k_q8_mmq_uop import describe_q4k_q8_mmq_role_sized_wmma, emit_q4k_q8_mmq_role_sized_wmma
  spec = describe_q4k_q8_mmq_role_sized_wmma(m, n, k)
  buffers = _ordered_buffers(admission, dtypes)
  sink = emit_q4k_q8_mmq_role_sized_wmma(spec)(*(UOp.placeholder(shape, dtype, slot) for slot, _, dtype, shape in buffers))
  return sink.replace(arg=KernelInfo(name=spec.name, opts_to_apply=sink.arg.opts_to_apply,
    candidate_context=admission.context)), admission


def compile_q4k_q8_five_buffer_program(payload:dict[str, Any], canonical_identity:str, *, target:str=AMD_ISA_TARGET):
  """Compile exactly one identity-bound PROGRAM for the fixed five-buffer ABI."""
  if target != AMD_ISA_TARGET: raise ValueError(f"compile target drift: expected {AMD_ISA_TARGET}")
  sink, admission = build_q4k_q8_five_buffer_sink(payload, canonical_identity)
  from tinygrad.codegen import to_program
  from tinygrad import dtypes
  from tinygrad.helpers import getenv, Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.uop.ops import Ops
  compile_environment = admission.normalized_payload["schedule"]["compile_environment"]
  old_environment = {key:os.environ.get(key) for key in compile_environment}
  try:
    os.environ.update({key:str(value) for key,value in compile_environment.items()}); getenv.cache_clear()
    program = to_program(sink, AMDISARenderer(Target.parse(target)))
  finally:
    for key,value in old_environment.items():
      if value is None: os.environ.pop(key, None)
      else: os.environ[key] = value
    getenv.cache_clear()
  programs = [u for u in program.toposort() if u.op is Ops.PROGRAM]
  if programs != [program]: raise RuntimeError(f"expected exactly one final PROGRAM, found {len(programs)}")
  if len(program.src) != 5 or program.src[3].op is not Ops.SOURCE or program.src[4].op is not Ops.BINARY:
    raise RuntimeError("final PROGRAM lost source or binary")
  context = getattr(program.src[0].arg, "candidate_context", None)
  if context is not admission.context or context.canonical_identity != admission.canonical_identity:
    raise RuntimeError("final PROGRAM candidate identity drift")
  buffers = _ordered_buffers(admission, dtypes)
  params = {u.arg.slot:u.dtype.base for u in program.src[0].toposort() if u.op is Ops.PARAM}
  expected = {slot:dtype for slot, _, dtype, _ in buffers}
  globals_ = tuple(slot for slot, _, _, _ in buffers)
  outs = tuple(slot for slot, direction, _, _ in buffers if direction == "out")
  ins = tuple(slot for slot, direction, _, _ in buffers if direction == "in")
  if params != expected or tuple(program.arg.globals) != globals_ or tuple(program.arg.outs) != outs or tuple(program.arg.ins) != ins:
    raise RuntimeError("final PROGRAM ABI mismatch")
  if program.arg.name != sink.arg.name: raise RuntimeError("final PROGRAM kernel identity drift")
  return program, admission


__all__ = ["AMD_ISA_TARGET", "admit_q4k_q8_five_buffer_compile",
  "build_q4k_q8_five_buffer_sink", "compile_q4k_q8_five_buffer_program"]
