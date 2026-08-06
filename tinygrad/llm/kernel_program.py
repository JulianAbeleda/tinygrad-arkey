"""Typed execution boundary for explicit LLM kernel programs.

The boundary carries declared provenance; it never infers provenance from a
route, program, or emitter name.  ``Tensor.uop_program`` remains the single
low-level transport used by these explicit program paths.

M5 typed boundary (docs/task_workflow/input/m5-variant-reopen-boundary-p0-scope-20260803.md):
the fp16 flash combine output may declare a typed layout, and the o-proj attn_qo Q4K GEMV
consumer may opt in to a typed input ABI so its activation prelude
``x[:, 0, :].reshape(K).cast(fp16).contiguous()`` folds to a zero-copy view of the combine
AFTER. The ABI is closed-default: the opt-in is issued only by that one consumer, and the
fold is accepted only when a fail-closed validator proves every requirement of the scope's
section 3.3/4. The generic flat-buffer ABI and ``UOp.custom_kernel``'s default
preserve-or-materialize rule are unchanged.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from enum import StrEnum
from functools import reduce
from operator import mul
from typing import Callable

from tinygrad import Tensor
from tinygrad.dtype import DType, dtypes
from tinygrad.helpers import getenv, prod
from tinygrad.uop import GroupOp, Ops
from tinygrad.uop.ops import UOp, graph_rewrite


def _prod(shape: tuple[int, ...]) -> int:
  return reduce(mul, shape, 1)


class KernelProgramProvenance(StrEnum):
  MACHINE_SEARCH_GENERATED = "machine_search_generated"
  TINYGRAD_SCHEDULER_GENERATED = "tinygrad_scheduler_generated"
  HAND_AUTHORED_ORACLE = "hand_authored_oracle"
  RESEARCH_ONLY = "research_only"


@dataclass(frozen=True)
class OutputSpec:
  """The kernel artifact's output ABI: the shape and dtype the emitted program writes."""
  shape: tuple[int, ...]
  dtype: DType
  typed_output: DeclaredTypedOutput | None = None

  def __post_init__(self):
    if not isinstance(self.shape, tuple) or not self.shape or not all(
        isinstance(dim, int) and dim > 0 for dim in self.shape):
      raise ValueError("output spec shape must be a non-empty tuple of positive ints")
    if not isinstance(self.dtype, DType):
      raise ValueError("output spec dtype must be a DType")
    if self.typed_output is not None and not isinstance(self.typed_output, DeclaredTypedOutput):
      raise ValueError("output spec typed_output must be a DeclaredTypedOutput")
    if self.typed_output is not None and (self.typed_output.layout.flat_shape != self.shape or
                                          self.typed_output.layout.dtype != self.dtype):
      raise ValueError("output spec typed_output must match the output spec shape and dtype")


@dataclass(frozen=True)
class TypedLayout:
  """Producer-declared output layout: a contiguous row-major flat buffer, optionally viewable
  as ``logical_shape``. Boundary ABI metadata only; the emitted kernel is unchanged. The
  dataclass has no fields for permutation, stride, or padding, so a declaration can never
  express them -- fail-closed by construction (scope section 3.3/4(a))."""
  dtype: DType
  flat_shape: tuple[int, ...]
  logical_shape: tuple[int, ...] | None = None
  row_major: bool = True

  def __post_init__(self):
    if not isinstance(self.dtype, DType):
      raise ValueError("typed layout dtype must be a DType")
    if not isinstance(self.flat_shape, tuple) or not self.flat_shape or not all(
        isinstance(dim, int) and dim > 0 for dim in self.flat_shape):
      raise ValueError("typed layout flat_shape must be a non-empty tuple of positive ints")
    if self.logical_shape is not None:
      if not isinstance(self.logical_shape, tuple) or not self.logical_shape or not all(
          isinstance(dim, int) and dim > 0 for dim in self.logical_shape):
        raise ValueError("typed layout logical_shape must be a non-empty tuple of positive ints")
      if _prod(self.flat_shape) != _prod(self.logical_shape):
        raise ValueError("typed layout flat and logical shapes must have equal numel")
    if self.row_major is not True:
      raise ValueError("only row-major typed layouts exist")


@dataclass(frozen=True)
class DeclaredTypedOutput:
  """Producer-side declaration: the program's typed output layout plus the M5 combine-fusion
  gate state recorded at production time. The consumer validator requires both (scope 4(d))."""
  layout: TypedLayout
  combine_fusion_admitted: bool

  def __post_init__(self):
    if not isinstance(self.layout, TypedLayout):
      raise TypeError("declared typed output layout must be a TypedLayout")
    if not isinstance(self.combine_fusion_admitted, bool):
      raise TypeError("declared typed output combine_fusion_admitted must be a bool")


@dataclass(frozen=True)
class TypedViewRequest:
  """Consumer typed-input ABI opt-in for one input slot: accept a declared view of an opaque
  program output instead of a materialized contiguous buffer. Closed default: only the o-proj
  attn_qo Q4K GEMV consumer (decode_routes.py) ever issues one; ffn_down, attn_kv, and every
  other route keep the generic flat-buffer input ABI (scope section 3.2)."""
  slot: int
  dtype: DType
  flat_shape: tuple[int, ...]
  route_role: str
  requires_combine_fusion: bool = True

  def __post_init__(self):
    if not isinstance(self.slot, int) or self.slot < 0:
      raise ValueError("typed view request slot must be a non-negative int")
    if not isinstance(self.dtype, DType):
      raise ValueError("typed view request dtype must be a DType")
    if not isinstance(self.flat_shape, tuple) or not self.flat_shape or not all(
        isinstance(dim, int) and dim > 0 for dim in self.flat_shape):
      raise ValueError("typed view request flat_shape must be a non-empty tuple of positive ints")
    if not isinstance(self.route_role, str) or not self.route_role:
      raise ValueError("typed view request route_role must be a non-empty string")
    if not isinstance(self.requires_combine_fusion, bool):
      raise TypeError("typed view request requires_combine_fusion must be a bool")


@dataclass(frozen=True)
class ResidualViewRequest:
  """M4 residual-slot typed-input opt-in (m4-resadd-landing-scope-20260806.md section 2.2): accept a
  zero-copy view of the ORDINARY block-output producer for the o-proj residual_add slot instead of the
  materialized flat-buffer copy. Deliberately distinct from TypedViewRequest so the M5 combine ABI is
  untouched: the residual producer is an ordinary precompiled-function output (or a buffer), not an
  opaque AFTER with a declared typed output. Closed default: only the attn_qo residual_add branch of
  decode_routes.py ever issues one."""
  slot: int
  dtype: DType
  flat_shape: tuple[int, ...]
  route_role: str
  kind: str = "residual_add"

  def __post_init__(self):
    if not isinstance(self.slot, int) or self.slot < 0:
      raise ValueError("residual view request slot must be a non-negative int")
    if not isinstance(self.dtype, DType):
      raise ValueError("residual view request dtype must be a DType")
    if not isinstance(self.flat_shape, tuple) or not self.flat_shape or not all(
        isinstance(dim, int) and dim > 0 for dim in self.flat_shape):
      raise ValueError("residual view request flat_shape must be a non-empty tuple of positive ints")
    if not isinstance(self.route_role, str) or not self.route_role:
      raise ValueError("residual view request route_role must be a non-empty string")
    if not isinstance(self.kind, str) or not self.kind:
      raise ValueError("residual view request kind must be a non-empty string")


@dataclass(frozen=True)
class KernelProgram:
  route_id: str
  program_id: str
  provenance: KernelProgramProvenance
  emitter: Callable
  output_spec: OutputSpec | None = None
  typed_input_views: tuple[TypedViewRequest, ...] = ()
  residual_input_views: tuple[ResidualViewRequest, ...] = ()

  def __post_init__(self):
    for name, value in (("route_id", self.route_id), ("program_id", self.program_id)):
      if not isinstance(value, str) or not value:
        raise ValueError(f"kernel program {name} must be a non-empty string")
    if not isinstance(self.provenance, KernelProgramProvenance):
      raise ValueError("kernel program provenance must be a KernelProgramProvenance")
    if not callable(self.emitter): raise ValueError("kernel program emitter must be callable")
    if self.output_spec is not None and not isinstance(self.output_spec, OutputSpec):
      raise ValueError("kernel program output_spec must be an OutputSpec")
    if not isinstance(self.typed_input_views, tuple) or not all(
        isinstance(request, TypedViewRequest) for request in self.typed_input_views):
      raise ValueError("kernel program typed_input_views must be a tuple of TypedViewRequest")
    slots = tuple(request.slot for request in self.typed_input_views)
    if len(slots) != len(set(slots)):
      raise ValueError("kernel program typed_input_views slots must be unique")
    if not isinstance(self.residual_input_views, tuple) or not all(
        isinstance(request, ResidualViewRequest) for request in self.residual_input_views):
      raise ValueError("kernel program residual_input_views must be a tuple of ResidualViewRequest")
    rslots = tuple(request.slot for request in self.residual_input_views)
    if len(rslots) != len(set(rslots)):
      raise ValueError("kernel program residual_input_views slots must be unique")
    if set(rslots) & set(slots):
      raise ValueError("kernel program typed and residual input view slots must not overlap")

  def to_dict(self) -> dict[str, str]:
    return {"route_id": self.route_id, "program_id": self.program_id, "provenance": self.provenance.value}


# UOps are interned and identity-hashable; the Tensor graph keeps every live AFTER alive, so a
# weak table prevents completed graphs from retaining declarations (same pattern as
# UOp._memory_semantic_owners).
_DECLARED_TYPED_OUTPUTS: weakref.WeakKeyDictionary[UOp, DeclaredTypedOutput] = weakref.WeakKeyDictionary()


def _is_pure_contiguous_view(chain: UOp) -> bool:
  """True when the movement chain maps the flat index identically (offset 0, no permutation,
  no stride, no padding). This is the mechanical "view of the AFTER" proof the validator uses;
  an identity permute of size-1 dims passes, a data-moving permute or offset view does not."""
  from tinygrad.schedule.rangeify import pm_mops
  from tinygrad.uop.symbolic import symbolic
  numel = chain.numel()
  rewritten = graph_rewrite(chain.index(UOp.range(numel, 0)), pm_mops + symbolic,
                            name="m5_typed_view_identity")
  return rewritten.op is Ops.INDEX and rewritten.src[1].op is Ops.RANGE


def _cancel_lossless_fp16_roundtrip(chain: UOp) -> tuple[UOp | None, str]:
  """Cancel a lossless fp16->fp32->fp16 cast pair over pure movement legs (the real
  o-proj chain: model.py:704 upcasts the fp16 combine output to fp32 because decode q is
  fp32, then the prelude casts back). fp16->fp32->fp16 is exact for every fp16 value, so
  the composed fp16 movement chain is a view of the AFTER. Returns ``(chain, None)`` on
  success, else ``(None, reason)``; a lossy shape (bf16 round trips, arithmetic between
  the casts, non-identity movement) rejects back to the generic flat-buffer ABI."""
  if chain.op is not Ops.CAST or chain.dtype != dtypes.float16: return None, "request chain is not an fp16 cast"
  outer, base_cast = chain.src[0], chain.src[0].base
  if base_cast.op is not Ops.CAST or base_cast.dtype != dtypes.float32: return None, "outer view leg does not bottom out at an fp32 cast"
  inner = base_cast.src[0]
  if inner.dtype != dtypes.float16 or inner.base.op is not Ops.AFTER or inner.base.dtype != dtypes.float16:
    return None, "inner view leg is not an fp16 view of an fp16 AFTER"
  ops: list[tuple[Ops, object, tuple[UOp, ...]]] = []
  while outer.op in GroupOp.Movement: ops.append((outer.op, outer.arg, outer.src[1:])); outer = outer.src[0]
  if outer is not base_cast: return None, "outer view leg is not a pure movement chain"
  return reduce(lambda cur, op: UOp(op[0], cur.dtype, (cur,) + op[2], op[1]), reversed(ops), inner), None


def _validated_typed_view(uop: UOp, request: TypedViewRequest, program: KernelProgram) -> tuple[UOp | None, str]:
  """M5 typed-boundary fail-closed validator (scope sections 3.3 and 4). Returns the producer
  AFTER node when every check passes, else ``(None, reason)``; a rejection keeps the generic
  flat-buffer ABI, which is byte-identical (the consumer simply materializes as before)."""
  # (b) the request must be a contiguous() request over a movement-only view chain
  if uop.op is not Ops.CONTIGUOUS: return None, "request is not a contiguous() request"
  # (b-cont) cancel a lossless fp16->fp32->fp16 cast pair over pure movement legs. The
  # pair is exact for every fp16 value; it exists only because decode q is fp32, so the
  # model upcasts the fp16 combine output at model.py:704 before the consumer casts back.
  # On success the remaining chain is the composed fp16 view of the AFTER and every
  # subsequent check runs unchanged on it; on failure the original chain (with its real
  # cast) is kept, which the checks below reject as not a pure fp16 view of the AFTER.
  cancelled = _cancel_lossless_fp16_roundtrip(uop.src[0])[0]
  chain = cancelled if cancelled is not None else uop.src[0]
  if chain.base.op is not Ops.AFTER: return None, "view base is not an opaque program AFTER"
  # (a) the request chain must be a pure row-major view: dtype/numel preserved, no permute/stride/pad
  if uop.dtype != request.dtype: return None, "request dtype does not match the view request"
  if uop.shape != request.flat_shape: return None, "request flat shape does not match the view request"
  if chain.dtype != uop.dtype or chain.base.dtype != uop.dtype:
    return None, "view dtype is not preserved through the request chain"
  if not _is_pure_contiguous_view(chain): return None, "view is not a contiguous offset-0 reshape"
  # producer declaration must exist and exactly match the request (a), and the gate must be open (d)
  declared = _DECLARED_TYPED_OUTPUTS.get(chain.base)
  if declared is None: return None, "producer declared no typed output layout"
  if declared.layout.dtype != request.dtype or declared.layout.flat_shape != request.flat_shape:
    return None, "producer declared layout does not exactly match the requested view"
  if not declared.layout.row_major: return None, "producer declared layout is not row-major"
  if not declared.combine_fusion_admitted: return None, "combine-fusion gate is closed"
  # (d) typed-ABI gate: the consumer must explicitly require the M5 combine contract
  if not request.requires_combine_fusion: return None, "typed-ABI gate is closed"
  # (c) single intended consumer: route_role attn_qo on the o-proj Q4K GEMV program
  if request.route_role != "attn_qo": return None, f"wrong consumer route_role {request.route_role!r}"
  if not program.route_id.startswith("decode_q4k") or not program.program_id.endswith(".gemv"):
    return None, "program is not a q4k GEMV consumer"
  return chain.base, "ok"


def _fold_typed_input_views(inputs: tuple[Tensor, ...], program: KernelProgram) -> tuple[Tensor, ...]:
  """Apply the consumer's typed-input ABI opt-ins. Each accepted request substitutes the
  producer AFTER (a view) for the materialized contiguous request; a rejection leaves the
  generic flat-buffer input untouched."""
  if not program.typed_input_views: return inputs
  new_inputs = list(inputs)
  for request in program.typed_input_views:
    if not 0 <= request.slot < len(inputs):
      if getenv("M5_TYPED_BOUNDARY_DEBUG"):
        print(f"M5_TYPED_BOUNDARY_DEBUG {program.route_id}/{program.program_id} "
              f"slot={request.slot} out of range, no fold")
      continue
    view, reason = _validated_typed_view(inputs[request.slot].uop, request, program)
    if view is not None:
      new_inputs[request.slot] = Tensor(view)
    elif getenv("M5_TYPED_BOUNDARY_DEBUG"):
      print(f"M5_TYPED_BOUNDARY_DEBUG {program.route_id}/{program.program_id} "
            f"slot={request.slot} rejected: {reason}")
  return tuple(new_inputs)


# Ops that never move data by themselves: CONTIGUOUS over a pure view is an identity physical
# layout, MEMORY_SEMANTIC is a role marker. Both are transparent to the residual fold proof.
_RESIDUAL_TRANSPARENT = frozenset((Ops.CONTIGUOUS, Ops.MEMORY_SEMANTIC))
_RESIDUAL_MOVEMENT = frozenset((Ops.RESHAPE, Ops.SLICE, Ops.PERMUTE, Ops.EXPAND))


def _is_pure_residual_view(chain: UOp) -> bool:
  """Probe purity proof (m4_residual_boundary_fold_probe): every leg from the request down to the
  producer base must be a pure view. RESHAPE is identity on the flat index by construction; SLICE
  must be offset-0; PERMUTE and EXPAND must be identity; CONTIGUOUS and MEMORY_SEMANTIC are
  transparent physical/role markers. dtype is preserved through every leg. This is intentionally
  structural: the M5 rewrite cannot see through the precompiled GETTUPLE boundary, so a rewrite-only
  proof would wrongly reject the real chain."""
  cur = chain
  while cur.op in _RESIDUAL_TRANSPARENT | _RESIDUAL_MOVEMENT:
    if cur.op is Ops.SLICE:
      offsets, _ = cur.arg
      if any(o != 0 for o in offsets): return False
    if cur.op is Ops.PERMUTE and tuple(cur.arg) != tuple(range(len(cur.shape))): return False
    if cur.op is Ops.EXPAND and tuple(cur.arg) != tuple(cur.shape): return False
    if cur.dtype is not chain.dtype: return False
    cur = cur.src[0]
  return True


def _residual_producer_identity(base: UOp) -> bool:
  """The residual contract's base rule: an ordinary producer with a concrete buffer identity
  (BUFFER/SLICE/PARAM, or a precompiled function output - the real block boundary), or an AFTER
  with a declared typed output (the M5 path, kept intact)."""
  if base.op is Ops.AFTER:
    return _DECLARED_TYPED_OUTPUTS.get(base) is not None
  if base.op is Ops.CONTIGUOUS:
    return base.src[0].has_precompiled_output_identity() or base.src[0].has_buffer_identity() or base.src[0].op is Ops.AFTER
  return base.has_buffer_identity() or base.has_precompiled_output_identity()


def _validated_residual_view(uop: UOp, request: ResidualViewRequest, program: KernelProgram) -> tuple[UOp | None, str]:
  """M4 residual-slot fail-closed validator (m4-resadd-landing-scope section 2.2, probe-1 contract).
  Returns ``(base_uop, "ok")`` when the residual chain is a pure offset-0 view of an ordinary
  producer and the residual_add opt-in is exact; else ``(None, reason)``. Every failure keeps the
  generic flat-buffer ABI (the boundary copy)."""
  # (a) residual-slot opt-in: slot 2, attn_qo, residual_add, q4k GEMV consumer
  if request.slot != 2: return None, "not the residual slot"
  if request.route_role != "attn_qo": return None, f"wrong consumer route_role {request.route_role!r}"
  if request.kind != "residual_add": return None, f"wrong epilogue kind {request.kind!r}"
  if not program.route_id.startswith("decode_q4k") or not program.program_id.endswith(".gemv"):
    return None, "program is not a q4k GEMV consumer"
  # (b) request chain is a movement-only view: dtype/numel preserved through every leg
  chain = uop.src[0] if uop.op is Ops.CONTIGUOUS else uop
  if chain.dtype is not request.dtype: return None, "request dtype mismatch"
  if chain.numel() != prod(request.flat_shape): return None, "request numel mismatch"
  if chain.dtype is not uop.dtype: return None, "view dtype is not preserved through the request chain"
  # (c) purity: offset-0 row-major identity view (transparent legs stripped, structural proof)
  if not _is_pure_residual_view(chain): return None, "view is not a contiguous offset-0 reshape"
  # (d) producer: ordinary producer with buffer identity (or declared typed output)
  base = chain.base
  if not _residual_producer_identity(base): return None, "producer has no buffer/precompiled-output identity"
  return base, "ok"


def _fold_residual_input_views(inputs: tuple[Tensor, ...], program: KernelProgram) -> tuple[Tensor, ...]:
  """Apply the residual-slot typed-input opt-ins. Each accepted request substitutes the validated
  producer base (a zero-copy view) for the materialized request; a rejection leaves the generic
  flat-buffer input untouched."""
  if not program.residual_input_views: return inputs
  new_inputs = list(inputs)
  for request in program.residual_input_views:
    if not 0 <= request.slot < len(inputs):
      continue
    view, reason = _validated_residual_view(inputs[request.slot].uop, request, program)
    if view is not None:
      new_inputs[request.slot] = Tensor(view)
    elif getenv("M4_RESADD_BOUNDARY_DEBUG"):
      print(f"M4_RESADD_BOUNDARY_DEBUG {program.route_id}/{program.program_id} "
            f"slot={request.slot} rejected: {reason}")
  return tuple(new_inputs)


def _execute_outputs(output: Tensor | None, inputs: tuple[Tensor, ...], program: KernelProgram,
                     allowed: frozenset[KernelProgramProvenance], boundary: str) -> tuple[Tensor, ...]:
  if program.provenance not in allowed:
    allowed_values = ", ".join(sorted(provenance.value for provenance in allowed))
    raise ValueError(f"{boundary} does not accept {program.provenance.value}; expected one of: {allowed_values}")
  if output is None:
    if program.output_spec is None:
      raise ValueError(f"{boundary} requires an output tensor or a program output_spec")
    if not inputs:
      raise ValueError(f"{boundary} cannot allocate program output without inputs")
    output = Tensor.empty(*program.output_spec.shape, dtype=program.output_spec.dtype, device=inputs[0].device)
  inputs = _fold_typed_input_views(inputs, program)
  inputs = _fold_residual_input_views(inputs, program)
  results = output.uop_program(*inputs, fxn=program.emitter)
  # record the producer's typed output declaration against the AFTER the program produced, so a
  # later consumer's validator can prove the declared layout (scope section 3.1/4(a)).
  if program.output_spec is not None and program.output_spec.typed_output is not None:
    for result in results:
      _DECLARED_TYPED_OUTPUTS[result.uop] = program.output_spec.typed_output
  return results


def _execute(output: Tensor | None, inputs: tuple[Tensor, ...], program: KernelProgram,
             allowed: frozenset[KernelProgramProvenance], boundary: str) -> Tensor:
  return _execute_outputs(output, inputs, program, allowed, boundary)[0]


def execute_promoted_program(output: Tensor | None = None, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.MACHINE_SEARCH_GENERATED,
    KernelProgramProvenance.TINYGRAD_SCHEDULER_GENERATED)), "execute_promoted_program")


def execute_oracle_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.HAND_AUTHORED_ORACLE,)), "execute_oracle_program")


def execute_research_program(output: Tensor, *inputs: Tensor, program: KernelProgram) -> Tensor:
  return _execute(output, inputs, program, frozenset((KernelProgramProvenance.RESEARCH_ONLY,)), "execute_research_program")


def execute_research_program_outputs(output: Tensor, *inputs: Tensor, program: KernelProgram) -> tuple[Tensor, ...]:
  return _execute_outputs(output, inputs, program, frozenset((KernelProgramProvenance.RESEARCH_ONLY,)),
                          "execute_research_program_outputs")


__all__ = ["DeclaredTypedOutput", "KernelProgram", "KernelProgramProvenance", "OutputSpec", "TypedLayout",
           "TypedViewRequest", "ResidualViewRequest", "execute_oracle_program", "execute_promoted_program",
           "execute_research_program", "execute_research_program_outputs"]
