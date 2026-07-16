"""Explicit semantic ownership for schedule buffers.

This module deliberately keeps the annotation outside ``UOp.tag``: tags are a
compiler-internal scratch channel (notably used by callify).  An annotation is
therefore meaningful only when explicitly attached or propagated as an alias.
"""
from dataclasses import dataclass
from enum import Enum
import weakref
from typing import Any
from tinygrad.uop.ops import GroupOp, UOp
from tinygrad.uop import Ops

class MemorySemanticClass(str, Enum):
  MODEL_PARAMETER = "model_parameter"
  KV_CACHE = "kv_cache"
  RUNTIME_PERSISTENT = "runtime_persistent"
  RUNTIME_INPUT = "runtime_input"
  RUNTIME_ACTIVATION = "runtime_activation"
  RUNTIME_OUTPUT = "runtime_output"
  RUNTIME_SCRATCH = "runtime_scratch"
  PREFILL_ACTIVATION = "prefill_activation"
  PREFILL_OUTPUT = "prefill_output"
  PREFILL_SCRATCH = "prefill_scratch"
  CANDIDATE_WORKSPACE = "candidate_workspace"

@dataclass(frozen=True)
class MemorySemanticOwner:
  semantic_class: MemorySemanticClass
  candidate_id: str|None = None

  def __post_init__(self):
    if self.semantic_class is MemorySemanticClass.CANDIDATE_WORKSPACE:
      if not isinstance(self.candidate_id, str) or not self.candidate_id: raise ValueError("candidate_workspace requires a non-empty candidate_id")
    elif self.candidate_id is not None: raise ValueError(f"{self.semantic_class.value} does not accept a candidate_id")

MODEL_PARAMETER = MemorySemanticOwner(MemorySemanticClass.MODEL_PARAMETER)
KV_CACHE = MemorySemanticOwner(MemorySemanticClass.KV_CACHE)
RUNTIME_PERSISTENT = MemorySemanticOwner(MemorySemanticClass.RUNTIME_PERSISTENT)
RUNTIME_INPUT = MemorySemanticOwner(MemorySemanticClass.RUNTIME_INPUT)
RUNTIME_ACTIVATION = MemorySemanticOwner(MemorySemanticClass.RUNTIME_ACTIVATION)
RUNTIME_OUTPUT = MemorySemanticOwner(MemorySemanticClass.RUNTIME_OUTPUT)
RUNTIME_SCRATCH = MemorySemanticOwner(MemorySemanticClass.RUNTIME_SCRATCH)
PREFILL_ACTIVATION = MemorySemanticOwner(MemorySemanticClass.PREFILL_ACTIVATION)
PREFILL_OUTPUT = MemorySemanticOwner(MemorySemanticClass.PREFILL_OUTPUT)
PREFILL_SCRATCH = MemorySemanticOwner(MemorySemanticClass.PREFILL_SCRATCH)
_source_owners:weakref.WeakKeyDictionary[UOp, MemorySemanticOwner] = weakref.WeakKeyDictionary()

# UOps are interned and identity-hashable. The Tensor/JIT graph itself keeps every
# live mark alive; weak tables prevent completed searches from retaining graphs.
def candidate_workspace(candidate_id:str) -> MemorySemanticOwner:
  return MemorySemanticOwner(MemorySemanticClass.CANDIDATE_WORKSPACE, candidate_id)

def bind_memory_semantic_owner(value:Any, owner:MemorySemanticOwner) -> None:
  """Bind ownership to a concrete allocation identity without changing its graph."""
  uop = _uop(value)
  existing = memory_semantic_owner(uop)
  if existing is not None and existing != owner:
    raise ValueError(f"allocation already has semantic owner {existing!r}")
  _source_owners[uop] = owner

def _uop(value:Any) -> UOp:
  uop = value if isinstance(value, UOp) else getattr(value, "uop", None)
  if not isinstance(uop, UOp): raise TypeError("memory semantics can only mark a Tensor or UOp result")
  return uop

def mark_memory_semantic(value:Any, owner:MemorySemanticOwner) -> Any:
  """Wrap one exact result in structural scheduler ownership metadata."""
  if not isinstance(owner, MemorySemanticOwner): raise TypeError("owner must be a MemorySemanticOwner")
  uop = _uop(value)
  if (old := memory_semantic_owner(uop)) is not None and old != owner and uop.op is not Ops.CONTIGUOUS:
    raise ValueError(f"result already has semantic owner {old!r}")
  wrapped = uop if uop.op is Ops.MEMORY_SEMANTIC else UOp(Ops.MEMORY_SEMANTIC, uop.dtype, (uop,), owner)
  if isinstance(value, UOp): return wrapped
  value.uop = wrapped
  return value

def model_parameter(value:Any) -> Any: return mark_memory_semantic(value, MODEL_PARAMETER)
def kv_cache(value:Any) -> Any: return mark_memory_semantic(value, KV_CACHE)
def runtime_persistent(value:Any) -> Any: return mark_memory_semantic(value, RUNTIME_PERSISTENT)
def runtime_input(value:Any) -> Any: return mark_memory_semantic(value, RUNTIME_INPUT)
def runtime_input_materialization(value:Any) -> Any:
  """Own every buffer participating in construction of one runtime input.

  This is for request/literal ingestion graphs (for example PYTHON -> device
  COPY -> CONTIGUOUS), where the host source and device destinations are all
  parts of the same external input. Callers must not use it on a computation
  graph containing model or persistent state.
  """
  uop = _uop(value)
  for source in uop.toposort():
    if source.op is not Ops.BUFFER: continue
    existing = memory_semantic_owner(source)
    if existing is not None and existing != RUNTIME_INPUT:
      raise ValueError(f"runtime input materialization contains conflicting source owner {existing!r}")
    if existing is None: _source_owners[source] = RUNTIME_INPUT
  return mark_memory_semantic(value, RUNTIME_INPUT)
def materialize_runtime_input(value:Any) -> Any:
  """Realize a complete request-input graph and bind its final Buffer too."""
  value = runtime_input_materialization(value)
  realize = getattr(value, "realize", None)
  if not callable(realize): raise TypeError("materialize_runtime_input requires a realizable Tensor")
  realize()
  return runtime_input_materialization(value)
def runtime_activation(value:Any) -> Any: return mark_memory_semantic(value, RUNTIME_ACTIVATION)
def runtime_output(value:Any) -> Any: return mark_memory_semantic(value, RUNTIME_OUTPUT)
def runtime_scratch(value:Any) -> Any: return mark_memory_semantic(value, RUNTIME_SCRATCH)
def prefill_activation(value:Any) -> Any: return mark_memory_semantic(value, PREFILL_ACTIVATION)
def prefill_output(value:Any) -> Any: return mark_memory_semantic(value, PREFILL_OUTPUT)
def prefill_scratch(value:Any) -> Any: return mark_memory_semantic(value, PREFILL_SCRATCH)
def mark_candidate_workspace(value:Any, candidate_id:str) -> Any: return mark_memory_semantic(value, candidate_workspace(candidate_id))

def propagate_memory_semantic(source:UOp, target:UOp) -> UOp:
  """Declare that *target* is the concrete allocation identity of *source*."""
  owner = memory_semantic_owner(source)
  if owner is None: return target
  if (other := memory_semantic_owner(target)) is not None and other != owner:
    raise ValueError(f"conflicting explicit semantic owners: {owner!r} and {other!r}")
  return target if target.op is Ops.MEMORY_SEMANTIC else UOp(Ops.MEMORY_SEMANTIC, target.dtype, (target,), owner)

def memory_semantic_owner(value:Any) -> MemorySemanticOwner|None:
  uop, seen = _uop(value), set()
  while uop not in seen:
    seen.add(uop)
    if (source_owner := _source_owners.get(uop)) is not None: return source_owner
    if uop.op is Ops.MEMORY_SEMANTIC: return uop.arg if isinstance(uop.arg, MemorySemanticOwner) else None
    if len(uop.src) and uop.op in GroupOp.Movement | {Ops.CAST, Ops.BITCAST, Ops.CONTIGUOUS, Ops.STAGE, Ops.AFTER}:
      uop = uop.src[0]
    else: break
  return None
