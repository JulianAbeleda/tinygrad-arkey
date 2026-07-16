"""Production vocabulary for explicit physical-allocation ownership."""
from __future__ import annotations
import contextlib, contextvars, weakref
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class AllocationOwner:
  kind:str
  lifetime:str
  candidate_id:str|None = None
  semantic_owner_id:str|None = None
  def __post_init__(self):
    if not self.kind or not self.lifetime: raise ValueError("ownership kind and lifetime must be non-empty")

def allocation_owner_from_semantic(owner) -> AllocationOwner:
  from tinygrad.llm.memory_semantics import MemorySemanticClass, MemorySemanticOwner
  if not isinstance(owner, MemorySemanticOwner): raise TypeError("semantic owner must be a MemorySemanticOwner")
  cls = owner.semantic_class
  if cls in {MemorySemanticClass.MODEL_PARAMETER, MemorySemanticClass.KV_CACHE, MemorySemanticClass.RUNTIME_PERSISTENT}: lifetime = "model"
  elif cls is MemorySemanticClass.CANDIDATE_WORKSPACE: lifetime = "candidate"
  elif cls in {MemorySemanticClass.PREFILL_ACTIVATION, MemorySemanticClass.PREFILL_OUTPUT, MemorySemanticClass.PREFILL_SCRATCH}: lifetime = "prefill"
  else: lifetime = "invocation"
  return AllocationOwner(cls.value, lifetime, candidate_id=owner.candidate_id)

_owners:contextvars.ContextVar[tuple[AllocationOwner, ...]] = contextvars.ContextVar("physical_allocation_owners", default=())
_bound_owners:weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def bind_allocation_owner(buffer, owner:AllocationOwner) -> AllocationOwner:
  """Persistently bind explicit ownership evidence to a Buffer's physical base."""
  if not isinstance(owner, AllocationOwner): raise TypeError("owner must be an AllocationOwner")
  base = buffer.base
  if (existing := _bound_owners.get(base)) is not None and existing != owner: raise ValueError("buffer base already has different ownership")
  _bound_owners[base] = owner
  return owner

@contextlib.contextmanager
def allocation_owner(*, kind:str, lifetime:str, candidate_id:str|None=None, semantic_owner_id:str|None=None) -> Iterator[AllocationOwner]:
  """Explicitly classify allocations made inside this context."""
  owner = AllocationOwner(kind, lifetime, candidate_id, semantic_owner_id)
  token = _owners.set(_owners.get() + (owner,))
  try: yield owner
  finally: _owners.reset(token)

__all__ = ["AllocationOwner", "allocation_owner", "allocation_owner_from_semantic", "bind_allocation_owner"]
