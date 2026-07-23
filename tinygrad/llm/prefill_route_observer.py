"""Generic, context-local observation seam for model route dispatch."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

@dataclass(frozen=True)
class PrefillRouteAttachment:
  invocation_id: str
  route_id: str
  tensor_identity: str
  selected_policy: object
  scanned_target_facts: object
  allocation_owner_identity: str | None = None

@dataclass(frozen=True)
class PrefillRouteExecution:
  """Identity reported by a route only after it has selected its executable path."""
  invocation_id: str
  executed_route_id: str
  candidate_identity: str
  program_identity: str
  fallback_used: bool
  fallback_reason: str | None = None
  execution_evidence: object | None = None

_OBSERVER: ContextVar[Callable[[object], None] | None] = ContextVar("tinygrad_prefill_route_observer", default=None)
_EXECUTION_OBSERVER: ContextVar[Callable[[object, PrefillRouteExecution], None] | None] = ContextVar(
  "tinygrad_prefill_route_execution_observer", default=None)
_ACTIVE: ContextVar[bool] = ContextVar("tinygrad_prefill_route_active", default=False)

@contextmanager
def observe_prefill_routes(observer: Callable[[object], None]) -> Iterator[None]:
  if not callable(observer): raise TypeError("prefill route observer must be callable")
  token = _OBSERVER.set(observer)
  try: yield
  finally: _OBSERVER.reset(token)

@contextmanager
def observe_prefill_route_executions(observer: Callable[[object, PrefillRouteExecution], None]) -> Iterator[None]:
  if not callable(observer): raise TypeError("prefill route execution observer must be callable")
  token = _EXECUTION_OBSERVER.set(observer)
  try: yield
  finally: _EXECUTION_OBSERVER.reset(token)

@contextmanager
def prefill_route_scope(enabled: bool = True) -> Iterator[None]:
  token = _ACTIVE.set(bool(enabled))
  try: yield
  finally: _ACTIVE.reset(token)

def notify_prefill_route(linear: object) -> None:
  observer = _OBSERVER.get()
  if observer is not None and _ACTIVE.get(): observer(linear)

def notify_prefill_route_execution(linear: object, execution: PrefillRouteExecution) -> None:
  observer = _EXECUTION_OBSERVER.get()
  if observer is not None and _ACTIVE.get(): observer(linear, execution)

__all__ = ["PrefillRouteAttachment", "PrefillRouteExecution", "notify_prefill_route",
           "notify_prefill_route_execution", "observe_prefill_routes", "observe_prefill_route_executions",
           "prefill_route_scope"]
