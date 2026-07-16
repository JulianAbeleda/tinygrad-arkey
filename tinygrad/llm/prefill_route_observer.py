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

_OBSERVER: ContextVar[Callable[[object], None] | None] = ContextVar("tinygrad_prefill_route_observer", default=None)
_ACTIVE: ContextVar[bool] = ContextVar("tinygrad_prefill_route_active", default=False)

@contextmanager
def observe_prefill_routes(observer: Callable[[object], None]) -> Iterator[None]:
  if not callable(observer): raise TypeError("prefill route observer must be callable")
  token = _OBSERVER.set(observer)
  try: yield
  finally: _OBSERVER.reset(token)

@contextmanager
def prefill_route_scope(enabled: bool = True) -> Iterator[None]:
  token = _ACTIVE.set(bool(enabled))
  try: yield
  finally: _ACTIVE.reset(token)

def notify_prefill_route(linear: object) -> None:
  observer = _OBSERVER.get()
  if observer is not None and _ACTIVE.get(): observer(linear)

__all__ = ["PrefillRouteAttachment", "notify_prefill_route", "observe_prefill_routes", "prefill_route_scope"]
