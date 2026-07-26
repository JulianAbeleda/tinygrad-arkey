"""Opt-in, context-local sidecar for decode route and output-UOp observation.

The observer is deliberately not a route selector.  A caller must install both
an observer and ``decode_route_scope``; without those positive controls this
module is inert and decode execution retains its normal semantics.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Callable, Iterator

@dataclass(frozen=True)
class DecodeRouteExecution:
  route_id: str
  candidate_id: str
  model_identity: str | None
  shape: tuple[int, ...]
  tile_name: str
  combine_name: str
  output_path: str
  output_uop: object | None

_OBSERVER: ContextVar[Callable[[DecodeRouteExecution], None] | None] = ContextVar(
  "tinygrad_decode_route_execution_observer", default=None)
_ACTIVE: ContextVar[bool] = ContextVar("tinygrad_decode_route_observer_active", default=False)

@contextmanager
def observe_decode_route_executions(observer: Callable[[DecodeRouteExecution], None]) -> Iterator[None]:
  if not callable(observer): raise TypeError("decode route observer must be callable")
  token = _OBSERVER.set(observer)
  try: yield
  finally: _OBSERVER.reset(token)

@contextmanager
def decode_route_scope(enabled: bool = True) -> Iterator[None]:
  token = _ACTIVE.set(bool(enabled))
  try: yield
  finally: _ACTIVE.reset(token)

def notify_decode_route_execution(*, route_id: str, candidate_id: str, model_identity: str | None,
                                  shape: tuple[int, ...], tile_name: str, combine_name: str,
                                  output_path: str, output: object) -> None:
  """Report an already-selected route.  This function has no return authority."""
  observer = _OBSERVER.get()
  if observer is None or not _ACTIVE.get(): return
  observer(DecodeRouteExecution(route_id, candidate_id, model_identity, shape, tile_name, combine_name,
                                 output_path, getattr(output, "uop", None)))

__all__ = ["DecodeRouteExecution", "decode_route_scope", "notify_decode_route_execution",
           "observe_decode_route_executions"]
