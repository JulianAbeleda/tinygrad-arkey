from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class RouteRequest:
  op: Any
  request_id: str = ""
  attrs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteContext:
  linear: Any = None
  x: Any = None
  fallback: Callable[..., Any] | None = None
  arch_ok: bool = False
  getenv_fn: Callable[[str, Any], Any] | None = None


@dataclass(frozen=True)
class RouteBinding:
  request: RouteRequest
  candidate: Any
  concrete_spec: Any
  output_shape: Any
  resources: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouteSelection:
  status: str
  candidate: Any | None = None
  binding: RouteBinding | None = None
  reason: str = ""

  @property
  def selected(self) -> bool:
    return self.status == "selected" and self.binding is not None
