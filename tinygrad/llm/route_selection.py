"""Shared policy vocabulary for model route selection.

Candidate discovery remains phase-specific. This module owns the stable
contract shared by prefill and decode: route modes and candidate lifecycle.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Sequence

from tinygrad.helpers import getenv


class RouteLifecycle(str, Enum):
  PROMOTED = "promoted"
  FALLBACK = "fallback"
  RESEARCH = "research"
  QUARANTINED = "quarantined"


@dataclass(frozen=True)
class RouteCandidatePolicy:
  candidate_id: str
  lifecycle: RouteLifecycle
  reason: str | None = None

  def require_usable(self) -> None:
    if self.lifecycle is RouteLifecycle.QUARANTINED:
      raise RuntimeError(f"route candidate {self.candidate_id!r} is quarantined: {self.reason or 'no reason recorded'}")


def parse_route_mode(env_name: str, *, allowed: Sequence[str], default: str = "auto",
                     aliases: Mapping[str, str] | None = None,
                     getenv_fn: Callable[[str, object], object] = getenv) -> str:
  raw = str(getenv_fn(env_name, default)).strip().lower()
  mode = (aliases or {}).get(raw, raw)
  if mode not in allowed:
    raise ValueError(f"{env_name}={raw!r} is invalid; expected one of: {', '.join(allowed)}")
  return mode


__all__ = ["RouteCandidatePolicy", "RouteLifecycle", "parse_route_mode"]
