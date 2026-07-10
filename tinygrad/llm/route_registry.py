from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from tinygrad.llm.route_request import RouteBinding, RouteContext, RouteRequest, RouteSelection


@runtime_checkable
class RouteCandidate(Protocol):
  route_id: str

  def available(self, request: RouteRequest, ctx: RouteContext) -> RouteSelection: ...
  def bind(self, request: RouteRequest, ctx: RouteContext) -> RouteSelection | RouteBinding: ...


@dataclass
class RouteCandidateRegistry:
  _candidates: list[RouteCandidate] = field(default_factory=list)

  def register(self, candidate: RouteCandidate) -> RouteCandidate:
    self._candidates.append(candidate)
    return candidate

  def candidates(self, preferred: Sequence[str] | None = None) -> tuple[RouteCandidate, ...]:
    if preferred is None:
      return tuple(self._candidates)

    preferred_pos = {route_id: i for i, route_id in enumerate(preferred)}
    return tuple(sorted(self._candidates, key=lambda c: (preferred_pos.get(c.route_id, len(preferred_pos)), self._candidates.index(c))))

  def select(self, request: RouteRequest, ctx: RouteContext, preferred: Sequence[str] | None = None) -> RouteSelection:
    rejected: list[str] = []
    for candidate in self.candidates(preferred):
      available = candidate.available(request, ctx)
      if available.status != "available":
        rejected.append(_reason(candidate, available.reason or available.status))
        continue

      bound = candidate.bind(request, ctx)
      if isinstance(bound, RouteBinding):
        return RouteSelection(status="selected", candidate=candidate, binding=bound, reason="selected")
      if bound.status == "selected" and bound.binding is not None:
        return RouteSelection(status="selected", candidate=bound.candidate or candidate, binding=bound.binding, reason=bound.reason or "selected")
      rejected.append(_reason(candidate, bound.reason or bound.status))

    return RouteSelection(status="unavailable", reason="; ".join(rejected) if rejected else "no candidates registered")


def _reason(candidate: RouteCandidate, reason: str) -> str:
  return f"{candidate.route_id}: {reason}"
