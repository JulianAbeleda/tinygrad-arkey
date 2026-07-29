"""Fail-closed production route selection and trace interface.

This module owns no kernel emitter and makes no provenance judgment.  A caller
uses its selected route metadata to bind an implementation; absent an exact
match, the trace explicitly selects ordinary tinygrad graph lowering.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from tinygrad.llm.production_route_policy import GENERIC_ROUTE_ID, ProductionRoute, ProductionRouteRequest, select_current_route


@dataclass(frozen=True)
class ProductionRouteSelection:
  route: ProductionRoute | None
  trace: Mapping[str, object]

  @property
  def uses_selected_route(self) -> bool: return self.route is not None


def select_production_route(request: ProductionRouteRequest) -> ProductionRouteSelection:
  route = select_current_route(request)
  if route is None:
    return ProductionRouteSelection(None, {"route_id": GENERIC_ROUTE_ID, "selection": "generic_fallback",
      "fallback_reason": "no_exact_current_route", "workload": request.workload, "role": request.role,
      "quant": request.quant, "target": {"backend": request.target_backend, "architecture": request.target_architecture},
      "shape": dict(request.shape), "selected_config": {}})
  return ProductionRouteSelection(route, {"route_id": route.route_id, "selection": "current_selected_configuration",
    "fallback_reason": None, "workload": request.workload, "role": request.role, "quant": request.quant,
    "target": {"backend": request.target_backend, "architecture": request.target_architecture},
    "shape": dict(request.shape), "selected_config": route.config()})


__all__ = ["ProductionRouteSelection", "select_production_route"]
