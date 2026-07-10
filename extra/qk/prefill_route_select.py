"""Canonical prefill route wrapper -- manifest-backed `select(name) -> env dict`.

This is the CANONICAL route wrapper of the two-wrapper flag-collapse (docs/prefill-flag-classification.md):
it maps a named prefill route to its EXACT flag env, sourced verbatim from the single source of truth
extra/qk/route_manifest.ROUTES[route_id]["env"]. It exposes ONLY the four canonical routes (plus the empty
`default` baseline); it has NO raw-flag parameter, so invalid flag combinations are unrepresentable.

It lives in extra/qk (NOT tinygrad core): it reads the route manifest, and tinygrad core must not import
extra.qk (test/unit/test_tinygrad_boundary.py). The CORE consumer is the resolver
tinygrad.codegen.opt.prefill_route_spec.resolve_prefill_route, which round-trips these env dicts back to a
PrefillRouteSpec.route_name.
"""
from __future__ import annotations

from extra.qk.route_manifest import route_env
from tinygrad.codegen.opt.prefill_route_spec import (
  ROUTE_HYBRID, ROUTE_PURE, ROUTE_MIXED, ROUTE_PIPE_MVP, ROUTE_DEFAULT, ROUTE_NAMES,
)

# route_name -> manifest route_id. The named route is the stable public identity; the route_id is the manifest
# key that owns its defining env / rollback. Single source of truth for the mapping (Naming Reflects Actuals).
ROUTE_NAME_TO_ID: dict[str, str] = {
  ROUTE_HYBRID:   "prefill_pipe_role_selective_generated",       # GRAPH_GEMM only (~4413 pp512)
  ROUTE_PURE:     "prefill_wmma_pipe_lds_dbuf_primitive_generated",  # +PIPE+LDS+DBUF (~1332)
  ROUTE_MIXED:    "prefill_wmma_lds_dbuf_primitive_mixed",        # +LDS+DBUF (PIPE off)
  ROUTE_PIPE_MVP: "prefill_wmma_pipe_primitive_generated",        # +PIPE only
  ROUTE_DEFAULT:  "prefill_v2_scheduler_matmul_default",          # {} baseline (GRAPH_GEMM=0)
}

# The four FORCING routes (default is the empty baseline, not a forcing route).
CANONICAL_ROUTE_NAMES = (ROUTE_HYBRID, ROUTE_PURE, ROUTE_MIXED, ROUTE_PIPE_MVP)


class CanonicalRoute:
  """Canonical route selector. `select(name)` returns the exact env dict to force `name` onto the active path,
  read straight from the manifest -- so the wrapper and the manifest can never drift."""

  NAMES = CANONICAL_ROUTE_NAMES

  @staticmethod
  def select(name: str) -> dict[str, str]:
    """The env dict that forces route `name` (one of hybrid|pure|mixed|pipe_mvp, or the `default` baseline -> {}).

    Sourced from route_manifest.ROUTES[route_id]["env"]; a copy, so callers cannot mutate the manifest.
    """
    if name not in ROUTE_NAME_TO_ID:
      raise KeyError(f"unknown prefill route {name!r}; known: {sorted(ROUTE_NAMES)}")
    return route_env(ROUTE_NAME_TO_ID[name])
