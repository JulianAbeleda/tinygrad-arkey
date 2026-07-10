"""Prefill route SPEC + resolver -- the core object the prefill flag-collapse converges on.

Two wrappers, one resolver (docs/prefill-flag-classification.md):

  * canonical route wrapper -- `select(hybrid|pure|mixed|pipe_mvp)`, manifest-backed, invalid combos
    unrepresentable. It lives in `extra/qk/prefill_route_select.py` (NOT core: it reads the route manifest).
  * debug wrapper (`DebugFlags`, below) -- the single home for the surviving raw survivor flags.

Both produce ONE `PrefillRouteSpec`; downstream (`prefill_graph_gemm_route`, the lowerer, the renderer) is
unchanged. This module is CORE so tinygrad can consume the spec WITHOUT importing `extra.qk`
(test/unit/test_tinygrad_boundary.py enforces that no `extra.qk` import leaks into `tinygrad/`).

Phase 1 (SKELETON) is additive and behaviour-neutral: it stands up the spec, the resolver, and the debug-wrapper
surface. It does NOT rewire the scattered `getenv` reads in amd.py / postrange.py and it bakes no promotes -- those
are Phases 2-3 (see the execution order in docs/prefill-flag-classification.md).
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Mapping

# The five prefill route names. Four are the canonical FORCING routes selected by the graph-GEMM primitive flags;
# `default` is the shipped baseline (no flag, ordinary PREFILL_V2 scheduler matmul).
ROUTE_HYBRID = "hybrid"
ROUTE_PURE = "pure"
ROUTE_MIXED = "mixed"
ROUTE_PIPE_MVP = "pipe_mvp"
ROUTE_DEFAULT = "default"
ROUTE_NAMES = (ROUTE_HYBRID, ROUTE_PURE, ROUTE_MIXED, ROUTE_PIPE_MVP, ROUTE_DEFAULT)

# The four SELECTOR-OWNED flags. These are consumed by the resolver/canonical wrapper, not user toggles
# (docs/prefill-flag-classification.md::SELECTOR_OWNED).
FLAG_GRAPH_GEMM = "PREFILL_GRAPH_GEMM"
FLAG_WMMA_PIPE_PRIMITIVE = "PREFILL_WMMA_PIPE_PRIMITIVE"
FLAG_WMMA_LDS_PRIMITIVE = "PREFILL_WMMA_LDS_PRIMITIVE"
FLAG_DBUF = "PREFILL_DBUF"

_FALSE_TOKENS = ("", "0", "false", "off", "no")


@dataclass(frozen=True)
class PrefillRouteSpec:
  """The resolved prefill route as data. Every route/staging decision in the flag-collapse eventually collapses
  into this object.

  Fields:
    route_name: one of ROUTE_NAMES (hybrid | pure | mixed | pipe_mvp | default).
    graph_gemm / wmma_pipe_primitive / wmma_lds_primitive / dbuf: the four selector-owned booleans that the
      graph-GEMM primitive route reads (their exact env is single-sourced in extra/qk/route_manifest.ROUTES).

  Phase 1 carries only the selector booleans; the promote/debug fields land in Phases 2-3.
  """
  route_name: str
  graph_gemm: bool
  wmma_pipe_primitive: bool
  wmma_lds_primitive: bool
  dbuf: bool


def _graph_gemm_on(env: Mapping[str, str]) -> bool:
  # Mirror tinygrad/llm/model.py::_prefill_graph_gemm_default: absent -> off; present -> truthy unless a false token.
  if FLAG_GRAPH_GEMM not in env: return False
  raw = str(env.get(FLAG_GRAPH_GEMM, "0")).strip().lower()
  if raw in _FALSE_TOKENS: return False
  if raw in ("1", "true", "on", "yes"): return True
  try: return bool(int(raw))
  except ValueError: return False


def _primitive_on(env: Mapping[str, str], name: str) -> bool:
  # Mirror extra/qk/prefill_graph_gemm_route.py: the pipe/lds primitive gates read strict `== "1"`.
  return env.get(name) == "1"


def _dbuf_on(env: Mapping[str, str]) -> bool:
  # Mirror extra/qk/prefill_graph_gemm_route.py::_env_enabled for the bare PREFILL_DBUF selector flag.
  return str(env.get(FLAG_DBUF, "0")).strip().lower() not in _FALSE_TOKENS


def resolve_prefill_route(env: Mapping[str, str] | None = None) -> PrefillRouteSpec:
  """Read the four selector-owned flags and return the PrefillRouteSpec, reproducing today's selection exactly.

  Decision tree (docs/prefill-flag-classification.md route map):
    GRAPH_GEMM=0                          -> default
    GRAPH_GEMM + PIPE + LDS + DBUF        -> pure
    GRAPH_GEMM + LDS + DBUF (PIPE off)    -> mixed
    GRAPH_GEMM + PIPE (only)              -> pipe_mvp
    GRAPH_GEMM only (no primitive flags)  -> hybrid
  """
  if env is None: env = os.environ
  graph_gemm = _graph_gemm_on(env)
  pipe = _primitive_on(env, FLAG_WMMA_PIPE_PRIMITIVE)
  lds = _primitive_on(env, FLAG_WMMA_LDS_PRIMITIVE)
  dbuf = _dbuf_on(env)

  if not graph_gemm:
    name = ROUTE_DEFAULT
  elif pipe and lds and dbuf:
    name = ROUTE_PURE
  elif lds and dbuf:            # PIPE off
    name = ROUTE_MIXED
  elif pipe:                    # pipe on, not the composed pure combo
    name = ROUTE_PIPE_MVP
  else:                         # graph-GEMM only, no primitive flags
    name = ROUTE_HYBRID

  return PrefillRouteSpec(route_name=name, graph_gemm=graph_gemm, wmma_pipe_primitive=pipe,
                          wmma_lds_primitive=lds, dbuf=dbuf)


class DebugFlags:
  """Debug wrapper PLACEHOLDER (Phase 1 stub) -- the single future home for the surviving raw survivor flags.

  docs/prefill-flag-classification.md classifies ~53 KEEP_DEBUG survivors (pure-machine-path knobs, DBUF
  LDS-addressing knobs, the TC_LOCAL_STAGE base ladder, AMD_ISA policy knobs, V2/serving knobs) plus the
  KEEP_DEBUG--HAZARD gates. Today those live as ~110 scattered `getenv`/`os.environ` reads across
  tinygrad/codegen/opt/amd.py and postrange.py.

  Phase 1 establishes ONLY this surface + contract. It intentionally holds no flags yet and rewires no read site;
  Phase 3 relocates the survivors here so production code can never reach a raw flag except through an explicit
  debug opt-in. Kept CORE (no extra.qk import) so the core renderer/lowerer can consume it at the boundary.
  """
  #: Phase 3 populates this from the KEEP_DEBUG bucket. Empty in Phase 1 by design (surface-only).
  SURVIVORS: tuple[str, ...] = ()
