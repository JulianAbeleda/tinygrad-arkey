"""Runtime-capable kernel surfaces outside the route manifest.

These rows are not manifest routes yet: they are shape/env-capable execution
surfaces that can still reach handwritten UOp templates.  Keeping them here
lets audits, docs generators, and future strict-mode guards share one
inventory instead of copying local blocker lists.
"""

from __future__ import annotations

from typing import Any, TypedDict


class RuntimeSurface(TypedDict):
  surface_id: str
  surface_class: str
  writer_files: list[str]
  reason: str
  replacement_scope: str


_RUNTIME_SURFACES: tuple[RuntimeSurface, ...] = (
  {
    "surface_id": "prefill_q6k_direct_packed_default_capable",
    "surface_class": "route_local_custom_kernel",
    "writer_files": [
      "tinygrad/llm/prefill_routes.py",
      "extra/qk/quant/q6_k_gemv_primitive.py",
    ],
    "reason": "PREFILL_DIRECT_QUANTS defaults to Q4_K,Q6_K and Q6_K direct prefill calls q6k_gemm_packed_load_* hand UOp templates.",
    "replacement_scope": "Add Q6KPrefillRouteSpec or explicit manifest debt row.",
  },
  {
    "surface_id": "decode_q4k_smallk_batched",
    "surface_class": "route_local_custom_kernel",
    "writer_files": [
      "tinygrad/llm/decode_routes.py",
      "extra/qk/quant/q4_k_gemv_primitive.py",
    ],
    "reason": "q4k_primitive_linear_call routes non-decode K<=32 through q4k_gemm_kernel hand UOp template.",
    "replacement_scope": "Add Q4KSmallBatchGEMMSpec or block under PURE_MACHINE_SEARCH_ONLY.",
  },
  {
    "surface_id": "decode_q6k_smallk_batched",
    "surface_class": "route_local_custom_kernel",
    "writer_files": [
      "tinygrad/llm/decode_routes.py",
      "extra/qk/quant/q6_k_gemv_primitive.py",
    ],
    "reason": "q6k_primitive_linear_call routes non-decode K<=32 through q6k_gemm_kernel hand UOp template.",
    "replacement_scope": "Add Q6KSmallBatchGEMMSpec or block under PURE_MACHINE_SEARCH_ONLY.",
  },
)


def _sanitize_row(row: RuntimeSurface) -> dict[str, Any]:
  return {
    "surface_id": row["surface_id"],
    "surface_class": row["surface_class"],
    "writer_files": list(row["writer_files"]),
    "reason": row["reason"],
    "replacement_scope": row["replacement_scope"],
  }


def rows() -> list[dict[str, Any]]:
  return [_sanitize_row(r) for r in _RUNTIME_SURFACES]


def surface_ids() -> tuple[str, ...]:
  return tuple(r["surface_id"] for r in _RUNTIME_SURFACES)


def row(surface_id: str) -> dict[str, Any]:
  for r in _RUNTIME_SURFACES:
    if r["surface_id"] == surface_id:
      return _sanitize_row(r)
  raise KeyError(f"unknown runtime surface {surface_id!r}")


def build() -> dict[str, Any]:
  all_rows = rows()
  return {
    "schema": "runtime-surface-registry.v1",
    "total_surfaces": len(all_rows),
    "surfaces": all_rows,
  }
