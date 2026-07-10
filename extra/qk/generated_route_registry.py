"""Generated-route descriptor registry scaffold.

This module keeps a minimal, JSON-serializable inventory of generated routes that are
owned by descriptors (L3).  The scaffold is intentionally narrow for now and only
contains currently positively validated rows.
"""

from __future__ import annotations

from typing import Any, TypedDict

from extra.qk import route_manifest


class GeneratedRouteDescriptor(TypedDict):
  route_id: str
  descriptor_artifact: str
  lowering_level: str
  owner: str
  writer_files: list[str]
  emitter: str
  rollback_route: str | None


_GENERATION_REGISTRY: tuple[GeneratedRouteDescriptor, ...] = (
  {
    "route_id": "decode_q4k_g3_generated",
    "descriptor_artifact": "Q4KGateUpLaneMap",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/gemv_g3_codegen_lowering.py",
      "extra/qk/gemv_g2_lanemap.py",
    ],
    "emitter": "extra/qk/gemv_g3_codegen_lowering.py q4k_g3_lanemap_gemv_kernel",
    "rollback_route": None,
  },
  {
    "route_id": "decode_q6k_coop_generated",
    "descriptor_artifact": "Q6KGEMVRouteSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/q6k_route_spec.py",
      "extra/qk/quant/q6_k_gemv_primitive.py",
    ],
    "emitter": "extra/qk/q6k_route_spec.py emit_q6k_gemv_kernel",
    "rollback_route": None,
  },
  {
    "route_id": "decode_flash_live_split_g4_8b_kvboth",
    "descriptor_artifact": "FlashDecodeAttentionSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/live_split_geometry.py",
      "extra/qk/flash_kernels.py",
      "extra/qk/flash_decode_attention_spec.py",
    ],
    "emitter": "extra/qk/flash_decode_attention_spec.py describe_flash_decode_attention",
    "rollback_route": None,
  },
  {
    "route_id": "decode_flash_block_tile_g5_konly",
    "descriptor_artifact": "FlashDecodeAttentionSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/live_split_geometry.py",
      "extra/qk/flash_kernels.py",
      "extra/qk/flash_decode_attention_spec.py",
    ],
    "emitter": "extra/qk/flash_decode_attention_spec.py describe_flash_decode_attention",
    "rollback_route": None,
  },
  {
    "route_id": "prefill_q6k_direct_generated",
    "descriptor_artifact": "Q6KPrefillRouteSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/q6k_prefill_route_spec.py",
      "extra/qk/quant/q6_k_gemv_primitive.py",
    ],
    "emitter": "extra/qk/q6k_prefill_route_spec.py emit_q6k_packed_prefill_kernel",
    "rollback_route": None,
  },
  {
    "route_id": "prefill_q4k_direct_tile4x4_default",
    "descriptor_artifact": "Q4KPrefillRouteSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/q4k_prefill_route_spec.py",
      "extra/qk/quant/q4_k_gemv_primitive.py",
    ],
    "emitter": "extra/qk/q4k_prefill_route_spec.py emit_q4k_packed_prefill_kernel",
    "rollback_route": None,
  },
  {
    "route_id": "prefill_q4k_reduce_out_research",
    "descriptor_artifact": "Q4KPrefillRouteSpec",
    "lowering_level": "L3",
    "owner": "descriptor",
    "writer_files": [
      "extra/qk/q4k_prefill_route_spec.py",
      "extra/qk/quant/q4_k_gemv_primitive.py",
    ],
    "emitter": "extra/qk/q4k_prefill_route_spec.py emit_q4k_packed_prefill_kernel",
    "rollback_route": "prefill_q4k_direct_tile4x4_default",
  },
)


def _manifest(route_id: str) -> dict[str, Any]:
  if route_id not in route_manifest.ROUTES:
    raise KeyError(f"generated route {route_id!r} is missing from route_manifest.ROUTES")
  return route_manifest.ROUTES[route_id]


def _shape_role_policy(manifest: dict[str, Any]) -> dict[str, Any]:
  return {
    "roles": list(manifest.get("roles", ())),
    "excluded_roles": list(manifest.get("excluded_roles", ())),
    "quant": list(manifest.get("quant", ())),
    "shape_guards": [dict(g) for g in manifest.get("shape_guards", ())],
  }


def _required_gates(manifest: dict[str, Any]) -> list[str]:
  gate = str(manifest.get("authority_gate", ""))
  return [part.strip() for part in gate.split(" + ") if part.strip()]


def _sanitize_row(row: GeneratedRouteDescriptor) -> dict[str, Any]:
  """Return only JSON-safe primitive values."""
  manifest = _manifest(row["route_id"])
  return {
    "route_id": row["route_id"],
    "descriptor_artifact": row["descriptor_artifact"],
    "lowering_level": row["lowering_level"],
    "owner": row["owner"],
    "writer_files": list(row["writer_files"]),
    "emitter": row["emitter"],
    "emitted_kernel_patterns": list(manifest.get("expected_kernels", ())),
    "authority_gate": str(manifest.get("authority_gate", "")),
    "authority_artifacts": list(manifest.get("promotion_artifacts", ())),
    "selector_binding": str(manifest.get("selector", "")),
    "shape_role_policy": _shape_role_policy(manifest),
    "manifest_profile_id": str(manifest.get("profile_id", "")),
    "manifest_status": str(manifest.get("status", "")),
    "manifest_provenance": route_manifest.route_provenance(row["route_id"]),
    "manifest_purity_status": str(manifest.get("purity_status", "")),
    "route_attribution": str(manifest.get("route_attribution", "")),
    "required_gates": _required_gates(manifest),
    "rollback_route": row["rollback_route"],
  }


def row(route_id: str) -> dict[str, Any]:
  for r in _GENERATION_REGISTRY:
    if r["route_id"] == route_id:
      return _sanitize_row(r)
  raise KeyError(f"unknown generated route {route_id!r}")


def route_ids() -> tuple[str, ...]:
  return tuple(r["route_id"] for r in _GENERATION_REGISTRY)


def rows() -> list[dict[str, Any]]:
  return [_sanitize_row(r) for r in _GENERATION_REGISTRY]


def build() -> dict[str, Any]:
  all_rows = rows()
  return {
    "schema": "generated-route-descriptor-registry.v2",
    "total_routes": len(all_rows),
    "l3_routes": len([r for r in all_rows if r["lowering_level"] == "L3"]),
    "routes": all_rows,
  }
