#!/usr/bin/env python3
"""Strict pure-machine-search route surface audit.

This is the route-aware companion to generated_quant_binding_audit. The older audit inventories source markers; this one
answers the stricter question from docs/pure-machine-search.md: does the selected runtime route execute through generated
codegen, or through a hand-authored/raw kernel surface?
"""
from __future__ import annotations

import json, pathlib
from dataclasses import dataclass
from typing import Any

from extra.qk import generated_route_registry, route_manifest, runtime_surface_registry

ROOT = pathlib.Path(__file__).resolve().parents[2]

PURE_CLASSES = {"ordinary_tinygrad_graph", "descriptor_owned_uop_codegen", "backend_owned_intrinsic_lowering"}
IMPURE_CLASSES = {"descriptor_wrapped_hand_kernel", "route_local_custom_kernel", "external_raw_or_binary",
                  "rollback_oracle", "unknown"}

RAW_MARKERS = ("Ops.INS", "Ops.BINARY", "asm volatile")
CUSTOM_MARKERS = (".custom_kernel(", "Tensor.custom_kernel", "Ops.CUSTOM", "Ops.CUSTOMI")
SOURCE_MARKERS = ("asm volatile", "__builtin_amdgcn", "Ops.BINARY")


@dataclass(frozen=True)
class RouteSurface:
  route_id: str
  surface_class: str
  writer_files: tuple[str, ...]
  reason: str
  replacement_scope: str = ""
  descriptor_artifact: str = ""

  @property
  def pure(self) -> bool:
    return self.surface_class in PURE_CLASSES


# Runtime-relevant surfaces from tinygrad/llm/route_ops.py + route_manifest.py. This is intentionally explicit: a route
# cannot claim strict purity just because its name contains "generated".
ROUTE_SURFACES: dict[str, RouteSurface] = {
  # decode_q4k_owned_warp REMOVED 2026-07-06 (no backups): rollback route deleted; bubblebeam-off -> ordinary graph.
  # decode_q6k_coop_shipped + decode_q6k_direct_refuted RouteSurfaces REMOVED 2026-07-06 (no backups): kernels deleted.
  # decode_attention_owned_two_kernel, decode_flash_block_tile_g5_8b_refuted, decode_attention_generic_flash_generated
  # RouteSurfaces REMOVED 2026-07-06 (no backups): their manifest ROUTES rows were deleted (kernels/files gone).
  "decode_flash_live_split_g4_8b_kvboth": RouteSurface(
    "decode_flash_live_split_g4_8b_kvboth", "route_local_custom_kernel",
    ("extra/qk/live_split_geometry.py", "extra/qk/flash_kernels.py"),
    "Promoted attention route still executes hand-authored flash/live-split UOp templates until descriptor artifacts prove topology ownership.",
    replacement_scope="Add FlashDecodeTileSpec, LiveSplitGeometrySpec, FlashCombineSpec, and generated-only binding gate."),
  "decode_flash_block_tile_g5_konly": RouteSurface(
    "decode_flash_block_tile_g5_konly", "route_local_custom_kernel",
    ("extra/qk/live_split_geometry.py", "extra/qk/flash_kernels.py"),
    "G5 attention route shares live-split/block-tile hand UOp template surface until descriptor artifacts prove topology ownership.",
    replacement_scope="Add FlashDecodeTileSpec, LiveSplitGeometrySpec, FlashCombineSpec, and generated-only binding gate."),
  "prefill_pipe_role_selective_generated": RouteSurface(
    "prefill_pipe_role_selective_generated", "external_raw_or_binary",
    ("extra/qk/prefill_graph_gemm_route.py", "extra/qk/prefill/wmma.py", "extra/qk/prefill_schedule_spec.py"),
    "Schedule selection is spec-shaped, but executing substrate wraps raw RDNA3 instruction lists with Ops.INS.",
    replacement_scope="Route B: generated LDS+WMMA codegen substrate replacing extra/qk/prefill/wmma.py."),
  "prefill_q4k_generated_tile_research": RouteSurface(
    "prefill_q4k_generated_tile_research", "route_local_custom_kernel",
    ("extra/qk/prefill_packed_tile_spec.py", "tinygrad/llm/prefill_routes.py"),
    "Descriptor-shaped research route still returns hand-written UOp bodies; generated-only provenance not proven.",
    replacement_scope="Descriptor-owned generated emitter plus binding audit."),
  "prefill_q4k_int8_wmma_generated_research": RouteSurface(
    "prefill_q4k_int8_wmma_generated_research", "ordinary_tinygrad_graph",
    ("extra/qk/prefill_int8_wmma_spec.py", "tinygrad/llm/prefill_routes.py"),
    "Q4_K/Q8_1 int WMMA research expresses core dot as Tensor.matmul(dtype=int), relying on codegen TC matching.",
    descriptor_artifact="Q4KInt8WMMAPrefillSpec"),
  "prefill_q4k_int8_wmma_tiled_research": RouteSurface(
    "prefill_q4k_int8_wmma_tiled_research", "ordinary_tinygrad_graph",
    ("extra/qk/prefill_int8_wmma_spec.py", "tinygrad/llm/prefill_routes.py"),
    "Tiled Q4_K/Q8_1 WMMA research route is intended to use Tensor/codegen lowering; full route remains blocked.",
    descriptor_artifact="Q4KInt8WMMATiledPrefillSpec"),
  "prefill_pipe_global_rollback": RouteSurface(
    "prefill_pipe_global_rollback", "rollback_oracle",
    ("extra/qk/prefill_graph_gemm_route.py", "extra/qk/prefill/wmma.py"),
    "Rollback comparator still uses raw WMMA instruction-list substrate."),
}

def _read(path: str) -> str:
  p = ROOT / path
  return p.read_text() if p.exists() else ""


def _writer_scan(paths: tuple[str, ...]) -> tuple[dict[str, list[str]], dict[str, bool], list[str]]:
  markers: dict[str, list[str]] = {}
  writer_file_exists: dict[str, bool] = {}
  missing: list[str] = []
  for path in paths:
    p = ROOT / path
    exists = p.exists()
    writer_file_exists[path] = exists
    if not exists:
      missing.append(path)
      continue
    src = _read(path)
    found = sorted({m for m in RAW_MARKERS + CUSTOM_MARKERS + SOURCE_MARKERS if m in src})
    if found: markers[path] = found
  return markers, writer_file_exists, sorted(missing)


def _markers(paths: tuple[str, ...]) -> dict[str, list[str]]:
  # Backward-compatible helper for previous callers.
  return _writer_scan(paths)[0]


def route_surface(route_id: str) -> RouteSurface:
  if route_id in ROUTE_SURFACES: return ROUTE_SURFACES[route_id]
  if route_id in generated_route_registry.route_ids():
    reg = generated_route_registry.row(route_id)
    return RouteSurface(route_id, "descriptor_owned_uop_codegen", tuple(reg["writer_files"]),
                        f"{route_id} is L3 descriptor-owned generated codegen via {reg['descriptor_artifact']}.",
                        descriptor_artifact=str(reg["descriptor_artifact"]))
  r = route_manifest.ROUTES.get(route_id, {})
  prov = str(r.get("provenance", "unknown"))
  if prov == "rollback_oracle":
    cls = "rollback_oracle"
  elif prov == "external_handwritten_kernel":
    cls = "external_raw_or_binary"
  elif prov == "hand_authored_uop_template":
    cls = "route_local_custom_kernel"
  elif prov in route_manifest.FINAL_DEFAULT_PROVENANCE:
    cls = "unknown"
  else:
    cls = "unknown"
  return RouteSurface(route_id, cls, (), f"No explicit strict route-surface row for manifest provenance {prov!r}.")


def route_surface_row(route_id: str) -> dict[str, Any]:
  surface = route_surface(route_id)
  manifest = route_manifest.ROUTES.get(route_id, {})
  prov = str(manifest.get("provenance", "unknown"))
  status = str(manifest.get("status", "unknown"))
  manifest_pure = prov in route_manifest.FINAL_DEFAULT_PROVENANCE
  markers, writer_file_exists, missing_writer_files = _writer_scan(surface.writer_files)
  surface_pure = surface.pure
  strict_pure = surface_pure and not missing_writer_files
  contradiction = manifest_pure and not strict_pure
  if route_id in generated_route_registry.route_ids():
    reg = generated_route_registry.row(route_id)
    expected_kernel_patterns = list(reg.get("emitted_kernel_patterns", ()))
  else:
    expected_kernel_patterns = list(manifest.get("expected_kernels", ()))
  has_expected_kernel_binding = bool(expected_kernel_patterns)
  return {"route_id": route_id, "status": status, "manifest_provenance": prov, "manifest_pure": manifest_pure,
          "surface_class": surface.surface_class, "surface_pure": surface_pure, "strict_pure": strict_pure,
          "contradiction": contradiction,
          "writer_files": list(surface.writer_files), "writer_file_exists": writer_file_exists,
          "writer_files_present": not missing_writer_files, "missing_writer_files": missing_writer_files, "markers": markers,
          "expected_kernel_patterns": expected_kernel_patterns, "has_expected_kernel_binding": has_expected_kernel_binding,
          "descriptor_artifact": surface.descriptor_artifact, "reason": surface.reason,
          "replacement_scope": surface.replacement_scope or str(manifest.get("replacement_scope", ""))}


def route_rows() -> list[dict[str, Any]]:
  return [route_surface_row(rid) for rid in sorted(route_manifest.ROUTES)]


def default_rows() -> list[dict[str, Any]]:
  return [route_surface_row(rid) for rid in route_manifest.default_routes()]


def strict_default_purity_report() -> dict[str, Any]:
  rows = default_rows()
  blockers = [r for r in rows if not r["strict_pure"]]
  contradictions = [r for r in rows if r["contradiction"]]
  missing_writer_file_blockers = [r for r in rows if r["missing_writer_files"]]
  return {"verdict": "STRICT_DEFAULT_PURITY_PASS" if not blockers else "STRICT_DEFAULT_PURITY_FAIL",
          "default_routes": [r["route_id"] for r in rows],
          "blockers": blockers, "manifest_contradictions": contradictions,
          "missing_writer_file_blockers": missing_writer_file_blockers}


def unmanifested_runtime_surface_rows() -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for surface in runtime_surface_registry.rows():
    writer_files = tuple(surface.get("writer_files", ()))
    markers, writer_file_exists, missing_writer_files = _writer_scan(writer_files)
    rows.append({**surface,
                 "writer_files": list(writer_files),
                 "writer_file_exists": writer_file_exists,
                 "writer_files_present": not missing_writer_files,
                 "missing_writer_files": missing_writer_files,
                 "markers": markers,
                 "manifested": False,
                 "strict_pure": False,
                 "audit_blocker": True})
  return rows


def build() -> dict[str, Any]:
  rows = route_rows()
  report = strict_default_purity_report()
  registry_route_ids = set(generated_route_registry.route_ids())
  missing = [rid for rid in route_manifest.ROUTES if rid not in ROUTE_SURFACES and rid not in registry_route_ids]
  unmanifested_rows = unmanifested_runtime_surface_rows()
  routes_with_missing_writer_files = sorted([r["route_id"] for r in rows if r["missing_writer_files"]])
  missing_writer_files = sorted({path for row in rows for path in row["missing_writer_files"]})
  unmanifested_ids = [s["surface_id"] for s in unmanifested_rows]
  unmanifested_blocker_ids = sorted(unmanifested_ids)
  audit_blockers = {
    "strict_default_route_blockers": sorted([r["route_id"] for r in report["blockers"]]),
    "missing_writer_file_routes": routes_with_missing_writer_files,
    "unmanifested_runtime_surfaces": unmanifested_blocker_ids,
    "routes_missing_explicit_surface_rows": sorted(missing),
  }
  audit_pass = not any(audit_blockers.values())
  by_surface: dict[str, int] = {}
  for row in rows: by_surface[row["surface_class"]] = by_surface.get(row["surface_class"], 0) + 1
  return {"schema": "pure_kernel_surface_audit.v1",
          "verdict": "PURE_KERNEL_SURFACE_AUDIT_PASS" if audit_pass else "PURE_KERNEL_SURFACE_AUDIT_DEBT_FOUND",
          "strict_default_purity": report,
          "audit_blockers": audit_blockers,
          "summary": {"routes_by_surface_class": by_surface,
                      "manifest_contradictions": [r["route_id"] for r in report["manifest_contradictions"]],
                      "unmanifested_runtime_surfaces": unmanifested_ids,
                      "unmanifested_runtime_surface_blockers": unmanifested_blocker_ids,
                      "missing_writer_files": missing_writer_files,
                      "routes_with_missing_writer_files": routes_with_missing_writer_files,
                      "routes_missing_explicit_surface_rows": sorted(missing)},
          "routes": rows,
          "unmanifested_runtime_surfaces": unmanifested_rows,
          "next": ["align route_manifest provenance with strict selected-surface classifications",
                   "add explicit manifest rows for unmanifested runtime-capable surfaces",
                   "replace strict blockers with ordinary tinygrad graph or descriptor-owned generated codegen"]}


if __name__ == "__main__":
  print(json.dumps(build(), indent=2))
