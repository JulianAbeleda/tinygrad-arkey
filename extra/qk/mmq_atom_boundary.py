#!/usr/bin/env python3
"""Fail-closed boundary for the 14B Q4_K x Q8_1 hybrid MMQ backend atom.

This file intentionally does not implement or select a GPU kernel. It is the
typed contract for the one allowed handwritten atom described in
docs/14b-prefill-hybrid-mmq-machine-search-scope-20260710.md. Importing this
module does not register a route, select a route, or provide a fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from extra.qk.layout import Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS


PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION = "hybrid_machine_search_mmq_backend_atom"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_PROVENANCE = "compiler_primitive_spec_owned"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_STATUS = "research_boundary_stub"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_PHASE = "prefill"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_QUANT = "Q4_K"
PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ACTIVATION = "Q8_1"

_SUPPORTED_ROLES = ("ffn_gate_up", "attn_qo", "attn_kv")
_PACKED_WEIGHT_LAYOUT = "ggml_q4_k_bytes_row_major_nk"
_ACTIVATION_LAYOUT = "q8_1_row_major_mk_scales_per_32"
_OUTPUT_LAYOUT = "row_major_mn_tile"
_PARTS_SPLIT_POLICY = "single_k_tile"
_HAND_SURFACE = "one_parameterized_q4_k_q8_1_mmq_tile_atom"


@dataclass(frozen=True)
class Prefill14BHybridMMQAtomSpec:
  role: str
  m: int
  n: int
  k: int
  tile_m: int = 16
  tile_n: int = 16
  tile_k: int = Q4_K_BLOCK_ELEMS
  quant_format: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_QUANT
  activation_format: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ACTIVATION
  packed_weight_layout: str = _PACKED_WEIGHT_LAYOUT
  activation_layout: str = _ACTIVATION_LAYOUT
  output_layout: str = _OUTPUT_LAYOUT
  parts_split_policy: str = _PARTS_SPLIT_POLICY
  route_id: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID
  classification: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION
  promoted: bool = False
  pure_generated: bool = False

  @property
  def k_groups(self) -> int:
    return self.k // Q8_1_BLOCK_ELEMS

  @property
  def tile_k_groups(self) -> int:
    return self.tile_k // Q8_1_BLOCK_ELEMS

  def validate(self) -> None:
    if self.route_id != PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID:
      raise ValueError(f"route_id must be {PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID!r}, got {self.route_id!r}")
    if self.classification != PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION:
      raise ValueError(f"classification must be hybrid MMQ atom, got {self.classification!r}")
    if self.promoted:
      raise ValueError("14B hybrid MMQ atom boundary is not promoted")
    if self.pure_generated:
      raise ValueError("14B hybrid MMQ atom boundary cannot claim pure_generated=True")
    if self.role not in _SUPPORTED_ROLES:
      raise ValueError(f"unsupported role {self.role!r}; supported={_SUPPORTED_ROLES}")
    if self.quant_format != "Q4_K":
      raise ValueError(f"quant_format must be Q4_K, got {self.quant_format!r}")
    if self.activation_format != "Q8_1":
      raise ValueError(f"activation_format must be Q8_1, got {self.activation_format!r}")
    if self.packed_weight_layout != _PACKED_WEIGHT_LAYOUT:
      raise ValueError(f"unsupported packed_weight_layout={self.packed_weight_layout!r}")
    if self.activation_layout != _ACTIVATION_LAYOUT:
      raise ValueError(f"unsupported activation_layout={self.activation_layout!r}")
    if self.output_layout != _OUTPUT_LAYOUT:
      raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.parts_split_policy != _PARTS_SPLIT_POLICY:
      raise ValueError(f"unsupported parts_split_policy={self.parts_split_policy!r}")
    if self.m <= 0 or self.n <= 0 or self.k <= 0:
      raise ValueError(f"m/n/k must be positive, got m={self.m} n={self.n} k={self.k}")
    if self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems {Q4_K_BLOCK_ELEMS}")
    if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
      raise ValueError(f"tile_m/tile_n/tile_k must be positive, got {self.tile_m}/{self.tile_n}/{self.tile_k}")
    if self.tile_k % Q8_1_BLOCK_ELEMS:
      raise ValueError(f"tile_k={self.tile_k} must be Q8_1 block aligned")
    if self.tile_k > self.k:
      raise ValueError(f"tile_k={self.tile_k} cannot exceed k={self.k}")

  def to_json(self) -> dict[str, Any]:
    return {
      "route_id": self.route_id, "classification": self.classification, "promoted": self.promoted,
      "pure_generated": self.pure_generated, "role": self.role, "phase": PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_PHASE,
      "M": self.m, "N": self.n, "K": self.k, "tile_m": self.tile_m, "tile_n": self.tile_n,
      "tile_k": self.tile_k, "k_groups": self.k_groups, "tile_k_groups": self.tile_k_groups,
      "quant_format": self.quant_format, "activation_format": self.activation_format,
      "packed_weight_layout": self.packed_weight_layout, "activation_layout": self.activation_layout,
      "output_layout": self.output_layout, "parts_split_policy": self.parts_split_policy,
    }


@dataclass(frozen=True)
class Prefill14BHybridMMQAtomDescriptor:
  route_id: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID
  status: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_STATUS
  classification: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_CLASSIFICATION
  provenance: str = PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_PROVENANCE
  promoted: bool = False
  pure_generated: bool = False
  strict_fallback: bool = True
  live_default_route: bool = False
  selector_env: dict[str, str] = field(default_factory=dict)
  fallback_route_id: str | None = None
  roles: tuple[str, ...] = _SUPPORTED_ROLES
  supported_quant_formats: tuple[str, ...] = ("Q4_K",)
  supported_activation_formats: tuple[str, ...] = ("Q8_1",)
  hand_surface: str = _HAND_SURFACE
  authority_gate: str = "not_implemented"

  def validate(self) -> None:
    if self.route_id != PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID:
      raise ValueError(f"unexpected route_id={self.route_id!r}")
    if self.promoted or self.live_default_route:
      raise ValueError("14B hybrid MMQ atom descriptor is a non-promoted boundary, not a live default route")
    if self.pure_generated:
      raise ValueError("14B hybrid MMQ atom descriptor cannot claim pure_generated=True")
    if self.selector_env:
      raise ValueError("14B hybrid MMQ atom boundary must not expose selector env")
    if self.fallback_route_id is not None:
      raise ValueError("14B hybrid MMQ atom boundary must not declare a fallback route")
    if not self.strict_fallback:
      raise ValueError("14B hybrid MMQ atom boundary must be strict/fail-closed")

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {
      "route_id": self.route_id, "status": self.status, "classification": self.classification,
      "provenance": self.provenance, "promoted": self.promoted, "pure_generated": self.pure_generated,
      "strict_fallback": self.strict_fallback, "live_default_route": self.live_default_route,
      "selector_env": dict(self.selector_env), "fallback_route_id": self.fallback_route_id,
      "roles": list(self.roles), "supported_quant_formats": list(self.supported_quant_formats),
      "supported_activation_formats": list(self.supported_activation_formats), "hand_surface": self.hand_surface,
      "authority_gate": self.authority_gate,
    }


class Prefill14BHybridMMQAtomUnsupported(NotImplementedError):
  """Raised whenever the non-promoted 14B hybrid MMQ atom boundary is called."""


def describe_prefill_14b_q4k_q8_1_hybrid_mmq_atom(
  *, role: str, m: int = 512, n: int, k: int = 5120, tile_m: int = 16, tile_n: int = 16,
  tile_k: int = Q4_K_BLOCK_ELEMS,
) -> Prefill14BHybridMMQAtomSpec:
  spec = Prefill14BHybridMMQAtomSpec(role=role, m=m, n=n, k=k, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k)
  spec.validate()
  return spec


def prefill_14b_q4k_q8_1_hybrid_mmq_atom_descriptor() -> Prefill14BHybridMMQAtomDescriptor:
  desc = Prefill14BHybridMMQAtomDescriptor()
  desc.validate()
  return desc


def prefill_14b_q4k_q8_1_hybrid_mmq_atom(*_args: Any, spec: Prefill14BHybridMMQAtomSpec, **_kwargs: Any) -> Any:
  spec.validate()
  raise Prefill14BHybridMMQAtomUnsupported(
    f"{PREFILL_14B_Q4K_Q8_1_HYBRID_MMQ_ATOM_ROUTE_ID} is a non-promoted boundary stub; "
    "GPU MMQ atom body is not implemented and no fallback route is permitted"
  )
