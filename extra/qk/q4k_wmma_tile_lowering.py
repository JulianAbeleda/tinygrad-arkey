#!/usr/bin/env python3
"""Data-only full-role lowering contract for Q4_K/Q8_1 tiled WMMA prefill.

This module intentionally does not emit kernels. It centralizes the role-shape and tile-lifecycle
contract that a future scheduler/codegen-owned implementation must satisfy before `wmma_tiled`
can bind full 14B prefill shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.layout import Q4_K_BLOCK_ELEMS, Q8_1_BLOCK_ELEMS
from extra.qk.model_profiles import prefill_role_shapes, qwen3_14b_q4k_m_gfx1100_profile

VALID_WMMA_SURFACES = ("tc_matcher_tile", "shaped_wmma_tile")
VALID_OUTPUT_LAYOUTS = ("direct",)
SCHEDULER_OWNED_TILE_LOOP_CONTRACT = "scheduler_owned_tiled_wmma_contract_v1"
SCHEDULER_OWNED_TILE_LOOP_BLOCKER = "scheduler_owned_tile_loop_missing"


def build_scheduler_owned_tile_loop_contract(roles: tuple[Int8WMMATileLoweringSpec, ...], *, route_id: str) -> dict[str, Any]:
  required_roles = [role.role for role in roles if role.requires_scheduler_owned_loop]
  return {
    "contract": SCHEDULER_OWNED_TILE_LOOP_CONTRACT,
    "route_id": route_id,
    "required": bool(required_roles),
    "required_roles": required_roles,
    "required_axes": ("m_tile", "n_tile", "group_tile"),
    "requires_scheduler_owned_loop": required_roles != [],
    "remaining_blocker": SCHEDULER_OWNED_TILE_LOOP_BLOCKER if required_roles else None,
  }


@dataclass(frozen=True)
class Int8WMMATileLoweringSpec:
  m: int
  n: int
  k: int
  role: str
  m_tile: int = 16
  n_tile: int = 16
  group_tile: int = 1
  group_elems: int = Q8_1_BLOCK_ELEMS
  wmma_m: int = 16
  wmma_n: int = 16
  wmma_k: int = 16
  waves_per_block: int = 1
  output_layout: str = "direct"
  wmma_surface: str = "shaped_wmma_tile"

  @property
  def groups(self) -> int:
    return self.k // self.group_elems

  @property
  def k_blocks(self) -> int:
    return self.k // Q4_K_BLOCK_ELEMS

  @property
  def m_tiles(self) -> int:
    return self.m // self.m_tile

  @property
  def n_tiles(self) -> int:
    return self.n // self.n_tile

  @property
  def group_tiles(self) -> int:
    return self.groups // self.group_tile

  @property
  def output_tiles(self) -> int:
    return self.m_tiles * self.n_tiles

  @property
  def raw_tile_steps(self) -> int:
    return self.output_tiles * self.group_tiles

  @property
  def wmma_fragments_per_raw_tile(self) -> int:
    return self.group_elems // self.wmma_k

  @property
  def wmma_fragment_ops(self) -> int:
    return self.raw_tile_steps * self.wmma_fragments_per_raw_tile

  @property
  def live_raw_elems(self) -> int:
    return self.m_tile * self.n_tile * self.group_tile

  @property
  def forbidden_full_raw_elems(self) -> int:
    return self.groups * self.m * self.n

  @property
  def bounded_raw_ok(self) -> bool:
    return self.live_raw_elems < self.forbidden_full_raw_elems

  @property
  def kernel_name(self) -> str:
    role = f"_{self.role}" if self.role else ""
    return f"prefill_q4k_q8_1_wmma_tiled_generated_gemm{role}_{self.n}_{self.k}_{self.m}_{self.m_tile}x{self.n_tile}x{self.group_tile}"

  @property
  def requires_scheduler_owned_loop(self) -> bool:
    return self.output_tiles > 1 or self.group_tiles > 1

  def validate(self) -> None:
    if min(self.m, self.n, self.k) <= 0:
      raise ValueError(f"shape must be positive, got m={self.m} n={self.n} k={self.k}")
    if self.group_elems != Q8_1_BLOCK_ELEMS:
      raise ValueError(f"group_elems must be {Q8_1_BLOCK_ELEMS}, got {self.group_elems}")
    if self.k % Q4_K_BLOCK_ELEMS:
      raise ValueError(f"k={self.k} must be a multiple of Q4_K block elems {Q4_K_BLOCK_ELEMS}")
    if self.m % self.m_tile or self.n % self.n_tile:
      raise ValueError(f"m/n must divide tile sizes exactly, got m={self.m} n={self.n} tile={self.m_tile}x{self.n_tile}")
    if self.groups % self.group_tile:
      raise ValueError(f"groups={self.groups} must divide group_tile={self.group_tile}")
    if self.m_tile % self.wmma_m or self.n_tile % self.wmma_n or self.group_elems % self.wmma_k:
      raise ValueError("tile/group geometry must align with WMMA geometry")
    if self.wmma_surface not in VALID_WMMA_SURFACES:
      raise ValueError(f"unknown wmma_surface={self.wmma_surface!r}")
    if self.output_layout not in VALID_OUTPUT_LAYOUTS:
      raise ValueError(f"unsupported output_layout={self.output_layout!r}")
    if self.waves_per_block <= 0:
      raise ValueError(f"waves_per_block must be positive, got {self.waves_per_block}")

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {
      "role": self.role,
      "m": self.m,
      "n": self.n,
      "k": self.k,
      "groups": self.groups,
      "k_blocks": self.k_blocks,
      "tile": {
        "m_tile": self.m_tile,
        "n_tile": self.n_tile,
        "group_tile": self.group_tile,
        "group_elems": self.group_elems,
        "wmma_m": self.wmma_m,
        "wmma_n": self.wmma_n,
        "wmma_k": self.wmma_k,
      },
      "grid": {
        "m_tiles": self.m_tiles,
        "n_tiles": self.n_tiles,
        "group_tiles": self.group_tiles,
        "output_tiles": self.output_tiles,
        "raw_tile_steps": self.raw_tile_steps,
        "wmma_fragments_per_raw_tile": self.wmma_fragments_per_raw_tile,
        "wmma_fragment_ops": self.wmma_fragment_ops,
      },
      "bounds": {
        "live_raw_elems": self.live_raw_elems,
        "forbidden_full_raw_elems": self.forbidden_full_raw_elems,
        "bounded_raw_ok": self.bounded_raw_ok,
      },
      "lowering": {
        "wmma_surface": self.wmma_surface,
        "waves_per_block": self.waves_per_block,
        "output_layout": self.output_layout,
        "requires_scheduler_owned_loop": self.requires_scheduler_owned_loop,
        "kernel_name": self.kernel_name,
      },
    }


@dataclass(frozen=True)
class Q4KWMMAFullRoleLoweringSpec:
  roles: tuple[Int8WMMATileLoweringSpec, ...]
  route_id: str = "prefill_q4k_int8_wmma_tiled_research"
  target: str = "amd_gfx1100"
  implementation: str = "scheduler_owned_tiled_wmma_contract_v1"

  def validate(self) -> None:
    if not self.roles:
      raise ValueError("full-role lowering spec needs at least one role")
    seen = set[str]()
    for role in self.roles:
      role.validate()
      if role.role in seen:
        raise ValueError(f"duplicate role {role.role!r}")
      seen.add(role.role)

  def to_json(self) -> dict[str, Any]:
    self.validate()
    return {
      "route_id": self.route_id,
      "target": self.target,
      "implementation": self.implementation,
      "roles": [role.to_json() for role in self.roles],
      "role_count": len(self.roles),
      "total_output_tiles": sum(role.output_tiles for role in self.roles),
      "total_raw_tile_steps": sum(role.raw_tile_steps for role in self.roles),
      "max_forbidden_full_raw_elems": max(role.forbidden_full_raw_elems for role in self.roles),
      "max_live_raw_elems": max(role.live_raw_elems for role in self.roles),
    }


def _role_shape_tuple(role_shape: Any) -> tuple[str, int, int, int]:
  if isinstance(role_shape, dict):
    return (
      str(role_shape["role"]),
      int(role_shape["M"] if "M" in role_shape else role_shape["m"]),
      int(role_shape["N"] if "N" in role_shape else role_shape["n"]),
      int(role_shape["K"] if "K" in role_shape else role_shape["k"]),
    )
  if isinstance(role_shape, tuple):
    if len(role_shape) < 4:
      raise ValueError(f"role shape tuple must include role,M,N,K, got {role_shape!r}")
    role, m, n, k = role_shape[:4]
    return str(role), int(m), int(n), int(k)
  return (
    str(getattr(role_shape, "role")),
    int(role_shape.M if hasattr(role_shape, "M") else role_shape.m),
    int(role_shape.N if hasattr(role_shape, "N") else role_shape.n),
    int(role_shape.K if hasattr(role_shape, "K") else role_shape.k),
  )


def q4k_prefill_role_shape_tuples(profile: Any) -> tuple[tuple[str, int, int, int], ...]:
  return tuple(_role_shape_tuple(role_shape) for role_shape in prefill_role_shapes(profile))


QWEN3_14B_Q4K_ROLE_SHAPES: tuple[tuple[str, int, int, int], ...] = q4k_prefill_role_shape_tuples(
  qwen3_14b_q4k_m_gfx1100_profile()
)


def describe_int8_wmma_tile_lowering(m:int, n:int, k:int, *, role:str, m_tile:int=16, n_tile:int=16,
                                     group_tile:int=1, wmma_surface:str="shaped_wmma_tile") -> Int8WMMATileLoweringSpec:
  spec = Int8WMMATileLoweringSpec(m=m, n=n, k=k, role=role, m_tile=m_tile, n_tile=n_tile,
                                  group_tile=group_tile, wmma_surface=wmma_surface)
  spec.validate()
  return spec


def describe_q4k_full_role_lowering(profile: Any, *, wmma_surface:str="shaped_wmma_tile") -> Q4KWMMAFullRoleLoweringSpec:
  spec = Q4KWMMAFullRoleLoweringSpec(tuple(
    describe_int8_wmma_tile_lowering(m, n, k, role=role, wmma_surface=wmma_surface)
    for role, m, n, k in q4k_prefill_role_shape_tuples(profile)
  ))
  spec.validate()
  return spec


def describe_qwen3_14b_q4k_full_role_lowering(*, wmma_surface:str="shaped_wmma_tile") -> Q4KWMMAFullRoleLoweringSpec:
  return describe_q4k_full_role_lowering(qwen3_14b_q4k_m_gfx1100_profile(), wmma_surface=wmma_surface)


def main() -> None:
  import json
  print(json.dumps(describe_qwen3_14b_q4k_full_role_lowering().to_json(), indent=2))


if __name__ == "__main__":
  main()
