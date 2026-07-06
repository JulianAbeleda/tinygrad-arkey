from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IntrinsicRow:
  name: str
  scope: str
  intrinsic: str
  allow_markers: tuple[str, ...] = ()
  banned_markers: tuple[str, ...] = ()
  backend_emitters: tuple[str, ...] = ()
  route_local_emitters: tuple[str, ...] = ()
  note: str = ""

  def to_json(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "scope": self.scope,
      "intrinsic": self.intrinsic,
      "allow_markers": list(self.allow_markers),
      "banned_markers": list(self.banned_markers),
      "backend_emitters": list(self.backend_emitters),
      "route_local_emitters": list(self.route_local_emitters),
      "note": self.note,
    }


_BACKEND_OWNED = (
  IntrinsicRow(
    name="wmma_mfma",
    scope="backend_owned",
    intrinsic="WMMA/MFMA",
    allow_markers=(
      "Ops.WMMA",
      "__builtin_amdgcn_wmma_",
      "__builtin_amdgcn_mfma_",
    ),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel(", "Ops.CUSTOM", "Ops.CUSTOMI"),
    backend_emitters=("tinygrad/renderer/cstyle.py", "tinygrad/renderer/llvmir.py", "tinygrad/renderer/isa/amd.py"),
    route_local_emitters=(),
    note="Allowed only when emitted by tensor-core/codegen lowering.",
  ),
  IntrinsicRow(
    name="dot4",
    scope="backend_owned",
    intrinsic="dot4",
    allow_markers=("__builtin_amdgcn_udot4", "_dp4a", "udot4"),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Ops.CUSTOM", "Ops.CUSTOMI", "Tensor.custom_kernel"),
    backend_emitters=("tinygrad/renderer/cstyle.py",),
    route_local_emitters=("extra/qk/quant/q4_k_gemv_primitive.py",),
    note="Dot4 can be backend-owned when lowered through generated path; source-string variants stay non-pure.",
  ),
  IntrinsicRow(
    name="v_dot2_fdot2",
    scope="backend_owned",
    intrinsic="v_dot2/fdot2",
    allow_markers=(
      "__builtin_amdgcn_fdot2",
      "__builtin_bit_cast(float, __builtin_amdgcn_ds_bpermute",
      "fdot2",
    ),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel(", "Ops.CUSTOM", "Ops.CUSTOMI"),
    backend_emitters=("extra/qk/fdot2_lowering.py", "tinygrad/renderer/isa/amd.py", "tinygrad/renderer/cstyle.py"),
    route_local_emitters=("extra/qk/flash_kernels.py",),
    note="Route-local fdot2/ds_bpermute string emission is permitted only through backend-owned helper lowering.",
  ),
  IntrinsicRow(
    name="cross_lane_reduction",
    scope="backend_owned",
    intrinsic="cross-lane reduction",
    allow_markers=("__builtin_amdgcn_ds_bpermute", "bpermute", "ds_bpermute"),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel(", "Ops.CUSTOM", "Ops.CUSTOMI"),
    backend_emitters=("tinygrad/renderer/isa/amd.py", "tinygrad/renderer/cstyle.py", "extra/qk/warp_reduce_lowering.py"),
    route_local_emitters=("extra/qk/amd_warp_reduce.py",),
    note="Cross-lane shuffles are pure only when owned by shared lowering, not inline route logic.",
  ),
  IntrinsicRow(
    name="exp2_fast_math",
    scope="backend_owned",
    intrinsic="exp2/fast math",
    allow_markers=(
      "Ops.EXP2",
      "__builtin_amdgcn_exp2f",
      "exp2",
    ),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel(", "Ops.CUSTOM", "Ops.CUSTOMI"),
    backend_emitters=("tinygrad/renderer/cstyle.py", "tinygrad/renderer/llvmir.py", "extra/qk/flash_common.py"),
    route_local_emitters=("extra/qk/flash_common.py", "tinygrad/llm/prefill_routes.py"),
    note="Fast-exp route must be the codegen helper path, not route-local source strings.",
  ),
  IntrinsicRow(
    name="route_local_raw_markers",
    scope="route_local",
    intrinsic="raw string/raw instruction emission",
    allow_markers=(),
    banned_markers=("asm volatile", "Ops.INS", "Ops.BINARY", "Tensor.custom_kernel", ".custom_kernel(", "Ops.CUSTOM", "Ops.CUSTOMI"),
    backend_emitters=(),
    route_local_emitters=(
      "extra/qk/flash_kernels.py",
      "extra/qk/codegen_outer_b_lds_split.py",
      "extra/qk/asm_scheduler.py",
      "tinygrad/llm/prefill_routes.py",
    ),
    note="When these markers are route-local, the path is non-pure even if marker text appears in backend source.",
  ),
)


def rows() -> list[dict[str, Any]]:
  return [r.to_json() for r in _BACKEND_OWNED]


def row(name: str) -> dict[str, Any]:
  key = {r.name: r for r in _BACKEND_OWNED}.get(name)
  if key is None:
    raise KeyError(name)
  return key.to_json()


def marker_conflicts() -> dict[str, list[str]]:
  conflicts: dict[str, list[str]] = {}
  for r in rows():
    overlap = sorted(set(r["allow_markers"]) & set(r["banned_markers"]))
    if overlap:
      conflicts[r["name"]] = overlap
  return conflicts


def build() -> dict[str, Any]:
  rows_out = rows()
  by_scope = {"backend_owned": [], "route_local": []}
  for r in rows_out:
    by_scope.setdefault(r["scope"], []).append(r["name"])
  banned = next((r for r in rows_out if r["scope"] == "route_local"), None)
  conflicts = marker_conflicts()
  return {
    "schema": "backend_intrinsic_lowering_allowlist.v1",
    "rows": rows_out,
    "marker_conflicts": conflicts,
    "summary": {
      "row_count": len(rows_out),
      "backend_owned_count": len(by_scope["backend_owned"]),
      "route_local_count": len(by_scope["route_local"]),
      "banned_marker_count": len(banned["banned_markers"]) if banned else 0,
      "marker_conflict_row_count": len(conflicts),
    },
  }
