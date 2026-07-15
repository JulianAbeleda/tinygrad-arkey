"""Independent Q4_K/Q8_1 role contracts for the beyond-parity workstream.

These are data-only research contracts.  They describe the complete logical
shape and layout at each role boundary; they do not select or promote a
kernel.  In particular, ``direct-packed`` remains the rollback route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mmq_logical_vocabulary import (
  Axis, BackendCapability, DotOp, EdgePredicate, LogicalMMQDescriptor,
  MMQCandidate, PhysicalMapping,
)
from .model_profiles import QWEN3_14B_Q4_K_M_GFX1100

Q4_ROLES = ("ffn_gate_up", "ffn_down", "attn_qo", "attn_kv")


@dataclass(frozen=True)
class Q4RoleContract:
  role: str
  shape: tuple[int, int, int]
  weight_layout: str = "q4_k_blocks_n_k"
  activation_layout: str = "q8_1_ds4_m_k"
  output_layout: str = "tokens_rows"
  edge_axes: tuple[str, ...] = ("m", "n", "k")
  route: str = "direct_packed"
  research_only: bool = True

  def __post_init__(self) -> None:
    if self.role not in Q4_ROLES:
      raise ValueError(f"unsupported Q4 role {self.role!r}")
    if len(self.shape) != 3 or any(x <= 0 for x in self.shape):
      raise ValueError("role shape must be positive M,N,K")
    if self.route != "direct_packed" or self.research_only is not True:
      raise ValueError("Q4 MMQ contracts are research-only with direct-packed rollback")
    if set(self.edge_axes) != {"m", "n", "k"}:
      raise ValueError("edge metadata must cover m, n, and k")

  @property
  def M(self) -> int: return self.shape[0]
  @property
  def N(self) -> int: return self.shape[1]
  @property
  def K(self) -> int: return self.shape[2]

  def to_dict(self) -> dict[str, Any]:
    return {"role": self.role, "shape": {"M": self.M, "N": self.N, "K": self.K},
            "weight_layout": self.weight_layout, "activation_layout": self.activation_layout,
            "output_layout": self.output_layout, "edge_axes": list(self.edge_axes),
            "route": self.route, "research_only": self.research_only}

  def candidate(self) -> MMQCandidate:
    axes = (Axis("m", self.M, 16), Axis("n", self.N, 16), Axis("k", self.K, 256),
            Axis("group", f"k/256"), Axis("activation_block", "k/32"))
    descriptor = LogicalMMQDescriptor(
      axes=axes,
      edge_predicates=tuple(EdgePredicate(axis) for axis in self.edge_axes),
      abi={"role": self.role, "shape": {"M": self.M, "N": self.N, "K": self.K},
           "weight_layout": self.weight_layout, "activation_layout": self.activation_layout,
           "output_layout": self.output_layout, "edge_predicates": list(self.edge_axes)},
    )
    return MMQCandidate(descriptor, PhysicalMapping(32, 64),
                        BackendCapability("AMD", "gfx1100", (DotOp.WMMA_I8_I8_I32,), (32,), 256, 64 * 1024))


Q4_ROLE_CONTRACTS = tuple(
  Q4RoleContract(role, QWEN3_14B_Q4_K_M_GFX1100.role_shape(role).mnk)
  for role in Q4_ROLES
)
_BY_ROLE = {contract.role: contract for contract in Q4_ROLE_CONTRACTS}


def q4_role_contract(role: str) -> Q4RoleContract:
  try: return _BY_ROLE[role]
  except KeyError as exc: raise ValueError(f"unsupported Q4 role {role!r}") from exc


def q4_role_matrix() -> dict[str, dict[str, Any]]:
  return {contract.role: contract.to_dict() | {"candidate_identity": contract.candidate().identity()}
          for contract in Q4_ROLE_CONTRACTS}


__all__ = ["Q4_ROLES", "Q4RoleContract", "Q4_ROLE_CONTRACTS", "q4_role_contract", "q4_role_matrix"]
