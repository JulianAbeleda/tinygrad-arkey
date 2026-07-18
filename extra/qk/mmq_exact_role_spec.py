"""Canonical full-role and K256 program geometry for exact Q4_K/Q8_1 MMQ evidence."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from extra.qk.prefill.q4k_q8_five_buffer_role_gate import admitted_q4k_non_fitting_roles
from extra.qk.runtime_specs import full_kernel_workload


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INVENTORY = REPO_ROOT / "bench/prefill-pure-full-kernel/qwen3-14b-mixed-quant-candidate-inventory-v1.json"
EPOCH_K = 256


@dataclass(frozen=True)
class ExactProgramGeometry:
  """Role-neutral shape compiled once and launched for each K256 epoch."""
  m: int
  n: int
  k: int = EPOCH_K

  def __post_init__(self) -> None:
    if self.k != EPOCH_K or self.m <= 0 or self.n <= 0 or self.m % 128 or self.n % 128:
      raise ValueError("exact MMQ program geometry requires aligned M/N and K=256")

  @property
  def shape(self) -> tuple[int, int, int]: return self.m, self.n, self.k

  @property
  def grid(self) -> tuple[int, int, int]: return self.n // 128, self.m // 128, 1

  @property
  def abi_elements(self) -> tuple[int, int, int, int, int]:
    return self.m * self.n, self.n * 36, 2 * self.m * 128, 2 * self.m * 4, 2 * self.m * 4


@dataclass(frozen=True)
class ExactRoleSpec:
  """One exact inventory-admitted full role and its reusable epoch program."""
  role: str
  m: int
  n: int
  k: int
  candidate_canonical_identity: str

  def __post_init__(self) -> None:
    if not self.role or min(self.m, self.n, self.k) <= 0 or self.m % 128 or self.n % 128 or self.k % EPOCH_K:
      raise ValueError("exact MMQ role requires a named, aligned M/N/K workload")
    if len(self.candidate_canonical_identity) != 64:
      raise ValueError("exact MMQ role requires its canonical admitted five-buffer candidate identity")

  @property
  def shape(self) -> tuple[int, int, int]: return self.m, self.n, self.k

  @property
  def program(self) -> ExactProgramGeometry: return ExactProgramGeometry(self.m, self.n)

  @property
  def epochs(self) -> int: return self.k // EPOCH_K


def load_exact_role_specs(inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY) -> tuple[ExactRoleSpec, ...]:
  artifact = json.loads(Path(inventory).read_text()) if isinstance(inventory, (str, Path)) else inventory
  rows = []
  for entry, admission in admitted_q4k_non_fitting_roles(artifact):
    workload = full_kernel_workload(admission.normalized_payload)
    rows.append(ExactRoleSpec(workload.role, *workload.shape, entry.canonical_identity))
  return tuple(rows)


def exact_role_spec(role: str, *, shape: tuple[int, int, int] | None = None,
                    inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY) -> ExactRoleSpec:
  matches = [row for row in load_exact_role_specs(inventory) if row.role == role]
  if len(matches) != 1: raise ValueError(f"expected one admitted exact Q4 role {role!r}, got {len(matches)}")
  row = matches[0]
  if shape is not None and tuple(shape) != row.shape:
    raise ValueError(f"requested shape {tuple(shape)!r} differs from admitted {role!r} shape {row.shape!r}")
  return row


def exact_role_spec_for_shape(role: str, shape: tuple[int, int, int],
                              *, inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY) -> ExactRoleSpec:
  return exact_role_spec(role, shape=shape, inventory=inventory)


def exact_role_spec_from_shape(shape: tuple[int, int, int], *,
                               inventory: str | Path | Mapping[str, Any] = DEFAULT_INVENTORY) -> ExactRoleSpec:
  matches = [row for row in load_exact_role_specs(inventory) if row.shape == tuple(shape)]
  if len(matches) != 1: raise ValueError(f"expected one admitted exact Q4 role for shape {tuple(shape)!r}, got {len(matches)}")
  return matches[0]


DEFAULT_EXACT_ROLE_SPEC = exact_role_spec("ffn_gate_up")


__all__ = ["DEFAULT_EXACT_ROLE_SPEC", "DEFAULT_INVENTORY", "EPOCH_K", "ExactProgramGeometry", "ExactRoleSpec",
           "exact_role_spec", "exact_role_spec_for_shape", "exact_role_spec_from_shape", "load_exact_role_specs"]
