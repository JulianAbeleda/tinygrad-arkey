"""Typed AMD backend resource/artifact join.

This module records facts about one lowered program without participating in
allocation or route selection.  It is deliberately independent of the
research-plane candidate schemas so a backend capture can be reused by any
compiler-owned route.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib, re
from typing import Any
from tinygrad.codegen.opt.register_contracts import RegisterBank

AMD_ARTIFACT_SCHEMA = "tinygrad.amd.resource_artifact.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _hash(value: bytes | str, name: str) -> str:
  if isinstance(value, str): value = value.encode()
  if not isinstance(value, bytes): raise TypeError(f"{name} must be bytes or str")
  return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class AMDPhysicalInterval:
  """Half-open physical register interval owned by one logical role."""
  logical_role: str
  bank: RegisterBank | str
  start: int
  end: int
  purpose: str = "value"

  def __post_init__(self):
    if not isinstance(self.logical_role, str) or not self.logical_role:
      raise ValueError("physical interval logical_role must be non-empty")
    try: bank = self.bank if isinstance(self.bank, RegisterBank) else RegisterBank(self.bank)
    except (TypeError, ValueError) as exc: raise ValueError("physical interval bank must be vgpr or sgpr") from exc
    object.__setattr__(self, "bank", bank)
    if any(not isinstance(x, int) or isinstance(x, bool) for x in (self.start, self.end)) or self.start < 0 or self.end <= self.start:
      raise ValueError("physical interval must satisfy 0 <= start < end")
    if not isinstance(self.purpose, str) or not self.purpose:
      raise ValueError("physical interval purpose must be non-empty")

  def to_json(self) -> dict[str, Any]:
    return {"logical_role": self.logical_role, "bank": self.bank.value, "start": self.start,
            "end": self.end, "purpose": self.purpose}


@dataclass(frozen=True)
class AMDResourceFacts:
  """Final backend resource facts; zero is explicit, unknown is rejected."""
  vgpr: int
  sgpr: int
  lds_bytes: int = 0
  scratch_bytes: int = 0
  vgpr_spills: int = 0
  sgpr_spills: int = 0
  workgroup_threads: int | None = None
  wavefront_size: int | None = None

  def __post_init__(self):
    for name in ("vgpr", "sgpr", "lds_bytes", "scratch_bytes", "vgpr_spills", "sgpr_spills"):
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative int")
    for name in ("workgroup_threads", "wavefront_size"):
      value = getattr(self, name)
      if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
        raise ValueError(f"{name} must be a positive int when provided")

  def to_json(self) -> dict[str, Any]:
    return {"vgpr": self.vgpr, "sgpr": self.sgpr, "lds_bytes": self.lds_bytes,
            "scratch_bytes": self.scratch_bytes, "vgpr_spills": self.vgpr_spills,
            "sgpr_spills": self.sgpr_spills, "workgroup_threads": self.workgroup_threads,
            "wavefront_size": self.wavefront_size}


@dataclass(frozen=True)
class AMDResourceArtifact:
  """Identity-joined target, code, candidate, register map, and resources."""
  target: str
  abi: str
  source_sha256: str
  binary_sha256: str
  candidate_identity: str
  resources: AMDResourceFacts
  intervals: tuple[AMDPhysicalInterval, ...]
  schema: str = AMD_ARTIFACT_SCHEMA

  def __post_init__(self):
    if self.schema != AMD_ARTIFACT_SCHEMA: raise ValueError("unsupported AMD resource artifact schema")
    for name, value in (("target", self.target), ("abi", self.abi), ("candidate_identity", self.candidate_identity)):
      if not isinstance(value, str) or not value: raise ValueError(f"{name} must be non-empty")
    for name, value in (("source_sha256", self.source_sha256), ("binary_sha256", self.binary_sha256),
                        ("candidate_identity", self.candidate_identity)):
      if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase sha256 identity")
    if not isinstance(self.resources, AMDResourceFacts): raise TypeError("resources must be AMDResourceFacts")
    if not isinstance(self.intervals, tuple) or not self.intervals or any(not isinstance(x, AMDPhysicalInterval) for x in self.intervals):
      raise ValueError("artifact requires typed physical intervals")
    by_bank = {bank: sorted((x for x in self.intervals if x.bank is bank), key=lambda x: (x.start, x.end, x.logical_role))
               for bank in (RegisterBank.VGPR, RegisterBank.SGPR)}
    for bank, rows in by_bank.items():
      limit = getattr(self.resources, bank)
      for row in rows:
        if row.end > limit: raise ValueError(f"{bank.value} interval {row} exceeds allocated {limit}")
      for previous, current in zip(rows, rows[1:]):
        if current.start < previous.end:
          raise ValueError(f"overlapping {bank.value} intervals: {previous} and {current}")

  def to_json(self) -> dict[str, Any]:
    role_map: dict[str, list[dict[str, Any]]] = {}
    for interval in self.intervals: role_map.setdefault(interval.logical_role, []).append(interval.to_json())
    return {"schema": self.schema, "target": self.target, "abi": self.abi,
            "source_sha256": self.source_sha256, "binary_sha256": self.binary_sha256,
            "candidate_identity": self.candidate_identity, "resources": self.resources.to_json(),
            "logical_role_intervals": {role: role_map[role] for role in sorted(role_map)},
            "physical_intervals": [x.to_json() for x in self.intervals]}


def join_amd_resource_artifact(*, target: str, abi: str, source: bytes | str, binary: bytes,
                               candidate_identity: str, resources: AMDResourceFacts,
                               intervals: tuple[AMDPhysicalInterval, ...] | list[AMDPhysicalInterval]) -> AMDResourceArtifact:
  """Build the one authoritative join; all identity/resource checks are fail-closed."""
  if not isinstance(intervals, (tuple, list)): raise TypeError("intervals must be a tuple or list")
  return AMDResourceArtifact(target, abi, _hash(source, "source"), _hash(binary, "binary"), candidate_identity,
                             resources, tuple(intervals))


def validate_amd_resource_artifact(artifact: AMDResourceArtifact, *, expected_target: str | None = None,
                                   expected_abi: str | None = None, expected_candidate_identity: str | None = None) -> AMDResourceArtifact:
  """Revalidate an artifact and optional external identity bindings."""
  if not isinstance(artifact, AMDResourceArtifact): raise TypeError("expected AMDResourceArtifact")
  if expected_target is not None and artifact.target != expected_target: raise ValueError("AMD target identity mismatch")
  if expected_abi is not None and artifact.abi != expected_abi: raise ValueError("AMD ABI identity mismatch")
  if expected_candidate_identity is not None and artifact.candidate_identity != expected_candidate_identity:
    raise ValueError("AMD candidate identity mismatch")
  return artifact
