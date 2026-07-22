"""Shared, non-promotional workload and geometry evidence for prefill attention.

This module deliberately enumerates the same geometry domain for both model
routes.  It does not score or select a kernel: ranking needs measured GPU
evidence from a real fused schedule, and no such schedule exists yet.
"""
from __future__ import annotations

from dataclasses import dataclass

from extra.qk.model_profiles import MODEL_PROFILES, ModelProfile
from extra.qk.prefill_harness import prefill_authority_argv, prefill_run_profile, resolve_prefill_model_profile

def fused_wmma_role_report(source: str) -> dict[str, object]:
  """Fail-closed diagnostic requiring explicit QK/PV WMMA in one CALL."""
  lines = tuple(line for line in source.splitlines() if "WMMA" in line.upper())
  upper = "\n".join(lines).upper()
  calls = source.upper().count("CALL")
  qk, pv = "QK" in upper, "PV" in upper
  return {"wmma_lines": len(lines), "qk": qk, "pv": pv,
          "single_call": calls == 1, "promotable": bool(qk and pv and calls == 1)}

ATTENTION_EVIDENCE_SCHEMA = "tinygrad.shared_attention_evidence.v1"
DEFAULT_CONTEXTS = (512, 2048, 4096)
_GEOMETRIES = ((16, 32, 1, 1), (16, 64, 1, 2), (32, 64, 2, 1), (32, 128, 2, 2), (64, 64, 4, 1))


@dataclass(frozen=True)
class AttentionWorkload:
  profile_id: str
  device: str
  activation_dtype: str
  B: int
  Hq: int
  Hkv: int
  G: int
  T: int
  KV: int
  Hd: int
  causal: bool = True

  def to_json(self) -> dict[str, int | str | bool]:
    return {"profile_id": self.profile_id, "device": self.device, "activation_dtype": self.activation_dtype,
            "B": self.B, "Hq": self.Hq, "Hkv": self.Hkv, "G": self.G, "T": self.T, "KV": self.KV,
            "Hd": self.Hd, "causal": self.causal}


@dataclass(frozen=True)
class AttentionGeometry:
  Bq: int
  Bkv: int
  waves: int
  stages: int

  @property
  def candidate_id(self) -> str:
    return f"bq{self.Bq}-bkv{self.Bkv}-w{self.waves}-s{self.stages}"

  def to_json(self) -> dict[str, int | str]:
    return {"candidate_id": self.candidate_id, "Bq": self.Bq, "Bkv": self.Bkv,
            "waves": self.waves, "stages": self.stages}


def attention_workloads(*, contexts: tuple[int, ...] = DEFAULT_CONTEXTS) -> tuple[AttentionWorkload, ...]:
  """Return one route-neutral attention workload row per real profile/context."""
  if not contexts or any(not isinstance(x, int) or x <= 0 for x in contexts):
    raise ValueError("contexts must be non-empty positive integers")
  rows = []
  for profile in MODEL_PROFILES:
    shape = profile.attention
    if shape.Hq % shape.Hkv: raise ValueError(f"profile {profile.id} has non-integral GQA")
    for context in contexts:
      rows.append(AttentionWorkload(profile.id, profile.device_profile, "float16", shape.B, shape.Hq, shape.Hkv,
                                    shape.Hq // shape.Hkv, context, context, shape.Hd))
  return tuple(rows)


def geometry_candidates(workload: AttentionWorkload) -> tuple[AttentionGeometry, ...]:
  """Enumerate common feasible geometry labels; timing owns ranking and promotion."""
  if workload.Hd % 16: return ()
  return tuple(AttentionGeometry(*row) for row in _GEOMETRIES if row[0] <= workload.T and row[1] <= workload.KV)


def authority_command(profile: ModelProfile, *, artifact_path: str) -> list[str]:
  """Build the sole whole-prefill command shape used for baseline and candidate runs."""
  resolved = resolve_prefill_model_profile(profile.id)
  run = prefill_run_profile("authority")
  return prefill_authority_argv(resolved.default_model, run, model_profile_id=profile.id, pin_clock=True,
                                artifact_path=artifact_path)


__all__ = ["ATTENTION_EVIDENCE_SCHEMA", "DEFAULT_CONTEXTS", "AttentionGeometry", "AttentionWorkload",
           "attention_workloads", "authority_command", "fused_wmma_role_report", "geometry_candidates"]
