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


def dual_wmma_fused_call_report(source: str, allocation_shapes: tuple[tuple[int, ...], ...] = ()) -> dict[str, object]:
  """Return the strict compiler-side dual-WMMA diagnostic.

  This is intentionally only an evidence parser: it never enables a lowering.  A
  candidate is promotable only when one fused CALL contains explicit QK and PV
  WMMA markers, shaped-fragment construction, and no materialized score or
  probability tensor.  Missing or ambiguous evidence fails closed.
  """
  upper = source.upper()
  calls = upper.count("CALL")
  # Role attribution must be present on the same generated-code line as WMMA;
  # a generic WMMA elsewhere must never satisfy either contraction gate.
  source_wmma_lines = tuple(line.upper() for line in source.splitlines() if "WMMA" in line.upper())
  qk_source = sum("QK" in line for line in source_wmma_lines)
  pv_source = sum("PV" in line for line in source_wmma_lines)
  qk, pv = qk_source > 0, pv_source > 0
  shaped = "SHAPED_WMMA" in upper or "TILE_GATHER" in upper
  # The census supplies logical attention buffers, which have a batch/head
  # prefix in addition to the T x KV matrix.  Small 2-D fragments are not
  # score/probability materializations and must not trip this gate.
  score_probability = any(len(shape) >= 3 and shape[-2] > 1 and shape[-1] > 1 for shape in allocation_shapes)
  report = {"single_call": calls == 1, "qk_wmma": qk, "pv_wmma": pv,
            "qk_source_wmma_lines": qk_source, "pv_source_wmma_lines": pv_source,
            "shaped_fragments": shaped, "full_score_probability_buffers": score_probability}
  report["promotable"] = bool(report["single_call"] and qk and pv and shaped and not score_probability)
  return report

def dual_wmma_fused_call_fixture(*, isa: str | None = None,
                                 allocation_shapes: tuple[tuple[int, ...], ...] = ((16, 16), (16, 16))) -> dict[str, object]:
  """Build a deterministic source/ISA fixture for the dual-WMMA gate.

  This is intentionally synthetic: it exercises the evidence contract without
  pretending that the production composite scheduler already emits both
  instructions.  ``isa`` is optional so real compiler captures can be plugged
  in later; absent ISA keeps the fixture non-promotable.
  """
  source = ("CALL fused_attention\n"
            "// QK WMMA score tile\n"
            "// PV WMMA value tile\n"
            "SHAPED_WMMA(TILE_GATHER score,value,acc)\n")
  report = dual_wmma_fused_call_report(source, allocation_shapes)
  isa_lines = tuple(line.upper() for line in (isa or "").splitlines() if "WMMA" in line.upper())
  qk_isa = sum("QK" in line for line in isa_lines)
  pv_isa = sum("PV" in line for line in isa_lines)
  report.update({"source": source, "isa_lines": isa_lines,
                 "qk_isa_wmma_instructions": qk_isa, "pv_isa_wmma_instructions": pv_isa,
                 "isa_captured": bool(isa_lines),
                 # The fixture is deliberately not promotable unless both roles
                 # are attributed in the real ISA capture, not merely any WMMA.
                 "role_attributed_isa": bool(qk_isa and pv_isa),
                 "promotable": bool(report["promotable"] and qk_isa and pv_isa)})
  return report

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
           "attention_workloads", "authority_command", "dual_wmma_fused_call_report",
           "dual_wmma_fused_call_fixture",
           "fused_wmma_role_report", "geometry_candidates"]
