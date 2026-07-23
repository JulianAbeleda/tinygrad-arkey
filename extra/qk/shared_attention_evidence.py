"""Shared, non-promotional workload and geometry evidence for prefill attention.

This module deliberately enumerates the same geometry domain for both model
routes.  It does not score or select a kernel: ranking needs measured GPU
evidence from a real fused schedule, and no such schedule exists yet.
"""
from __future__ import annotations

from dataclasses import dataclass

from extra.qk.model_profiles import MODEL_PROFILES, ModelProfile
from extra.qk.prefill_harness import prefill_authority_argv, prefill_run_profile, resolve_prefill_model_profile
from extra.qk.shared_attention_capture import (ACC_SLICE_CAPTURE_SCHEMA, CAPTURE_SCHEMA, PHASE_CAPTURE_SCHEMA,
  SharedAttentionCompilerCapture)

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

def _shared_attention_route_key(capture:SharedAttentionCompilerCapture) -> tuple[str,str,str]:
  ctx = capture.candidate_context
  if ctx.start_pos == 0 and ctx.kv_tokens != ctx.q_tokens: raise ValueError("first-chunk capture has prefix KV geometry")
  if ctx.start_pos > 0 and ctx.kv_tokens != ctx.start_pos+ctx.q_tokens: raise ValueError("prefix capture KV geometry is not exact")
  return (ctx.profile,ctx.strategy,"first" if ctx.start_pos == 0 else "prefix")

def _shared_attention_v2_proof_artifact(captures:tuple[SharedAttentionCompilerCapture,...]) -> dict[str, object]:
  """Aggregate only four validated, content-addressed legacy compiler captures.

  Raw caller-provided source, ISA, ownership, or route claims are deliberately
  not accepted. Every fact in the proof originates in a capture constructor.
  """
  if not isinstance(captures,tuple) or any(not isinstance(x,SharedAttentionCompilerCapture) for x in captures):
    raise TypeError("shared attention proof requires immutable compiler captures")
  captures = tuple(x.validate() for x in captures)
  required = {
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","first"),
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","prefix"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","first"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","prefix"),
  }
  keys = tuple(_shared_attention_route_key(x) for x in captures)
  if len(captures) != 4 or len(set(keys)) != 4 or set(keys) != required:
    raise ValueError("shared attention proof requires exact 8B/14B first/prefix coverage")
  rows = sorted(zip(keys,captures),key=lambda x:x[0])
  return {"schema":"tinygrad.shared_attention_proof.v2","status":"PASS","passed":True,
    "captures":[{"profile":key[0],"strategy":key[1],"position":key[2],
      "capture_sha256":capture.capture_sha256,"canonical_graph_sha256":capture.canonical_graph_sha256,
      "candidate_context":{name:getattr(capture.candidate_context,name) for name in capture.candidate_context._fields},
      "wmma":{"qk":8,"pv":8},"numeric":{"max_abs":capture.numeric_max_abs,"max_rel":capture.numeric_max_rel,
        "rel_l2":capture.numeric_rel_l2,"reference_sha256":capture.reference_sha256}}
      for key,capture in rows]}

def _shared_attention_acc_slice_proof_artifact(captures:tuple[SharedAttentionCompilerCapture,...]) -> dict[str, object]:
  required = {
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","first"),
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","prefix"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","first"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","prefix"),
  }
  groups:dict[tuple[str,str,str],list[SharedAttentionCompilerCapture]] = {}
  for capture in captures: groups.setdefault(_shared_attention_route_key(capture),[]).append(capture)
  if len(captures) != 8 or set(groups) != required or any(len(group) != 2 for group in groups.values()):
    raise ValueError("accumulator-slice proof requires exactly two passes for each 8B/14B first/prefix route")
  rows:list[dict[str,object]] = []
  for key in sorted(groups):
    pair = groups[key]
    intervals = sorted((capture.acc_slice_pass.output_block_base,
                        capture.acc_slice_pass.output_block_base+capture.acc_slice_pass.acc_blocks) for capture in pair)
    if intervals[0][0] != 0: raise ValueError("accumulator-slice output ownership has a gap")
    if intervals[0][1] > intervals[1][0]: raise ValueError("accumulator-slice output ownership overlaps")
    if intervals[0][1] < intervals[1][0] or intervals[1][1] != 8:
      raise ValueError("accumulator-slice output ownership has a gap")
    contexts = {tuple((name,getattr(capture.candidate_context,name)) for name in capture.candidate_context._fields
                      if name not in {"output_block_base","acc_blocks"}) for capture in pair}
    if len(contexts) != 1: raise ValueError("accumulator-slice candidate contexts do not match")
    if len({capture.acc_slice_pass.logical_graph_sha256 for capture in pair}) != 1:
      raise ValueError("accumulator-slice logical graphs do not match")
    if len({capture.param_ownership for capture in pair}) != 1:
      raise ValueError("accumulator-slice parameter ownership does not match")
    numerics = {(capture.numeric_max_abs,capture.numeric_max_rel,capture.numeric_rel_l2,capture.reference_sha256) for capture in pair}
    if len(numerics) != 1: raise ValueError("accumulator-slice numeric records do not match")
    resources = {(capture.hip_resources,capture.highest_vgpr,capture.highest_sgpr,capture.spill_count,
                  capture.scratch_bytes,capture.lds_bytes,capture.synchronization) for capture in pair}
    if len(resources) != 1: raise ValueError("accumulator-slice resource records do not match")
    if any(not capture.allocation_complete or capture.expanded_kv_buffers or capture.score_probability_buffers or
           capture.spill_count or capture.scratch_bytes for capture in pair):
      raise ValueError("accumulator-slice proof forbids incomplete, materialized, or spilled captures")
    numeric = pair[0]
    rows.append({"profile":key[0],"strategy":key[1],"position":key[2],
      "logical_graph_sha256":pair[0].acc_slice_pass.logical_graph_sha256,"output_blocks":list(range(8)),
      "wmma":{"qk":16,"pv":8,"qk_recomputed_passes":2},
      "passes":[{"output_block_base":capture.acc_slice_pass.output_block_base,"acc_blocks":capture.acc_slice_pass.acc_blocks,
        "qk_recomputed":capture.acc_slice_pass.qk_recomputed,"capture_sha256":capture.capture_sha256,
        "canonical_graph_sha256":capture.canonical_graph_sha256} for capture in sorted(pair,key=lambda x:x.acc_slice_pass.output_block_base)],
      "numeric":{"max_abs":numeric.numeric_max_abs,"max_rel":numeric.numeric_max_rel,"rel_l2":numeric.numeric_rel_l2,
        "reference_sha256":numeric.reference_sha256}})
  return {"schema":"tinygrad.shared_attention_proof.acc_slice_v3","status":"PASS","passed":True,"captures":rows}

def _shared_attention_phase_proof_artifact(captures:tuple[SharedAttentionCompilerCapture,...]) -> dict[str,object]:
  required = {
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","first"),
    ("qwen3_8b_q4k_m_gfx1100","FULL_RESIDENT_OVERLAY","prefix"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","first"),
    ("qwen3_14b_q4k_m_gfx1100","BOUNDED_PACKED_TILES","prefix"),
  }
  keyed={_shared_attention_route_key(capture):capture for capture in captures}
  if len(captures) != 4 or len(keyed) != 4 or set(keyed) != required:
    raise ValueError("phase proof requires exact 8B/14B first/prefix coverage")
  phase_ids={capture.phase_plan.phase_ids for capture in captures}
  if len(phase_ids) != 1: raise ValueError("compiler phase IDs differ across attention routes")
  rows=[]
  for key,capture in sorted(keyed.items()):
    plan=capture.phase_plan.validate()
    rows.append({"profile":key[0],"strategy":key[1],"position":key[2],"capture_sha256":capture.capture_sha256,
      "logical_graph_sha256":plan.logical_graph_sha256,"phase_ids":list(plan.phase_ids),
      "state_handles":[handle.to_json() for handle in plan.state_handles],
      "numeric":{"max_abs":capture.numeric_max_abs,"max_rel":capture.numeric_max_rel,"rel_l2":capture.numeric_rel_l2,
                 "reference_sha256":capture.reference_sha256},
      "resources":{"vgpr":capture.highest_vgpr,"scratch_bytes":capture.scratch_bytes,"spill_count":capture.spill_count,
                   "lds_bytes":capture.lds_bytes}})
  return {"schema":"tinygrad.shared_attention_proof.phase_v4","status":"PASS","passed":True,
          "phase_ids":list(next(iter(phase_ids))),"captures":rows}

def shared_attention_proof_artifact(captures:tuple[SharedAttentionCompilerCapture,...]) -> dict[str, object]:
  """Aggregate complete v2 captures or paired accumulator-slice v3 captures, failing closed on mixed evidence."""
  if not isinstance(captures,tuple) or any(not isinstance(x,SharedAttentionCompilerCapture) for x in captures):
    raise TypeError("shared attention proof requires immutable compiler captures")
  captures = tuple(x.validate() for x in captures)
  schemas = {capture.schema for capture in captures}
  if schemas == {CAPTURE_SCHEMA}: return _shared_attention_v2_proof_artifact(captures)
  if schemas == {ACC_SLICE_CAPTURE_SCHEMA}: return _shared_attention_acc_slice_proof_artifact(captures)
  if schemas == {PHASE_CAPTURE_SCHEMA}: return _shared_attention_phase_proof_artifact(captures)
  raise ValueError("shared attention proof does not accept mixed or unknown capture schemas")

ATTENTION_EVIDENCE_SCHEMA = "tinygrad.shared_attention_evidence.v1"
SHARED_ATTENTION_PROOF_SCHEMA = "tinygrad.shared_attention_proof.v2"
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


__all__ = ["ATTENTION_EVIDENCE_SCHEMA", "SHARED_ATTENTION_PROOF_SCHEMA", "DEFAULT_CONTEXTS", "AttentionGeometry", "AttentionWorkload",
           "attention_workloads", "authority_command", "dual_wmma_fused_call_report",
           "dual_wmma_fused_call_fixture", "shared_attention_proof_artifact",
           "fused_wmma_role_report", "geometry_candidates"]
