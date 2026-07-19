#!/usr/bin/env python3
"""Bounded machine-search surface for the completed 14B Q4_K/Q8_1 MMQ pieces.

This does not bind production prefill. It turns the pieces that are already
implemented into explicit candidate rows, and records the unfinished llama-style
pieces as blocked rows instead of treating them as selectable variants.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import pathlib
import sys
from typing import Any, Callable

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, ACTIVATION_LAYOUT_ROW_MAJOR, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID,
  AMD_DS4_COOP_TILE_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, FULL_GRID_BACKEND_ID, BoundedMMQConfig, CANDIDATE_ROUTE_ID,
  COMPARATOR_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, LLAMA_MMQ_GEOMETRY, PUBLIC_LABEL, QUANT, ROLE,
  STAGED_DS4_BACKEND_ID,
  candidate_metadata, coop_tile_blocked_translation_evidence, run_bounded_harness,
)
from extra.qk.mmq_llama_five_buffer_gpu_harness import TARGET_IN_PLACE_ACCUMULATION, run_full_grid_r5_benchmark
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT
from extra.qk.mmq_llama_oracle import llama_mma_writeback_owners
from extra.qk.mmq_llama_store_probe import lowered_tinygrad_r4_store_owner_trace_rows
from extra.qk.mmq_owner_coverage import (
  build_mmq_owner_coverage_artifact, observed_stores_from_amd_isa_proof_rows,
  validate_mmq_owner_coverage_artifact,
)
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile
from extra.qk.mmq_staging_evidence import build_mmq_staging_evidence_bundle


SCHEMA = "q4k-q8-1-mmq-machine-search.v1"
MILESTONE_EVIDENCE_SCHEMA = "tinygrad.mmq_milestone_evidence.v1"
MILESTONE_NAMES = (
  "M1_owner_coverage", "M2_q4_q8_staging", "M3_resource_scratch",
  "M4_distinct_binary", "M5_correctness", "M6_same_session_timing", "M7_no_fallback",
)
DEFAULT_OUTPUT = pathlib.Path("bench/prefill-14b-mmq-machine-search/search-report.json")
FULL_GPU_PROBE_CANDIDATE_ID = "prefill_14b_q4k_q8_1_hybrid_mmq_atom"
FULL_GPU_PROBE_ROUTE_ID = "prefill_q4k_q8_1_hybrid_mmq_atom"
FULL_GPU_PROBE_ROLE = "ffn_gate_up"
FULL_GRID_R5_SHAPE = {"M": 128, "N": 128, "K": 256}
R6_TARGET_ROLE_SHAPE = {"role": "ffn_gate_up", "M": 512, "N": 17408, "K": 5120}
R5_GEOMETRY_SCHEMA = "q4k-q8-1-mmq-r5-geometry-search.v2"
R5_RETAINED_VALIDATION_SCHEMA = "q4k-q8-1-mmq-r5-retained-evidence-validation.v2"
R5_IDENTICAL_WORKLOAD_SCOPE = "identical_workload"
R5_WORKLOAD_FIELDS = frozenset(("role", "M", "N", "K", "k_launches", "complete_role", "output_elements"))
R5_MEASUREMENT_DEFINITION_FIELDS = frozenset((
  "preparation_scope", "allocation_scope", "readback_scope", "sync_scope",
))
R5_MATCHED_MEASUREMENT_DEFINITION = {
  "preparation_scope": "excluded_prepared_identical_logical_inputs",
  "allocation_scope": "excluded_persistent_preallocated_buffers",
  "readback_scope": "excluded_timed_region",
  "sync_scope": "outer_synchronize_after_complete_role",
}
R6_TARGET_EVIDENCE_SCHEMA = "q4k-q8-1-mmq-r6-target-role-evidence.v1"
R6_INDEPENDENT_EVIDENCE_SCHEMA = "q4k-q8-1-mmq-r6-independent-epoch-evidence.v1"
R7_REQUIRED_SOURCE_ANCHORS = ("mmq.cuh:mul_mat_q_process_tile", "mmq.cuh:load_tiles_q4_K")
LLAMA_KERNEL_SOURCES: tuple[dict[str, Any], ...] = (
  {
    "component": "mmq_route_and_launch",
    "path": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cu",
    "anchors": ["GGML_TYPE_Q4_K", "mul_mat_q_case"],
  },
  {
    "component": "mmq_tile_geometry_and_layout",
    "path": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh",
    "anchors": ["MMQ_ITER_K", "MMQ_NWARPS", "block_q8_1_mmq", "load_tiles_q4_K"],
  },
  {
    "component": "q8_1_mmq_ds4_quantizer",
    "path": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/quantize.cu",
    "anchors": ["quantize_mmq_q8_1", "MMQ_Q8_1_DS_LAYOUT_DS4"],
  },
  {
    "component": "q4k_q8_1_dot_formula",
    "path": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh",
    "anchors": ["vec_dot_q4_K_q8_1_impl_mmq", "VDR_Q4_K_Q8_1_MMQ"],
  },
  {
    "component": "q4k_block_format",
    "path": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-common.h",
    "anchors": ["block_q4_K"],
  },
)

DONE_COMPONENTS: tuple[dict[str, Any], ...] = (
  {
    "component": "DS4 layout",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_reference.Q81MMQDS4ActivationSpec",
    "proof": "test/unit/test_mmq_q4k_q8_reference.py",
  },
  {
    "component": "DS4 reference correctness",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_reference.q8_1_mmq_ds4_quantize_reference",
    "proof": "test/unit/test_mmq_q4k_q8_reference.py",
  },
  {
    "component": "Q4_K x DS4 formula",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_reference.q4k_q8_1_mmq_ds4_tile_reference",
    "proof": "test/unit/test_mmq_q4k_q8_reference.py",
  },
  {
    "component": "Q4_K tile loader",
    "status": "done",
    "implementation": "extra.qk.q4k_tile_loader.load_q4k_256_tile",
    "proof": "test/unit/test_mmq_q4k_q8_reference.py",
  },
  {
    "component": "sudot4 primitive availability",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_atom._sudot4",
    "proof": "test/unit/test_mmq_q4k_q8_atom.py",
  },
  {
    "component": "direct DS4 GPU atom",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_atom.run_q4k_q8_1_mmq_bounded_amd_ds4_warp",
    "proof": "test/unit/test_mmq_q4k_q8_atom.py",
  },
  {
    "component": "R3 LDS skeleton atom",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_atom.run_q4k_q8_1_mmq_bounded_amd_ds4_lds_skeleton",
    "proof": "test/unit/test_mmq_q4k_q8_atom.py",
  },
  {
    "component": "R4 cooperative multi-wave output ownership",
    "status": "done",
    "implementation": AMD_DS4_COOP_TILE_BACKEND_ID,
    "proof": "test/unit/test_mmq_machine_search.py",
    "evidence": "lowered AMD ISA proof manifest covers 16x16 R4 owner map as eight spill-free 32-store fragments",
  },
  {
    "component": "R4 llama cooperative tile oracle",
    "status": "done",
    "implementation": "extra.qk.mmq_llama_oracle.run_llama_mmq_coop_tile_oracle",
    "proof": "test/unit/test_mmq_llama_oracle.py",
  },
  {
    "component": "R5 bounded cooperative numeric atom",
    "status": "done",
    "implementation": "extra.qk.mmq_q4k_q8_atom.run_q4k_q8_1_mmq_bounded_amd_ds4_coop_tile",
    "proof": "test/unit/test_mmq_bounded_harness.py",
    "evidence": "16x16x256 DS4 coop numeric atom emits and passes bounded correctness; store-owner proof remains separate and route promotion is not claimed",
  },
)


@dataclass(frozen=True)
class MMQSearchCandidate:
  candidate_id: str
  backend: str
  activation_layout: str
  status: str
  search_class: str
  promotion_eligible: bool
  reason: str
  m_tile: int = 4
  n_tile: int = 5
  k_groups: int = 8
  measure_direct_packed: bool = True

  def config(self, *, warmups: int = 0, rounds: int = 1) -> BoundedMMQConfig:
    return BoundedMMQConfig(m_tile=self.m_tile, n_tile=self.n_tile, k_groups=self.k_groups, warmups=warmups,
                            rounds=rounds, backend=self.backend, activation_layout=self.activation_layout,
                            measure_direct_packed=self.measure_direct_packed)

  def to_json(self) -> dict[str, Any]:
    cfg = self.config()
    return {
      "candidate_id": self.candidate_id,
      "backend": self.backend,
      "activation_layout": self.activation_layout,
      "status": self.status,
      "search_class": self.search_class,
      "promotion_eligible": self.promotion_eligible,
      "reason": self.reason,
      "bounded_config": {
        "m_tile": cfg.m_tile, "n_tile": cfg.n_tile, "k_groups": cfg.k_groups,
        "m_tiles": cfg.m_tiles, "n_tiles": cfg.n_tiles,
      },
      "metadata": candidate_metadata(cfg),
    }


SEARCHABLE_CANDIDATES: tuple[MMQSearchCandidate, ...] = (
  MMQSearchCandidate(
    candidate_id="direct_packed_comparator",
    backend="direct_packed",
    activation_layout=ACTIVATION_LAYOUT_ROW_MAJOR,
    status="searchable",
    search_class="baseline_comparator",
    promotion_eligible=False,
    reason="current rollback/comparator path; included so every search report has a same-session baseline",
    measure_direct_packed=False,
  ),
  MMQSearchCandidate(
    candidate_id="ds4_reference_formula",
    backend="reference",
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="searchable",
    search_class="reference_correctness",
    promotion_eligible=False,
    reason="completed DS4 layout plus Q4_K x DS4 formula reference",
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="amd_ds4_warp_direct",
    backend=AMD_DS4_WARP_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="searchable",
    search_class="working_gpu_atom",
    promotion_eligible=False,
    reason="completed real AMD DS4 direct warp atom; correctness candidate, not llama-style cooperative tile",
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="staged_ds4_reference_probe",
    backend=STAGED_DS4_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="evidence_only",
    search_class="reference_backed_lifecycle_probe",
    promotion_eligible=False,
    reason="completed lifecycle/reference probe; no real staged AMD tile kernel emitted",
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="amd_ds4_dot4x4_packed",
    backend=AMD_DS4_DOT4X4_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="searchable",
    search_class="packed_dot_candidate",
    promotion_eligible=False,
    reason="R1 packed DS4 dot4x4 lane mapping passes bounded proof; still bounded-only until cooperative tile passes",
    m_tile=4,
    n_tile=5,
    k_groups=8,
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="amd_ds4_lds_skeleton",
    backend=AMD_DS4_LDS_SKELETON_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="evidence_only",
    search_class="bounded_shared_lds_skeleton",
    promotion_eligible=False,
    reason="R3 real tinygrad custom kernel stages DS4 q8 values through LOCAL memory and a barrier; bounded-only skeleton, not production dispatch",
    m_tile=4,
    n_tile=5,
    k_groups=8,
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="llama_mmq_coop_tile_oracle",
    backend=LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="oracle_only",
    search_class="llama_structure_oracle",
    promotion_eligible=False,
    reason="translated llama cooperative writeback ownership oracle; not a production atom and not a selectable route",
    m_tile=16,
    n_tile=16,
    k_groups=8,
    measure_direct_packed=True,
  ),
  MMQSearchCandidate(
    candidate_id="amd_ds4_coop_tile_bounded",
    backend=AMD_DS4_COOP_TILE_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="searchable",
    search_class="bounded_cooperative_numeric_atom",
    promotion_eligible=False,
    reason="R5 bounded 16x16x256 coop numeric atom emits and passes correctness; store-owner proof is separate and promotion requires a same-session speed win",
    m_tile=16,
    n_tile=16,
    k_groups=8,
    measure_direct_packed=True,
  ),
)

R5_GEOMETRY_CANDIDATES: tuple[MMQSearchCandidate, ...] = (
  MMQSearchCandidate(
    candidate_id="r5_ds4_warp_4x5",
    backend=AMD_DS4_WARP_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="r5_candidate",
    search_class="bounded_geometry_existing_atom",
    promotion_eligible=False,
    reason="existing DS4 warp atom geometry probe; not llama cooperative tile",
    m_tile=4,
    n_tile=5,
  ),
  MMQSearchCandidate(
    candidate_id="r5_ds4_dot4x4_8x7",
    backend=AMD_DS4_DOT4X4_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="r5_candidate",
    search_class="bounded_geometry_existing_atom",
    promotion_eligible=False,
    reason="existing DS4 dot4x4 geometry probe; not llama cooperative tile",
    m_tile=8,
    n_tile=7,
  ),
  MMQSearchCandidate(
    candidate_id="r5_ds4_lds_skeleton_4x5",
    backend=AMD_DS4_LDS_SKELETON_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="r5_candidate",
    search_class="bounded_geometry_existing_atom",
    promotion_eligible=False,
    reason="existing LDS-staged DS4 skeleton geometry probe; not llama cooperative tile",
    m_tile=4,
    n_tile=5,
  ),
  MMQSearchCandidate(
    candidate_id="r5_ds4_coop_tile_16x16",
    backend=AMD_DS4_COOP_TILE_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="r5_candidate",
    search_class="bounded_geometry_emitted_coop_atom",
    promotion_eligible=False,
    reason="bounded coop numeric atom; promotion only if it beats direct_packed in the same R5 report",
    m_tile=16,
    n_tile=16,
  ),
  MMQSearchCandidate(
    candidate_id="r5_llama_coop_oracle_16x16",
    backend=LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="oracle_only",
    search_class="llama_structure_oracle",
    promotion_eligible=False,
    reason="numeric/ownership oracle only; no emitted AMD candidate",
    m_tile=16,
    n_tile=16,
  ),
  MMQSearchCandidate(
    candidate_id="r5_full_grid_128x128",
    backend=FULL_GRID_BACKEND_ID,
    activation_layout=ACTIVATION_LAYOUT_MMQ_DS4,
    status="r5_candidate",
    search_class="bounded_geometry_emitted_full_grid",
    promotion_eligible=False,
    reason="emitted 128x128x256 full-grid GPU probe; bounded timing/correctness evidence only, no production role or shape integration",
    m_tile=128,
    n_tile=128,
    k_groups=8,
  ),
)

BLOCKED_CANDIDATES: tuple[dict[str, Any], ...] = (
  {
    "candidate_id": "full_14b_prefill_route",
    "backend": "not_implemented",
    "activation_layout": ACTIVATION_LAYOUT_MMQ_DS4,
    "status": "blocked",
    "search_class": "production_route",
    "promotion_eligible": False,
    "reason": "full llama-style MMQ route is not implemented; default remains direct_packed",
  },
)


def evaluate_milestone_evidence(*, owner_coverage: bool = False, q4_q8_staging: bool = False,
                                resource_scratch: bool = False, distinct_binary: bool = False,
                                correctness: bool = False, same_session_timing: bool = False,
                                no_fallback: bool = False, complete_atom: bool = False,
                                default_route: str = "direct_packed",
                                production_dispatch_changed: bool = False) -> dict[str, Any]:
  """Return the promotion contract; missing evidence is deliberately false."""
  gates = dict(zip(MILESTONE_NAMES, (owner_coverage, q4_q8_staging, resource_scratch,
                                    distinct_binary, correctness, same_session_timing, no_fallback)))
  blockers = [name for name, passed in gates.items() if passed is not True]
  if complete_atom is not True:
    blockers.append("complete_atom")
  if default_route != "direct_packed":
    blockers.append("default_route_must_remain_direct_packed")
  if production_dispatch_changed is not False:
    blockers.append("production_dispatch_changed")
  return {
    "schema": MILESTONE_EVIDENCE_SCHEMA,
    "milestones": {name: bool(value) for name, value in gates.items()},
    "complete_atom": bool(complete_atom),
    "default_route": default_route,
    "production_dispatch_changed": production_dispatch_changed,
    "promotion_eligible": not blockers,
    "verdict": "PASS" if not blockers else "BLOCKED_FAIL_CLOSED",
    "blockers": blockers,
  }


def _default_milestone_evidence() -> dict[str, Any]:
  return evaluate_milestone_evidence()


def evaluate_candidate_promotion(*, owner_coverage: dict[str, Any] | None = None,
                                 cooperative_tile: dict[str, Any] | None = None,
                                 correctness: bool = False, distinct_binary: bool = False,
                                 same_session_timing: bool = False, no_fallback: bool = False,
                                 q4_q8_staging: bool = False, resource_scratch: bool = False,
                                 default_route: str = "direct_packed",
                                 production_dispatch_changed: bool = False) -> dict[str, Any]:
  """Join producer evidence at the guarded boundary; absent evidence blocks."""
  owner_ok = False
  owner_blocker = "missing owner coverage evidence"
  if owner_coverage is not None:
    try:
      checked = validate_mmq_owner_coverage_artifact(owner_coverage)
      owner_ok = checked["status"] == "PASS"
      owner_blocker = None if owner_ok else checked.get("exact_blocker", "owner coverage did not pass")
    except (TypeError, ValueError) as exc:
      owner_blocker = f"invalid owner coverage evidence: {exc}"
  coop_ok = False
  coop_blocker = "missing bounded cooperative tile evidence"
  if cooperative_tile is not None:
    required_coop_contract = (
      cooperative_tile.get("status") in ("PASS", "bounded_numeric_pass") and
      cooperative_tile.get("bounded_only") is True and
      cooperative_tile.get("production_dispatch_changed") is False and
      cooperative_tile.get("default_route") == "direct_packed" and
      cooperative_tile.get("exact_blocker") is None and
      cooperative_tile.get("distinct_binary_identity") is True and
      cooperative_tile.get("same_session_timing") is True and
      cooperative_tile.get("no_fallback") is True
    )
    coop_ok = required_coop_contract
    coop_blocker = None if coop_ok else cooperative_tile.get(
      "exact_blocker", "cooperative evidence lacks distinct-binary, same-session, or no-fallback proof")
  result = evaluate_milestone_evidence(
    owner_coverage=owner_ok, q4_q8_staging=q4_q8_staging, resource_scratch=resource_scratch,
    distinct_binary=distinct_binary, correctness=correctness, same_session_timing=same_session_timing,
    no_fallback=no_fallback, default_route=default_route,
    production_dispatch_changed=production_dispatch_changed,
  )
  if owner_blocker is not None: result["blockers"].append(owner_blocker)
  if coop_blocker is not None: result["blockers"].append(coop_blocker)
  result["milestones"]["M1_owner_coverage"] = owner_ok
  result["bounded_cooperative_tile"] = coop_ok
  result["promotion_eligible"] = not result["blockers"]
  result["verdict"] = "PASS" if result["promotion_eligible"] else "BLOCKED_FAIL_CLOSED"
  return result


def build_r4_evidence_artifacts() -> dict[str, Any]:
  spec = describe_q4k_q8_1_mmq_tile(role=ROLE, m=16, n=16, k=256, m_tile=16, n_tile=16,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  lowered_rows = lowered_tinygrad_r4_store_owner_trace_rows(spec)
  lowered_observed = observed_stores_from_amd_isa_proof_rows(lowered_rows)
  owner = build_mmq_owner_coverage_artifact(
      spec, lowered_observed, candidate_id="cooperative_multi_wave_tile",
      backend="lowered_amd_isa_fragmented_store_owner_manifest")
  validate_mmq_owner_coverage_artifact(owner)
  return {
    "owner_coverage": owner,
    "gpu_owner_trace": {
      "schema": "tinygrad.mmq_gpu_owner_trace.v1",
      "candidate_id": "cooperative_multi_wave_tile",
      "backend": "lowered_amd_isa_fragmented_store_owner_manifest",
      "status": "PASS",
      "shape": {"M": 16, "N": 16, "K": 256},
      "store_rows": len(lowered_rows),
      "unique_store_owners": len({(row["store_owner"]["m"], row["store_owner"]["n"]) for row in lowered_rows if "store_owner" in row}),
      "fragment_count": len({row["trace_fragment"] for row in lowered_rows}),
      "gated_store_rows": sum(1 for row in lowered_rows if row.get("gated") is True),
      "production_dispatch_changed": False,
    },
    "staging_sum_slots": build_mmq_staging_evidence_bundle(
      candidate_id="cooperative_multi_wave_tile",
      backend=AMD_DS4_COOP_TILE_BACKEND_ID,
      shape={"M": 16, "N": 16, "K": 256},
      notes="R4 structural staging evidence plus R5 bounded numeric PASS; route promotion still blocks on same-session coop speed win and unified production binding",
    ),
  }


def build_full_gpu_probe_candidate(probe: dict[str, Any], *, r4_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  """Adapt the full-grid GPU probe into the candidate evidence schema.

  This is deliberately an evidence join, not an admission or route selector.
  The probe proves the emitted 128x128x256 kernel's dispatch, artifact identity,
  resources, and numerical output.  R4 owner/staging artifacts are joined for
  M1/M2, while M6 (same-session timing) and M7 (route/no-fallback census) stay
  false until the production-sized candidate is integrated.  Keeping those
  gates explicit prevents a passing probe from becoming a promotion claim.
  """
  if not isinstance(probe, dict): raise TypeError("probe must be a mapping")
  evidence = probe.get("evidence")
  if not isinstance(evidence, dict): evidence = {}
  r4 = build_r4_evidence_artifacts() if r4_evidence is None else r4_evidence
  owner = r4.get("owner_coverage") if isinstance(r4, dict) else None
  staging = r4.get("staging_sum_slots") if isinstance(r4, dict) else None
  owner_ok = isinstance(owner, dict) and owner.get("status") == "PASS" and owner.get("production_dispatch_changed") is False
  staging_ok = isinstance(staging, dict) and staging.get("status") == "PASS" and staging.get("production_dispatch_changed") is False
  resources = evidence.get("resources") if isinstance(evidence.get("resources"), dict) else {}
  resource_ok = resources.get("scratch_bytes") == 0 and resources.get("vgpr") is not None
  distinct_ok = all(isinstance(evidence.get(k), str) and len(evidence[k]) == 64 for k in ("source_sha256", "binary_sha256"))
  comparison = evidence.get("comparison") if isinstance(evidence.get("comparison"), dict) else {}
  correctness_ok = probe.get("passed") is True and comparison.get("status") == "pass" and comparison.get("mismatch_count") == 0
  milestones = {"M1": owner_ok, "M2": staging_ok, "M3": resource_ok, "M4": distinct_ok,
                "M5": correctness_ok, "M6": False, "M7": False}
  admission_payload = {
    "candidate_id": FULL_GPU_PROBE_CANDIDATE_ID, "route_id": FULL_GPU_PROBE_ROUTE_ID,
    "role": FULL_GPU_PROBE_ROLE, "quant_format": "Q4_K", "activation_format": "Q8_1",
    "evidence": milestones,
  }
  canonical_identity = hashlib.sha256(json.dumps(admission_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
  return {
    **admission_payload, "canonical_identity": canonical_identity,
    "promotion_eligible": False, "complete_atom": False,
    "default_route": "direct_packed", "production_dispatch_changed": False,
    "probe": probe, "owner_coverage": owner, "staging_sum_slots": staging,
    "resource_evidence": resources,
    "exact_blocker": "M6 same-session timing and M7 route/no-fallback evidence are not present; complete_atom is false",
  }


def _ranked_r5_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
  def key(row: dict[str, Any]) -> tuple[int, float]:
    speedup = row.get("speedup_vs_direct_packed")
    return (0 if row.get("status") == "PASS" and isinstance(speedup, (int, float)) else 1, -float(speedup or 0.0))
  return sorted(rows, key=key)


def _r5_workload_descriptor(shape: dict[str, int], *, role: str, k_launches: int,
                            complete_role: bool) -> dict[str, Any]:
  return {
    "role": role, "M": shape["M"], "N": shape["N"], "K": shape["K"],
    "k_launches": k_launches, "complete_role": complete_role,
    "output_elements": shape["M"] * shape["N"],
  }


def _r5_candidate_measurement_definition() -> dict[str, str]:
  return {
    "preparation_scope": "excluded_prepared_inputs",
    "allocation_scope": "excluded_persistent_buffers",
    "readback_scope": "excluded_timed_region",
    "sync_scope": "runtime_wait_true_each_sample",
  }


def _r5_legacy_direct_measurement_definition() -> dict[str, str]:
  return {
    "preparation_scope": "included_q4_q8_conversion_and_dequant",
    "allocation_scope": "included_tensor_construction_and_realization",
    "readback_scope": "included_numpy_readback",
    "sync_scope": "numpy_readback_synchronization",
  }


def _r5_timing_descriptors(shape: dict[str, int], timing: Any, *,
                           role: str = "bounded_r5_geometry", k_launches: int = 1,
                           complete_role: bool = False) -> dict[str, Any]:
  # A future collector may supply exact descriptors. Preserve them verbatim;
  # the retained-evidence validator below decides whether they are complete,
  # identical, and production-comparable.
  if isinstance(timing, dict) and all(key in timing for key in (
      "comparison_scope", "candidate_measurement", "direct_packed_measurement")):
    return {
      "comparison_scope": timing["comparison_scope"],
      "candidate_measurement": timing["candidate_measurement"],
      "direct_packed_measurement": timing["direct_packed_measurement"],
    }
  workload = _r5_workload_descriptor(
    shape, role=role, k_launches=k_launches, complete_role=complete_role)
  return {
    "comparison_scope": "legacy_mismatched_measurement_definition",
    "candidate_measurement": {
      "workload": dict(workload),
      "measurement_definition": _r5_candidate_measurement_definition(),
    },
    "direct_packed_measurement": {
      "workload": dict(workload),
      "measurement_definition": _r5_legacy_direct_measurement_definition(),
    },
  }


def _valid_r5_workload_descriptor(value: Any) -> bool:
  if not isinstance(value, dict) or set(value) != R5_WORKLOAD_FIELDS:
    return False
  positive_ints = tuple(value.get(key) for key in ("M", "N", "K", "k_launches", "output_elements"))
  return (
    isinstance(value.get("role"), str) and bool(value["role"]) and
    isinstance(value.get("complete_role"), bool) and
    all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in positive_ints) and
    value["output_elements"] == value["M"] * value["N"])


def _valid_r5_measurement_definition(value: Any) -> bool:
  return (
    isinstance(value, dict) and set(value) == R5_MEASUREMENT_DEFINITION_FIELDS and
    value == R5_MATCHED_MEASUREMENT_DEFINITION)


def _r5_identical_timing_descriptors(timing: Any) -> bool:
  if not isinstance(timing, dict) or timing.get("comparison_scope") != R5_IDENTICAL_WORKLOAD_SCOPE:
    return False
  candidate = timing.get("candidate_measurement")
  direct = timing.get("direct_packed_measurement")
  if not isinstance(candidate, dict) or not isinstance(direct, dict):
    return False
  candidate_workload, direct_workload = candidate.get("workload"), direct.get("workload")
  candidate_definition, direct_definition = (
    candidate.get("measurement_definition"), direct.get("measurement_definition"))
  if not (_valid_r5_workload_descriptor(candidate_workload) and
          _valid_r5_workload_descriptor(direct_workload) and
          _valid_r5_measurement_definition(candidate_definition) and
          _valid_r5_measurement_definition(direct_definition)):
    return False
  return (
    candidate_workload == direct_workload and
    candidate_definition == direct_definition)


def _r5_comparable_complete_target_timing(timing: Any) -> bool:
  if not _r5_identical_timing_descriptors(timing):
    return False
  candidate_workload = timing["candidate_measurement"]["workload"]
  target_workload = _r5_workload_descriptor(
    {key: R6_TARGET_ROLE_SHAPE[key] for key in ("M", "N", "K")},
    role=R6_TARGET_ROLE_SHAPE["role"], k_launches=R6_TARGET_ROLE_SHAPE["K"] // FULL_GRID_R5_SHAPE["K"],
    complete_role=True)
  return candidate_workload == target_workload


def build_r5_geometry_search_report(
  *,
  run: bool = False,
  warmups: int = 0,
  rounds: int = 1,
  runner: Callable[[BoundedMMQConfig], dict[str, Any]] = run_bounded_harness,
) -> dict[str, Any]:
  rows: list[dict[str, Any]] = []
  for candidate in R5_GEOMETRY_CANDIDATES:
    cfg = candidate.config(warmups=warmups, rounds=rounds)
    row = {
      "candidate_id": candidate.candidate_id,
      "backend": candidate.backend,
      "search_class": candidate.search_class,
      "shape": {"M": cfg.bounded_m, "N": cfg.bounded_n, "K": cfg.bounded_k},
      "promotion_eligible": False,
      "production_dispatch_changed": False,
      "reason": candidate.reason,
    }
    if run:
      try:
        # The full-grid candidate is emitted by a different kernel builder and
        # ABI than the bounded atom runner.  Only invoke it on the real default
        # runner; injected test runners still receive every bounded config so
        # ranking tests remain deterministic and side-effect free.
        result = (run_full_grid_r5_benchmark(warmups=warmups, rounds=rounds)
                  if candidate.backend == FULL_GRID_BACKEND_ID and runner is run_bounded_harness
                  else runner(cfg))
        direct = result["timing"].get("direct_packed")
        direct_min = None if direct is None else direct.get("min_ms")
        cand_min = result["timing"].get("min_ms")
        speedup = None if not (isinstance(direct_min, (int, float)) and isinstance(cand_min, (int, float)) and cand_min > 0) else float(direct_min) / float(cand_min)
        row.update({
          "status": result["status"],
          "correctness": result["correctness"],
          "timing": {
            "min_ms": cand_min,
            "median_ms": result["timing"].get("median_ms"),
            "samples_ms": result["timing"].get("samples_ms"),
            "direct_packed_min_ms": direct_min,
            "direct_packed": direct,
            "comparator_status": result["timing"].get("comparator_status"),
            **_r5_timing_descriptors(row["shape"], result["timing"]),
          },
          "artifacts": result.get("artifacts"),
          "distinct_binary_identity": result.get("distinct_binary_identity"),
          "same_session_timing": result.get("same_session_timing"),
          "no_fallback": result.get("no_fallback"),
          "speedup_vs_direct_packed": speedup,
          "exact_blocker": result.get("exact_blocker") if result["status"] != "PASS" else None,
        })
      except Exception as exc:
        row.update({"status": "BLOCKED", "exact_blocker": str(exc), "speedup_vs_direct_packed": None})
    else:
      row.update({"status": "NOT_RUN", "exact_blocker": "pass run=True to execute bounded R5 geometry search"})
    rows.append(row)

  ranked = _ranked_r5_rows(rows)
  best = ranked[0] if ranked and ranked[0].get("status") == "PASS" else None
  # Bounded timings remain useful for geometry ranking, but they cannot feed
  # R6. A production-facing speed win must compare the exact complete target
  # role under byte-identical workload and measurement-definition descriptors.
  emitted_rows = [row for row in rows if row.get("status") == "PASS" and
                  row.get("backend") in (AMD_DS4_COOP_TILE_BACKEND_ID, FULL_GRID_BACKEND_ID)]
  best_emitted = max(emitted_rows, key=lambda row: float(row.get("speedup_vs_direct_packed") or 0.0), default=None)
  bounded_emitted_win = (
    best_emitted is not None and (best_emitted.get("speedup_vs_direct_packed") or 0) > 1.0 and
    _r5_identical_timing_descriptors(best_emitted.get("timing")))
  emitted_win = bounded_emitted_win and _r5_comparable_complete_target_timing(best_emitted.get("timing"))
  coop_winner = emitted_win
  role_shape_integration = False
  return {
    "schema": R5_GEOMETRY_SCHEMA,
    "status": "PASS_NON_PROMOTABLE" if best is not None else ("NOT_RUN" if not run else "BLOCKED"),
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "promotion_eligible": bool(coop_winner and role_shape_integration),
    "emitted_backend_win": bool(emitted_win),
    "bounded_emitted_backend_win": bool(bounded_emitted_win),
    "role_shape_integration": role_shape_integration,
    "promotion_verdict": "R5_COMPARABLE_FULL_ROLE_WIN_READY_FOR_R6" if coop_winner else "NO_PROMOTION_WITHOUT_COMPARABLE_FULL_ROLE_WIN",
    "ranking": ranked,
    "best_candidate_id": None if best is None else best["candidate_id"],
    "exact_blocker": (None if role_shape_integration else "comparable full-role emitted backend win awaits route integration") if coop_winner else
      "no emitted cooperative MMQ candidate has an identical-workload exact complete target-role win",
  }


def _validate_r5_evidence(r5_report: dict[str, Any] | None) -> dict[str, Any]:
  """Validate a retained R5 win from its measured row, not summary booleans."""
  r5 = r5_report if isinstance(r5_report, dict) else {}
  ranking = r5.get("ranking")
  qualifying_rows: list[str] = []
  if isinstance(ranking, list):
    for row in ranking:
      if not isinstance(row, dict) or row.get("status") != "PASS":
        continue
      candidate_id, backend, shape = row.get("candidate_id"), row.get("backend"), row.get("shape")
      expected_identity = {
        AMD_DS4_COOP_TILE_BACKEND_ID: ("r5_ds4_coop_tile_16x16", {"M": 16, "N": 16, "K": 256}),
        FULL_GRID_BACKEND_ID: ("r5_full_grid_128x128", dict(FULL_GRID_R5_SHAPE)),
      }.get(backend)
      if expected_identity is None or (candidate_id, shape) != expected_identity:
        continue
      if row.get("promotion_eligible") is not False or row.get("production_dispatch_changed") is not False:
        continue
      timing = row.get("timing")
      if not isinstance(timing, dict):
        continue
      candidate_ms, direct_ms, speedup = (
        timing.get("min_ms"), timing.get("direct_packed_min_ms"), row.get("speedup_vs_direct_packed"))
      numeric = (candidate_ms, direct_ms, speedup)
      if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in numeric):
        continue
      if not all(math.isfinite(float(value)) for value in numeric):
        continue
      if candidate_ms <= 0 or direct_ms <= 0 or speedup <= 1.0:
        continue
      expected_speedup = float(direct_ms) / float(candidate_ms)
      if abs(float(speedup) - expected_speedup) > max(1e-9, abs(expected_speedup) * 1e-9):
        continue
      correctness = row.get("correctness")
      if not isinstance(correctness, dict):
        continue
      if backend == FULL_GRID_BACKEND_ID:
        comparison = correctness.get("comparison")
        error_values = () if not isinstance(comparison, dict) else tuple(
          comparison.get(key) for key in ("max_abs_error", "mean_abs_error", "rtol", "atol"))
        correctness_ok = (
          timing.get("comparator_status") == "pass" and
          correctness.get("status") == "PASS" and
          correctness.get("authority") == "full_grid_r5_same_session_reference" and
          isinstance(comparison, dict) and comparison.get("status") == "pass" and
          comparison.get("mismatch_count") == 0 and
          comparison.get("nan_got") == comparison.get("nan_reference") == 0 and
          comparison.get("inf_got") == comparison.get("inf_reference") == 0 and
          comparison.get("joint_finite") == comparison.get("got_size") ==
            comparison.get("reference_size") == 128 * 128 and
          comparison.get("got_shape") == comparison.get("reference_shape") == [128, 128] and
          all(isinstance(value, (int, float)) and not isinstance(value, bool) and
              math.isfinite(float(value)) and value >= 0 for value in error_values))
      else:
        max_abs, atol, tiles = correctness.get("max_abs"), correctness.get("atol"), correctness.get("tiles")
        correctness_ok = (
          timing.get("comparator_status") == "measured" and
          all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
              for value in (max_abs, atol)) and
          0 <= max_abs <= atol and tiles == 1)
      if backend == FULL_GRID_BACKEND_ID:
        samples = timing.get("samples_ms")
        direct = timing.get("direct_packed")
        direct_samples = direct.get("samples_ms") if isinstance(direct, dict) else None
        artifacts = row.get("artifacts")
        resources = artifacts.get("resources") if isinstance(artifacts, dict) else None
        def _valid_samples(values: Any, expected_min: float) -> bool:
          return (isinstance(values, list) and bool(values) and
                  all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                      math.isfinite(float(value)) and value > 0 for value in values) and
                  min(values) == expected_min)
        def _sha256(value: Any) -> bool:
          return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
        provenance_ok = (
          _valid_samples(samples, candidate_ms) and isinstance(direct, dict) and
          direct.get("status") == "measured" and direct.get("min_ms") == direct_ms and
          _valid_samples(direct_samples, direct_ms) and
          row.get("distinct_binary_identity") is True and row.get("same_session_timing") is True and
          row.get("no_fallback") is True and isinstance(artifacts, dict) and
          artifacts.get("backend_id") == FULL_GRID_BACKEND_ID and
          artifacts.get("distinct_binary_identity") is True and artifacts.get("same_session_timing") is True and
          artifacts.get("no_fallback") is True and _sha256(artifacts.get("source_sha256")) and
          _sha256(artifacts.get("binary_sha256")) and isinstance(resources, dict) and
          isinstance(resources.get("vgpr"), int) and resources.get("vgpr") > 0 and
          isinstance(resources.get("lds_bytes"), int) and resources.get("lds_bytes") > 0 and
          resources.get("scratch_bytes") == 0)
      else:
        provenance_ok = True
      comparable_complete_target = _r5_comparable_complete_target_timing(timing)
      if correctness_ok and comparable_complete_target:
        if provenance_ok:
          qualifying_rows.append(candidate_id)
  full_grid_lineage = "r5_full_grid_128x128" in qualifying_rows
  all_ranking_ids = [row.get("candidate_id") for row in ranking] if isinstance(ranking, list) else []
  checks = {
    "schema": r5.get("schema") == R5_GEOMETRY_SCHEMA,
    "non_promotable_status": r5.get("status") == "PASS_NON_PROMOTABLE",
    "default_route": r5.get("default_route") == "direct_packed",
    "production_dispatch_unchanged": r5.get("production_dispatch_changed") is False,
    "not_directly_promotable": (
      r5.get("promotion_eligible") is False and r5.get("role_shape_integration") is False),
    "unique_candidate_ids": (
      bool(all_ranking_ids) and all(isinstance(value, str) and value for value in all_ranking_ids) and
      len(all_ranking_ids) == len(set(all_ranking_ids))),
    "measured_emitted_win": bool(qualifying_rows),
    "identical_workload_complete_target_win": bool(qualifying_rows),
    "target_backend_lineage": full_grid_lineage,
    "summary_consistent": (
      r5.get("emitted_backend_win") is bool(qualifying_rows) and
      r5.get("promotion_verdict") ==
        ("R5_COMPARABLE_FULL_ROLE_WIN_READY_FOR_R6" if qualifying_rows else
         "NO_PROMOTION_WITHOUT_COMPARABLE_FULL_ROLE_WIN")),
  }
  missing = [name for name, passed in checks.items() if not passed]
  return {
    "schema": R5_RETAINED_VALIDATION_SCHEMA,
    "status": "PASS" if not missing else "BLOCKED",
    "checks": checks,
    "qualifying_candidate_ids": qualifying_rows,
    "exact_blocker": None if not missing else "R5 evidence missing/failed: " + ", ".join(missing),
  }


def _validate_r6_target_role_evidence(target_evidence: dict[str, Any] | None) -> dict[str, Any]:
  """Validate measured target-role evidence without trusting summary booleans.

  R6 is intentionally stricter than the bounded R5 ranking.  A target probe
  must identify the exact role/shape, cover all K=256 epochs, report numeric
  correctness and resources, and prove that it did not silently use the
  direct-packed fallback.  Missing or malformed evidence remains blocked.
  """
  if not isinstance(target_evidence, dict):
    return {"schema": R6_TARGET_EVIDENCE_SCHEMA, "status": "BLOCKED",
            "exact_blocker": "target-role GPU evidence is missing"}
  def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

  def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0

  def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0

  evidence_identity_ok = (
    target_evidence.get("schema") == "tinygrad.mmq_q4k_q8_1_full_grid_target_role_probe.v1" and
    target_evidence.get("status") == "PASS" and target_evidence.get("exact_blocker") is None and
    target_evidence.get("bounded_only") is True and target_evidence.get("default_route") == "direct_packed" and
    target_evidence.get("production_dispatch_changed") is False)
  shape_ok = (target_evidence.get("role") == R6_TARGET_ROLE_SHAPE["role"] and
              target_evidence.get("shape") == [R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")])
  correctness = target_evidence.get("correctness")
  comparison = correctness.get("comparison") if isinstance(correctness, dict) else None
  numeric_ok = (isinstance(correctness, dict) and correctness.get("status") == "PASS" and
                correctness.get("authority") == "same_session_fp16_rounded_ds4_reference" and
                isinstance(comparison, dict) and comparison.get("status") == "pass" and
                comparison.get("mismatch_count") == 0 and
                comparison.get("nan_got") == 0 and comparison.get("nan_reference") == 0 and
                comparison.get("inf_got") == 0 and comparison.get("inf_reference") == 0 and
                comparison.get("joint_finite") == comparison.get("got_size") == 512 * 17408 and
                comparison.get("reference_size") == 512 * 17408 and
                comparison.get("got_shape") == [512, 17408] and
                comparison.get("reference_shape") == [512, 17408])
  timing = target_evidence.get("timing")
  epoch_checks = timing.get("epoch_checks") if isinstance(timing, dict) else None
  metadata = timing.get("metadata_staging") if isinstance(timing, dict) else None
  metadata_rows = metadata.get("per_epoch_vas") if isinstance(metadata, dict) else None
  metadata_ok = (
    isinstance(metadata, dict) and metadata.get("mode") == "fixed_va_gpu_sdma" and
    metadata.get("fixed_va") is True and metadata.get("transfer") == "gpu_sdma" and
    metadata.get("source_preloaded") is True and
    isinstance(metadata_rows, list) and len(metadata_rows) == R6_TARGET_ROLE_SHAPE["K"] // 256 and
    [row.get("epoch") for row in metadata_rows if isinstance(row, dict)] ==
      list(range(R6_TARGET_ROLE_SHAPE["K"] // 256)) and
    all(all(_positive_int(row.get(key)) for key in
            ("source_scales_va", "source_sums_va", "stage_scales_va", "stage_sums_va"))
        for row in metadata_rows if isinstance(row, dict)) and
    len({row["source_scales_va"] for row in metadata_rows}) == len(metadata_rows) and
    len({row["source_sums_va"] for row in metadata_rows}) == len(metadata_rows) and
    len({row["stage_scales_va"] for row in metadata_rows}) == 1 and
    len({row["stage_sums_va"] for row in metadata_rows}) == 1 and
    target_evidence.get("metadata_staging") == metadata)
  timing_ok = (isinstance(timing, dict) and timing.get("k_epoch_launches") == R6_TARGET_ROLE_SHAPE["K"] // 256 and
               timing.get("total_k_epoch_launches") == R6_TARGET_ROLE_SHAPE["K"] // 256 and
               timing.get("n_chunk_tiles") == R6_TARGET_ROLE_SHAPE["N"] // 128 and
               timing.get("accumulation") == TARGET_IN_PLACE_ACCUMULATION and
               timing.get("persistent_buffers") is True and
               timing.get("preloaded_epochs") is True and
               timing.get("stable_metadata_staging") is True and
               timing.get("sync_each_epoch") is False and
               isinstance(timing.get("samples_ms"), list) and len(timing["samples_ms"]) > 0 and
               all(isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0 for v in timing["samples_ms"]) and
               epoch_checks == [] and metadata_ok)
  artifacts = target_evidence.get("artifacts")
  artifacts_map = artifacts if isinstance(artifacts, dict) else {}
  resources = artifacts.get("resources") if isinstance(artifacts, dict) else None
  frozen = artifacts.get("frozen_bundle") if isinstance(artifacts, dict) else None
  frozen_ok = (
    isinstance(frozen, dict) and
    frozen.get("manifest_schema") == "tinygrad.mmq_q4k_q8_1.frozen_target_artifact.v1" and
    frozen.get("fixture_schema") == "tinygrad.mmq_q4k_q8_1_target_fixture.v1" and
    frozen.get("state") == "FROZEN" and
    all(_sha256(frozen.get(key)) for key in ("program_key", "serialized_program_sha256", "fixture_sha256")) and
    isinstance(frozen.get("path"), str) and bool(frozen["path"]) and
    frozen.get("compile_performed") is False and frozen.get("requires_recompile") is False and
    artifacts.get("compile_performed") is False and artifacts.get("requires_recompile") is False and
    target_evidence.get("compile_performed") is False and target_evidence.get("requires_recompile") is False)
  resource_ok = (isinstance(artifacts, dict) and isinstance(resources, dict) and
                 all(_nonnegative_int(resources.get(k)) for k in ("vgpr", "lds_bytes", "scratch_bytes")) and
                 _positive_int(resources.get("vgpr")) and _positive_int(resources.get("lds_bytes")) and
                 resources.get("scratch_bytes") == 0 and resources.get("wavefront_size") == 32 and
                 resources.get("authority") == "native_elf_descriptor" and
                 resources.get("kernarg_bytes") == 40 and
                 artifacts.get("backend_id") == FULL_GRID_BACKEND_ID and
                 artifacts.get("distinct_binary_identity") is True and
                 artifacts.get("same_session_timing") is True and
                 target_evidence.get("distinct_binary_identity") is True and
                 target_evidence.get("same_session_timing") is True and
                 _sha256(artifacts.get("source_sha256")) and _sha256(artifacts.get("binary_sha256")) and
                 frozen_ok)
  repack = target_evidence.get("repack")
  repack_ok = (isinstance(repack, dict) and
               all(_sha256(repack.get(key))
                   for key in ("q4_sha256", "q8_values_sha256", "q8_scales_sha256", "q8_sums_sha256")) and
               repack.get("q4_layout") == "q4_k_bytes[n, k_epoch, 144]" and
               repack.get("q8_layout") == "q8_ds4[epoch, m, groups]" and
               repack.get("q4_epoch_major_layout") == "q4_k_bytes[k_epoch, n, 144]" and
               repack.get("q4_epoch_major_dtype") == "uint32" and
               repack.get("q4_epoch_major_elements") == 20 * 17408 * 144 // 4 and
               _sha256(repack.get("q4_epoch_major_sha256")))

  runtime = target_evidence.get("runtime_evidence")
  runtime_map = runtime if isinstance(runtime, dict) else {}
  launches = runtime.get("launches") if isinstance(runtime, dict) else None
  runtime_identity_ok = (
    isinstance(runtime, dict) and runtime.get("launch_count") == 20 and
    runtime.get("intermediate_readback") is False and runtime.get("external_accumulation_add") is False and
    runtime.get("binary_sha256") == artifacts_map.get("binary_sha256") and
    runtime.get("queue_mode") in ("PM4", "AQL") and
    runtime.get("amd_aql_env") in ("0", "1") and
    runtime.get("amd_aql_effective") is (runtime.get("amd_aql_env") == "1") and
    runtime.get("queue_mode") == ("AQL" if runtime.get("amd_aql_env") == "1" else "PM4") and
    runtime.get("runtime_class") == "tinygrad.runtime.ops_amd.AMDProgram" and
    runtime.get("queue_class") == ("tinygrad.runtime.ops_amd.AMDComputeAQLQueue"
                                   if runtime.get("amd_aql_env") == "1"
                                   else "tinygrad.runtime.ops_amd.AMDComputeQueue") and
    _positive_int(runtime.get("lib_va")) and _positive_int(runtime.get("lib_nbytes")) and
    _positive_int(runtime.get("entry_va")) and _positive_int(runtime.get("descriptor_va")) and
    _positive_int(runtime.get("program_va")) and runtime.get("entry_va") == runtime.get("program_va") and
    runtime["lib_va"] <= runtime["entry_va"] < runtime["lib_va"] + runtime["lib_nbytes"] and
    runtime["lib_va"] <= runtime["descriptor_va"] < runtime["lib_va"] + runtime["lib_nbytes"] and
    target_evidence.get("child_env_overrides") == {"AMD_AQL": runtime.get("amd_aql_env")})
  launch_rows_ok = metadata_ok and isinstance(launches, list) and len(launches) == 20
  if launch_rows_ok:
    expected_names = ("output", "q4", "q8_values", "q8_scales", "q8_original_sums")
    output_va, q4_base, q8_base = None, None, None
    stage_scales_va, stage_sums_va = metadata_rows[0]["stage_scales_va"], metadata_rows[0]["stage_sums_va"]
    for epoch, launch in enumerate(launches):
      args = launch.get("arguments") if isinstance(launch, dict) else None
      kernarg = launch.get("kernarg") if isinstance(launch, dict) else None
      row_ok = (
        launch.get("epoch") == epoch and launch.get("global_size") == [136, 4, 1] and
        launch.get("local_size") == [256, 1, 1] and launch.get("n0") == 0 and
        launch.get("n1") == 17408 and launch.get("tile_count") == 136 and
        isinstance(args, list) and len(args) == 5 and isinstance(kernarg, dict))
      if not row_ok:
        launch_rows_ok = False
        break
      for slot, (arg, name) in enumerate(zip(args, expected_names)):
        if not (isinstance(arg, dict) and arg.get("name") == name and arg.get("slot") == slot and
                arg.get("call_index") == slot and _positive_int(arg.get("va")) and
                _positive_int(arg.get("base_va")) and _positive_int(arg.get("nbytes")) and
                _positive_int(arg.get("base_nbytes")) and _nonnegative_int(arg.get("offset_bytes")) and
                arg.get("va_matches_base_offset") is True and
                arg["va"] == arg["base_va"] + arg["offset_bytes"] and
                arg["offset_bytes"] + arg["nbytes"] <= arg["base_nbytes"]):
          launch_rows_ok = False
          break
      if not launch_rows_ok:
        break
      if epoch == 0:
        output_va, q4_base, q8_base = args[0]["va"], args[1]["base_va"], args[2]["base_va"]
      launch_rows_ok = (
        args[0]["va"] == output_va and args[0]["offset_bytes"] == 0 and
        args[1]["base_va"] == q4_base and args[1]["offset_bytes"] == epoch * args[1]["nbytes"] and
        args[2]["base_va"] == q8_base and args[2]["offset_bytes"] == epoch * args[2]["nbytes"] and
        args[3]["va"] == stage_scales_va and args[3]["offset_bytes"] == 0 and
        args[4]["va"] == stage_sums_va and args[4]["offset_bytes"] == 0 and
        kernarg.get("size") == 40 and _positive_int(kernarg.get("va")) and
        kernarg.get("pointer_words") == [arg["va"] for arg in args] and
        kernarg.get("bound_pointer_words") == [arg["va"] for arg in args] and
        kernarg.get("pointer_words_match_bound") is True)
      if not launch_rows_ok:
        break
  runtime_ok = runtime_identity_ok and launch_rows_ok

  health_mode = target_evidence.get("health_mode")
  health_ok = (
    target_evidence.get("health_before") is True and target_evidence.get("health_after") is True and
    target_evidence.get("mode_health_before") is True and target_evidence.get("mode_health_after") is True and
    target_evidence.get("kernel_faults") == [] and isinstance(health_mode, dict) and
    health_mode.get("before") is True and health_mode.get("after") is True and
    health_mode.get("amd_aql_env") == runtime_map.get("amd_aql_env"))
  fallback_ok = (target_evidence.get("no_fallback") is True and
                 isinstance(artifacts, dict) and artifacts.get("no_fallback") is True and
                 target_evidence.get("accumulation") == TARGET_IN_PLACE_ACCUMULATION and
                 artifacts.get("accumulation") == TARGET_IN_PLACE_ACCUMULATION and
                 target_evidence.get("production_dispatch_changed") is False)
  checks = {"evidence_identity": evidence_identity_ok, "exact_role_shape": shape_ok, "numeric_correctness": numeric_ok,
            "all_k_epochs": timing_ok, "resource_artifact": resource_ok,
            "repack_identity": repack_ok, "runtime_dispatch_evidence": runtime_ok,
            "clean_gpu_health": health_ok, "no_hidden_fallback": fallback_ok}
  missing = [name for name, passed in checks.items() if not passed]
  return {"schema": R6_TARGET_EVIDENCE_SCHEMA, "status": "PASS" if not missing else "BLOCKED",
          "checks": checks, "exact_blocker": None if not missing else "target evidence missing/failed: " + ", ".join(missing)}


def _validate_r6_independent_epoch_evidence(independent_evidence: dict[str, Any] | None) -> dict[str, Any]:
  """Validate the process-per-epoch overwrite proof used beside the strict accumulator proof."""
  if not isinstance(independent_evidence, dict):
    return {"schema": R6_INDEPENDENT_EVIDENCE_SCHEMA, "status": "BLOCKED",
            "exact_blocker": "independent all-epoch GPU evidence is missing"}

  def _sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)

  expected_epochs = list(range(R6_TARGET_ROLE_SHAPE["K"] // 256))
  expected_output_shape = [R6_TARGET_ROLE_SHAPE["M"], R6_TARGET_ROLE_SHAPE["N"]]
  expected_size = R6_TARGET_ROLE_SHAPE["M"] * R6_TARGET_ROLE_SHAPE["N"]
  identity_ok = (
    independent_evidence.get("schema") == "tinygrad.mmq_q4k_q8_1_target_epoch_orchestrator.v1" and
    independent_evidence.get("status") == "PASS" and independent_evidence.get("passed") is True and
    independent_evidence.get("diagnostic_only") is True and independent_evidence.get("promotion_eligible") is False and
    independent_evidence.get("failed_epoch") is None and independent_evidence.get("stop_reason") is None and
    independent_evidence.get("default_route") == "direct_packed" and
    independent_evidence.get("production_dispatch_changed") is False)
  shape_ok = (
    independent_evidence.get("role") == R6_TARGET_ROLE_SHAPE["role"] and
    independent_evidence.get("shape") == [R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")])

  fixture = independent_evidence.get("fixture")
  repack = fixture.get("repack") if isinstance(fixture, dict) else None
  fixture_ok = (
    isinstance(fixture, dict) and fixture.get("schema") == "tinygrad.mmq_q4k_q8_1_target_fixture.v1" and
    fixture.get("role") == R6_TARGET_ROLE_SHAPE["role"] and
    fixture.get("shape") == [R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")] and
    fixture.get("total_epochs") == len(expected_epochs) and
    fixture.get("seeds") == {"q4": 20260721, "q8_source": 20260722} and
    _sha256(fixture.get("source_sha256")) and isinstance(repack, dict) and
    all(_sha256(repack.get(key)) for key in
        ("q4_sha256", "q4_epoch_major_sha256", "q8_values_sha256", "q8_scales_sha256", "q8_sums_sha256")) and
    repack.get("q4_layout") == "q4_k_bytes[n, k_epoch, 144]" and
    repack.get("q4_epoch_major_layout") == "q4_k_bytes[k_epoch, n, 144]" and
    repack.get("q4_epoch_major_dtype") == "uint32" and
    repack.get("q4_epoch_major_elements") == 20 * 17408 * 144 // 4 and
    repack.get("q8_layout") == "q8_ds4[epoch, m, groups]")

  program = independent_evidence.get("program")
  program_resources = program.get("resources") if isinstance(program, dict) else None
  program_ok = (
    isinstance(program, dict) and program.get("backend_id") == FULL_GRID_BACKEND_ID and
    program.get("source_revision") == LLAMA_SOURCE_COMMIT and
    program.get("program_globals") == [0, 1, 2, 3, 4] and
    program.get("program_global_size") == [136, 4, 1] and program.get("program_local_size") == [256, 1, 1] and
    program.get("compile_only_parent") is True and program.get("distinct_binary_identity") is True and
    program.get("no_fallback") is True and
    isinstance(program_resources, dict) and program_resources.get("authority") == "native_elf_descriptor" and
    program_resources.get("vgpr") == 256 and program_resources.get("lds_bytes") == 57856 and
    program_resources.get("scratch_bytes") == 0 and program_resources.get("wavefront_size") == 32 and
    program_resources.get("kernarg_bytes") == 40 and
    all(_sha256(program.get(key)) for key in ("source_sha256", "binary_sha256", "serialized_program_sha256")))

  coverage = independent_evidence.get("coverage")
  coverage_ok = (
    independent_evidence.get("completed_epochs") == expected_epochs and
    isinstance(coverage, dict) and coverage.get("verified_epochs") == expected_epochs and
    coverage.get("verified_k") == R6_TARGET_ROLE_SHAPE["K"] and
    coverage.get("target_epochs") == len(expected_epochs) and coverage.get("complete_target") is True and
    independent_evidence.get("aggregate_shape") == expected_output_shape and
    _sha256(independent_evidence.get("aggregate_sha256")) and
    isinstance(independent_evidence.get("aggregate_sum"), (int, float)) and
    not isinstance(independent_evidence.get("aggregate_sum"), bool))

  epoch_results = independent_evidence.get("epoch_results")
  numerical_ok = isinstance(epoch_results, list) and len(epoch_results) == len(expected_epochs)
  if numerical_ok:
    for epoch, row in enumerate(epoch_results):
      comparison = row.get("comparison") if isinstance(row, dict) else None
      if not (
        isinstance(row, dict) and row.get("schema") == "tinygrad.mmq_q4k_q8_1_target_epoch_orchestrator.v1.epoch" and
        row.get("epoch") == epoch and row.get("shape") == [512, 17408, 256] and
        row.get("status") == "PASS" and row.get("passed") is True and row.get("no_fallback") is True and
        _sha256(row.get("output_sha256")) and
        isinstance(row.get("gpu_ms"), (int, float)) and not isinstance(row.get("gpu_ms"), bool) and row["gpu_ms"] >= 0 and
        isinstance(comparison, dict) and comparison.get("status") == "pass" and
        comparison.get("mismatch_count") == 0 and comparison.get("nan_got") == comparison.get("nan_reference") == 0 and
        comparison.get("inf_got") == comparison.get("inf_reference") == 0 and
        comparison.get("joint_finite") == comparison.get("got_size") == comparison.get("reference_size") == expected_size and
        comparison.get("got_shape") == comparison.get("reference_shape") == expected_output_shape):
        numerical_ok = False
        break

  health = independent_evidence.get("health_attestation")
  epoch_health = independent_evidence.get("epoch_health")
  health_ok = (
    independent_evidence.get("preflight_health") is True and independent_evidence.get("kernel_faults") == [] and
    isinstance(health, dict) and health.get("schema") == "tinygrad.mmq_q4k_q8_1_target_epoch_attestation.v1" and
    health.get("status") == "PASS" and health.get("preflight_passed") is True and
    health.get("all_post_epoch_healthy") is True and health.get("all_kernel_faults_clear") is True and
    isinstance(epoch_health, list) and health.get("epochs") == epoch_health and len(epoch_health) == len(expected_epochs))
  if health_ok:
    health_ok = all(
      isinstance(row, dict) and row.get("epoch") == epoch and row.get("status") == "PASS" and
      row.get("worker_passed") is True and row.get("kernel_log_checked") is True and row.get("kernel_faults") == [] and
      row.get("post_health_checked") is True and row.get("post_health") is True and
      row.get("partial_verified") is True and row.get("stop_stage") is None
      for epoch, row in enumerate(epoch_health))

  fallback_ok = independent_evidence.get("no_fallback") is True and program_ok and numerical_ok
  checks = {
    "evidence_identity": identity_ok, "exact_role_shape": shape_ok, "deterministic_fixture": fixture_ok,
    "pinned_program_source": program_ok, "all_epoch_coverage": coverage_ok,
    "all_epoch_numerical_correctness": numerical_ok, "all_epoch_clean_health": health_ok,
    "no_hidden_fallback": fallback_ok,
  }
  missing = [name for name, passed in checks.items() if not passed]
  return {"schema": R6_INDEPENDENT_EVIDENCE_SCHEMA, "status": "PASS" if not missing else "BLOCKED",
          "checks": checks, "exact_blocker": None if not missing else
          "independent epoch evidence missing/failed: " + ", ".join(missing)}


def _validate_r6_evidence_composition(target_evidence: dict[str, Any] | None,
                                      independent_evidence: dict[str, Any] | None) -> dict[str, Any]:
  """Join two independently executed proofs by fixture identity, never by kernel binary."""
  target_status = _validate_r6_target_role_evidence(target_evidence)
  independent_status = _validate_r6_independent_epoch_evidence(independent_evidence)
  target = target_evidence if isinstance(target_evidence, dict) else {}
  independent = independent_evidence if isinstance(independent_evidence, dict) else {}
  fixture = independent.get("fixture") if isinstance(independent.get("fixture"), dict) else {}
  target_frozen = target.get("artifacts", {}).get("frozen_bundle", {}) if isinstance(target.get("artifacts"), dict) else {}
  canonical_fixture_sha = hashlib.sha256(
    (json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()
  checks = {
    "strict_target_evidence": target_status["status"] == "PASS",
    "independent_epoch_evidence": independent_status["status"] == "PASS",
    "exact_role_shape_join": (
      target.get("role") == independent.get("role") == R6_TARGET_ROLE_SHAPE["role"] and
      target.get("shape") == independent.get("shape") == [R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")]),
    "fixture_hash_join": target_frozen.get("fixture_sha256") == canonical_fixture_sha,
    "repack_hash_join": isinstance(target.get("repack"), dict) and target.get("repack") == fixture.get("repack"),
    "source_revision_join": (
      target.get("reduction", {}).get("source_revision") == LLAMA_SOURCE_COMMIT and
      independent.get("program", {}).get("source_revision") == LLAMA_SOURCE_COMMIT),
  }
  missing = [name for name, passed in checks.items() if not passed]
  return {
    "schema": "q4k-q8-1-mmq-r6-evidence-composition.v1",
    "status": "PASS" if not missing else "BLOCKED",
    "checks": checks,
    "binary_identity_policy": "overwrite and accumulate binaries are independently validated and may differ",
    "target_binary_sha256": target.get("artifacts", {}).get("binary_sha256") if isinstance(target.get("artifacts"), dict) else None,
    "independent_binary_sha256": independent.get("program", {}).get("binary_sha256") if isinstance(independent.get("program"), dict) else None,
    "exact_blocker": None if not missing else "R6 evidence composition missing/failed: " + ", ".join(missing),
  }


def build_r6_route_gate_status(r5_report: dict[str, Any] | None = None,
                               target_evidence: dict[str, Any] | None = None,
                               independent_epoch_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  r5 = r5_report if r5_report is not None else build_r5_geometry_search_report(run=False)
  r5_status = _validate_r5_evidence(r5)
  target_status = _validate_r6_target_role_evidence(target_evidence)
  independent_status = _validate_r6_independent_epoch_evidence(independent_epoch_evidence)
  composition = _validate_r6_evidence_composition(target_evidence, independent_epoch_evidence)
  shape_artifact = build_r6_role_shape_integration_artifact(
    r5, target_evidence=target_evidence, independent_epoch_evidence=independent_epoch_evidence)
  smoke = build_r6_negative_role_fallback_smoke_artifact()
  ready = (r5_status["status"] == "PASS" and shape_artifact.get("status") == "PASS" and
           smoke.get("status") == "PASS_STATIC_DESCRIPTOR")
  role_blocked = r5_status["status"] == "PASS" and not ready
  required_evidence = {
    "comparable_full_role_candidate_win": r5_status["status"] == "PASS",
    "ffn_gate_up_only": smoke["ffn_gate_up_only"],
    "static_negative_role_scope": smoke["static_negative_role_scope"],
    "static_direct_packed_rollback": smoke["static_direct_packed_rollback"],
    "live_route_census": False,
    "target_role_gpu_evidence": target_status["status"] == "PASS",
    "independent_epoch_gpu_evidence": independent_status["status"] == "PASS",
    "evidence_composition": composition["status"] == "PASS",
  }
  return {
    "schema": "q4k-q8-1-mmq-r6-route-gate-status.v1",
    "status": "READY_FOR_ONE_ROLE_OPT_IN" if ready else (
      "BLOCKED_ROLE_SHAPE_INTEGRATION" if role_blocked else "BLOCKED_NO_COMPARABLE_FULL_ROLE_WIN"),
    "role": ROLE,
    "quant": QUANT,
    "default_route": "direct_packed",
    "production_dispatch_changed": False,
    "route_binding_implemented": False,
    "live_route_census_performed": False,
    "required_evidence": required_evidence,
    "r5_evidence_validation": r5_status,
    "role_shape_integration": shape_artifact,
    "target_role_evidence": target_status,
    "independent_epoch_evidence": independent_status,
    "evidence_composition": composition,
    "negative_role_fallback_smoke": smoke,
    "exact_blocker": None if ready else (
      "R6 route binding is illegal until the comparable complete-role winner is integrated"
      if role_blocked else
      "R6 route binding is illegal until R5 reports an identical-workload exact complete target-role win"),
  }


def build_r6_negative_role_fallback_smoke_artifact() -> dict[str, Any]:
  """Verify only static descriptor scope and rollback; this is not a live route census."""
  from extra.qk.route_manifest import ROUTES
  candidate = ROUTES.get("prefill_q4k_q8_1_hybrid_mmq_atom", {})
  default = ROUTES.get("prefill_q4k_direct_tile4x4_default", {})
  roles = tuple(candidate.get("roles", ()))
  excluded = tuple(candidate.get("excluded_roles", ()))
  rejected = ("attn_qo", "ffn_down", "attn_kv")
  role_scope_ok = roles == ("ffn_gate_up",) and all(role in excluded for role in rejected)
  rollback = candidate.get("rollback") if isinstance(candidate.get("rollback"), dict) else {}
  rollback_ok = candidate.get("baseline_route_id") == "direct_packed" and (
    candidate.get("rollback_route") == "direct_packed" or rollback.get("route") == "direct_packed")
  default_ok = default.get("status") == "promoted_default" and default.get("baseline_route_id") == "prefill_q4k_direct_packed_load_direct_out"
  research_only_ok = (
    candidate.get("status") == "research" and candidate.get("purity_status") == "research" and
    candidate.get("selector") == "research_descriptor_only" and candidate.get("research_only") is True and
    "no tinygrad/llm/prefill_routes.py binding" in candidate.get("route_attribution", ""))
  passed = role_scope_ok and rollback_ok and default_ok and research_only_ok
  return {
    "schema": "q4k-q8-1-mmq-r6-negative-role-fallback-smoke.v1",
    "status": "PASS_STATIC_DESCRIPTOR" if passed else "BLOCKED",
    "evidence_scope": "static_route_manifest_descriptor",
    "ffn_gate_up_only": role_scope_ok,
    "static_negative_role_scope": role_scope_ok,
    "static_direct_packed_rollback": rollback_ok and default_ok,
    "research_descriptor_unbound": research_only_ok,
    "negative_role_tests": False,
    "no_hidden_direct_packed_fallback": False,
    "live_route_census_performed": False,
    "accepted_roles": list(roles), "rejected_roles": list(rejected),
    "candidate_route": "prefill_q4k_q8_1_hybrid_mmq_atom", "rollback_route": "direct_packed",
    "runtime_default_route": "prefill_q4k_direct_tile4x4_default",
    "production_dispatch_changed": False,
    "exact_blocker": None if passed else "research descriptor scope/rollback drift",
  }


def build_r6_role_shape_integration_artifact(r5_report: dict[str, Any] | None = None,
                                             *, target_evidence: dict[str, Any] | None = None,
                                             independent_epoch_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  """Record the exact shape/role gap before any one-role route opt-in.

  The emitted probe is a numerically passing bounded kernel, but it covers a
  single 128x128x256 tile.  The first proposed production role is the 14B
  ffn_gate_up GEMM (512x17408x5120), which needs a tiled multi-dispatch
  adapter and negative-role/fallback census.  Keeping this artifact explicit
  prevents a winning microbenchmark from being mistaken for route coverage.
  """
  r5 = r5_report if r5_report is not None else build_r5_geometry_search_report(run=False)
  r5_status = _validate_r5_evidence(r5)
  full_grid_row = next(
    (row for row in r5.get("ranking", ()) if row.get("candidate_id") == "r5_full_grid_128x128"), None)
  winner = next(
    iter(r5_status["qualifying_candidate_ids"]),
    "r5_full_grid_128x128" if full_grid_row is not None else r5.get("best_candidate_id"))
  winner_row = next((row for row in r5.get("ranking", ()) if row.get("candidate_id") == winner), None)
  candidate_shape = None if winner_row is None else winner_row.get("shape")
  shape_matches = candidate_shape == {k: R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")}
  tile_plan = build_full_grid_k_tiled_dispatch_plan(R6_TARGET_ROLE_SHAPE)
  target_status = _validate_r6_target_role_evidence(target_evidence)
  independent_status = _validate_r6_independent_epoch_evidence(independent_epoch_evidence)
  composition = _validate_r6_evidence_composition(target_evidence, independent_epoch_evidence)
  target_shape_matches = target_status.get("checks", {}).get("exact_role_shape") is True
  target_ok = target_status["status"] == "PASS"
  smoke = build_r6_negative_role_fallback_smoke_artifact()
  # The R5 candidate is the 128x128x256 kernel; R6 admission is an adapter
  # claim over the exact role shape, so the target probe—not equality with the
  # R5 microkernel shape—proves this dimension.
  role_ready = (r5_status["status"] == "PASS" and target_shape_matches and target_ok and
                independent_status["status"] == "PASS" and composition["status"] == "PASS" and
                smoke.get("status") == "PASS_STATIC_DESCRIPTOR")
  return {
    "schema": "q4k-q8-1-mmq-r6-role-shape-integration.v1",
    "status": "PASS" if role_ready else "BLOCKED",
    "candidate_id": winner,
    "candidate_shape": candidate_shape,
    "target": dict(R6_TARGET_ROLE_SHAPE),
    "shape_matches": shape_matches,
    "target_shape_matches": target_shape_matches,
    "role_scope": ["ffn_gate_up"],
    "static_negative_role_scope": smoke.get("static_negative_role_scope") is True,
    "static_direct_packed_rollback": smoke.get("static_direct_packed_rollback") is True,
    "live_route_census_performed": False,
    "negative_role_fallback_smoke": smoke,
    "target_role_evidence": target_status,
    "independent_epoch_evidence": independent_status,
    "evidence_composition": composition,
    "tile_plan": tile_plan,
    "production_dispatch_changed": False,
    "exact_blocker": None if role_ready else ("full-grid probe shape is bounded 128x128x256; no 14B ffn_gate_up multi-tile adapter exists"
      if not target_shape_matches else target_status["exact_blocker"] or independent_status["exact_blocker"] or
      composition["exact_blocker"] or "target role evidence is not admitted"),
  }


def build_full_grid_k_tiled_dispatch_plan(shape: dict[str, Any]) -> dict[str, Any]:
  """Plan (without dispatch) how the bounded full-grid kernel would cover a role.

  This is an executable shape audit, not a route selector. The full-grid
  kernel covers all 4x136 M/N tiles in one launch; the 14B role therefore needs
  20 K-epoch launches. Each epoch needs Q4/DS4 repacking plus accumulation.
  Naming those obligations gives R7 a concrete conversion target while keeping
  runtime admission blocked.
  """
  if not isinstance(shape, dict): raise TypeError("shape must be a mapping")
  try: m, n, k = (int(shape[key]) for key in ("M", "N", "K"))
  except (KeyError, TypeError, ValueError) as exc:
    raise ValueError("shape must contain integer M/N/K") from exc
  if min(m, n, k) <= 0 or m % 128 or n % 128 or k % 256:
    return {"schema": "q4k-q8-1-mmq-full-grid-tile-plan.v1", "status": "BLOCKED",
            "shape": {"M": m, "N": n, "K": k},
            "exact_blocker": "role dimensions must be multiples of bounded M/N=128 and K=256"}
  m_tiles, n_tiles, k_epochs = m // 128, n // 128, k // 256
  launches = k_epochs
  return {
    "schema": "q4k-q8-1-mmq-full-grid-tile-plan.v1", "status": "PLANNED",
    "shape": {"M": m, "N": n, "K": k}, "kernel_shape": dict(FULL_GRID_R5_SHAPE),
    "tile_counts": {"M": m_tiles, "N": n_tiles, "K_epochs": k_epochs},
    "launch_count": launches, "source_layout": "full_role_buffers",
    "requires_q4_repack": True, "requires_q8_ds4_repack": True,
    "requires_k_epoch_accumulate": k_epochs > 1,
    "requires_output_tile_scatter": False,
    "production_dispatch_changed": False,
    "monolithic_k512_compile": {
      "status": "BLOCKED", "exception": "NotImplementedError",
      "exact_blocker": "vgpr lease exceeds virtual pool",
      "implication": "K must be split into 256-wide launches with explicit output accumulation",
    },
    "target_shape_k256_compile": {
      "status": "PASS_EMITTED", "shape": {"M": 512, "N": 17408, "K": 256},
      "owner_manifest": "FullGridOwnerCoordinates", "owner_count": 8912896,
      "compile_seconds": 182.23,
      "resources": {"vgpr": 256, "lds_bytes": 57856, "scratch_bytes": 0, "wavefront_size": 32},
      "source_sha256": "b89239852bfaca2709e93a425a40c45e774bd61cd07189f7d3867c40f06fb196",
      "binary_sha256": "21908e0bff83e5b7f7d4796cfb8a15d377d20ebfa7cefc63117607c5f03d0143",
      "exact_blocker": "target-shape GPU dispatch/correctness and 20-epoch accumulation evidence are still absent",
    },
    "reduced_grid_256_probe": {
      "status": "BLOCKED_NUMERIC", "shape": {"M": 256, "N": 256, "K": 256},
      "mismatch_count": 65425, "output_size": 65536,
      "max_abs_error": 840.8983764648438, "mean_abs_error": 120.01234436035156,
      "timing_ms": 7.6224,
      "resources": {"vgpr": 256, "lds_bytes": 57856, "scratch_bytes": 0, "wavefront_size": 32},
      "binary_sha256": "03fc97ef67921c2a5546e1fa709101426c42d028a7ceabad96f4656914f762d2",
      "exact_blocker": "single-grid 128x128 passes but 2x2 M/N grid has writeback/address mismatches; target dispatch is not numerically admissible",
    },
    "per_store_accumulate_sink_probe": {
      "status": "BLOCKED_TIMEOUT", "timeout_seconds": 360,
      "exact_blocker": "overwrite/accumulate two-launch per-store LOAD+ADD sink exceeded hard compile deadline before structured output",
      "next_action": "use fresh partial outputs and tinygrad elementwise accumulation",
    },
    "k_tiled_accumulate_probe": {
      "status": "PASS_BOUNDED", "shape": {"M": 128, "N": 128, "K": 512},
      "k_epoch_launches": 2, "mismatch_count": 0, "max_abs_error": 2.44140625e-4,
      "accumulation": "tinygrad_elementwise_add",
      "resources": {"vgpr": 256, "lds_bytes": 57856, "scratch_bytes": 0, "wavefront_size": 32},
      "converted_slice": "K_epoch_accumulation",
      "exact_blocker": "bounded two-epoch proof only; production role tiling/repack/fallback census remain absent",
    },
    "exact_blocker": "production role adapter, Q4/DS4 repack, and negative-role/fallback census are not implemented; bounded K-epoch accumulation is proven",
  }


def build_r7_reduction_status(target_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  rows = [
    {
      "source_component": "cooperative tile loop",
      "source": "mmq.cuh:mul_mat_q_process_tile",
      "status": "blocked_translation",
      "next_action": "promote bounded elementwise K-epoch adapter into a production-shape repack/dispatch harness; current owner trace is still proof-only",
      "blocking_evidence": "bounded 128x128x512 elementwise K-epoch proof passes, but monolithic K=512 fails vgpr lease exceeds virtual pool and 14B repack/dispatch is absent",
    },
    {
      "source_component": "Q4_K tile_x staging",
      "source": "mmq.cuh:load_tiles_q4_K",
      "status": "oracle_backed_not_converted",
      "next_action": "translate bounded tile_x staging after cooperative numeric skeleton exists",
      "blocking_evidence": "Q4_K staging has no production-shape launch/ownership adapter beyond the bounded probe",
    },
    {
      "source_component": "Q8_1 tile_y two-panel lifecycle",
      "source": "mmq.cuh:mul_mat_q_process_tile",
      "status": "partially_converted",
      "next_action": "existing DS4 LDS skeleton stages values once; llama two-panel lifecycle still unconverted",
      "blocking_evidence": "Q8_1 two-panel lifecycle and fallback census are absent for the 14B role route",
    },
  ]
  target_status = _validate_r6_target_role_evidence(target_evidence)
  reduction = target_evidence.get("reduction") if isinstance(target_evidence, dict) else None
  owned = (isinstance(reduction, dict) and reduction.get("source_revision") == LLAMA_SOURCE_COMMIT and
           reduction.get("source_anchors") == list(R7_REQUIRED_SOURCE_ANCHORS) and
           isinstance(reduction.get("owned_components"), list) and
           {row["source_component"] for row in rows}.issubset(set(reduction["owned_components"])))
  if target_status["status"] == "PASS" and owned:
    converted = [
      {**{key: value for key, value in row.items() if key not in ("next_action", "blocking_evidence")},
       "status": "owned_atom",
       "evidence": {"target_role_probe": True, "source_revision": reduction["source_revision"],
                    "source_anchors": list(reduction["source_anchors"])}}
      for row in rows]
    return {
      "schema": "q4k-q8-1-mmq-r7-reduction-status.v1", "status": "PASS_TARGET_ROLE_REDUCTION",
      "production_dispatch_changed": False, "remaining_rows": [], "converted_rows": converted,
      "target_role_evidence": target_status, "exact_blocker": None,
    }
  blocker = target_status["exact_blocker"] if target_status["status"] != "PASS" else "target reduction/source ownership evidence is missing"
  return {
    "schema": "q4k-q8-1-mmq-r7-reduction-status.v1",
    "status": "BLOCKED_REMAINING_SOURCE_CLONE_ROWS",
    "production_dispatch_changed": False,
    "remaining_rows": rows,
    "target_role_evidence": target_status,
    "exact_blocker": blocker,
  }


def build_search_report(*, run: bool = False, warmups: int = 0, rounds: int = 1,
                        runner: Callable[[BoundedMMQConfig], dict[str, Any]] = run_bounded_harness,
                        full_gpu_probe: dict[str, Any] | None = None,
                        target_role_probe: dict[str, Any] | None = None,
                        independent_epoch_evidence: dict[str, Any] | None = None,
                        r5_evidence: dict[str, Any] | None = None) -> dict[str, Any]:
  rows = []
  for candidate in SEARCHABLE_CANDIDATES:
    row = candidate.to_json()
    if run:
      try:
        result = runner(candidate.config(warmups=warmups, rounds=rounds))
        row["run"] = {
          "status": result["status"],
          "correctness": result["correctness"],
          "timing": result["timing"],
          "artifacts": result["artifacts"],
          "blockers": result["blockers"],
        }
      except Exception as exc:
        row["run"] = {"status": "ERROR", "error": str(exc)}
    rows.append(row)

  r4_evidence = build_r4_evidence_artifacts()
  coop_evidence = coop_tile_blocked_translation_evidence(
    BoundedMMQConfig(m_tile=16, n_tile=16, k_groups=8, backend=AMD_DS4_COOP_TILE_BACKEND_ID))
  r5_report = r5_evidence if r5_evidence is not None else build_r5_geometry_search_report(run=False)
  r6_status = build_r6_route_gate_status(
    r5_report, target_evidence=target_role_probe, independent_epoch_evidence=independent_epoch_evidence)
  r7_status = build_r7_reduction_status(target_role_probe)
  r5_ready = r6_status["r5_evidence_validation"]["status"] == "PASS"
  one_role_evidence_ready = (
    r6_status["status"] == "READY_FOR_ONE_ROLE_OPT_IN" and
    r7_status["status"] == "PASS_TARGET_ROLE_REDUCTION")
  role_candidate = {
    "candidate_id": FULL_GPU_PROBE_CANDIDATE_ID,
    "backend": FULL_GRID_BACKEND_ID,
    "role": FULL_GPU_PROBE_ROLE,
    "shape": dict(R6_TARGET_ROLE_SHAPE),
    "status": "one_role_evidence_ready" if one_role_evidence_ready else "evidence_only",
    "one_role_opt_in_eligible": False,
    "research_opt_in_implementation_eligible": one_role_evidence_ready,
    "route_binding_implemented": False,
    "live_route_census_performed": False,
    "promotion_eligible": False,
    "default_route": "direct_packed",
    "production_dispatch_changed": False,
    "r6_route_gate_status": r6_status["status"],
    "r7_reduction_status": r7_status["status"],
    "evidence": target_role_probe,
  }
  blocked_candidates = [dict(row) for row in BLOCKED_CANDIDATES]
  if one_role_evidence_ready:
    blocked_candidates[0]["reason"] = (
      "ffn_gate_up evidence is ready to implement a research opt-in, but the live route binding, "
      "remaining six-row policy, whole-model correctness/memory census, and production rollback proof are absent")
  promotion_gate = evaluate_candidate_promotion(
    owner_coverage=r4_evidence["owner_coverage"], cooperative_tile=coop_evidence)
  promotion_gate.update({
    "scope": "full_14b_prefill_production_route",
    "one_role_evidence_ready": one_role_evidence_ready,
    "exact_blocker": (
      "one ffn_gate_up role is evidence-ready; full production route admission remains incomplete"
      if one_role_evidence_ready else
      "full production route admission remains incomplete"),
  })
  return {
    "schema": SCHEMA,
    "status_semantics": {
      "one_role_evidence_ready": (
        "the exact ffn_gate_up kernel passed retained R5/R6/R7 evidence; no live route is bound"),
      "ready_for_one_role_opt_in": (
        "evidence gate only; implementation plus a live negative-role/fallback census are still required"),
      "production_promotion": "requires the complete multi-role policy and whole-model gates",
    },
    "candidate_route_id": CANDIDATE_ROUTE_ID,
    "public_label": PUBLIC_LABEL,
    "comparator_id": COMPARATOR_ID,
    "llama_mmq_geometry": LLAMA_MMQ_GEOMETRY,
    "llama_kernel_source_policy": {
      "mode": "point_to_local_clone_do_not_vendor",
      "handcoded_translation": True,
      "reduction_model": "unconverted_parts_point_to_clone_converted_parts_become_bounded_atoms",
      "atom_definition": "the atom is the minimized hand-coded tinygrad translation of the cloned llama kernel pieces that pass bounded machine-search proof",
      "sources": list(LLAMA_KERNEL_SOURCES),
    },
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "done_components": list(DONE_COMPONENTS),
    "searchable_components": [row["component"] for row in DONE_COMPONENTS if row["status"] == "done"],
    "searchable_candidates": rows,
    "blocked_candidates": blocked_candidates,
    "r4_evidence_artifacts": r4_evidence,
    "r5_geometry_search": r5_report,
    "r5_geometry_search_status": {
      "status": ("complete_for_one_role_evidence" if one_role_evidence_ready else
                 "comparable_full_role_win_ready_for_r6" if r5_ready else "ready_for_bounded_geometry_search"),
      "reason": (
        "R5 identical-workload complete-role win and the independent R6/R7 one-role evidence are retained; production promotion remains blocked"
        if one_role_evidence_ready else
        "R5 identical-workload complete-role win is retained; R6/R7 role evidence remains required"
        if r5_ready else
        "R4 lowered owner trace, staging evidence, and R5 bounded correctness pass; R6 remains blocked until R5 reports an identical-workload exact complete target-role win"),
      "required_r4_evidence": ["owner_coverage:PASS", "staging_sum_slots:PASS", "gpu_owner_trace:PASS"],
    },
    "r6_route_gate_status": r6_status,
    "r7_reduction_status": r7_status,
    "role_candidates": [role_candidate],
    # Optional because the base machine search is compile/evidence-only.  When
    # supplied, this joins the exact GPU artifact without changing the default
    # route or making the incomplete probe promotable.
    "full_gpu_probe_candidate": None if full_gpu_probe is None else build_full_gpu_probe_candidate(full_gpu_probe),
    "target_role_probe": target_role_probe,
    "independent_epoch_evidence": independent_epoch_evidence,
    "promotion_verdict": (
      "ONE_ROLE_EVIDENCE_READY_PRODUCTION_PROMOTION_BLOCKED" if one_role_evidence_ready else
      "R5_COMPARABLE_FULL_ROLE_WIN_READY_FOR_R6" if r5_ready else
      "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN"),
    "production_promotion_verdict": "BLOCKED",
    "milestone_evidence": _default_milestone_evidence(),
    "promotion_gate": promotion_gate,
  }


def build_boltbeam_oracle_trace(*, context: int = 512) -> dict[str, Any]:
  spec = describe_q4k_q8_1_mmq_tile(role="ffn_gate_up", m=128, n=128, k=256, m_tile=128, n_tile=128,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  owners = list(llama_mma_writeback_owners(spec))
  owner_hash = hashlib.sha256(json.dumps(owners, sort_keys=True).encode()).hexdigest()[:16]
  return {
    "schema": "boltbeam.hw_trace.v1",
    "model_id": "qwen3-14b-q4k-mmq-oracle",
    "target_id": "amd_gfx1100",
    "workload": "prefill",
    "provider_id": "tinygrad/mmq-llama-oracle",
    "source_schema": SCHEMA,
    "contexts": [context],
    "metadata": {
      "production_dispatch_changed": False,
      "default_route": "direct_packed",
      "promotion_eligible": False,
      "promotion_verdict": "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN",
    },
    "rows": [
      {
        "scope": "kernel",
        "context": context,
        "kernel": "llama_mmq_coop_tile_oracle",
        "kind": "gemm",
        "role": ROLE,
        "quant": QUANT,
        "shape": [spec.tile_m, spec.tile_n, spec.k],
        "calls": 1,
        "tile_oracle": {
          "kind": "cooperative_tile",
          "source": "extra.qk.mmq_llama_oracle.llama_mma_writeback_owners",
          "candidate_id": "llama_mmq_coop_tile_oracle",
          "backend_id": LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID,
          "target_backend_atom_id": AMD_DS4_COOP_TILE_BACKEND_ID,
          "route_family": "llama_mmq_cooperative_tile",
          "geometry": {
            **LLAMA_MMQ_GEOMETRY,
            "warp_size": 32,
            "tile_c_i": 16,
            "tile_c_j": 16,
            "tile_c_ne": 256,
          },
          "wave_ownership": {
            "owner": "warp_id_owns_16x16_output_fragment",
            "mapping": "8 warps cover 128 M rows; each warp owns 16-row stripes across 16-column fragments",
            "requires_single_store_per_output": True,
          },
          "writeback_owner_count": len(owners),
          "expected_writeback_owners_hash": owner_hash,
          "writeback_owners": owners,
        },
        "candidate_geometry": {
          "mmq_x": 128,
          "mmq_y": 128,
          "iter_k": 256,
          "nwarps": 8,
          "warp_size": 32,
          "tile_m": 128,
          "tile_n": 128,
          "tile_k": 256,
          "tile_c_i": 16,
          "tile_c_j": 16,
        },
        "resource_constraints": {
          "scratch_bytes": {"eq": 0},
          "duplicate_store_count": {"eq": 0},
          "missing_store_count": {"eq": 0},
          "production_dispatch_changed": {"eq": False},
        },
        "sources": {
          "llama_mmq_source": "/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh",
          "tinygrad_oracle": "extra/qk/mmq_llama_oracle.py",
        },
      }
    ],
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Bounded machine-search report for completed 14B Q4_K/Q8_1 MMQ pieces")
  ap.add_argument("--run", action="store_true", help="execute searchable bounded candidates")
  ap.add_argument("--warmups", type=int, default=0)
  ap.add_argument("--rounds", type=int, default=1)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  ap.add_argument("--boltbeam-oracle-trace", action="store_true",
                  help="emit a boltbeam.hw_trace.v1 cooperative-tile oracle evidence trace")
  ap.add_argument("--r5-geometry-search", action="store_true",
                  help=f"emit {R5_GEOMETRY_SCHEMA} instead of the base search report")
  ap.add_argument("--experiment", type=pathlib.Path,
                  help="canonical tinygrad.mmq_candidate_spec.v1 to execute")
  ap.add_argument("--bundle-out", type=pathlib.Path, help="atomic experiment bundle output directory")
  ap.add_argument("--experiment-id", help="immutable BoltBeam experiment identity")
  ap.add_argument("--system-snapshot-id", help="closed-system snapshot identity")
  ap.add_argument("--r5-evidence", type=pathlib.Path,
                  help=f"retained {R5_GEOMETRY_SCHEMA} JSON for the base report")
  ap.add_argument("--target-role-probe", type=pathlib.Path,
                  help="strict full-role target GPU evidence JSON for the base report")
  ap.add_argument("--independent-epoch-evidence", type=pathlib.Path,
                  help="independent process-per-epoch GPU evidence JSON for the base report")
  return ap.parse_args()


def _load_json_object(path: pathlib.Path, label: str) -> dict[str, Any]:
  def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")
  try:
    value = json.loads(path.read_text(), parse_constant=_reject_nonfinite)
  except (OSError, json.JSONDecodeError, ValueError) as exc:
    raise SystemExit(f"{label}: unable to load JSON object from {path}: {exc}") from exc
  if not isinstance(value, dict):
    raise SystemExit(f"{label}: expected a JSON object in {path}")
  return value


def main() -> None:
  args = _parse_args()
  evidence_paths = (args.r5_evidence, args.target_role_probe, args.independent_epoch_evidence)
  if any(path is not None for path in evidence_paths) and (
      args.experiment is not None or args.boltbeam_oracle_trace or args.r5_geometry_search):
    raise SystemExit("evidence inputs are valid only for the base machine-search report")
  if args.experiment is not None:
    if args.bundle_out is None or not args.experiment_id or not args.system_snapshot_id:
      raise SystemExit("--experiment requires --bundle-out, --experiment-id, and --system-snapshot-id")
    from extra.qk.mmq_experiment import MMQCandidateSpec, produce_experiment_bundle
    spec = MMQCandidateSpec.from_json(json.loads(args.experiment.read_text()))
    output = produce_experiment_bundle(spec, args.bundle_out, experiment_id=args.experiment_id,
                                       system_snapshot_id=args.system_snapshot_id)
    print(json.dumps({"bundle": str(output), "candidate_id": spec.candidate_id}, sort_keys=True))
    return
  if args.boltbeam_oracle_trace:
    report = build_boltbeam_oracle_trace()
  elif args.r5_geometry_search:
    report = build_r5_geometry_search_report(run=args.run, warmups=args.warmups, rounds=args.rounds)
  else:
    report = build_search_report(
      run=args.run, warmups=args.warmups, rounds=args.rounds,
      r5_evidence=None if args.r5_evidence is None else _load_json_object(args.r5_evidence, "--r5-evidence"),
      target_role_probe=None if args.target_role_probe is None else
        _load_json_object(args.target_role_probe, "--target-role-probe"),
      independent_epoch_evidence=None if args.independent_epoch_evidence is None else
        _load_json_object(args.independent_epoch_evidence, "--independent-epoch-evidence"))
  text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
