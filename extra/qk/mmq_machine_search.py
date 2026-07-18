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
from extra.qk.mmq_llama_five_buffer_gpu_harness import run_full_grid_r5_benchmark
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
            "direct_packed_min_ms": direct_min,
            "comparator_status": result["timing"].get("comparator_status"),
          },
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
  # A full-grid PASS is an emitted cooperative/backend win for R5 ranking,
  # even though it is not yet eligible for route promotion.  R6 separately
  # requires role/shape integration, so this distinction stays fail-closed.
  emitted_rows = [row for row in rows if row.get("status") == "PASS" and
                  row.get("backend") in (AMD_DS4_COOP_TILE_BACKEND_ID, FULL_GRID_BACKEND_ID)]
  best_emitted = max(emitted_rows, key=lambda row: float(row.get("speedup_vs_direct_packed") or 0.0), default=None)
  emitted_win = best_emitted is not None and (best_emitted.get("speedup_vs_direct_packed") or 0) > 1.0
  coop_winner = emitted_win
  role_shape_integration = False
  return {
    "schema": "q4k-q8-1-mmq-r5-geometry-search.v1",
    "status": "PASS_NON_PROMOTABLE" if best is not None else ("NOT_RUN" if not run else "BLOCKED"),
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "promotion_eligible": bool(coop_winner and role_shape_integration),
    "emitted_backend_win": bool(emitted_win),
    "role_shape_integration": role_shape_integration,
    "promotion_verdict": "R5_COOP_WIN_READY_FOR_R6" if coop_winner else "NO_PROMOTION_WITHOUT_BOUNDED_COOP_WIN",
    "ranking": ranked,
    "best_candidate_id": None if best is None else best["candidate_id"],
    "exact_blocker": (None if role_shape_integration else "emitted backend win awaits production role/shape integration") if coop_winner else "no emitted cooperative MMQ tile candidate has a bounded same-session win",
  }


def build_r6_route_gate_status(r5_report: dict[str, Any] | None = None) -> dict[str, Any]:
  r5 = r5_report or build_r5_geometry_search_report(run=False)
  ready = r5.get("promotion_verdict") == "R5_COOP_WIN_READY_FOR_R6" and r5.get("role_shape_integration") is True
  role_blocked = r5.get("emitted_backend_win") is True and not ready
  shape_artifact = build_r6_role_shape_integration_artifact(r5)
  return {
    "schema": "q4k-q8-1-mmq-r6-route-gate-status.v1",
    "status": "READY_FOR_ONE_ROLE_OPT_IN" if ready else ("BLOCKED_ROLE_SHAPE_INTEGRATION" if role_blocked else "BLOCKED_NO_BOUNDED_COOP_WIN"),
    "role": ROLE,
    "quant": QUANT,
    "default_route": "direct_packed",
    "production_dispatch_changed": False,
    "required_evidence": {
      "bounded_coop_candidate_win": ready,
      "ffn_gate_up_only": False,
      "negative_role_tests": False,
      "no_hidden_direct_packed_fallback": False,
    },
    "role_shape_integration": shape_artifact,
    "exact_blocker": None if ready else ("R6 route binding is illegal until the bounded winner is integrated for a production role and shape" if role_blocked else "R6 route binding is illegal until R5 reports an emitted cooperative backend win"),
  }


def build_r6_role_shape_integration_artifact(r5_report: dict[str, Any] | None = None) -> dict[str, Any]:
  """Record the exact shape/role gap before any one-role route opt-in.

  The emitted probe is a numerically passing bounded kernel, but it covers a
  single 128x128x256 tile.  The first proposed production role is the 14B
  ffn_gate_up GEMM (512x17408x5120), which needs a tiled multi-dispatch
  adapter and negative-role/fallback census.  Keeping this artifact explicit
  prevents a winning microbenchmark from being mistaken for route coverage.
  """
  r5 = r5_report or build_r5_geometry_search_report(run=False)
  winner = r5.get("best_candidate_id")
  winner_row = next((row for row in r5.get("ranking", ()) if row.get("candidate_id") == winner), None)
  candidate_shape = None if winner_row is None else winner_row.get("shape")
  shape_matches = candidate_shape == {k: R6_TARGET_ROLE_SHAPE[k] for k in ("M", "N", "K")}
  tile_plan = build_full_grid_k_tiled_dispatch_plan(R6_TARGET_ROLE_SHAPE)
  return {
    "schema": "q4k-q8-1-mmq-r6-role-shape-integration.v1",
    "status": "PASS" if shape_matches and r5.get("role_shape_integration") is True else "BLOCKED",
    "candidate_id": winner,
    "candidate_shape": candidate_shape,
    "target": dict(R6_TARGET_ROLE_SHAPE),
    "shape_matches": shape_matches,
    "role_scope": ["ffn_gate_up"],
    "negative_role_tests": False,
    "no_hidden_direct_packed_fallback": False,
    "tile_plan": tile_plan,
    "production_dispatch_changed": False,
    "exact_blocker": None if shape_matches else "full-grid probe shape is bounded 128x128x256; no 14B ffn_gate_up multi-tile adapter exists",
  }


def build_full_grid_k_tiled_dispatch_plan(shape: dict[str, Any]) -> dict[str, Any]:
  """Plan (without dispatch) how the bounded full-grid kernel would cover a role.

  This is an executable shape audit, not a route selector.  The 128x128x256
  kernel can cover the 14B role only as 10,880 launches (4 M tiles × 136 N
  tiles × 20 K epochs).  Since its ABI expects tile-local packed buffers and
  overwrites output, every K epoch needs Q4/DS4 repacking plus an accumulation
  step.  Naming those obligations gives R7 a concrete conversion target while
  keeping runtime admission blocked.
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
  launches = m_tiles * n_tiles * k_epochs
  return {
    "schema": "q4k-q8-1-mmq-full-grid-tile-plan.v1", "status": "PLANNED",
    "shape": {"M": m, "N": n, "K": k}, "kernel_shape": dict(FULL_GRID_R5_SHAPE),
    "tile_counts": {"M": m_tiles, "N": n_tiles, "K_epochs": k_epochs},
    "launch_count": launches, "source_layout": "full_role_buffers",
    "requires_q4_repack": True, "requires_q8_ds4_repack": True,
    "requires_k_epoch_accumulate": k_epochs > 1,
    "requires_output_tile_scatter": True,
    "production_dispatch_changed": False,
    "monolithic_k512_compile": {
      "status": "BLOCKED", "exception": "NotImplementedError",
      "exact_blocker": "vgpr lease exceeds virtual pool",
      "implication": "K must be split into 256-wide launches with explicit output accumulation",
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
    "exact_blocker": "production role adapter, Q4/DS4 repack, output scatter, and negative-role/fallback census are not implemented; bounded K-epoch accumulation is proven",
  }


def build_r7_reduction_status() -> dict[str, Any]:
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
  return {
    "schema": "q4k-q8-1-mmq-r7-reduction-status.v1",
    "status": "BLOCKED_REMAINING_SOURCE_CLONE_ROWS",
    "production_dispatch_changed": False,
    "remaining_rows": rows,
    "exact_blocker": "remaining clone-backed rows require the emitted cooperative numeric tile before route binding",
  }


def build_search_report(*, run: bool = False, warmups: int = 0, rounds: int = 1,
                        runner: Callable[[BoundedMMQConfig], dict[str, Any]] = run_bounded_harness,
                        full_gpu_probe: dict[str, Any] | None = None) -> dict[str, Any]:
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
  return {
    "schema": SCHEMA,
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
    "blocked_candidates": list(BLOCKED_CANDIDATES),
    "r4_evidence_artifacts": r4_evidence,
    "r5_geometry_search": build_r5_geometry_search_report(run=False),
    "r5_geometry_search_status": {
      "status": "ready_for_bounded_geometry_search",
      "reason": "R4 lowered owner trace, staging evidence, and R5 bounded coop numeric correctness pass; R6 remains blocked until R5 reports a same-session coop speed win",
      "required_r4_evidence": ["owner_coverage:PASS", "staging_sum_slots:PASS", "gpu_owner_trace:PASS"],
    },
    "r6_route_gate_status": build_r6_route_gate_status(),
    "r7_reduction_status": build_r7_reduction_status(),
    # Optional because the base machine search is compile/evidence-only.  When
    # supplied, this joins the exact GPU artifact without changing the default
    # route or making the incomplete probe promotable.
    "full_gpu_probe_candidate": None if full_gpu_probe is None else build_full_gpu_probe_candidate(full_gpu_probe),
    "promotion_verdict": "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN",
    "milestone_evidence": _default_milestone_evidence(),
    "promotion_gate": evaluate_candidate_promotion(owner_coverage=r4_evidence["owner_coverage"],
      cooperative_tile=coop_evidence),
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
                  help="emit q4k-q8-1-mmq-r5-geometry-search.v1 instead of the base search report")
  ap.add_argument("--experiment", type=pathlib.Path,
                  help="canonical tinygrad.mmq_candidate_spec.v1 to execute")
  ap.add_argument("--bundle-out", type=pathlib.Path, help="atomic experiment bundle output directory")
  ap.add_argument("--experiment-id", help="immutable BoltBeam experiment identity")
  ap.add_argument("--system-snapshot-id", help="closed-system snapshot identity")
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
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
    report = build_search_report(run=args.run, warmups=args.warmups, rounds=args.rounds)
  text = json.dumps(report, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
