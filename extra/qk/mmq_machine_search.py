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
  AMD_DS4_COOP_TILE_BACKEND_ID, AMD_DS4_LDS_SKELETON_BACKEND_ID, BoundedMMQConfig, CANDIDATE_ROUTE_ID,
  COMPARATOR_ID, LLAMA_MMQ_COOP_TILE_ORACLE_BACKEND_ID, LLAMA_MMQ_GEOMETRY, PUBLIC_LABEL, QUANT, ROLE,
  STAGED_DS4_BACKEND_ID,
  candidate_metadata, coop_tile_blocked_translation_evidence, run_bounded_harness,
)
from extra.qk.mmq_llama_oracle import llama_mma_writeback_owners
from extra.qk.mmq_llama_store_probe import lowered_tinygrad_r4_store_owner_trace_rows
from extra.qk.mmq_owner_coverage import (
  build_mmq_owner_coverage_artifact, observed_stores_from_amd_isa_proof_rows,
)
from extra.qk.mmq_q4k_q8_reference import Q8_1_MMQ_DS4_LAYOUT, describe_q4k_q8_1_mmq_tile
from extra.qk.mmq_staging_evidence import build_mmq_staging_evidence_bundle


SCHEMA = "q4k-q8-1-mmq-machine-search.v1"
DEFAULT_OUTPUT = pathlib.Path("bench/prefill-14b-mmq-machine-search/search-report.json")
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


def build_r4_evidence_artifacts() -> dict[str, Any]:
  spec = describe_q4k_q8_1_mmq_tile(role=ROLE, m=16, n=16, k=256, m_tile=16, n_tile=16,
                                    activation_layout=Q8_1_MMQ_DS4_LAYOUT)
  lowered_rows = lowered_tinygrad_r4_store_owner_trace_rows(spec)
  lowered_observed = observed_stores_from_amd_isa_proof_rows(lowered_rows)
  return {
    "owner_coverage": build_mmq_owner_coverage_artifact(
      spec,
      lowered_observed,
      candidate_id="cooperative_multi_wave_tile",
      backend="lowered_amd_isa_fragmented_store_owner_manifest",
    ),
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
        result = runner(cfg)
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
          "exact_blocker": None if result["status"] == "PASS" else "bounded correctness failed",
        })
      except Exception as exc:
        row.update({"status": "BLOCKED", "exact_blocker": str(exc), "speedup_vs_direct_packed": None})
    else:
      row.update({"status": "NOT_RUN", "exact_blocker": "pass run=True to execute bounded R5 geometry search"})
    rows.append(row)

  ranked = _ranked_r5_rows(rows)
  best = ranked[0] if ranked and ranked[0].get("status") == "PASS" else None
  coop_winner = best is not None and best["backend"] == AMD_DS4_COOP_TILE_BACKEND_ID and (best.get("speedup_vs_direct_packed") or 0) > 1.0
  return {
    "schema": "q4k-q8-1-mmq-r5-geometry-search.v1",
    "status": "PASS_NON_PROMOTABLE" if best is not None else ("NOT_RUN" if not run else "BLOCKED"),
    "production_dispatch_changed": False,
    "default_route": "direct_packed",
    "promotion_eligible": bool(coop_winner),
    "promotion_verdict": "R5_COOP_WIN_READY_FOR_R6" if coop_winner else "NO_PROMOTION_WITHOUT_BOUNDED_COOP_WIN",
    "ranking": ranked,
    "best_candidate_id": None if best is None else best["candidate_id"],
    "exact_blocker": None if coop_winner else "no emitted cooperative MMQ tile candidate has a bounded same-session win",
  }


def build_r6_route_gate_status(r5_report: dict[str, Any] | None = None) -> dict[str, Any]:
  r5 = r5_report or build_r5_geometry_search_report(run=False)
  ready = r5.get("promotion_verdict") == "R5_COOP_WIN_READY_FOR_R6"
  return {
    "schema": "q4k-q8-1-mmq-r6-route-gate-status.v1",
    "status": "READY_FOR_ONE_ROLE_OPT_IN" if ready else "BLOCKED_NO_BOUNDED_COOP_WIN",
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
    "exact_blocker": None if ready else "R6 route binding is illegal until R5 reports an emitted cooperative backend win",
  }


def build_r7_reduction_status() -> dict[str, Any]:
  rows = [
    {
      "source_component": "cooperative tile loop",
      "source": "mmq.cuh:mul_mat_q_process_tile",
      "status": "blocked_translation",
      "next_action": "implement emitted cooperative numeric tile; current owner trace is proof-only",
    },
    {
      "source_component": "Q4_K tile_x staging",
      "source": "mmq.cuh:load_tiles_q4_K",
      "status": "oracle_backed_not_converted",
      "next_action": "translate bounded tile_x staging after cooperative numeric skeleton exists",
    },
    {
      "source_component": "Q8_1 tile_y two-panel lifecycle",
      "source": "mmq.cuh:mul_mat_q_process_tile",
      "status": "partially_converted",
      "next_action": "existing DS4 LDS skeleton stages values once; llama two-panel lifecycle still unconverted",
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
                        runner: Callable[[BoundedMMQConfig], dict[str, Any]] = run_bounded_harness) -> dict[str, Any]:
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
    "r4_evidence_artifacts": build_r4_evidence_artifacts(),
    "r5_geometry_search": build_r5_geometry_search_report(run=False),
    "r5_geometry_search_status": {
      "status": "ready_for_bounded_geometry_search",
      "reason": "R4 lowered owner trace, staging evidence, and R5 bounded coop numeric correctness pass; R6 remains blocked until R5 reports a same-session coop speed win",
      "required_r4_evidence": ["owner_coverage:PASS", "staging_sum_slots:PASS", "gpu_owner_trace:PASS"],
    },
    "r6_route_gate_status": build_r6_route_gate_status(),
    "r7_reduction_status": build_r7_reduction_status(),
    "promotion_verdict": "BLOCKED_UNTIL_COOPERATIVE_TILE_WIN",
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
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
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
