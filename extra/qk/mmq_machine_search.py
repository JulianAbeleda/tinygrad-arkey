#!/usr/bin/env python3
"""Bounded machine-search surface for the completed 14B Q4_K/Q8_1 MMQ pieces.

This does not bind production prefill. It turns the pieces that are already
implemented into explicit candidate rows, and records the unfinished llama-style
pieces as blocked rows instead of treating them as selectable variants.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import pathlib
import sys
from typing import Any, Callable

if __package__ in (None, ""):
  sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from extra.qk.mmq_bounded_harness import (
  ACTIVATION_LAYOUT_MMQ_DS4, ACTIVATION_LAYOUT_ROW_MAJOR, AMD_DS4_DOT4X4_BACKEND_ID, AMD_DS4_WARP_BACKEND_ID,
  BoundedMMQConfig, CANDIDATE_ROUTE_ID, COMPARATOR_ID, LLAMA_MMQ_GEOMETRY, PUBLIC_LABEL, STAGED_DS4_BACKEND_ID,
  candidate_metadata, run_bounded_harness,
)


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
)

BLOCKED_CANDIDATES: tuple[dict[str, Any], ...] = (
  {
    "candidate_id": "cooperative_shared_lds_tile",
    "backend": "not_implemented",
    "activation_layout": ACTIVATION_LAYOUT_MMQ_DS4,
    "status": "blocked",
    "search_class": "llama_style_tile_structure",
    "promotion_eligible": False,
    "reason": "cooperative multi-wave shared/LDS tile ownership is not implemented",
  },
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
    "searchable_components": [row["component"] for row in DONE_COMPONENTS],
    "searchable_candidates": rows,
    "blocked_candidates": list(BLOCKED_CANDIDATES),
    "promotion_verdict": "BLOCKED_UNTIL_COOPERATIVE_TILE_PASS",
  }


def _parse_args() -> argparse.Namespace:
  ap = argparse.ArgumentParser(description="Bounded machine-search report for completed 14B Q4_K/Q8_1 MMQ pieces")
  ap.add_argument("--run", action="store_true", help="execute searchable bounded candidates")
  ap.add_argument("--warmups", type=int, default=0)
  ap.add_argument("--rounds", type=int, default=1)
  ap.add_argument("--out", type=pathlib.Path, default=None)
  return ap.parse_args()


def main() -> None:
  args = _parse_args()
  report = build_search_report(run=args.run, warmups=args.warmups, rounds=args.rounds)
  text = json.dumps(report, indent=2, sort_keys=True)
  if args.out is not None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n")
  print(text)


if __name__ == "__main__":
  main()
