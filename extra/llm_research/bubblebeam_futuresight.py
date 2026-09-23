#!/usr/bin/env python3
"""BubbleBeam dimension proposals and FutureSight static ordering, under the fork's names.

The policy now lives in BoltBeam: boltbeam/search/bubblebeam.py proposes legal
dimension values, and boltbeam/search/futuresight.py rejects and orders canonical
candidates, including flash decode candidates in BoltBeam's one flash schema
(boltbeam/search/flash_decode_candidate.py). This module re-exports those names
so fork callers keep their imports. extra/llm_research/boltbeam_checkout.py finds
the BoltBeam checkout and fails loudly when there is none. BoltBeam alone owns
candidate schema, identity, finite expansion, measured ranking, and promotion.

The Q4_K lane map that used to sit at the bottom of this file now lives in
extra/llm_research/q4k_lane_map.py, beside the other lane-partition code. It reasons over
tinygrad UOps, so it stays in the fork; this module does not, so it imports no tinygrad at all.
"""
from __future__ import annotations

from extra.llm_research.boltbeam_checkout import require_boltbeam

require_boltbeam("boltbeam.search.bubblebeam", "boltbeam.search.futuresight")
from boltbeam.search.bubblebeam import (  # noqa: E402
  JSONValue, LegalDimensionProposal, ScheduleVocabulary, dimension_mapping, propose_legal_dimensions,
  target_schedule_vocabulary,
)
from boltbeam.search.futuresight import (  # noqa: E402
  CanonicalCandidate, Legality, Priority, StaticAssessment, StaticRejection, apply_coupled_row, build_flash_legality,
  build_flash_static_priority, build_static_legality, build_static_priority, candidate_report, classify_candidates,
  classify_coupled_rows, rank_static_candidates,
)

__all__ = [
  "CanonicalCandidate", "JSONValue", "LegalDimensionProposal", "Legality", "Priority", "ScheduleVocabulary", "StaticAssessment",
  "StaticRejection", "apply_coupled_row", "build_flash_legality", "build_flash_static_priority", "build_static_legality",
  "build_static_priority", "candidate_report", "classify_candidates", "classify_coupled_rows", "dimension_mapping",
  "propose_legal_dimensions", "rank_static_candidates", "target_schedule_vocabulary",
]
