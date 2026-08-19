"""CPU-only contracts for BubbleBeam/FutureSight's static boundary."""
from copy import deepcopy

import pytest

import extra.llm_research.bubblebeam_futuresight as bf
from extra.llm_research.bubblebeam_futuresight import (
  build_flash_legality, build_flash_static_priority, build_static_legality, build_static_priority, candidate_report,
  classify_candidates, dimension_mapping, propose_legal_dimensions, q4k_lane_partition_candidates,
  rank_candidates, rank_static_candidates,
  should_route_q4k_lane_partition, target_schedule_vocabulary,
)
from tinygrad.uop.ops import UOp


def _candidate(candidate_hash: str, threads: int, *, tile_k: int = 32, local_memory: int | None = 16384) -> dict:
  # This is a minimal stand-in for a candidate already validated and hashed by
  # BoltBeam.  FutureSight consumes its schedule and opaque identity only.
  return {"schema_version": "fixture.candidate.v2", "candidate_hash": candidate_hash,
          "schedule": {"plan_kind": "tinygrad_opt_sequence.v1", "transforms": [{"op": "UPCAST", "axis": 0, "arg": 4}],
                       "tile": {"m": 1, "n": 32, "k": tile_k}, "launch": {"threads": threads},
                       "layout": "packed", "quant_storage": "Q4_K", "scalar_dtype": "float16",
                       "accumulator_dtype": "float32"},
          "static_constraints": {"max_local_memory_bytes": local_memory}}


def _facts():
  return ({"shape": {"m": 1, "n": 4096, "k": 4096}, "quant_storage": "Q4_K",
           "scalar_dtype": "float16", "accumulator_dtype": "float32", "layout": "packed"},
          {"compiler_transforms": ["LOCAL", "UPCAST"], "max_threads_per_threadgroup": 64,
           "max_threadgroup_memory_bytes": 32768,
           "supported_plan_kinds": ["tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1"],
           "generic_control_plan_kind": "tinygrad_heuristic.v1"})


def test_dimension_proposal_is_derived_from_supplied_target_facts():
  vocabulary = target_schedule_vocabulary({"backend": "metal", "compiler_transforms": ["LOCAL", "UPCAST"]})
  proposal = vocabulary.propose_dimension("transform", ("LOCAL", "UNSUPPORTED", "UPCAST"))
  assert proposal.path == "transform"
  assert proposal.values == ("LOCAL", "UPCAST")


def test_proposals_are_deterministic_boltbeam_paths_filtered_by_generic_facts():
  workload, target = _facts()
  choices = {
    "workload.operands.b.quantization": ("q4_k", "q6_k"),
    "schedule.plan_kind": ("tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1", "unsupported"),
    "schedule.transforms": ((), ({"op": "UPCAST", "axis": 0, "arg": 4},),
                            ({"op": "UNROLL", "axis": 0, "arg": 8},)),
    "schedule.tile.k": (30, 32, 64), "schedule.launch.threads": (0, 32, 128),
    "schedule.memory.b.vector_width": (0, 4), "static_constraints.max_local_memory_bytes": (None, 16384, 65536),
  }
  first = propose_legal_dimensions(workload, target, choices)
  second = propose_legal_dimensions(workload, target, dict(reversed(tuple(choices.items()))))
  assert first == second
  assert dimension_mapping(first) == {
    "schedule.launch.threads": (32,), "schedule.memory.b.vector_width": (4,),
    "schedule.plan_kind": ("tinygrad_heuristic.v1", "tinygrad_opt_sequence.v1"), "schedule.tile.k": (32, 64),
    "schedule.transforms": ((), ({"op": "UPCAST", "axis": 0, "arg": 4},)),
    "static_constraints.max_local_memory_bytes": (None, 16384),
    "workload.operands.b.quantization": ("q4_k", "q6_k"),
  }


def test_static_report_preserves_rejections_and_uses_supplied_hashes_only():
  workload, target = _facts()
  candidates = [_candidate("boltbeam-hash-a", 32), _candidate("boltbeam-hash-b", 128)]
  report = candidate_report(candidates, (build_static_legality(workload, target),),
    build_static_priority({"schedule.launch.threads": (64, 32), "schedule.tile.k": (64, 32)}))
  assert report["assessments"] == [{"candidate_hash": "boltbeam-hash-a", "static_score": 2,
                                     "static_reason": "preferred:schedule.launch.threads,schedule.tile.k"}]
  assert report["rejections"] == [{"candidate_hash": "boltbeam-hash-b", "reason": "over_threads"}]
  assert "promotion" not in repr(report).lower()
  assert all(set(row) == {"candidate_hash", "static_score", "static_reason"} for row in report["assessments"])


def test_legality_builder_checks_transforms_divisibility_and_local_memory():
  workload, target = _facts(); check = build_static_legality(workload, target)
  unsupported = _candidate("unsupported", 32); unsupported["schedule"]["transforms"][0]["op"] = "UNROLL"
  assert check(unsupported) == "unsupported_or_invalid_transform"
  assert check(_candidate("nondivisible", 32, tile_k=30)) == "non_divisible_tile_k"
  assert check(_candidate("local", 32, local_memory=65536)) == "over_local_memory"
  assert check(_candidate("legal", 32)) is None
  heuristic = _candidate("control", 32); heuristic["schedule"].update(plan_kind="tinygrad_heuristic.v1", transforms=[])
  assert check(heuristic) is None
  heuristic["schedule"]["transforms"] = [{"op": "UPCAST", "axis": 0, "arg": 4}]
  assert check(heuristic) == "heuristic_control_has_explicit_transforms"
  no_opt = _candidate("no-opt", 32); no_opt["schedule"]["transforms"] = []
  assert check(no_opt) is None


def test_assessment_does_not_mutate_canonical_candidate_payloads():
  candidates = [_candidate("boltbeam-hash-b", 64), _candidate("boltbeam-hash-a", 32)]
  before = deepcopy(candidates)
  accepted, rejected = classify_candidates(candidates)
  candidate_report(candidates, (), lambda row: (row["schedule"]["launch"]["threads"], "threads"))
  assert candidates == before
  assert accepted[0] is candidates[0] and accepted[1] is candidates[1]
  assert rejected == []


def test_module_does_not_own_candidate_schema_hash_or_population_expansion():
  assert not hasattr(bf, "StaticCandidate")
  assert not hasattr(bf, "CandidateDimension")
  assert not hasattr(bf, "enumerate_candidates")
  assert not hasattr(bf, "sha256")
  assert "boltbeam.full_kernel_candidate" not in bf.__doc__


def test_historical_q4_wrapper_still_selects_the_lane_partition():
  best = rank_candidates(q4k_lane_partition_candidates(UOp.range(32, 0)))[0]
  assert best.candidate.name == "lane_partition_q4k"
  assert should_route_q4k_lane_partition(12288, 4096) is True

def test_proposals_serialize_to_the_plain_boltbeam_dimension_boundary():
  proposals = (bf.LegalDimensionProposal("schedule.launch.threads", (32, 64)),
               bf.LegalDimensionProposal("schedule.tile.k", (256,)))
  plain = dimension_mapping(proposals)
  assert plain == {"schedule.launch.threads": (32, 64), "schedule.tile.k": (256,)}
  import json
  assert json.loads(json.dumps(plain)) == {"schedule.launch.threads": [32, 64], "schedule.tile.k": [256]}


# flash_decode_candidate.v1 legality and ordering ----------------------------
def _flash_envelope(**tile_kwargs):
  from extra.llm_research import flash_candidate_schema as schema
  defaults = {"Hq": 8, "Hd": 128, "Hkv": 8, "MAXC": 4096, "split_count": 1}
  defaults.update(tile_kwargs)
  # Build without validating so invalid-geometry fixtures reach the legality check.
  descriptor = {"schema_version": schema.SCHEMA_VERSION, "tile": schema.tile_fields(**defaults), "combine": None}
  return schema.candidate_envelope(descriptor)


def _sm120_facts(**overrides):
  facts = {"subgroup_size": 32, "max_threads_per_threadgroup": 1024,
           "max_threadgroup_memory_bytes": 227 * 1024}
  facts.update(overrides)
  return facts


def test_flash_legality_classifies_an_enumerated_sm120_like_space_without_vendor_branches():
  from extra.llm_research import flash_candidate_schema as schema
  facts = _sm120_facts()
  check = build_flash_legality({}, facts)
  candidates, expected_legal = [], set()
  for lane_width in (8, 16, 32, 64):
    for reduce_structure in ("staged", "inline"):
      for token_block in (16, 128, 512):
        for stage_width in (1, 4):
          for score_group_width in (None, 8):
            for warps in (1, 8):
              envelope = _flash_envelope(lane_width=lane_width, reduce_structure=reduce_structure,
                                         token_block=token_block, stage_width=stage_width,
                                         score_group_width=score_group_width, warps=warps)
              candidates.append(envelope)
              tile = envelope["tile"]
              try:
                schema.validate(envelope)
                geometry_ok = True
              except schema.SchemaError:
                geometry_ok = False
              ladder = schema.ladder_spans_lanes(tile)
              subgroup_ok = not ladder or facts["subgroup_size"] % lane_width == 0
              threads_ok = schema.derived_threads(tile) <= facts["max_threads_per_threadgroup"]
              local_ok = schema.local_memory_bytes(envelope) <= facts["max_threadgroup_memory_bytes"]
              if geometry_ok and subgroup_ok and threads_ok and local_ok:
                expected_legal.add(envelope["candidate_hash"])
  accepted, rejected = classify_candidates(candidates, (check,))
  assert {candidate["candidate_hash"] for candidate in accepted} == expected_legal
  assert {row.candidate_hash for row in rejected} == {c["candidate_hash"] for c in candidates} - expected_legal
  assert {row.reason for row in rejected} <= {"lane_width_not_in_subgroup", "over_threads", "over_local_memory",
                                              "invalid_flash_geometry"}


def test_flash_legality_rejects_specific_invalid_geometries():
  check = build_flash_legality({}, _sm120_facts())
  assert check(_flash_envelope()) is None
  assert check(_flash_envelope(lane_width=64)) == "lane_width_not_in_subgroup"
  assert check(_flash_envelope(warps=64)) == "over_threads"
  assert check(_flash_envelope(token_block=512)) == "over_local_memory"
  assert check(_flash_envelope(reduce_structure="recursive")) == "invalid_flash_geometry"
  wrong = _flash_envelope()
  wrong["schema_version"] = "other.v1"
  assert check(wrong) == "unsupported_schema_version"


def test_flash_legality_needs_no_vendor_identity_keys():
  minimal = {"subgroup_size": 32, "max_threads_per_threadgroup": 1024, "max_threadgroup_memory_bytes": 227 * 1024}
  check = build_flash_legality({}, minimal)
  assert check(_flash_envelope()) is None
  assert check(_flash_envelope(lane_width=64)) == "lane_width_not_in_subgroup"
  with pytest.raises(ValueError, match="subgroup_size"):
    build_flash_legality({}, {"max_threads_per_threadgroup": 1024})


def test_flash_static_priority_prefers_wider_lanes_and_penalizes_small_stage_width():
  priority = build_flash_static_priority(_sm120_facts())
  wide, narrow = _flash_envelope(lane_width=32), _flash_envelope(lane_width=8)
  assert priority(wide)[0] > priority(narrow)[0]
  assert "column=32" in priority(wide)[1] and "column=8" in priority(narrow)[1]
  wide_stage = _flash_envelope(lane_width=32, stage_width=4)
  assert priority(wide_stage)[0] > priority(wide)[0]  # stage_width 1 relaunches more staging passes
  inline = _flash_envelope(lane_width=32, reduce_structure="inline")
  assert priority(inline)[0] > priority(wide)[0]  # shorter ladder beats staged


def test_flash_ranking_is_deterministic_and_tie_breaks_by_candidate_hash():
  facts = _sm120_facts()
  check = build_flash_legality({}, facts)
  priority = build_flash_static_priority(facts)
  legal = _flash_envelope(lane_width=32)
  rejected = _flash_envelope(lane_width=64)
  accepted, rejections = classify_candidates([legal, rejected], (check,))
  assert [c["candidate_hash"] for c in accepted] == [legal["candidate_hash"]]
  assert rejections[0].reason == "lane_width_not_in_subgroup"
  first = dict(legal, candidate_hash="a" * 64)
  second = dict(legal, candidate_hash="b" * 64)
  ranked = rank_static_candidates([second, first], priority)
  assert [row.candidate_hash for row in ranked] == ["a" * 64, "b" * 64]
  assert ranked[0].score == ranked[1].score
