#!/usr/bin/env python3
"""BubbleBeam dimension proposals and FutureSight static ordering.

This module is deliberately CPU-only policy-free research infrastructure.  It
proposes target-neutral legal dimension values and can reject/order canonical
candidate payloads supplied by BoltBeam.  BoltBeam alone owns candidate schema,
identity, finite expansion, measured ranking, and promotion.  This module does
not execute candidates or import route policy in its shared path.

The Q4_K lane-map functions at the bottom are a compatibility wrapper for the
historical experiment.  New workloads should use the data-driven primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import UOp
from tinygrad.codegen.late.warp_reduce import WARP
from extra.llm_research.lane_partition_reduce import LanePartition, q4k_packed_word_index
from tinygrad.codegen.late.coalesced_load import axis_stride, vector_width


JSONValue = str | int | float | bool | None | tuple["JSONValue", ...] | Mapping[str, "JSONValue"]
CanonicalCandidate = Mapping[str, Any]


def _value_key(value: Any) -> str:
  """Stable equality key for dimension values; never a candidate identity."""
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class LegalDimensionProposal:
  """Legal values proposed for one independent axis; not a population."""
  path: str
  values: tuple[JSONValue, ...]

  def __post_init__(self) -> None:
    if not self.path or any(not part for part in self.path.split(".")): raise ValueError("dimension proposal path is required")
    if not self.values: raise ValueError(f"dimension proposal {self.path!r} has no legal values")
    if len({_value_key(value) for value in self.values}) != len(self.values):
      raise ValueError(f"dimension proposal {self.path!r} contains duplicate values")


def dimension_mapping(proposals: Iterable[LegalDimensionProposal]) -> dict[str, tuple[JSONValue, ...]]:
  """Adapt proposal rows to ``BoltBeam.instantiate_candidates`` dimensions."""
  rows = tuple(proposals)
  if len({row.path for row in rows}) != len(rows): raise ValueError("dimension proposal paths must be unique")
  return {row.path: row.values for row in sorted(rows, key=lambda row: row.path)}


@dataclass(frozen=True)
class StaticRejection:
  candidate_hash: str
  reason: str


@dataclass(frozen=True)
class StaticAssessment:
  candidate_hash: str
  score: int
  reason: str


Legality = Callable[[CanonicalCandidate], str | None]
Priority = Callable[[CanonicalCandidate], tuple[int, str]]


def _at_path(value: Mapping[str, Any], path: str) -> Any:
  current: Any = value
  for part in path.split("."):
    if not isinstance(current, Mapping) or part not in current: return None
    current = current[part]
  return current


def _positive_int(value: Any) -> bool:
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _legal_transform_sequence(value: Any, supported: frozenset[str]) -> bool:
  if not isinstance(value, (list, tuple)): return False
  for transform in value:
    if not isinstance(transform, Mapping) or set(transform) != {"op", "axis", "arg"}: return False
    if transform["op"] not in supported: return False
    if transform["axis"] is not None and (not isinstance(transform["axis"], int) or isinstance(transform["axis"], bool)): return False
  return True


@dataclass(frozen=True)
class _StaticFacts:
  shape: Mapping[str, Any]
  vocabulary: "ScheduleVocabulary"
  max_threads: int | None
  max_local_memory: int | None


def _static_facts(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> _StaticFacts:
  shape = workload_facts.get("shape", {})
  if not isinstance(shape, Mapping): raise ValueError("workload shape must be a mapping")
  max_threads = target_facts.get("max_threads_per_threadgroup")
  max_local = target_facts.get("max_threadgroup_memory_bytes")
  if max_threads is not None and not _positive_int(max_threads): raise ValueError("max_threads_per_threadgroup must be positive")
  if max_local is not None and not _positive_int(max_local): raise ValueError("max_threadgroup_memory_bytes must be positive")
  return _StaticFacts(dict(shape), target_schedule_vocabulary(target_facts), max_threads, max_local)


def propose_legal_dimensions(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any],
                             axis_choices: Mapping[str, Sequence[JSONValue]]) -> tuple[LegalDimensionProposal, ...]:
  """Filter caller-declared axes using only supplied compiler/resource facts.

  The result is a sorted set of path/value rows, not a candidate population.
  Quantization, scalar/accumulator dtype, layout, and schedule remain separate
  paths; this function never infers one axis from another.
  """
  facts = _static_facts(workload_facts, target_facts); shape, vocabulary = facts.shape, facts.vocabulary
  supported = vocabulary.transforms

  proposals = []
  for path in sorted(axis_choices):
    raw_values = axis_choices[path]
    if isinstance(raw_values, (str, bytes)) or not isinstance(raw_values, Sequence):
      raise ValueError(f"axis choices for {path!r} must be a sequence")
    values: list[JSONValue] = []
    for value in raw_values:
      legal = True
      if path == "schedule.plan_kind": legal = isinstance(value, str) and value in vocabulary.plan_kinds
      elif path == "schedule.transforms": legal = _legal_transform_sequence(value, supported)
      elif path == "schedule.launch.threads": legal = _positive_int(value) and (facts.max_threads is None or value <= facts.max_threads)
      elif path.startswith("schedule.tile.") and path.rsplit(".", 1)[-1] in {"m", "n", "k"}:
        axis = path.rsplit(".", 1)[-1]; extent = shape.get(axis)
        legal = _positive_int(value) and _positive_int(extent) and extent % value == 0
      elif path.endswith((".vector_width", ".alignment", ".stage_count")): legal = _positive_int(value)
      elif path == "static_constraints.max_local_memory_bytes":
        legal = value is None or (_positive_int(value) and (facts.max_local_memory is None or value <= facts.max_local_memory))
      if legal: values.append(value)
    proposals.append(LegalDimensionProposal(path, tuple(values)))
  return tuple(proposals)


def build_static_legality(workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> Legality:
  """Build a pure generic legality check from the same supplied live facts."""
  facts = _static_facts(workload_facts, target_facts)
  shape, vocabulary, supported = facts.shape, facts.vocabulary, facts.vocabulary.transforms

  def check(candidate: CanonicalCandidate) -> str | None:
    schedule = candidate.get("schedule", {})
    plan_kind = schedule.get("plan_kind") if isinstance(schedule, Mapping) else None
    if vocabulary.plan_kinds and plan_kind not in vocabulary.plan_kinds: return "unsupported_plan_kind"
    transforms = schedule.get("transforms") if isinstance(schedule, Mapping) else None
    if not _legal_transform_sequence(transforms, supported): return "unsupported_or_invalid_transform"
    if vocabulary.generic_control_plan_kind == plan_kind and transforms: return "heuristic_control_has_explicit_transforms"
    threads = _at_path(candidate, "schedule.launch.threads")
    if not _positive_int(threads): return "non_positive_threads"
    if facts.max_threads is not None and threads > facts.max_threads: return "over_threads"
    for axis in ("m", "n", "k"):
      tile, extent = _at_path(candidate, f"schedule.tile.{axis}"), shape.get(axis)
      if not _positive_int(tile): return f"non_positive_tile_{axis}"
      if _positive_int(extent) and extent % tile: return f"non_divisible_tile_{axis}"
    local_limit = _at_path(candidate, "static_constraints.max_local_memory_bytes")
    if local_limit is not None and not _positive_int(local_limit): return "non_positive_local_memory"
    if facts.max_local_memory is not None and _positive_int(local_limit) and local_limit > facts.max_local_memory: return "over_local_memory"
    return None
  return check


def build_static_priority(preferences: Mapping[str, Sequence[JSONValue]]) -> Priority:
  """Build deterministic preference scoring without target or backend policy."""
  for path, values in preferences.items():
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
      raise ValueError(f"priority preferences for {path!r} must be a non-empty sequence")
  ordered = tuple((path, tuple(values)) for path, values in sorted(preferences.items()))

  def priority(candidate: CanonicalCandidate) -> tuple[int, str]:
    score, matched = 0, []
    for path, values in ordered:
      actual = _at_path(candidate, path)
      try: index = next(index for index, value in enumerate(values) if actual == value)
      except StopIteration: continue
      score += len(values) - index
      matched.append(path)
    return score, "preferred:" + ",".join(matched) if matched else "no_static_preference"
  return priority


def _candidate_hash(candidate: CanonicalCandidate) -> str:
  """Read BoltBeam's identity without deriving or normalizing it locally."""
  if not isinstance(candidate.get("schema_version"), str) or not candidate["schema_version"]:
    raise ValueError("canonical candidate requires schema_version")
  if not isinstance(candidate.get("candidate_hash"), str) or not candidate["candidate_hash"]:
    raise ValueError("canonical candidate requires candidate_hash")
  if not isinstance(candidate.get("schedule"), Mapping):
    raise ValueError("canonical candidate requires a schedule mapping")
  return candidate["candidate_hash"]


def classify_candidates(candidates: Iterable[CanonicalCandidate], legalities: Iterable[Legality] = ()) -> tuple[list[CanonicalCandidate], list[StaticRejection]]:
  """Classify an authoritative population without changing or expanding it."""
  accepted, rejected = [], []
  predicates = tuple(legalities)
  for candidate in candidates:
    candidate_hash = _candidate_hash(candidate)
    reason = next((result for check in predicates if (result := check(candidate)) is not None), None)
    (rejected if reason is not None else accepted).append(StaticRejection(candidate_hash, reason) if reason is not None else candidate)
  return accepted, rejected

def apply_coupled_row(baseline: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
  """Apply dotted schedule fields to a JSON baseline for static row legality."""
  value = json.loads(json.dumps(baseline));
  for path, replacement in sorted(row.items()):
    current = value
    parts = path.split(".")
    if not isinstance(path, str) or not parts or any(not part for part in parts): raise ValueError("invalid coupled row path")
    for part in parts[:-1]:
      if not isinstance(current, dict) or part not in current: raise ValueError("coupled row path does not exist")
      current = current[part]
    if not isinstance(current, dict) or parts[-1] not in current: raise ValueError("coupled row path does not exist")
    current[parts[-1]] = replacement
  return value

def classify_coupled_rows(baseline: Mapping[str, Any], proposed_rows: Iterable[Mapping[str, Any]], workload_facts: Mapping[str, Any], target_facts: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
  check = build_static_legality(workload_facts, target_facts); legal, rejected = [], []
  for row in proposed_rows:
    candidate = apply_coupled_row(baseline, row); reason = check(candidate)
    (legal if reason is None else rejected).append(dict(row) if reason is None else {"row":dict(row),"reason":reason})
  return legal, rejected


def rank_static_candidates(candidates: Iterable[CanonicalCandidate], priority: Priority) -> list[StaticAssessment]:
  """Deterministically order candidates for measurement; this is never a verdict."""
  scored = [StaticAssessment(_candidate_hash(candidate), *priority(candidate)) for candidate in candidates]
  return sorted(scored, key=lambda row: (-row.score, row.candidate_hash))


def candidate_report(candidates: Iterable[CanonicalCandidate], legalities: Iterable[Legality], priority: Priority) -> dict[str, Any]:
  """Report only static assessments keyed by BoltBeam-supplied identity."""
  accepted, rejected = classify_candidates(candidates, legalities)
  ranked = rank_static_candidates(accepted, priority)
  return {"assessment_version": "bubblebeam.futuresight.static.v1",
          "assessments": [{"candidate_hash": row.candidate_hash, "static_score": row.score,
                           "static_reason": row.reason} for row in ranked],
          "rejections": [{"candidate_hash": row.candidate_hash, "reason": row.reason} for row in rejected]}


@dataclass(frozen=True)
class ScheduleVocabulary:
  """Adapter from target-admitted tinygrad transforms to data dimensions."""
  transforms: frozenset[str]
  plan_kinds: frozenset[str]
  generic_control_plan_kind: str | None

  def propose_dimension(self, path: str, values: Iterable[JSONValue]) -> LegalDimensionProposal:
    legal = tuple(value for value in values if isinstance(value, str) and value in self.transforms)
    return LegalDimensionProposal(path, legal)


def target_schedule_vocabulary(target_facts: Mapping[str, JSONValue]) -> ScheduleVocabulary:
  """Consume target facts supplied by an adapter; no backend table is embedded here."""
  transforms = target_facts.get("compiler_transforms", ())
  if not isinstance(transforms, (list, tuple, frozenset)): raise ValueError("target compiler_transforms must be a sequence")
  if not all(isinstance(transform, str) for transform in transforms): raise ValueError("target compiler_transforms must contain strings")
  plan_kinds = target_facts.get("supported_plan_kinds", ())
  if not isinstance(plan_kinds, (list, tuple, frozenset)) or not all(isinstance(kind, str) for kind in plan_kinds):
    raise ValueError("target supported_plan_kinds must be a sequence of strings")
  control = target_facts.get("generic_control_plan_kind")
  if control is not None and (not isinstance(control, str) or control not in plan_kinds):
    raise ValueError("generic_control_plan_kind must name a supported plan kind")
  return ScheduleVocabulary(frozenset(transforms), frozenset(plan_kinds), control)


# Historical Q4_K lane-map compatibility wrapper ---------------------------------
@dataclass(frozen=True)
class CoalesceCandidate:
  name: str
  index: UOp
  lane: UOp
  requires_lane_partition: bool = False

@dataclass(frozen=True)
class CoalesceScore:
  candidate: CoalesceCandidate
  stride: int|None
  vector_width: int
  score: int
  reason: str


def score_candidate(c:CoalesceCandidate) -> CoalesceScore:
  stride = axis_stride(c.index, c.lane)
  vw = vector_width(c.index, c.lane)
  score = (1000 if stride == 1 else 0) + vw
  reason = "unit_stride_lane" if stride == 1 else f"non_coalesced_stride_{stride}"
  return CoalesceScore(c, stride, vw, score, reason)


def rank_candidates(cands:list[CoalesceCandidate]) -> list[CoalesceScore]:
  return sorted((score_candidate(c) for c in cands), key=lambda s: (s.score, s.vector_width, s.candidate.name), reverse=True)


def q4k_lane_partition_candidates(lane:UOp, base:UOp|None=None) -> list[CoalesceCandidate]:
  base = UOp.const(dtypes.weakint, 0) if base is None else base
  part = LanePartition(lane)
  # The losing candidate models row-per-lane/default packed word access: adjacent lanes jump by a full block.
  uncoalesced = base + 4 + lane * 36
  return [
    CoalesceCandidate("lane_partition_q4k", q4k_packed_word_index(base, 0, part), lane, True),
    CoalesceCandidate("row_serial_q4k", uncoalesced, lane, False),
  ]


def choose_q4k_candidate(lane:UOp|None=None) -> CoalesceScore:
  lane = UOp.range(WARP, 0) if lane is None else lane
  return rank_candidates(q4k_lane_partition_candidates(lane))[0]

def score_layout_transform(name:str, lane:UOp|None=None) -> CoalesceScore:
  if name != "q4k_lane_partition": raise ValueError(f"unknown layout transform {name!r}")
  return choose_q4k_candidate(lane)

def _manifest_q4k_g3_shapes() -> frozenset[tuple[int, int]]:
  # Compatibility-only boundary: shared generation above intentionally never imports route policy.
  from extra.llm_research.route_manifest import ROUTES
  return frozenset((int(g["N"]), int(g["K"])) for g in ROUTES["decode_q4k_g3_generated"].get("shape_guards", [])
                   if isinstance(g.get("N"), int) and isinstance(g.get("K"), int))

def q4k_g3_manifest_shape(out_features:int, in_features:int) -> bool:
  return (out_features, in_features) in _manifest_q4k_g3_shapes()

def should_route_q4k_lane_partition(out_features:int, in_features:int) -> bool:
  """Search-owned q4k route selector for manifest-tracked Q4_K GEMV roles.

  The original P3.3 selector covered only FFN gate/up. It now covers the promoted G3 LaneMap Q4_K roles:
  gate/up, FFN down, and projection shapes declared by the route manifest.
  """
  if not q4k_g3_manifest_shape(out_features, in_features): return False
  return choose_q4k_candidate().candidate.requires_lane_partition
