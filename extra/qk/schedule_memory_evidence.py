"""Fail-closed semantic accounting for :mod:`tinygrad.schedule.memory` manifests.

This adapter deliberately uses only explicit ``semantic_owner`` metadata and
physical arena ranges.  It contains no device-, model-, phase-, or size-based
classification rules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.schedule_memory_manifest import ScheduleMemoryManifest


from tinygrad.llm.memory_semantics import MemorySemanticClass

SEMANTIC_CLASSES = tuple(item.value for item in MemorySemanticClass)


@dataclass(frozen=True, order=True)
class SemanticMemoryKey:
  semantic_class: str
  candidate_id: str | None = None

  def __post_init__(self) -> None:
    if self.semantic_class not in SEMANTIC_CLASSES: raise ValueError(f"unknown semantic class {self.semantic_class!r}")
    if self.semantic_class == "candidate_workspace":
      if not isinstance(self.candidate_id, str) or not self.candidate_id: raise ValueError("candidate_workspace requires candidate_id")
    elif self.candidate_id is not None: raise ValueError(f"candidate_id is invalid for {self.semantic_class}")


@dataclass(frozen=True)
class SemanticPhysicalBytes:
  semantic_class: str
  candidate_id: str | None
  physical_bytes: int

  @property
  def key(self) -> SemanticMemoryKey: return SemanticMemoryKey(self.semantic_class, self.candidate_id)


@dataclass(frozen=True)
class ScheduleMemoryIndexEvidence:
  index: int
  physical_bytes: int
  by_semantic_class: tuple[SemanticPhysicalBytes, ...]

  @property
  def physical_byte_union(self) -> int: return self.physical_bytes


@dataclass(frozen=True)
class ScheduleMemoryEvidence:
  complete: bool
  blockers: tuple[str, ...]
  indices: tuple[ScheduleMemoryIndexEvidence, ...]
  peak_physical_bytes: int
  peak_by_semantic_class: tuple[SemanticPhysicalBytes, ...]

  @property
  def completeness(self) -> bool: return self.complete


def _owner(owner:Any) -> SemanticMemoryKey:
  # memory.py freezes mappings into sorted key/value tuples.  Accept that
  # canonical representation as well as a structural (class, candidate-id)
  # pair; candidate IDs therefore need not encode model or machine facts.
  # The preferred runtime vocabulary is typed; legacy manifests remain JSON-shaped.
  from tinygrad.llm.memory_semantics import MemorySemanticOwner
  if isinstance(owner, MemorySemanticOwner): return SemanticMemoryKey(owner.semantic_class.value, owner.candidate_id)
  if isinstance(owner, str): return SemanticMemoryKey(owner)
  if isinstance(owner, tuple):
    if len(owner) == 2 and isinstance(owner[0], str) and owner[0] in SEMANTIC_CLASSES:
      return SemanticMemoryKey(owner[0], owner[1])
    if all(isinstance(x, tuple) and len(x) == 2 and isinstance(x[0], str) for x in owner):
      fields = dict(owner)
      names = [fields[x] for x in ("semantic_class", "kind", "class") if x in fields]
      if len(names) != 1: raise ValueError("ownership mapping requires exactly one of semantic_class, kind, class")
      return SemanticMemoryKey(names[0], fields.get("candidate_id"))
  raise ValueError("ownership must be an explicit semantic class or structural ownership record")


def _merge(ranges:list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
  out:list[tuple[int, int]] = []
  for start, end in sorted(ranges):
    if out and start <= out[-1][1]: out[-1] = (out[-1][0], max(out[-1][1], end))
    else: out.append((start, end))
  return tuple(out)


def schedule_memory_evidence(manifest:ScheduleMemoryManifest) -> ScheduleMemoryEvidence:
  """Adapt a planner manifest into exact semantic physical-byte evidence.

  Bad input is represented by ``complete=False`` and precise blockers instead
  of partial evidence being presented as authoritative.
  """
  blockers:list[str] = []
  def block(message:str) -> None:
    if message not in blockers: blockers.append(message)

  try: buffers, arenas, source_indices = tuple(manifest.buffers), tuple(manifest.arenas), tuple(manifest.indices)
  except Exception as exc:
    return ScheduleMemoryEvidence(False, (f"malformed manifest: {exc}",), (), 0, ())

  arena_sizes:dict[str, int] = {}
  for n, arena in enumerate(arenas):
    if not isinstance(arena.identity, str) or not arena.identity: block(f"arena[{n}] has invalid identity")
    elif arena.identity in arena_sizes: block(f"duplicate arena identity: {arena.identity}")
    if not isinstance(arena.size, int) or isinstance(arena.size, bool) or arena.size < 0: block(f"arena[{n}] has invalid size")
    elif isinstance(arena.identity, str): arena_sizes[arena.identity] = arena.size

  rows:list[tuple[Any, SemanticMemoryKey | None]] = []
  identities:set[str] = set()
  for n, row in enumerate(buffers):
    ident = getattr(row, "identity", None)
    if not isinstance(ident, str) or not ident: block(f"buffer[{n}] has invalid identity")
    elif ident in identities: block(f"duplicate buffer identity: {ident}")
    else: identities.add(ident)
    aid, br = getattr(row, "arena_identity", None), getattr(row, "byte_range", None)
    if aid not in arena_sizes: block(f"buffer {ident!r} references unknown arena {aid!r}")
    valid_range = isinstance(br, tuple) and len(br) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in br)
    if not valid_range or br[0] < 0 or br[0] >= br[1]: block(f"buffer {ident!r} has invalid byte range {br!r}")
    elif aid in arena_sizes and br[1] > arena_sizes[aid]: block(f"buffer {ident!r} byte range exceeds arena {aid!r}")
    first, last = getattr(row, "first_index", None), getattr(row, "last_index", None)
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (first, last)) or first < 0 or last < first:
      block(f"buffer {ident!r} has invalid live indices {first!r}..{last!r}")
    try: key = _owner(getattr(row, "semantic_owner", None))
    except (TypeError, ValueError) as exc:
      block(f"buffer {ident!r} has unknown or malformed ownership: {exc}"); key = None
    rows.append((row, key))

  source_by_index:dict[int, Any] = {}
  for n, idx in enumerate(source_indices):
    i = getattr(idx, "index", None)
    if not isinstance(i, int) or isinstance(i, bool) or i < 0: block(f"manifest index[{n}] has invalid index {i!r}")
    elif i in source_by_index: block(f"duplicate manifest index: {i}")
    else: source_by_index[i] = idx
  expected = set(range(len(source_indices)))
  if set(source_by_index) != expected: block(f"manifest indices are not contiguous 0..{len(source_indices)-1}")

  evidence:list[ScheduleMemoryIndexEvidence] = []
  peaks:dict[SemanticMemoryKey, int] = {}
  for i in range(len(source_indices)):
    by_arena:dict[str, list[tuple[int, int, SemanticMemoryKey | None, str]]] = {}
    for row, key in rows:
      if isinstance(row.first_index, int) and isinstance(row.last_index, int) and row.first_index <= i <= row.last_index:
        br = row.byte_range
        if isinstance(br, tuple) and len(br) == 2 and all(isinstance(x, int) and not isinstance(x, bool) for x in br):
          by_arena.setdefault(row.arena_identity, []).append((br[0], br[1], key, row.identity))
    class_ranges:dict[SemanticMemoryKey, list[tuple[int, int]]] = {}
    all_ranges:list[tuple[int, int]] = []
    for aid, live in by_arena.items():
      for n, (start, end, key, ident) in enumerate(live):
        for ostart, oend, other, oident in live[:n]:
          if max(start, ostart) < min(end, oend) and key != other:
            block(f"conflicting ownership at index {i} in arena {aid!r}: buffers {oident!r} and {ident!r} overlap")
        all_ranges.append((start, end))
        if key is not None: class_ranges.setdefault(key, []).append((start, end))
    # Arena address spaces are independent: merge within each arena, then sum.
    physical = sum(sum(e-s for s,e in _merge([(s,e) for s,e,_,_ in live])) for live in by_arena.values())
    values:dict[SemanticMemoryKey, int] = {}
    for key in class_ranges:
      values[key] = sum(sum(e-s for s,e in _merge([(s,e) for s,e,k,_ in live if k == key])) for live in by_arena.values())
      peaks[key] = max(peaks.get(key, 0), values[key])
    semantic = tuple(SemanticPhysicalBytes(k.semantic_class, k.candidate_id, values[k]) for k in sorted(values))
    evidence.append(ScheduleMemoryIndexEvidence(i, physical, semantic))
    source = source_by_index.get(i)
    if source is not None and getattr(source, "physical_byte_union", None) != physical:
      block(f"index {i} physical union mismatch: manifest={getattr(source, 'physical_byte_union', None)!r}, computed={physical}")

  computed_peak = max((x.physical_bytes for x in evidence), default=0)
  if getattr(manifest, "peak_physical_bytes", None) != computed_peak:
    block(f"peak physical bytes mismatch: manifest={getattr(manifest, 'peak_physical_bytes', None)!r}, computed={computed_peak}")
  peak_values = tuple(SemanticPhysicalBytes(k.semantic_class, k.candidate_id, peaks[k]) for k in sorted(peaks))
  return ScheduleMemoryEvidence(not blockers, tuple(blockers), tuple(evidence), computed_peak, peak_values)


adapt_schedule_memory_manifest = schedule_memory_evidence

__all__ = ["SEMANTIC_CLASSES", "SemanticMemoryKey", "SemanticPhysicalBytes", "ScheduleMemoryIndexEvidence",
           "ScheduleMemoryEvidence", "schedule_memory_evidence", "adapt_schedule_memory_manifest"]
