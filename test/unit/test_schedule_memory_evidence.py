from dataclasses import replace

from extra.qk.schedule_memory_evidence import schedule_memory_evidence
from tinygrad.llm.memory_semantics import (KV_CACHE, MODEL_PARAMETER, RUNTIME_INPUT, RUNTIME_PERSISTENT,
                                            MemorySemanticClass, MemorySemanticOwner)
from extra.qk.schedule_memory_manifest import ScheduleMemoryArena, ScheduleMemoryBuffer, ScheduleMemoryIndex, ScheduleMemoryManifest


def _manifest(buffers, indices, peak, arenas=(ScheduleMemoryArena("arena", "CPU", 0, 512),)):
  return ScheduleMemoryManifest(tuple(buffers), tuple(arenas), tuple(indices), peak)


def _buffer(identity, owner, first, last, byte_range):
  return ScheduleMemoryBuffer(identity, "CPU", 0, byte_range[1]-byte_range[0], first, last,
                              "arena", 512, byte_range[0], byte_range, owner)


def test_exact_per_index_and_peak_semantic_physical_unions():
  buffers = (_buffer("activation", "prefill_activation", 0, 0, (0, 256)),
             _buffer("scratch", "prefill_scratch", 0, 1, (256, 512)),
             _buffer("workspace", ("candidate_workspace", "structural:tile-8"), 1, 1, (0, 256)))
  out = schedule_memory_evidence(_manifest(buffers, (ScheduleMemoryIndex(0, 512, ()), ScheduleMemoryIndex(1, 512, ())), 512))
  assert out.complete and not out.blockers and out.peak_physical_bytes == 512
  assert [(x.semantic_class, x.candidate_id, x.physical_bytes) for x in out.indices[1].by_semantic_class] == [
    ("candidate_workspace", "structural:tile-8", 256), ("prefill_scratch", None, 256)]
  assert {(x.semantic_class, x.candidate_id): x.physical_bytes for x in out.peak_by_semantic_class} == {
    ("candidate_workspace", "structural:tile-8"): 256, ("prefill_activation", None): 256, ("prefill_scratch", None): 256}


def test_shared_same_owner_ranges_use_physical_union_not_logical_sum():
  buffers = (_buffer("a", "prefill_output", 0, 0, (0, 256)), _buffer("b", "prefill_output", 0, 0, (128, 384)))
  out = schedule_memory_evidence(_manifest(buffers, (ScheduleMemoryIndex(0, 384, ()),), 384))
  assert out.complete
  assert out.indices[0].physical_bytes == out.indices[0].by_semantic_class[0].physical_bytes == 384


def test_unknown_malformed_and_conflicting_ownership_fail_closed_with_blockers():
  unknown = schedule_memory_evidence(_manifest((_buffer("u", "unknown", 0, 0, (0, 256)),),
                                                (ScheduleMemoryIndex(0, 256, ()),), 256))
  assert not unknown.complete and "buffer 'u' has unknown or malformed ownership" in unknown.blockers[0]
  malformed = schedule_memory_evidence(_manifest((_buffer("w", "candidate_workspace", 0, 0, (0, 256)),),
                                                  (ScheduleMemoryIndex(0, 256, ()),), 256))
  assert not malformed.complete and "candidate_workspace requires candidate_id" in malformed.blockers[0]
  conflict = schedule_memory_evidence(_manifest((_buffer("a", "prefill_activation", 0, 0, (0, 256)),
                                                 _buffer("b", "prefill_output", 0, 0, (128, 384))),
                                                (ScheduleMemoryIndex(0, 384, ()),), 384))
  assert not conflict.complete and any("conflicting ownership at index 0" in x for x in conflict.blockers)


def test_manifest_totals_are_independently_checked():
  good = _manifest((_buffer("a", "prefill_activation", 0, 0, (0, 256)),), (ScheduleMemoryIndex(0, 256, ()),), 256)
  out = schedule_memory_evidence(replace(good, peak_physical_bytes=255))
  assert not out.complete and out.peak_physical_bytes == 256
  assert out.blockers == ("peak physical bytes mismatch: manifest=255, computed=256",)

def test_all_typed_semantic_classes_are_accepted():
  classes = tuple(item for item in MemorySemanticClass if item is not MemorySemanticClass.CANDIDATE_WORKSPACE)
  buffers = tuple(_buffer(item.value, MemorySemanticOwner(item), 0, 0, (n * 16, (n + 1) * 16)) for n, item in enumerate(classes))
  total = len(buffers) * 16
  out = schedule_memory_evidence(_manifest(buffers, (ScheduleMemoryIndex(0, total, ()),), total))
  assert out.complete
  assert {x.semantic_class for x in out.indices[0].by_semantic_class} == {item.value for item in classes}
