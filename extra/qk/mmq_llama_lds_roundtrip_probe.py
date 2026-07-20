"""Compile-only LDS round-trip probe for the K256 five-buffer producer.

The device graph is deliberately narrower than the full MMQ kernel: it runs
the production hardware-local Q4 persistent producer and Q8 phase-0 producer,
waits at their production publish barrier, then copies every *defined* record
word to ABI slot 0.  No WMMA or numerical writeback is present.

The host oracle below is intentionally independent of the UOp producer
callbacks.  Its Q4 decode spells the source-pinned ``load_tiles_q4_K`` /
``unpack_scales_q45_K`` equations directly, while all physical offsets and
field sizes are read from the candidate-plan record descriptors.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from tinygrad import dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import KernelInfo, Ops, ProgramInfo, UOp

from extra.qk.mmq_llama_candidate_plan import llama_mmq_candidate_plan
from extra.qk.mmq_llama_five_buffer_graph import five_buffer_parameters
from extra.qk.mmq_llama_oracle_epoch import build_llama_oracle_epoch_stage_five_buffer
from extra.qk.mmq_llama_runtime_contract import LLAMA_SOURCE_COMMIT


SCHEMA = "tinygrad.mmq_llama_lds_roundtrip_probe.v1"
SHAPE = (128, 128, 256)
PHASE = 0


@dataclass(frozen=True)
class LDSRoundTripSegment:
  name: str
  output_word_start: int
  words_per_row: int
  rows: int
  lds_byte_start: int
  lds_row_stride_bytes: int

  @property
  def word_count(self) -> int: return self.words_per_row * self.rows

  @property
  def output_word_end(self) -> int: return self.output_word_start + self.word_count


def lds_roundtrip_segments() -> tuple[LDSRoundTripSegment, ...]:
  """Return the exact compact debug schema derived from production LDS records."""
  geometry = llama_mmq_candidate_plan().geometry
  q8, q4 = geometry.lds_region("q8"), geometry.lds_region("q4")
  if q8.records is None or q4.records is None: raise ValueError("llama LDS regions must retain record layouts")
  q8_defined = max(x.end_bytes for x in q8.records.components)
  q4_qs, q4_dm, q4_padding = (q4.records.component(x) for x in ("qs", "dm", "padding"))
  # Q8 phase 0 defines its complete 144-byte record. Q4 defines qs+dm; the
  # source kernel intentionally leaves the trailing 16-byte padding alone.
  # Keep qs and dm as separate compact segments: their 64- and 8-word rows
  # lower to power-of-two division on AMD, unlike an artificial 72-word row.
  if (q8_defined != q8.records.stride_bytes or q4_qs.offset_bytes != 0 or
      q4_qs.end_bytes != q4_dm.offset_bytes or q4_dm.end_bytes != q4_padding.offset_bytes):
    raise ValueError("unsupported llama LDS record layout")
  first = LDSRoundTripSegment("q8_phase0_record", 0, q8_defined//4, q8.records.rows,
                              q8.base, q8.records.stride_bytes)
  second = LDSRoundTripSegment("q4_persistent_qs", first.output_word_end,
                               q4_qs.size_bytes//4, q4.records.rows,
                               q4.base+q4_qs.offset_bytes, q4.records.stride_bytes)
  third = LDSRoundTripSegment("q4_persistent_dm", second.output_word_end,
                              q4_dm.size_bytes//4, q4.records.rows,
                              q4.base+q4_dm.offset_bytes, q4.records.stride_bytes)
  return first, second, third


try:
  DEBUG_WORDS = sum(x.word_count for x in lds_roundtrip_segments())
  DEBUG_BYTES = DEBUG_WORDS * 4
except (KeyError, ValueError):
  # phase2-fp16-dequant-q4k: the candidate-plan geometry no longer has "q8"/"q4"
  # LDS regions (two fp16 "A"/"B" K32-group regions replaced them; see
  # mmq_llama_candidate_plan.py _geometry()).  This whole probe targets the
  # retired int8 5-buffer record layout and has not been rewritten for the
  # per-K32-group fp16 design -- degrade to None at import time instead of
  # crashing test collection; callers of lds_roundtrip_segments() still fail
  # loudly with a clear KeyError.
  DEBUG_WORDS = DEBUG_BYTES = None


@dataclass(frozen=True)
class LlamaLDSRoundTripProbe:
  sink: UOp
  stage: object
  segments: tuple[LDSRoundTripSegment, ...]
  source_commit: str
  program: UOp|None = None
  emitted: bool = False

  def __post_init__(self) -> None:
    if self.source_commit != LLAMA_SOURCE_COMMIT: raise ValueError("source identity drift")
    if self.segments != lds_roundtrip_segments(): raise ValueError("debug-output schema drift")
    if self.emitted != (self.program is not None): raise ValueError("emitted must match PROGRAM presence")
    if ProgramInfo.from_sink(self.sink).globals != tuple(range(5)):
      raise ValueError("probe must retain the exact five-buffer pointer ABI")


def _five_buffer_sources() -> tuple[UOp, ...]:
  return tuple(UOp.param(x.slot, x.dtype.ptr(x.size)) for x in five_buffer_parameters(*SHAPE))


def build_llama_lds_roundtrip_probe() -> LlamaLDSRoundTripProbe:
  """Build the real phase-0 producer followed by a deterministic LDS export."""
  output, q4, values, scales, sums = _five_buffer_sources()
  stage = build_llama_oracle_epoch_stage_five_buffer(q4, values, scales, sums)
  segments = lds_roundtrip_segments()
  if stage.phases[PHASE].publish.src[0].op is not Ops.GROUP:
    raise ValueError("phase-0 publish no longer joins production producers")
  publish = stage.phases[PHASE].publish
  local = UOp.special(stage.geometry.threads, "lidx0")
  prior:UOp|None = None
  stores = []
  for iteration in range(DEBUG_WORDS // stage.geometry.threads):
    debug_word = local + iteration*stage.geometry.threads
    # Segment boundaries are workgroup aligned. Select the schema segment at
    # graph-construction time, avoiding a runtime address branch and ensuring
    # no inactive branch ever computes an out-of-segment row.
    chunk_start, chunk_end = iteration*stage.geometry.threads, (iteration+1)*stage.geometry.threads
    segment = next((x for x in segments if x.output_word_start <= chunk_start and chunk_end <= x.output_word_end), None)
    if segment is None: raise ValueError("debug segment must contain each complete workgroup copy chunk")
    segment_word = debug_word-segment.output_word_start
    lds_byte = segment.lds_byte_start + segment_word*4 if \
      segment.words_per_row*4 == segment.lds_row_stride_bytes else \
      segment.lds_byte_start + (segment_word//segment.words_per_row)*segment.lds_row_stride_bytes + \
      (segment_word%segment.words_per_row)*4
    dependencies = (publish,) if prior is None else (publish, prior)
    word = stage.allocation.after(*dependencies).index(lds_byte, dtype=dtypes.uint32).replace(
      tag=("llama_lds_roundtrip_load", segment.name, iteration)).load()
    pointer = output.after(*dependencies).index(debug_word, ptr=True)
    prior = pointer.store(word.bitcast(dtypes.float32)).replace(
      tag=("llama_lds_roundtrip_store", iteration))
    stores.append(prior)
  if len(stores)*stage.geometry.threads != DEBUG_WORDS:
    raise ValueError("debug schema must divide exactly across one workgroup")
  closed = prior
  assert closed is not None
  sink = UOp(Ops.SINK, dtypes.void, (closed,),
             KernelInfo(name="mmq_llama_lds_roundtrip_k256_phase0", opts_to_apply=()))
  return LlamaLDSRoundTripProbe(sink, stage, segments, LLAMA_SOURCE_COMMIT)


def compile_llama_lds_roundtrip_probe(probe:LlamaLDSRoundTripProbe,
                                      target:str="AMD:ISA:gfx1100") -> LlamaLDSRoundTripProbe:
  if not isinstance(probe, LlamaLDSRoundTripProbe): raise TypeError("expected LlamaLDSRoundTripProbe")
  program = to_program(probe.sink, AMDISARenderer(Target.parse(target)))
  if program.arg.globals != tuple(range(5)): raise ValueError("lowering changed the five-buffer ABI")
  return replace(probe, program=program, emitted=True)


def _reshape_exact(name:str, value:np.ndarray, shape:tuple[int, ...], dtype:np.dtype) -> np.ndarray:
  array = np.asarray(value)
  if array.dtype != dtype or array.size != int(np.prod(shape)):
    raise ValueError(f"{name} must be exact {dtype.name}{shape} storage")
  return array.reshape(shape)


def _host_q4_defined_records(q4_words:np.ndarray) -> np.ndarray:
  """Independent host spelling of llama's Q4_K decoded LDS qs+dm fields."""
  blocks = _reshape_exact("q4_words", q4_words, (128, 36), np.dtype(np.uint32))
  raw = blocks.view(np.uint8).reshape(128, 144)
  out = np.empty((128, 288), dtype=np.uint8)
  decoded = out[:, :256].view(np.uint32).reshape(128, 64)
  for txi in range(32):
    word = blocks[:, 4+txi]
    destination = 16*(txi//8) + txi%8
    decoded[:, destination] = word & np.uint32(0x0f0f0f0f)
    decoded[:, destination+8] = (word >> np.uint32(4)) & np.uint32(0x0f0f0f0f)

  scales, minimums = np.empty((128, 8), np.uint8), np.empty((128, 8), np.uint8)
  for group in range(8):
    if group < 4:
      scales[:, group] = raw[:, 4+group] & np.uint8(0x3f)
      minimums[:, group] = raw[:, 8+group] & np.uint8(0x3f)
    else:
      lo = group-4
      scales[:, group] = (raw[:, 12+lo] & np.uint8(0x0f)) | ((raw[:, 4+lo] >> 6) << 4)
      minimums[:, group] = (raw[:, 12+lo] >> 4) | ((raw[:, 8+lo] >> 6) << 4)
  halves = blocks[:, :1].view(np.float16).reshape(128, 2)
  dm = out[:, 256:288].view(np.float16).reshape(128, 16)
  dm[:, 0::2] = (halves[:, 0:1] * scales.astype(np.float16)).astype(np.float16)
  dm[:, 1::2] = ((-halves[:, 1:2]) * minimums.astype(np.float16)).astype(np.float16)
  return out


def expected_llama_lds_roundtrip(q4_words:np.ndarray, q8_values:np.ndarray,
                                 q8_scales:np.ndarray, q8_original_sums:np.ndarray) -> np.ndarray:
  """Construct the exact compact debug words without evaluating producer UOps."""
  if not np.little_endian: raise RuntimeError("probe schema requires little-endian host words")
  values = _reshape_exact("q8_values", q8_values, (2, 128, 128), np.dtype(np.int8))
  scales = _reshape_exact("q8_scales", q8_scales, (2, 128, 4), np.dtype(np.float32))
  sums = _reshape_exact("q8_original_sums", q8_original_sums, (2, 128, 4), np.dtype(np.float32))
  q8 = np.empty((128, 144), dtype=np.uint8)
  ds = q8[:, :16].view(np.float16).reshape(128, 8)
  ds[:, 0::2], ds[:, 1::2] = scales[PHASE].astype(np.float16), sums[PHASE].astype(np.float16)
  q8[:, 16:] = values[PHASE].view(np.uint8)
  q4 = _host_q4_defined_records(q4_words)
  result = np.concatenate((q8.reshape(-1), q4[:, :256].reshape(-1), q4[:, 256:].reshape(-1))).copy()
  if result.size != DEBUG_BYTES: raise ValueError("host debug schema byte count drift")
  return result.view(np.uint32)


def compare_llama_lds_roundtrip(output:np.ndarray, expected:np.ndarray) -> dict[str, Any]:
  """Fail closed over dtype/extent and report exact word-level mismatches."""
  actual = np.asarray(output)
  if actual.dtype == np.float32 and actual.size == SHAPE[0]*SHAPE[1]:
    actual_words = actual.view(np.uint32)[:DEBUG_WORDS]
  elif actual.dtype == np.uint32 and actual.size == DEBUG_WORDS:
    actual_words = actual.reshape(-1)
  else:
    raise ValueError(f"output must be float32[{SHAPE[0]*SHAPE[1]}] ABI storage or uint32[{DEBUG_WORDS}] debug words")
  expected_words = np.asarray(expected)
  if expected_words.dtype != np.uint32 or expected_words.size != DEBUG_WORDS:
    raise ValueError(f"expected must be exact uint32[{DEBUG_WORDS}] debug words")
  indices = np.flatnonzero(actual_words != expected_words.reshape(-1))
  first = None
  if indices.size:
    index = int(indices[0])
    segment = next(x for x in lds_roundtrip_segments() if x.output_word_start <= index < x.output_word_end)
    local_word = index-segment.output_word_start
    first = {"output_word": index, "segment": segment.name,
             "row": local_word//segment.words_per_row, "word_in_row": local_word%segment.words_per_row,
             "actual": int(actual_words[index]), "expected": int(expected_words.reshape(-1)[index])}
  return {"schema": SCHEMA, "passed": not indices.size, "compared_words": DEBUG_WORDS,
          "mismatch_count": int(indices.size), "first_mismatch": first}


__all__ = ["DEBUG_BYTES", "DEBUG_WORDS", "LDSRoundTripSegment", "LlamaLDSRoundTripProbe", "PHASE", "SCHEMA",
  "SHAPE", "build_llama_lds_roundtrip_probe", "compare_llama_lds_roundtrip",
  "compile_llama_lds_roundtrip_probe", "expected_llama_lds_roundtrip", "lds_roundtrip_segments"]
