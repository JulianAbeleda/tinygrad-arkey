"""Exact, source-derived record producers for llama's Q4_K/Q8_1 MMQ LDS ABI.

This is intentionally only generic packed-record vocabulary.  In particular it
does not introduce a Q4 operation or perform floating-point weight dequantization.
"""
from __future__ import annotations
from dataclasses import dataclass

from tinygrad.codegen.opt.kernel_lds import (PackedRecordCooperativeSchedule, PackedRecordCooperativeStore,
  PackedRecordFieldProducer, PackedRecordOperandTemplate, PackedRecordSource, PrecontractThreadAxes)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_packed_operands import Q4_K_DECODED_LDS_ROW, Q8_1_DS4_ROW

SOURCE_ANCHORS = (
  "ggml/src/ggml-cuda/mmq.cuh:2079-2089 unpack_scales_q45_K",
  "ggml/src/ggml-cuda/mmq.cuh:2093-2165 load_tiles_q4_K",
)

@dataclass(frozen=True)
class RecordProducerInstanceWitness:
  """Typed stage identity carried in the producer dependency graph."""
  schema: str
  role: str
  field: str
  phase: int
  slot: int
  iteration: int
  source_row: UOp
  source_k: UOp
  destination_row: UOp
  destination_vector: UOp

  def __post_init__(self):
    if self.schema != "llama-q8-ds4-producer-instance.v1": raise ValueError("unsupported producer witness schema")
    if not all(isinstance(x, str) and x for x in (self.role, self.field)): raise TypeError("invalid producer witness role/field")
    if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in (self.phase, self.slot, self.iteration)):
      raise TypeError("invalid producer witness ordinal")
    if not all(isinstance(x, UOp) for x in (self.source_row, self.source_k, self.destination_row, self.destination_vector)):
      raise TypeError("producer witness coordinates must be UOps")


def record_producer_instance_value(value: UOp, witness: RecordProducerInstanceWitness) -> UOp:
  """Carry one typed witness in the native value-ordering operation."""
  if not isinstance(value, UOp): raise TypeError("expected producer value UOp")
  if not isinstance(witness, RecordProducerInstanceWitness): raise TypeError("expected producer instance witness")
  return UOp(Ops.AFTER, value.dtype, (value,), witness)


def record_producer_instance_witnesses(root: UOp) -> tuple[RecordProducerInstanceWitness, ...]:
  """Read only witnesses protected by a GROUP dependency.

  A bare metadata NOOP is intentionally not proof: ordinary simplification is
  free to erase it. Only a typed payload on the native AFTER value carrier is
  accepted.
  """
  out = []
  for node in root.toposort():
    if node.op is Ops.AFTER and len(node.src) == 1 and isinstance(node.arg, RecordProducerInstanceWitness): out.append(node.arg)
    if node.op is Ops.STORE and isinstance(node.arg, RecordProducerInstanceWitness): out.append(node.arg)
  return tuple(out)


def is_record_producer_instance_dependency(node: UOp) -> bool:
  """True for a witness carrier or one of its compiler-built prefixes."""
  return node.op is Ops.AFTER and len(node.src) == 1 and isinstance(node.arg, RecordProducerInstanceWitness)

Q8_DS4_GLOBAL_RECORD = PackedOperandTransform("llama.q8_1.ds4.global_record.typed.v1", (
  PackedOperandComponent("ds", dtypes.half, 0, 16, "half[8]=4x_half2", 16, 4),
  PackedOperandComponent("qs", dtypes.int8, 16, 128, "int8[128]", 128, 16),
))
Q8_DS4_RECORD_COPY = PackedOperandRecordTransform("llama.q8_1.ds4.record_copy.v1", Q8_DS4_GLOBAL_RECORD, Q8_1_DS4_ROW)

Q4_K_UINT32_BLOCK = PackedOperandTransform("llama.q4_k.global_block.uint32x36.v1", (
  PackedOperandComponent("record", dtypes.uint32, 0, 144, "block_q4_K_uint32x36", 144, 16),
))
Q4_K_RECORD_DECODE = PackedOperandRecordTransform("llama.q4_k.load_tiles_q4_K.v1", Q4_K_UINT32_BLOCK, Q4_K_DECODED_LDS_ROW)


def _stack(dtype, values: tuple[UOp, ...]) -> UOp:
  return UOp(Ops.STACK, dtype.vec(len(values)), values)


def _q8_copy(dtype, field_offset_bytes: int):
  def produce(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    # ``k`` is the logical K coordinate within the 256-element outer epoch.
    # The source ABI is not two compact SoA arrays: every K128 block and row is
    # one physical 144-byte [half2 ds[4], int8 qs[128]] record.
    source, record = sources[0], k//128
    field_element = k % 128
    byte_base = (record*128 + row)*144 + field_offset_bytes + field_element*dtype.itemsize
    # INDEX coordinates are expressed in the requested view dtype, even when
    # the underlying allocation is carried as uint8.
    element_base = byte_base//dtype.itemsize
    return _stack(dtype, tuple(source.index(element_base+i, dtype=dtype).load() for i in range(width)))
  return produce


def q4_k_qs_record_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
  """Inverse destination map of the two x_qs stores in load_tiles_q4_K."""
  source = sources[0]
  vals = []
  for i in range(width):
    dst = k+i
    txi = (dst//16)*8 + dst%8
    high = (dst%16)//8
    word = source.index(row*36 + 4 + txi).load()
    vals.append(((word >> (high*4)) & UOp.const(dtypes.uint32, 0x0f0f0f0f)).cast(dtypes.int32))
  return _stack(dtypes.int32, tuple(vals))


def _packed_byte(source: UOp, row: UOp, byte: UOp|int) -> UOp:
  word, lane = byte//4, byte%4
  return (source.index(row*36 + word).load() >> (lane*8)) & UOp.const(dtypes.uint32, 0xff)


def _scale_or_min(source: UOp, row: UOp, group: UOp, minimum: bool) -> UOp:
  """Scalar spelling of unpack_scales_q45_K's exact packed six-bit ABI."""
  lo = group % 4
  direct = _packed_byte(source, row, 4 + lo + (4 if minimum else 0)) & UOp.const(dtypes.uint32, 0x3f)
  packed = _packed_byte(source, row, 12 + lo)
  low4 = (packed >> (4 if minimum else 0)) & UOp.const(dtypes.uint32, 0x0f)
  upper_src = _packed_byte(source, row, 8 + lo if minimum else 4 + lo)
  extended = low4 | ((upper_src >> 6) << 4)
  return (group < 4).where(direct, extended)


def q4_k_dm_record_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
  """Produce half[16] = eight half2(d*scale, -dmin*minimum) correction pairs."""
  source = sources[0]
  dm_word = source.index(row*36).load()
  d = (dm_word & UOp.const(dtypes.uint32, 0xffff)).cast(dtypes.uint16).bitcast(dtypes.half)
  dmin = (dm_word >> 16).cast(dtypes.uint16).bitcast(dtypes.half)
  vals = []
  for i in range(width):
    half_index, group = k+i, (k+i)//2
    is_min = half_index % 2
    scale, minimum = _scale_or_min(source, row, group, False), _scale_or_min(source, row, group, True)
    if isinstance(is_min, UOp):
      code, base = is_min.cast(dtypes.bool).where(minimum, scale), is_min.cast(dtypes.bool).where(-dmin, d)
    else:
      code, base = (minimum, -dmin) if is_min else (scale, d)
    vals.append(base * code.cast(dtypes.half))
  return _stack(dtypes.half, tuple(vals))


def _linear_q8_ds4_schedule(template: PackedRecordOperandTemplate, threads: PrecontractThreadAxes,
                            source_k: int) -> tuple[PackedRecordCooperativeStore, ...]:
  """Exact 256-thread linear copy used for each Q8 K128 phase."""
  thread = (threads.wave_m+threads.wave_n)*32+threads.lane
  stores = []
  rows = template.row_axis.vmax+1
  for binding in template.fields:
    field = template.transform.produced.component(binding.field)
    width, vectors_per_row = binding.vector_bytes//field.dtype.itemsize, field.size_bytes//binding.vector_bytes
    sources = tuple(template.source(x) for x in binding.sources)
    for iteration in range(rows*vectors_per_row//256):
      linear = thread+iteration*256
      row, vector = linear//vectors_per_row, linear%vectors_per_row
      logical_row, logical_k = template.row_tile_base+row, UOp.const(dtypes.weakint, source_k)+vector*width
      value = binding.producer(sources, logical_row, logical_k, width)
      witness = RecordProducerInstanceWitness("llama-q8-ds4-producer-instance.v1", template.role, binding.field,
        source_k//128, iteration, iteration, logical_row, logical_k, row, vector)
      value = record_producer_instance_value(value, witness)
      stores.append(PackedRecordCooperativeStore(binding.field, iteration, logical_row, logical_k, row, vector, value))
  return tuple(stores)


def _q4_k_oracle_schedule(template: PackedRecordOperandTemplate, threads: PrecontractThreadAxes,
                          source_k: int) -> tuple[PackedRecordCooperativeStore, ...]:
  """Exact wave-row payload fanout and paired-lane metadata ownership from load_tiles_q4_K."""
  source, base_k, stores = template.source("record"), UOp.const(dtypes.weakint, source_k), []
  # wave/threadIdx.y owns rows wave+8*i; lane is txi and one source word fans out to two decoded int32 destinations.
  for iteration in range(16):
    row = threads.wave_m+iteration*8
    logical_row = template.row_tile_base+row
    word = source.index(logical_row*36+4+base_k//8+threads.lane).load()
    for high in range(2):
      value = _stack(dtypes.int32, (((word >> (high*4)) & UOp.const(dtypes.uint32, 0x0f0f0f0f)).cast(dtypes.int32),))
      vector = 16*(threads.lane//8)+threads.lane%8+high*8
      stores.append(PackedRecordCooperativeStore("qs", iteration*2+high, logical_row, base_k+threads.lane,
                                                  row, vector, value))
  # Each pair of lanes owns one row; ksc selects groups 0..3 or 4..7 and emits four exact half2 corrections.
  row, ksc = threads.wave_m*16+threads.lane//2, threads.lane%2
  logical_row = template.row_tile_base+row
  for lane_group in range(4):
    group = 4*ksc+lane_group
    logical_k = base_k+group*2
    value = q4_k_dm_record_callback((source,), logical_row, logical_k, 2)
    stores.append(PackedRecordCooperativeStore("dm", lane_group, logical_row, logical_k, row, group, value))
  return tuple(stores)


LLAMA_Q4_K_COOPERATIVE_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-load-tiles-q4-k-wave-row-v1", _q4_k_oracle_schedule, ("wave_m", "lane"))
LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-q8-ds4-linear-256-v1", _linear_q8_ds4_schedule, ("wave_m", "wave_n", "lane"))


def build_q8_ds4_record_template(role: str, record_source: UOp, row_axis: UOp, k_axis: UOp,
                                  row_tile_base: UOp, *, source_layout: str = "Q8_1_MMQ_DS4",
                                  sum_semantics: str = "sum_original_fp") -> PackedRecordOperandTemplate:
  if source_layout != "Q8_1_MMQ_DS4": raise ValueError("Q8 record producer requires Q8_1_MMQ_DS4 source layout")
  if sum_semantics != "sum_original_fp": raise ValueError("Q8 DS4 record producer requires sum_original_fp semantic")
  if record_source.dtype.base != dtypes.uint8:
    raise TypeError("Q8 DS4 source must be one physical byte-addressed 144-byte record array")
  return PackedRecordOperandTemplate(role, Q8_DS4_RECORD_COPY,
    # The two typed source components are views of the same AoS record carrier.
    (PackedRecordSource("ds", record_source), PackedRecordSource("qs", record_source)),
    (PackedRecordFieldProducer("ds", ("ds",), _q8_copy(dtypes.half, 0), vector_bytes=4),
     PackedRecordFieldProducer("qs", ("qs",), _q8_copy(dtypes.int8, 16), vector_bytes=4)),
    (), "qs", row_axis, k_axis, row_tile_base, dtypes.char, LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE)


def build_q4_k_record_template(role: str, source: UOp, row_axis: UOp, k_axis: UOp,
                               row_tile_base: UOp, *, source_layout: str = "Q4_K_UINT32X36",
                               decode_semantics: str = "llama_load_tiles_q4_K") -> PackedRecordOperandTemplate:
  if source_layout != "Q4_K_UINT32X36": raise ValueError("Q4 record producer requires packed Q4_K uint32x36 source layout")
  if decode_semantics != "llama_load_tiles_q4_K": raise ValueError("Q4 record producer requires llama load_tiles_q4_K semantic")
  if source.dtype.base != dtypes.uint32: raise TypeError("Q4_K source must be physical uint32[36] blocks")
  return PackedRecordOperandTemplate(role, Q4_K_RECORD_DECODE, (PackedRecordSource("record", source),),
    (PackedRecordFieldProducer("qs", ("record",), q4_k_qs_record_callback, vector_bytes=4),
     PackedRecordFieldProducer("dm", ("record",), q4_k_dm_record_callback, vector_bytes=4)),
    ("padding",), "qs", row_axis, k_axis, row_tile_base, dtypes.char, LLAMA_Q4_K_COOPERATIVE_SCHEDULE)


# Explicit llama names plus short generic-template-friendly names.
build_llama_q8_ds4_record_template = build_q8_ds4_record_template
build_llama_q4_k_record_template = build_q4_k_record_template

__all__ = ["LLAMA_Q4_K_COOPERATIVE_SCHEDULE", "LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE",
  "Q4_K_RECORD_DECODE", "Q4_K_UINT32_BLOCK", "Q8_DS4_GLOBAL_RECORD", "Q8_DS4_RECORD_COPY",
  "SOURCE_ANCHORS", "RecordProducerInstanceWitness", "is_record_producer_instance_dependency",
  "record_producer_instance_value", "record_producer_instance_witnesses",
  "build_q4_k_record_template", "build_q8_ds4_record_template",
  "build_llama_q4_k_record_template", "build_llama_q8_ds4_record_template",
  "q4_k_dm_record_callback", "q4_k_qs_record_callback"]
