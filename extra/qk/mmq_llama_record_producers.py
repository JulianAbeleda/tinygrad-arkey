"""Exact, source-derived record producers for llama's Q4_K/Q8_1 MMQ LDS ABI.

This is intentionally only generic packed-record vocabulary.  In particular it
does not introduce a Q4 operation or perform floating-point weight dequantization.
"""
from __future__ import annotations
from dataclasses import dataclass

from extra.qk.kernel_lds import (PackedRecordCooperativeSchedule, PackedRecordCooperativeStore,
  PackedRecordFieldProducer, PackedRecordOperandTemplate, PackedRecordSource, PrecontractThreadAxes)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandRecordTransform, PackedOperandTransform
from tinygrad.dtype import PtrDType, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_packed_operands import (ACTIVATION_FP16_GROUP_ROW, Q4_K_DECODED_GROUP_LDS_ROW,
  Q4_K_DECODED_LDS_ROW, Q8_1_DS4_ROW)

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
Q8_DS4_SPLIT_GLOBAL_RECORD = PackedOperandTransform("llama.q8_1.ds4.split_global_record.typed.v1", (
  PackedOperandComponent("values", dtypes.int8, 0, 128, "int8[128]", 128, 16),
  PackedOperandComponent("scales", dtypes.float32, 128, 16, "float[4]", 16, 4),
  PackedOperandComponent("sums", dtypes.float32, 144, 16, "float[4]", 16, 4),
))
Q8_DS4_SPLIT_RECORD_ADAPTER = PackedOperandRecordTransform(
  "llama.q8_1.ds4.split_to_lds.v1", Q8_DS4_SPLIT_GLOBAL_RECORD, Q8_1_DS4_ROW)

Q4_K_UINT32_BLOCK = PackedOperandTransform("llama.q4_k.global_block.uint32x36.v1", (
  PackedOperandComponent("record", dtypes.uint32, 0, 144, "block_q4_K_uint32x36", 144, 16),
))
Q4_K_RECORD_DECODE = PackedOperandRecordTransform("llama.q4_k.load_tiles_q4_K.v1", Q4_K_UINT32_BLOCK, Q4_K_DECODED_LDS_ROW)

# fp16-dequant-in-register decode (implementation plan PART I.2/II.3): one K32
# group at a time, matching wmma.py decode_group/get_scale_min_k4 exactly.
Q4_K_RECORD_DECODE_GROUP = PackedOperandRecordTransform(
  "llama.q4_k.decode_group_fp16.v1", Q4_K_UINT32_BLOCK, Q4_K_DECODED_GROUP_LDS_ROW)
ACTIVATION_FP16_RECORD_COPY = PackedOperandRecordTransform(
  "llama.activation.fp16_group_copy.v1", ACTIVATION_FP16_GROUP_ROW, ACTIVATION_FP16_GROUP_ROW)


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


def _q8_split_qs(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int,
                 *, record_rows: int | None = None) -> UOp:
  """Copy split DS4 values from physical ``[K/128, M, 128]`` storage."""
  source, record, field_element = sources[0], k//128, k%128
  # Compact K256 callers carry exactly two records and can infer M from the
  # pointer. Full-role fixed-base callers must supply M explicitly: pointer
  # size alone cannot distinguish [2,M,128] from [K/128,M,128].
  if record_rows is None:
    if source.dtype.size % (2*128): raise ValueError("split Q8 values storage is not [2, M, 128]")
    record_rows = source.dtype.size // (2*128)
  element_base = (record*record_rows+row)*128+field_element
  return _stack(dtypes.int8, tuple(source.index(element_base+i).load() for i in range(width)))


def _q8_split_ds(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int,
                 *, record_rows: int | None = None) -> UOp:
  """Rebuild DS4 half2(scale, original sum) pairs from split fp32 metadata."""
  if width != 2: raise ValueError("split Q8 DS4 metadata producer requires one half2")
  scales, sums, record = sources[0], sources[1], k//128
  if record_rows is None:
    if scales.dtype.size % (2*4) or sums.dtype.size % (2*4): raise ValueError("split Q8 metadata storage is not [2, M, 4]")
    record_rows = scales.dtype.size // (2*4)
  group = (k%128)//2
  element = (record*record_rows+row)*4+group
  # These casts are intentionally independent: llama stores each fp32 operand
  # into its own half lane before constructing half2(scale, original sum).
  return _stack(dtypes.half, (scales.index(element).load().cast(dtypes.half),
                              sums.index(element).load().cast(dtypes.half)))


def q4_k_qs_record_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int,
                            *, row_stride_words: int = 36) -> UOp:
  """Inverse destination map of the two x_qs stores in load_tiles_q4_K."""
  source = sources[0]
  vals = []
  for i in range(width):
    dst = k+i
    txi = (dst//16)*8 + dst%8
    high = (dst%16)//8
    word = source.index(row*row_stride_words + 4 + txi).load()
    vals.append(((word >> (high*4)) & UOp.const(dtypes.uint32, 0x0f0f0f0f)).cast(dtypes.int32))
  return _stack(dtypes.int32, tuple(vals))


def _packed_byte(source: UOp, row: UOp, byte: UOp|int, row_stride_words: int = 36) -> UOp:
  word, lane = byte//4, byte%4
  return (source.index(row*row_stride_words + word).load() >> (lane*8)) & UOp.const(dtypes.uint32, 0xff)


def _scale_or_min(source: UOp, row: UOp, group: UOp, minimum: bool,
                  row_stride_words: int = 36) -> UOp:
  """Scalar spelling of unpack_scales_q45_K's exact packed six-bit ABI."""
  lo = group % 4
  direct = _packed_byte(source, row, 4 + lo + (4 if minimum else 0), row_stride_words) & UOp.const(dtypes.uint32, 0x3f)
  packed = _packed_byte(source, row, 12 + lo, row_stride_words)
  low4 = (packed >> (4 if minimum else 0)) & UOp.const(dtypes.uint32, 0x0f)
  upper_src = _packed_byte(source, row, 8 + lo if minimum else 4 + lo, row_stride_words)
  extended = low4 | ((upper_src >> 6) << 4)
  return (group < 4).where(direct, extended)


def q4_k_dm_record_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int,
                            *, row_stride_words: int = 36) -> UOp:
  """Produce half[16] = eight half2(d*scale, -dmin*minimum) correction pairs."""
  source = sources[0]
  dm_word = source.index(row*row_stride_words).load()
  d = (dm_word & UOp.const(dtypes.uint32, 0xffff)).cast(dtypes.uint16).bitcast(dtypes.half)
  dmin = (dm_word >> 16).cast(dtypes.uint16).bitcast(dtypes.half)
  vals = []
  for i in range(width):
    half_index, group = k+i, (k+i)//2
    is_min = half_index % 2
    scale, minimum = (_scale_or_min(source, row, group, False, row_stride_words),
                      _scale_or_min(source, row, group, True, row_stride_words))
    if isinstance(is_min, UOp):
      code, base = is_min.cast(dtypes.bool).where(minimum, scale), is_min.cast(dtypes.bool).where(-dmin, d)
    else:
      code, base = (minimum, -dmin) if is_min else (scale, d)
    vals.append(base * code.cast(dtypes.half))
  return _stack(dtypes.half, tuple(vals))


def q4_k_fp16_decode_group_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int, *,
                                    group_index: int, row_stride_words: int = 36) -> UOp:
  """Dequantize ``width`` consecutive local-K codes at ``k`` (0..31) within one K32 group.

  Mirrors ``wmma.py`` ``decode_group``/``get_scale_min_k4`` exactly (implementation
  plan PART I.2): per nibble ``value = d*sc*code - dmin*mn`` computed with a f32
  intermediate, then a single final cast to half.  ``group_index`` (0..7) selects
  the ggml Q4_K sub-group -- see ``tinygrad/llm/gguf.py`` ggml_type 12 decode
  (``sc``/``mn`` via the packed 6-bit scale bytes; nibble chunk ``group_index//2``,
  nibble half ``group_index%2``) -- both are re-derived here bit-for-bit rather
  than via the packed-int32x4 int8-MMQ callback above, since a scalar fp16
  decode needs one code at a time, not four packed char lanes.
  """
  if not 0 <= group_index < 8: raise ValueError("Q4_K group_index must be in [0, 8)")
  source = sources[0]
  dm_word = source.index(row*row_stride_words).load()
  d = (dm_word & UOp.const(dtypes.uint32, 0xffff)).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
  dmin = (dm_word >> 16).cast(dtypes.uint16).bitcast(dtypes.half).cast(dtypes.float32)
  group_u = UOp.const(dtypes.weakint, group_index)
  sc = _scale_or_min(source, row, group_u, False, row_stride_words).cast(dtypes.float32)
  mn = _scale_or_min(source, row, group_u, True, row_stride_words).cast(dtypes.float32)
  tdsc, tdmn = d*sc, dmin*mn
  nibble_shift = 4 if group_index % 2 else 0
  chunk_byte0 = 16 + (group_index//2)*32
  vals = []
  for i in range(width):
    local_k = k+i
    byte = _packed_byte(source, row, chunk_byte0+local_k, row_stride_words)
    code = (byte >> nibble_shift) & UOp.const(dtypes.uint32, 0xf)
    value = tdsc*code.cast(dtypes.float32) - tdmn
    vals.append(value.cast(dtypes.half))
  return _stack(dtypes.half, tuple(vals))


def activation_fp16_group_callback(sources: tuple[UOp, ...], row: UOp, k: UOp, width: int, *,
                                   k_total: int, k_base: int) -> UOp:
  """Plain fp16 activation copy (implementation plan I.6/II.6): no quantization, no ids gather."""
  source = sources[0]
  vals = [source.index(row*k_total+k_base+i+k).load() for i in range(width)]
  return _stack(dtypes.half, tuple(vals))


@dataclass(frozen=True)
class Q4KFp16DecodeGroupProducer:
  group_index: int
  row_stride_words: int = 36

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return q4_k_fp16_decode_group_callback(sources, row, k, width,
      group_index=self.group_index, row_stride_words=self.row_stride_words)


@dataclass(frozen=True)
class ActivationFp16GroupProducer:
  k_total: int
  k_base: int

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return activation_fp16_group_callback(sources, row, k, width, k_total=self.k_total, k_base=self.k_base)


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


@dataclass(frozen=True)
class Q4KOracleSchedule:
  row_stride_words: int = 36

  def __call__(self, template: PackedRecordOperandTemplate, threads: PrecontractThreadAxes,
               source_k: int) -> tuple[PackedRecordCooperativeStore, ...]:
    """Exact wave-row payload fanout and paired-lane metadata ownership from load_tiles_q4_K."""
    row_stride_words = self.row_stride_words
    source, base_k, stores = template.source("record"), UOp.const(dtypes.weakint, source_k), []
    # wave/threadIdx.y owns rows wave+8*i; lane is txi and one source word fans out to two decoded int32 destinations.
    for iteration in range(16):
      row = threads.wave_m+iteration*8
      logical_row = template.row_tile_base+row
      word = source.index(logical_row*row_stride_words+4+base_k//8+threads.lane).load()
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
      value = q4_k_dm_record_callback((source,), logical_row, logical_k, 2,
                                      row_stride_words=row_stride_words)
      stores.append(PackedRecordCooperativeStore("dm", lane_group, logical_row, logical_k, row, group, value))
    return tuple(stores)


@dataclass(frozen=True)
class SplitQ8DSProducer:
  record_rows: int | None = None

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return _q8_split_ds(sources, row, k, width, record_rows=self.record_rows)


@dataclass(frozen=True)
class SplitQ8QSProducer:
  record_rows: int | None = None

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return _q8_split_qs(sources, row, k, width, record_rows=self.record_rows)


@dataclass(frozen=True)
class Q4KQSProducer:
  row_stride_words: int = 36

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return q4_k_qs_record_callback(
      sources, row, k, width, row_stride_words=self.row_stride_words)


@dataclass(frozen=True)
class Q4KDMProducer:
  row_stride_words: int = 36

  def __call__(self, sources: tuple[UOp, ...], row: UOp, k: UOp, width: int) -> UOp:
    return q4_k_dm_record_callback(
      sources, row, k, width, row_stride_words=self.row_stride_words)


_q4_k_oracle_schedule = Q4KOracleSchedule()


LLAMA_Q4_K_COOPERATIVE_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-load-tiles-q4-k-wave-row-v1", _q4_k_oracle_schedule, ("wave_m", "lane"))
LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-q8-ds4-linear-256-v1", _linear_q8_ds4_schedule, ("wave_m", "wave_n", "lane"))
# _linear_q8_ds4_schedule is generic single-field-producer vocabulary (despite
# the historical name): reused verbatim for the fp16 Q4_K-decode and plain-fp16
# activation K32-group templates below.
LLAMA_Q4K_FP16_GROUP_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-q4k-fp16-decode-group-linear-256-v1", _linear_q8_ds4_schedule, ("wave_m", "wave_n", "lane"))
LLAMA_ACTIVATION_FP16_GROUP_SCHEDULE = PackedRecordCooperativeSchedule(
  "llama-activation-fp16-group-linear-256-v1", _linear_q8_ds4_schedule, ("wave_m", "wave_n", "lane"))


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


def build_split_q8_ds4_record_template(role: str, values_source: UOp, scales_source: UOp, sums_source: UOp,
                                       row_axis: UOp, k_axis: UOp, row_tile_base: UOp, *,
                                       source_layout: str = "Q8_1_MMQ_DS4_SPLIT",
                                       sum_semantics: str = "sum_original_fp",
                                       record_rows: int | None = None) -> PackedRecordOperandTemplate:
  """Adapt split five-buffer Q8 arrays to the existing interleaved Q8_1 DS4 LDS row."""
  if source_layout != "Q8_1_MMQ_DS4_SPLIT": raise ValueError("split Q8 record producer requires Q8_1_MMQ_DS4_SPLIT source layout")
  if sum_semantics != "sum_original_fp": raise ValueError("split Q8 DS4 record producer requires sum_original_fp semantic")
  if record_rows is not None and (not isinstance(record_rows, int) or isinstance(record_rows, bool) or record_rows <= 0):
    raise ValueError("split Q8 record row count must be a positive integer")
  rows = 128 if record_rows is None else record_rows
  required = (("values", values_source, dtypes.int8, 2*rows*128),
              ("scales", scales_source, dtypes.float32, 2*rows*4),
              ("sums", sums_source, dtypes.float32, 2*rows*4))
  for name, source, dtype, size in required:
    if not isinstance(source.dtype, PtrDType) or source.dtype.base != dtype or source.dtype.size < size:
      raise TypeError(f"split Q8 {name} source must cover physical {dtype.name}[{size}] storage")
  return PackedRecordOperandTemplate(role, Q8_DS4_SPLIT_RECORD_ADAPTER,
    (PackedRecordSource("values", values_source), PackedRecordSource("scales", scales_source), PackedRecordSource("sums", sums_source)),
    (PackedRecordFieldProducer("ds", ("scales", "sums"), SplitQ8DSProducer(record_rows), vector_bytes=4),
     PackedRecordFieldProducer("qs", ("values",), SplitQ8QSProducer(record_rows), vector_bytes=4)),
    (), "qs", row_axis, k_axis, row_tile_base, dtypes.char, LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE)


def build_q4_k_record_template(role: str, source: UOp, row_axis: UOp, k_axis: UOp,
                               row_tile_base: UOp, *, source_layout: str = "Q4_K_UINT32X36",
                               decode_semantics: str = "llama_load_tiles_q4_K",
                               row_stride_words: int = 36) -> PackedRecordOperandTemplate:
  if source_layout != "Q4_K_UINT32X36": raise ValueError("Q4 record producer requires packed Q4_K uint32x36 source layout")
  if decode_semantics != "llama_load_tiles_q4_K": raise ValueError("Q4 record producer requires llama load_tiles_q4_K semantic")
  if source.dtype.base != dtypes.uint32: raise TypeError("Q4_K source must be physical uint32[36] blocks")
  if not isinstance(row_stride_words, int) or isinstance(row_stride_words, bool) or row_stride_words < 36:
    raise ValueError("Q4_K row stride must be at least 36 uint32 words")
  if source.dtype.size < 128*row_stride_words:
    raise TypeError("Q4_K source does not cover 128 rows at the declared physical stride")
  schedule = LLAMA_Q4_K_COOPERATIVE_SCHEDULE if row_stride_words == 36 else PackedRecordCooperativeSchedule(
    f"llama-load-tiles-q4-k-wave-row-stride-{row_stride_words}-v1",
    Q4KOracleSchedule(row_stride_words), ("wave_m", "lane"))
  return PackedRecordOperandTemplate(role, Q4_K_RECORD_DECODE, (PackedRecordSource("record", source),),
    (PackedRecordFieldProducer("qs", ("record",), Q4KQSProducer(row_stride_words), vector_bytes=4),
     PackedRecordFieldProducer("dm", ("record",), Q4KDMProducer(row_stride_words), vector_bytes=4)),
    ("padding",), "qs", row_axis, k_axis, row_tile_base, dtypes.char, schedule)


def build_llama_q4_k_fp16_group_record_template(role: str, source: UOp, row_axis: UOp, k_axis: UOp,
                                                row_tile_base: UOp, *, group_index: int,
                                                row_stride_words: int = 36) -> PackedRecordOperandTemplate:
  """One K32-group fp16-dequant-in-register Q4_K decode (implementation plan I.2/II.3)."""
  if source.dtype.base != dtypes.uint32: raise TypeError("Q4_K source must be physical uint32[36] blocks")
  if not isinstance(row_stride_words, int) or isinstance(row_stride_words, bool) or row_stride_words < 36:
    raise ValueError("Q4_K row stride must be at least 36 uint32 words")
  if source.dtype.size < 128*row_stride_words:
    raise TypeError("Q4_K source does not cover 128 rows at the declared physical stride")
  return PackedRecordOperandTemplate(role, Q4_K_RECORD_DECODE_GROUP, (PackedRecordSource("record", source),),
    (PackedRecordFieldProducer("value", ("record",),
      Q4KFp16DecodeGroupProducer(group_index, row_stride_words), vector_bytes=16),),
    (), "value", row_axis, k_axis, row_tile_base, dtypes.half, LLAMA_Q4K_FP16_GROUP_SCHEDULE)


def build_llama_activation_fp16_group_record_template(role: str, source: UOp, row_axis: UOp, k_axis: UOp,
                                                       row_tile_base: UOp, *, k_total: int,
                                                       k_base: int) -> PackedRecordOperandTemplate:
  """One K32-group plain fp16 activation cooperative copy (implementation plan I.6/II.6)."""
  if source.dtype.base != dtypes.half: raise TypeError("activation source must be a physical half[M,K] array")
  if not isinstance(k_total, int) or isinstance(k_total, bool) or k_total <= 0: raise ValueError("k_total must be a positive int")
  if not isinstance(k_base, int) or isinstance(k_base, bool) or k_base < 0: raise ValueError("k_base must be a non-negative int")
  return PackedRecordOperandTemplate(role, ACTIVATION_FP16_RECORD_COPY, (PackedRecordSource("value", source),),
    (PackedRecordFieldProducer("value", ("value",), ActivationFp16GroupProducer(k_total, k_base), vector_bytes=16),),
    (), "value", row_axis, k_axis, row_tile_base, dtypes.half, LLAMA_ACTIVATION_FP16_GROUP_SCHEDULE)


# Explicit llama names plus short generic-template-friendly names.
build_llama_q8_ds4_record_template = build_q8_ds4_record_template
build_llama_split_q8_ds4_record_template = build_split_q8_ds4_record_template
build_llama_q4_k_record_template = build_q4_k_record_template

__all__ = ["ACTIVATION_FP16_RECORD_COPY", "LLAMA_ACTIVATION_FP16_GROUP_SCHEDULE", "LLAMA_Q4K_FP16_GROUP_SCHEDULE",
  "LLAMA_Q4_K_COOPERATIVE_SCHEDULE", "LLAMA_Q8_DS4_COOPERATIVE_SCHEDULE",
  "Q4_K_RECORD_DECODE", "Q4_K_RECORD_DECODE_GROUP", "Q4_K_UINT32_BLOCK", "Q8_DS4_GLOBAL_RECORD", "Q8_DS4_RECORD_COPY",
  "Q8_DS4_SPLIT_GLOBAL_RECORD", "Q8_DS4_SPLIT_RECORD_ADAPTER",
  "SOURCE_ANCHORS", "RecordProducerInstanceWitness", "is_record_producer_instance_dependency",
  "record_producer_instance_value", "record_producer_instance_witnesses",
  "activation_fp16_group_callback", "build_llama_activation_fp16_group_record_template",
  "build_llama_q4_k_fp16_group_record_template",
  "build_q4_k_record_template", "build_q8_ds4_record_template", "build_split_q8_ds4_record_template",
  "build_llama_q4_k_record_template", "build_llama_q8_ds4_record_template", "build_llama_split_q8_ds4_record_template",
  "q4_k_dm_record_callback", "q4_k_fp16_decode_group_callback", "q4_k_qs_record_callback"]
