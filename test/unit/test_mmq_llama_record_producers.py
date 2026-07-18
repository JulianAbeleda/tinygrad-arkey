import numpy as np
import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
from test.helpers import eval_uop

from extra.qk.mmq_llama_record_producers import (Q4_K_RECORD_DECODE, Q8_DS4_RECORD_COPY,
  build_q4_k_record_template, build_q8_ds4_record_template, q4_k_dm_record_callback, q4_k_qs_record_callback)


def _axes():
  return UOp.range(128, 1201, AxisType.LOOP), UOp.range(256, 1202, AxisType.REDUCE), UOp.const(dtypes.weakint, 0)


def _templates():
  row, k, zero = _axes()
  q4src = UOp.param(0, dtypes.uint32.ptr(36*128))
  q4 = build_q4_k_record_template("A", q4src, row, k, zero)
  q8 = build_q8_ds4_record_template("B", UOp.param(1, dtypes.uint8.ptr(2*128*144)), row, k, zero)
  return q4, q8


def _reference(block: np.ndarray):
  raw = block.view(np.uint8)
  words = block[4:36]
  qs = np.empty(64, dtype=np.uint32)
  for txi, word in enumerate(words):
    dst = 16*(txi//8) + txi%8
    qs[dst], qs[dst+8] = word & 0x0f0f0f0f, (word >> 4) & 0x0f0f0f0f
  scales = np.empty(8, dtype=np.uint8)
  mins = np.empty(8, dtype=np.uint8)
  for group in range(8):
    if group < 4:
      scales[group], mins[group] = raw[4+group] & 63, raw[8+group] & 63
    else:
      lo = group-4
      scales[group] = (raw[12+lo] & 15) | ((raw[4+lo] >> 6) << 4)
      mins[group] = (raw[12+lo] >> 4) | ((raw[8+lo] >> 6) << 4)
  d, dmin = block[:1].view(np.float16).astype(np.float16)
  dm = np.empty(16, dtype=np.float16)
  dm[0::2] = (d * scales.astype(np.float16)).astype(np.float16)
  dm[1::2] = ((-dmin) * mins.astype(np.float16)).astype(np.float16)
  return qs, dm


def _run_callbacks(block: np.ndarray):
  src, row = UOp.param(0, dtypes.uint32.ptr(36)), UOp.const(dtypes.weakint, 0)
  inputs = [(dtypes.uint32, block.tolist())]
  qs = np.asarray([eval_uop(q4_k_qs_record_callback((src,), row, UOp.const(dtypes.weakint, i), 1).gep(0), inputs)
                   for i in range(64)], dtype=np.uint32)
  dm_bits = []
  for i in range(0, 16, 2):
    pair = q4_k_dm_record_callback((src,), row, UOp.const(dtypes.weakint, i), 2)
    dm_bits.extend(eval_uop(pair.gep(lane).bitcast(dtypes.uint16), inputs) for lane in range(2))
  return qs, np.asarray(dm_bits, dtype=np.uint16).view(np.float16)


def test_exact_transforms_record_sizes_sources_and_four_byte_callback_units():
  q4, q8 = _templates()
  assert Q4_K_RECORD_DECODE is q4.transform and Q8_DS4_RECORD_COPY is q8.transform
  assert [(x.name, x.dtype, x.size_bytes) for x in q8.transform.source.components] == [
    ("ds", dtypes.half, 16), ("qs", dtypes.int8, 128)]
  assert [(x.name, x.dtype, x.offset_bytes, x.size_bytes) for x in q4.transform.produced.components] == [
    ("qs", dtypes.int32, 0, 256), ("dm", dtypes.half, 256, 32), ("padding", dtypes.int32, 288, 16)]
  assert sum(x.size_bytes for x in q4.transform.produced.components) == 256+32+16
  assert q4.sources[0].pointer is q4.source("record")
  assert q8.source("ds") is q8.source("qs")
  assert q4.cooperative_schedule.name == "llama-load-tiles-q4-k-wave-row-v1"
  assert q8.cooperative_schedule.name == "llama-q8-ds4-linear-256-v1"
  assert q4.fragment_dtype == q8.fragment_dtype == dtypes.char
  assert [(x.field, x.vector_bytes) for x in q4.fields] == [("qs", 4), ("dm", 4)]
  assert [(x.field, x.vector_bytes) for x in q8.fields] == [("ds", 4), ("qs", 4)]
  for template, expected in ((q4, (dtypes.int32, dtypes.half.vec(2))),
                             (q8, (dtypes.half.vec(2), dtypes.int8.vec(4)))):
    got = []
    for field in template.fields:
      component = template.transform.produced.component(field.field)
      width = field.vector_bytes // component.dtype.itemsize
      got.append(field.producer(tuple(template.source(x) for x in field.sources), template.row_axis,
                                template.k_axis, width).dtype)
    assert tuple(got) == expected


@pytest.mark.parametrize("seed", [0, 1, 20260715, 0xffffffff])
def test_q4_callbacks_match_independent_source_formula_random(seed):
  rng = np.random.default_rng(seed)
  block = rng.integers(0, 2**32, 36, dtype=np.uint32)
  # finite, deliberately nontrivial half scales
  block[0] = np.asarray([np.float16(-1.25), np.float16(0.375)], dtype=np.float16).view(np.uint32)[0]
  expected_qs, expected_dm = _reference(block)
  actual_qs, actual_dm = _run_callbacks(block)
  np.testing.assert_array_equal(actual_qs, expected_qs)
  np.testing.assert_array_equal(actual_dm.view(np.uint16), expected_dm.view(np.uint16))


@pytest.mark.parametrize("epoch", [0, 19])
def test_q4_full_role_explicit_row_stride_selects_same_epoch_next_row(epoch):
  epochs, rows = 20, 128
  row_axis, k_axis, zero = _axes()
  root = UOp.param(0, dtypes.uint32.ptr(rows*epochs*36))
  template = build_q4_k_record_template(
    "A", root.index(UOp.const(dtypes.weakint, epoch*36), ptr=True), row_axis, k_axis, zero,
    row_stride_words=epochs*36)
  words = np.zeros(rows*epochs*36, dtype=np.uint32)
  expected_index = (epochs+epoch)*36+4
  compact_wrong_index = epoch*36+36+4
  words[expected_index] = np.uint32(0x01010101)
  words[compact_wrong_index] = np.uint32(0x02020202)
  value = template.fields[0].producer(
    (template.source("record"),), UOp.const(dtypes.weakint, 1),
    UOp.const(dtypes.weakint, 0), 1)
  assert eval_uop(value.gep(0), [(dtypes.uint32, words.tolist())]) == 0x01010101


@pytest.mark.parametrize("fill", [0x00000000, 0xffffffff, 0xaaaaaaaa, 0x55555555])
def test_q4_callbacks_adversarial_payload_and_six_bit_codes(fill):
  block = np.full(36, fill, dtype=np.uint32)
  block[0] = np.asarray([np.float16(2), np.float16(-3)], dtype=np.float16).view(np.uint32)[0]
  expected_qs, expected_dm = _reference(block)
  actual_qs, actual_dm = _run_callbacks(block)
  np.testing.assert_array_equal(actual_qs, expected_qs)
  np.testing.assert_array_equal(actual_dm.view(np.uint16), expected_dm.view(np.uint16))


def test_callbacks_are_pure_source_anchored_integer_decode_without_q4_opcode_or_float_weights():
  q4, _ = _templates()
  nodes = set()
  for field in q4.fields:
    width = field.vector_bytes // q4.transform.produced.component(field.field).dtype.itemsize
    nodes.update(field.producer((q4.source("record"),), q4.row_axis, q4.k_axis, width).backward_slice_with_self)
  assert all(node.op.name not in {"Q4", "DEQUANTIZE"} for node in nodes)
  assert all(node.dtype.scalar() not in (dtypes.float, dtypes.double) for node in nodes)
  loads = [node for node in nodes if node.op is Ops.LOAD]
  assert loads and all(q4.source("record") in node.backward_slice_with_self for node in loads)


def test_q8_callbacks_read_one_physical_aos_record_across_both_k128_phases():
  row_axis, k_axis, zero = _axes()
  q8 = build_q8_ds4_record_template("B", UOp.param(1, dtypes.uint8.ptr(2*128*144)), row_axis, k_axis, zero)
  row = 37
  for phase in range(2):
    logical_k = phase*128
    ds_value = q8.fields[0].producer((q8.source("ds"),), UOp.const(dtypes.weakint, row),
                                     UOp.const(dtypes.weakint, logical_k+4), 2)
    qs_value = q8.fields[1].producer((q8.source("qs"),), UOp.const(dtypes.weakint, row),
                                     UOp.const(dtypes.weakint, logical_k+52), 4)
    record_byte_base = (phase*128+row)*144
    assert [x.src[0].src[1].vmin for x in ds_value.src] == [(record_byte_base+2*i)//2 for i in (4, 5)]
    assert [x.src[0].src[1].vmin for x in qs_value.src] == [record_byte_base+16+i for i in range(52, 56)]
    assert all(q8.source("ds") in x.backward_slice_with_self for x in ds_value.src)
    assert all(q8.source("qs") in x.backward_slice_with_self for x in qs_value.src)


def test_builders_fail_closed_on_wrong_source_layout_and_semantic():
  row, k, zero = _axes()
  q4 = UOp.param(0, dtypes.uint32.ptr(36))
  record = UOp.param(1, dtypes.uint8.ptr(144))
  with pytest.raises(ValueError, match="source layout"):
    build_q4_k_record_template("A", q4, row, k, zero, source_layout="bytes144")
  with pytest.raises(ValueError, match="load_tiles_q4_K semantic"):
    build_q4_k_record_template("A", q4, row, k, zero, decode_semantics="float_dequant")
  with pytest.raises(TypeError, match="uint32"):
    build_q4_k_record_template("A", UOp.param(3, dtypes.uint8.ptr(144)), row, k, zero)
  with pytest.raises(ValueError, match="source layout"):
    build_q8_ds4_record_template("B", record, row, k, zero, source_layout="row_major")
  with pytest.raises(ValueError, match="sum_original_fp"):
    build_q8_ds4_record_template("B", record, row, k, zero, sum_semantics="sum_quantized")
  with pytest.raises(TypeError, match="physical byte-addressed"):
    build_q8_ds4_record_template("B", UOp.param(2, dtypes.half.ptr(8)), row, k, zero)


def test_wmma_fragment_proof_normalizes_four_q8_carrier_records():
  from extra.qk.amd_isa_renderer_policy import PREFILL_AMD_ISA_RENDERER_POLICY
  common = dict(role="B", lds_buffer_id="lds0", dbuf_slot=1, k_phase=("stage_epoch", 0, 1),
               producer_epoch=("stage", "lds0", 0, 1), overwrite_epoch=("stage", "lds0", 0, 1, "next"),
               field="qs", iteration=0, schema="llama-q8-ds4-producer-instance.v1", byte_len=4)
  elems = tuple(UOp(Ops.NOOP, dtypes.half, (), tag=("wmma_frag_proof", *(tuple(common.items()) +
    (("logical_row_or_col", ("B", 544+i*4)), ("byte_start", 544+i*4)))),) for i in range(4) for _ in range(4))
  class H:
    def wmma_elems(self, _carrier, _count): return elems
  key = PREFILL_AMD_ISA_RENDERER_POLICY.wmma_frag_proof_key("B", UOp(Ops.NOOP, dtypes.half.vec(16)), H())
  assert dict(key)["logical_row_or_col"] == ("B", 544) and dict(key)["byte_len"] == 16


@pytest.mark.parametrize("starts", [(544, 548, 556, 560), (544, 548, 548, 556)])
def test_wmma_fragment_proof_rejects_carrier_gaps_and_overlaps(starts):
  from extra.qk.amd_isa_renderer_policy import PREFILL_AMD_ISA_RENDERER_POLICY
  common = {"role":"B", "lds_buffer_id":"lds0", "dbuf_slot":1, "k_phase":("stage_epoch",0,1),
            "producer_epoch":("stage","lds0",0,1), "overwrite_epoch":("stage","lds0",0,1,"next"),
            "field":"qs", "iteration":0, "schema":"llama-q8-ds4-producer-instance.v1", "byte_len":4}
  elems = tuple(UOp(Ops.NOOP, dtypes.half, (), tag=("wmma_frag_proof", *(tuple(common.items()) +
    (("logical_row_or_col", ("B", s)), ("byte_start", s)))),) for s in starts for _ in range(4))
  class H:
    def wmma_elems(self, _carrier, _count): return elems
  assert PREFILL_AMD_ISA_RENDERER_POLICY.wmma_frag_proof_key("B", UOp(Ops.NOOP, dtypes.half.vec(16)), H()) is None


def test_wmma_fragment_proof_rejects_mixed_epochs():
  from extra.qk.amd_isa_renderer_policy import PREFILL_AMD_ISA_RENDERER_POLICY
  elems = []
  for i in range(4):
    for lane in range(4):
      epoch = ("stage", "lds0", 0, 2) if i == 3 else ("stage", "lds0", 0, 1)
      tag = ("wmma_frag_proof", ("role", "B"), ("lds_buffer_id", "lds0"), ("dbuf_slot", 1),
             ("k_phase", ("stage_epoch", 0, 1)), ("logical_row_or_col", ("B", 544+i*4)),
             ("byte_start", 544+i*4), ("byte_len", 4), ("producer_epoch", epoch),
             ("overwrite_epoch", epoch + ("next",)), ("field", "qs"), ("iteration", 0),
             ("schema", "llama-q8-ds4-producer-instance.v1"))
      elems.append(UOp(Ops.NOOP, dtypes.half, (), tag=tag))
  class H:
    def wmma_elems(self, _carrier, _count): return tuple(elems)
  assert PREFILL_AMD_ISA_RENDERER_POLICY.wmma_frag_proof_key("B", UOp(Ops.NOOP, dtypes.half.vec(16)), H()) is None
