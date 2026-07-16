import numpy as np
import pytest

from tinygrad import dtypes
from tinygrad.uop.ops import AxisType, UOp
from test.helpers import eval_uop

from extra.qk.mmq_llama_packed_operands import Q8_1_DS4_ROW
from extra.qk.mmq_llama_record_producers import (Q8_DS4_RECORD_COPY, Q8_DS4_SPLIT_RECORD_ADAPTER,
  build_q8_ds4_record_template, build_split_q8_ds4_record_template)


def _axes():
  return UOp.range(128, 1250, AxisType.LOOP), UOp.range(256, 1251, AxisType.REDUCE), UOp.const(dtypes.weakint, 0)


def _templates():
  row, k, zero = _axes()
  interleaved = build_q8_ds4_record_template("B", UOp.param(0, dtypes.uint8.ptr(2*128*144)), row, k, zero)
  split = build_split_q8_ds4_record_template("B", UOp.param(0, dtypes.int8.ptr(2*128*128)),
    UOp.param(1, dtypes.float32.ptr(2*128*4)), UOp.param(2, dtypes.float32.ptr(2*128*4)), row, k, zero)
  return interleaved, split


def test_split_q8_staging_matches_interleaved_bytes_after_independent_half_rounding():
  interleaved, split = _templates()
  rng = np.random.default_rng(20260716)
  values = rng.integers(-128, 128, (2, 128, 128), dtype=np.int8)
  scales = rng.normal(size=(2, 128, 4)).astype(np.float32)
  sums = rng.normal(size=(2, 128, 4)).astype(np.float32)
  # Values straddling half-way points catch fp32 pair packing or shared-rounding shortcuts.
  scales[0, 37] = np.array([1.0004882, 1.0004884, -2.0009763, -2.0009768], dtype=np.float32)
  sums[0, 37] = np.array([65519.0, 65520.0, 2**-25, -2**-25], dtype=np.float32)
  records = np.zeros((2, 128, 144), dtype=np.uint8)
  records[..., 16:] = values.view(np.uint8)
  metadata = np.stack((scales.astype(np.float16), sums.astype(np.float16)), axis=-1)
  records[..., :16] = metadata.view(np.uint8).reshape(2, 128, 16)
  scale_inputs = [(dtypes.float32, scales.reshape(-1).tolist())]
  sum_inputs = [(dtypes.float32, sums.reshape(-1).tolist())]
  value_inputs = [(dtypes.int8, values.reshape(-1).tolist())]
  for phase, row in ((0, 37), (1, 91)):
    for group in range(4):
      k = UOp.const(dtypes.weakint, phase*128+group*2)
      new = split.fields[0].producer((split.source("scales"), split.source("sums")), UOp.const(dtypes.weakint, row), k, 2)
      record_bits = records[phase, row, group*4:group*4+4].copy().view(np.uint16).tolist()
      assert [eval_uop(new.gep(0).bitcast(dtypes.uint16), scale_inputs),
              eval_uop(new.gep(1).bitcast(dtypes.uint16), sum_inputs)] == record_bits
    k = UOp.const(dtypes.weakint, phase*128+52)
    new = split.fields[1].producer((split.source("values"),), UOp.const(dtypes.weakint, row), k, 4)
    assert values[phase, row, 52:56].tolist() == [eval_uop(new.gep(i), value_inputs) for i in range(4)]


def test_split_builder_reuses_destination_schedule_and_witness_contract():
  interleaved, split = _templates()
  assert Q8_DS4_RECORD_COPY.produced is Q8_1_DS4_ROW
  assert Q8_DS4_SPLIT_RECORD_ADAPTER.produced is Q8_1_DS4_ROW
  assert split.cooperative_schedule is interleaved.cooperative_schedule
  assert [(x.field, x.vector_bytes) for x in split.fields] == [("ds", 4), ("qs", 4)]


def test_split_builder_rejects_wrong_dtype_size_layout_and_sum_semantics():
  row, k, zero = _axes()
  good = (UOp.param(0, dtypes.int8.ptr(2*128*128)), UOp.param(1, dtypes.float32.ptr(2*128*4)),
          UOp.param(2, dtypes.float32.ptr(2*128*4)))
  with pytest.raises(ValueError, match="source layout"):
    build_split_q8_ds4_record_template("B", *good, row, k, zero, source_layout="row_major")
  with pytest.raises(ValueError, match="sum_original_fp"):
    build_split_q8_ds4_record_template("B", *good, row, k, zero, sum_semantics="sum_quantized")
  for bad, match in ((UOp.param(3, dtypes.uint8.ptr(2*128*128)), "values"),
                     (UOp.param(3, dtypes.float32.ptr(2*128*4-1)), "scales"),
                     (UOp.param(3, dtypes.half.ptr(2*128*4)), "sums")):
    args = list(good)
    args[(0 if match == "values" else 1 if match == "scales" else 2)] = bad
    with pytest.raises(TypeError, match=match): build_split_q8_ds4_record_template("B", *args, row, k, zero)
