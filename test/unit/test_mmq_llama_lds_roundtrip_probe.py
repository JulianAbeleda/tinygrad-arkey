import numpy as np
import pytest

from tinygrad.uop.ops import Ops, ProgramInfo

from extra.qk.mmq_llama_lds_roundtrip_probe import (DEBUG_BYTES, DEBUG_WORDS, SHAPE,
  build_llama_lds_roundtrip_probe, compare_llama_lds_roundtrip, expected_llama_lds_roundtrip,
  lds_roundtrip_segments)


def _inputs():
  rng = np.random.default_rng(20260717)
  q4 = rng.integers(0, 2**32, (128, 36), dtype=np.uint32)
  finite = np.stack((np.linspace(-2, 2, 128, dtype=np.float16),
                     np.linspace(.25, 1.25, 128, dtype=np.float16)), axis=1)
  q4[:, 0] = finite.view(np.uint32).reshape(128)
  values = rng.integers(-128, 128, (2, 128, 128), dtype=np.int8)
  scales = rng.standard_normal((2, 128, 4), dtype=np.float32)
  sums = rng.standard_normal((2, 128, 4), dtype=np.float32)
  return q4, values, scales, sums


def test_probe_uses_exact_real_phase0_producers_publish_and_five_buffer_abi():
  probe = build_llama_lds_roundtrip_probe()
  nodes = list(probe.sink.toposort())
  assert ProgramInfo.from_sink(probe.sink).globals == tuple(range(5))
  assert probe.sink.arg.name == "mmq_llama_lds_roundtrip_k256_phase0"
  assert {(x.arg, x.src[0].arg) for x in nodes if x.op is Ops.SPECIAL} == {("lidx0", 256)}
  producers = [x for x in nodes if isinstance(x.tag, tuple) and x.tag[:1] == ("hierarchical_record_producer",)]
  assert [x.tag for x in producers] == [
    ("hierarchical_record_producer", "A", None), ("hierarchical_record_producer", "B", 0)]
  assert probe.stage.persistent_producer in nodes
  assert probe.stage.phases[0].producer in nodes and probe.stage.phases[1].producer not in nodes
  assert probe.stage.phases[0].publish in nodes
  stores = [x for x in nodes if isinstance(x.tag, tuple) and x.tag[:1] == ("llama_lds_roundtrip_store",)]
  assert len(stores) == DEBUG_WORDS//256 == 54
  assert all(probe.stage.phases[0].publish in x.backward_slice for x in stores)
  loads = [x for x in nodes if isinstance(x.tag, tuple) and x.tag[:1] == ("llama_lds_roundtrip_load",)]
  assert len(loads) == 54
  assert {x.tag[1] for x in loads} == {"q8_phase0_record", "q4_persistent_qs", "q4_persistent_dm"}
  assert not [x for x in nodes if x.op is Ops.WMMA]


def test_debug_schema_exports_all_defined_q8_and_q4_record_fields_exactly():
  q8, q4_qs, q4_dm = lds_roundtrip_segments()
  assert (q8.name, q8.output_word_start, q8.words_per_row, q8.rows,
          q8.lds_byte_start, q8.lds_row_stride_bytes) == \
         ("q8_phase0_record", 0, 36, 128, 512, 144)
  assert (q4_qs.name, q4_qs.output_word_start, q4_qs.words_per_row, q4_qs.rows,
          q4_qs.lds_byte_start, q4_qs.lds_row_stride_bytes) == \
         ("q4_persistent_qs", 4608, 64, 128, 18944, 304)
  assert (q4_dm.name, q4_dm.output_word_start, q4_dm.words_per_row, q4_dm.rows,
          q4_dm.lds_byte_start, q4_dm.lds_row_stride_bytes) == \
         ("q4_persistent_dm", 12800, 8, 128, 19200, 304)
  assert q4_dm.output_word_end == DEBUG_WORDS == 13824 and DEBUG_BYTES == 55296


def test_independent_host_layout_reconstructs_phase0_q8_and_q4_records():
  q4_words, values, scales, sums = _inputs()
  words = expected_llama_lds_roundtrip(q4_words, values, scales, sums)
  raw = words.view(np.uint8)
  q8 = raw[:128*144].reshape(128, 144)
  np.testing.assert_array_equal(q8[:, 16:].view(np.int8), values[0])
  ds = q8[:, :16].view(np.float16).reshape(128, 8)
  np.testing.assert_array_equal(ds[:, 0::2].view(np.uint16), scales[0].astype(np.float16).view(np.uint16))
  np.testing.assert_array_equal(ds[:, 1::2].view(np.uint16), sums[0].astype(np.float16).view(np.uint16))
  q4_qs = raw[128*144:128*144+128*256].reshape(128, 256)
  block_row, decoded = q4_words[7], q4_qs[7].view(np.uint32)
  txi = 19
  destination = 16*(txi//8)+txi%8
  assert decoded[destination] == (block_row[4+txi] & np.uint32(0x0f0f0f0f))
  assert decoded[destination+8] == ((block_row[4+txi] >> np.uint32(4)) & np.uint32(0x0f0f0f0f))
  dm = raw[128*144+128*256:].view(np.float16).reshape(128, 16)
  source = block_row.view(np.uint8)
  d = block_row[:1].view(np.float16)[0]
  scale0 = np.float16(source[4] & 0x3f)
  assert dm[7, 0].view(np.uint16) == np.float16(d*scale0).view(np.uint16)
  changed = values.copy()
  changed[1] ^= np.int8(0x55)
  np.testing.assert_array_equal(words, expected_llama_lds_roundtrip(q4_words, changed, scales, sums))


def test_compare_is_bit_exact_reports_segment_coordinate_and_fails_closed():
  expected = expected_llama_lds_roundtrip(*_inputs())
  output = np.zeros(SHAPE[0]*SHAPE[1], dtype=np.float32)
  output.view(np.uint32)[:DEBUG_WORDS] = expected
  assert compare_llama_lds_roundtrip(output, expected) == {
    "schema": "tinygrad.mmq_llama_lds_roundtrip_probe.v1", "passed": True,
    "compared_words": DEBUG_WORDS, "mismatch_count": 0, "first_mismatch": None}
  output.view(np.uint32)[4608+3*64+9] ^= np.uint32(1)
  failed = compare_llama_lds_roundtrip(output, expected)
  assert not failed["passed"] and failed["mismatch_count"] == 1
  assert failed["first_mismatch"] == {"output_word": 4809, "segment": "q4_persistent_qs",
    "row": 3, "word_in_row": 9, "actual": int(output.view(np.uint32)[4809]), "expected": int(expected[4809])}
  with pytest.raises(ValueError, match="output must be"):
    compare_llama_lds_roundtrip(np.zeros(DEBUG_WORDS, dtype=np.float32), expected)
  with pytest.raises(ValueError, match="expected must be"):
    compare_llama_lds_roundtrip(output, expected.astype(np.uint64))


@pytest.mark.parametrize("which", range(4))
def test_host_expected_rejects_wrong_five_buffer_storage(which):
  inputs = list(_inputs())
  inputs[which] = inputs[which].reshape(-1)[:-1]
  with pytest.raises(ValueError, match="must be exact"):
    expected_llama_lds_roundtrip(*inputs)
