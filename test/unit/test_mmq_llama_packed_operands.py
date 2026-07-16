import json
from dataclasses import replace

import pytest

from extra.qk.mmq_llama_packed_operands import (
  LLAMA_SOURCE_COMMIT, Q4_K_DECODED_LDS_ROW, Q4_K_GLOBAL_BLOCK, Q8_1_DS4_ROW,
  llama_packed_operand_oracle, validate_llama_packed_operand_oracle,
)
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform
from tinygrad.dtype import dtypes


def test_exact_transforms_and_stable_identity_json():
  oracle = llama_packed_operand_oracle()
  assert oracle.identity == llama_packed_operand_oracle().identity
  assert oracle.to_json() == llama_packed_operand_oracle().to_json()
  assert json.loads(json.dumps(oracle.to_json()))["source"]["commit"] == LLAMA_SOURCE_COMMIT
  assert Q8_1_DS4_ROW.identity == (
    "llama.q8_1.ds4.global_and_lds_row.v1",
    (("ds", "half", 0, 16, "4x_half2(d,sum_original_fp)", 4, 4),
     ("qs", "signed char", 16, 128, "4x32_signed_int8", 32, 16)))
  assert Q4_K_GLOBAL_BLOCK.component("qs").offset_bytes == 16 and Q4_K_GLOBAL_BLOCK.component("qs").end_bytes == 144
  assert Q4_K_DECODED_LDS_ROW.component("qs").size_bytes == 64 * 4
  assert Q4_K_DECODED_LDS_ROW.component("dm").size_bytes == 8 * 4  # eight half2 correction pairs
  assert Q4_K_DECODED_LDS_ROW.component("padding").size_bytes == 4 * 4
  assert Q4_K_GLOBAL_BLOCK.identity != Q4_K_DECODED_LDS_ROW.identity


def test_exact_panels_arena_epochs_wmma_and_runtime_grid_formula():
  payload = llama_packed_operand_oracle().to_json()
  assert payload["tile"] == {"m": 128, "n": 128, "k_epoch": 256}
  assert payload["q8_ds4"]["row_bytes"] == 16 + 128 == 144
  assert payload["q8_ds4"]["panel_bytes"] == 128 * 144 == 18432
  assert payload["q4_decoded_lds"] == {"rows": 128, "row_int32": 76, "panel_bytes": 38912,
    "payload_int32": 64, "metadata_half2": 8, "metadata_int32": 8, "padding_int32": 4}
  assert payload["arena"] == {"ids": {"offset_bytes": 0, "size_bytes": 512},
    "q8": {"offset_bytes": 512, "size_bytes": 18432}, "q4": {"offset_bytes": 18944, "size_bytes": 38912},
    "size_bytes": 57856}
  assert payload["wmma"]["signed_A"] is payload["wmma"]["signed_B"] is True
  assert payload["epochs"]["per_k_iteration"] == 4 and len(payload["epochs"]["sequence"]) == 8
  assert payload["conventional_grid"] == {"x": "ceil(nrows_x/128)", "y": "ceil(ncols_max/128)",
    "z": "channels*samples"}


def test_sum_is_original_fp_not_dequantized_q8_sum():
  q = (127, 127, -127)
  d = 1.0 / 127.0
  original_fp = (1.0, 0.999, -1.0)
  assert sum(original_fp) == pytest.approx(0.999)
  assert d * sum(q) == pytest.approx(1.0)
  payload = llama_packed_operand_oracle().to_json()
  assert payload["q8_ds4"]["sum_semantic"] == "sum_original_fp"
  assert payload["q8_ds4"]["sum_timing"] == "before_quantization"
  assert "sum_original_fp" in Q8_1_DS4_ROW.component("ds").layout


@pytest.mark.parametrize("field,value", [
  ("q8_sum_semantic", "sum_dequantized_q8"), ("q8_offset_bytes", 516), ("q4_offset_bytes", 18940),
  ("q8_panel_bytes", 18436), ("q4_panel_bytes", 38908), ("arena_bytes", 57860), ("k_epoch", 128),
])
def test_altered_semantics_offsets_sizes_or_k_epoch_fail_closed(field, value):
  with pytest.raises(ValueError, match=field): validate_llama_packed_operand_oracle(replace(llama_packed_operand_oracle(), **{field: value}))


def test_altered_transform_fails_closed():
  altered = PackedOperandTransform("altered", (PackedOperandComponent("qs", dtypes.int8, 0, 144),))
  with pytest.raises(ValueError, match="q8_global_lds"):
    validate_llama_packed_operand_oracle(replace(llama_packed_operand_oracle(), q8_global_lds=altered))


def test_source_commit_and_line_anchors_are_explicit():
  source = llama_packed_operand_oracle().to_json()["source"]
  assert source["commit"] == "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
  assert all(":" in anchor and any(ch.isdigit() for ch in anchor) for anchor in source["line_anchors"])
  assert any("make_half2(d,sum)" in anchor for anchor in source["line_anchors"])
