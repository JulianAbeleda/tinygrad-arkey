"""Source-anchored packed operands for the exact llama gfx1100 Q4_K/Q8_1 MMQ oracle.

Descriptors only: this module deliberately owns no route, emitter, compiler, or
lowering behavior.  Transform components describe one source block/LDS row;
panel placement and repetition are pinned separately by the oracle descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from extra.qk.mmq_llama_differential import LLAMA_SOURCE_COMMIT, llama_q4k_q8_structural_descriptor
from tinygrad.codegen.opt.packed_weight import PackedOperandComponent, PackedOperandTransform
from tinygrad.dtype import dtypes


SCHEMA = "tinygrad.mmq_llama_packed_operands.v1"
SOURCE_ROOT = "ggml/src/ggml-cuda"
SOURCE_ANCHORS = (
  "mmq.cuh:13-15 MMQ_ITER_K=256; MMQ_NWARPS=8",
  "mmq.cuh:28-47 block_q8_1_mmq: half2 ds4[4] followed by int8_t qs[4*QK8_1]",
  "mmq.cuh:33-36 16-byte padding stores scales/partial sums; sums are prior to quantization",
  "mmq.cuh:57-58 sizeof(block_q8_1_mmq)=4*sizeof(block_q8_1)=144",
  "quantize.cu:322-330 sum is reduced from original float xi before quantization",
  "quantize.cu:333-369 quantize q then ds4[iqs/32]=make_half2(d,sum)",
  "mmq.cuh:2093-2160 load_tiles_q4_K decodes packed block_q4_K into x_qs/x_dm",
  "mmq.cuh:219-228 MMQ_MMA_TILE_X_K_Q8_1=64 payload int32 + 8 half2 correction entries + 4 padding int32",
  "mmq.cuh:3360-3363 Q4_K dispatches load_tiles_q4_K and vec_dot_q8_1_q8_1_mma",
  "mmq.cuh:3447-3517 mul_mat_q_process_tile has two Q8 stages/dots and four __syncthreads epochs",
  "mmq.cuh:3459-3461 LDS ids, tile_y, padded tile_x offsets",
  "mmq.cuh:3933-3939 shared bytes are ids + x + padded y",
)

# Global Q4_K is its packed GGML ABI.  It is not the decoded Q4 LDS transform below.
Q4_K_GLOBAL_BLOCK = PackedOperandTransform("llama.q4_k.global_block.v1", (
  PackedOperandComponent("dm", dtypes.half, 0, 4, "half2(d,dmin)", 4, 2),
  PackedOperandComponent("scales", dtypes.uint8, 4, 12, "packed_6bit_scales_and_mins", 12, 1),
  PackedOperandComponent("qs", dtypes.uint8, 16, 128, "packed_q4_nibbles_256_values", 128, 16),
))

Q8_1_DS4_ROW = PackedOperandTransform("llama.q8_1.ds4.global_and_lds_row.v1", (
  PackedOperandComponent("ds", dtypes.half, 0, 16, "4x_half2(d,sum_original_fp)", 4, 4),
  PackedOperandComponent("qs", dtypes.int8, 16, 128, "4x32_signed_int8", 32, 16),
))

Q4_K_DECODED_LDS_ROW = PackedOperandTransform("llama.q4_k.decoded_lds_row.v1", (
  PackedOperandComponent("qs", dtypes.int32, 0, 256, "64x_int32_decoded_q4_payload", 4, 16),
  PackedOperandComponent("dm", dtypes.half, 256, 32, "8x_half2_scale_min_corrections", 4, 16),
  PackedOperandComponent("padding", dtypes.int32, 288, 16, "4x_int32_padding", 4, 16),
))


@dataclass(frozen=True)
class LlamaPackedOperandOracle:
  q4_global: PackedOperandTransform = Q4_K_GLOBAL_BLOCK
  q8_global_lds: PackedOperandTransform = Q8_1_DS4_ROW
  q4_decoded_lds: PackedOperandTransform = Q4_K_DECODED_LDS_ROW
  tile_m: int = 128
  tile_n: int = 128
  k_epoch: int = 256
  ids_offset_bytes: int = 0
  ids_bytes: int = 512
  q8_offset_bytes: int = 512
  q8_rows: int = 128
  q8_row_bytes: int = 144
  q8_panel_bytes: int = 18432
  q4_offset_bytes: int = 18944
  q4_rows: int = 128
  q4_row_int32: int = 76
  q4_panel_bytes: int = 38912
  arena_bytes: int = 57856
  q8_sum_semantic: str = "sum_original_fp"

  def validate(self) -> None:
    expected = _canonical_fields()
    for name, value in expected.items():
      actual = getattr(self, name)
      if actual != value: raise ValueError(f"llama packed operand oracle mismatch for {name}: expected {value!r}, got {actual!r}")
    if self.q4_global.identity == self.q4_decoded_lds.identity:
      raise ValueError("Q4 global packed ABI must differ from decoded LDS transform")

  @property
  def identity(self) -> tuple[Any, ...]:
    return (SCHEMA, LLAMA_SOURCE_COMMIT) + tuple(getattr(self, name).identity if isinstance(getattr(self, name), PackedOperandTransform)
      else getattr(self, name) for name in _canonical_fields())

  def to_json(self) -> dict[str, Any]:
    self.validate()
    structural = llama_q4k_q8_structural_descriptor()
    return {
      "schema": SCHEMA, "descriptor_id": "llama.gfx1100.q4_k_q8_1.mmq.128x128x256",
      "source": {"commit": LLAMA_SOURCE_COMMIT, "root": SOURCE_ROOT, "line_anchors": SOURCE_ANCHORS},
      "tile": {"m": self.tile_m, "n": self.tile_n, "k_epoch": self.k_epoch},
      "transforms": {"q4_global_packed": self.q4_global.to_json(), "q8_global_and_lds_row": self.q8_global_lds.to_json(),
                     "q4_decoded_lds_row": self.q4_decoded_lds.to_json()},
      "q4_abi_distinction": "Q4_K global block is packed dm/scales/qs; LDS row is decoded qs/half metadata/padding",
      "q8_ds4": {"rows": self.q8_rows, "row_bytes": self.q8_row_bytes, "panel_bytes": self.q8_panel_bytes,
                 "metadata": "4x half2(d,sum_original_fp)", "values": "128 signed int8",
                 "sum_semantic": self.q8_sum_semantic, "sum_timing": "before_quantization"},
      "q4_decoded_lds": {"rows": self.q4_rows, "row_int32": self.q4_row_int32, "panel_bytes": self.q4_panel_bytes,
                         "payload_int32": 64, "metadata_half2": 8, "metadata_int32": 8, "padding_int32": 4},
      "arena": {"ids": {"offset_bytes": self.ids_offset_bytes, "size_bytes": self.ids_bytes},
                "q8": {"offset_bytes": self.q8_offset_bytes, "size_bytes": self.q8_panel_bytes},
                "q4": {"offset_bytes": self.q4_offset_bytes, "size_bytes": self.q4_panel_bytes},
                "size_bytes": self.arena_bytes},
      "wmma": {"isa": "v_wmma_i32_16x16x16_iu8", "A": "signed_int8", "B": "signed_int8",
               "signed_A": True, "signed_B": True},
      "epochs": structural["dimensions"]["barriers"],
      "conventional_grid": dict(structural["dimensions"]["stream_k"]["grid"]),
    }


def _canonical_fields() -> Mapping[str, Any]:
  return {
    "q4_global": Q4_K_GLOBAL_BLOCK, "q8_global_lds": Q8_1_DS4_ROW, "q4_decoded_lds": Q4_K_DECODED_LDS_ROW,
    "tile_m": 128, "tile_n": 128, "k_epoch": 256,
    "ids_offset_bytes": 0, "ids_bytes": 512, "q8_offset_bytes": 512, "q8_rows": 128,
    "q8_row_bytes": 144, "q8_panel_bytes": 18432, "q4_offset_bytes": 18944, "q4_rows": 128,
    "q4_row_int32": 76, "q4_panel_bytes": 38912, "arena_bytes": 57856,
    "q8_sum_semantic": "sum_original_fp",
  }


def llama_packed_operand_oracle() -> LlamaPackedOperandOracle:
  oracle = LlamaPackedOperandOracle()
  oracle.validate()
  return oracle


def validate_llama_packed_operand_oracle(candidate: LlamaPackedOperandOracle) -> None:
  if not isinstance(candidate, LlamaPackedOperandOracle): raise TypeError("candidate must be a LlamaPackedOperandOracle")
  candidate.validate()


__all__ = ["LLAMA_SOURCE_COMMIT", "LlamaPackedOperandOracle", "Q4_K_DECODED_LDS_ROW", "Q4_K_GLOBAL_BLOCK",
           "Q8_1_DS4_ROW", "SCHEMA", "SOURCE_ANCHORS", "llama_packed_operand_oracle", "validate_llama_packed_operand_oracle"]
