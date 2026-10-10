"""Q5_K in the block-quant GEMV family: 256-element blocks, packed 6-bit scales/mins, fifth bits, two code units a lane.
The serial spelling (same per-lane math as the warp route) against tinygrad's GGUF dequantizer on random blocks."""
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.decode_kernels import BLOCK_QUANT_LAYOUTS, BlockQuantGEMVRouteSpec, block_quant_gemv_geometry, emit_block_quant_gemv_kernel
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.qk_layout import GGML_TYPE_IDS, QUANT_FORMATS, packed_nbytes


def _random_blocks(quant, rows, k, seed=0):
  rng = np.random.default_rng(seed)
  fmt = QUANT_FORMATS[quant]
  raw = rng.integers(0, 256, packed_nbytes(fmt, rows * k), dtype=np.uint8).reshape(-1, fmt.block_bytes)
  raw[:, 0:2] = np.array([0.0125], dtype=np.float16).view(np.uint8)
  if quant == "Q5_K": raw[:, 2:4] = np.array([0.003], dtype=np.float16).view(np.uint8)
  return raw.reshape(-1)


@pytest.mark.parametrize("quant,rows,k,lanes", [("Q5_K", 5, 768, 32), ("Q5_K", 3, 512, 8), ("Q5_K", 4, 256, 64 // 64),
                                                ("Q8_0", 3, 96, 16), ("Q5_0", 2, 64, 8), ("Q4_0", 2, 160, 4)])
def test_serial_route_matches_the_gguf_dequantizer(quant, rows, k, lanes):
  raw = _random_blocks(quant, rows, k)
  x = np.random.default_rng(1).standard_normal(k).astype(np.float16)
  spec = BlockQuantGEMVRouteSpec(rows=rows, k=k, quant=QUANT_FORMATS[quant], lanes_per_block=lanes, lane_reduce="serial")
  out = Tensor.empty(rows, dtype=dtypes.float32).uop_program(Tensor(raw.view(np.uint16), dtype=dtypes.uint16), Tensor(x),
                                                            fxn=emit_block_quant_gemv_kernel(spec))[0].numpy()
  w = ggml_data_to_tensor(Tensor(raw), rows * k, GGML_TYPE_IDS[quant]).numpy().reshape(rows, k)
  want = w.astype(np.float64) @ x.astype(np.float64)
  np.testing.assert_allclose(out, want, rtol=2e-4, atol=2e-3 * np.abs(want).max())


def test_q5_k_layout_and_geometry():
  lay = BLOCK_QUANT_LAYOUTS["Q5_K"]
  assert (lay.block_elems, lay.block_halfwords, lay.units) == (256, 88, 64)
  spec = block_quant_gemv_geometry(QUANT_FORMATS["Q5_K"], 2048, 1024, wave_size=32, sm_count=8, max_threads=1024)
  assert spec.lanes_per_block == 32 and spec.units_per_lane == 2  # 64 units over a 32-wide warp: two units a lane
  spec.validate()
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(rows=1, k=300, quant=QUANT_FORMATS["Q5_K"]).validate()


def test_a_format_rolls_back_alone():
  from tinygrad.llm.model_route_plan import decode_q8_0_gemv_formats
  everything = decode_q8_0_gemv_formats(lambda k, d=0: "")
  assert decode_q8_0_gemv_formats(lambda k, d=0: "Q5_K") == everything - {"Q5_K"}
  with pytest.raises(ValueError, match="unknown block-quant format"):  # a misspelled rollback is an error, not a no-op
    decode_q8_0_gemv_formats(lambda k, d=0: "Q5_K, Q9_9")
