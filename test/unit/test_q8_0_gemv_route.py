"""Generic block-quant (Q8_0 / Q5_0 / Q4_0) decode GEMV route: kernel numerics on CPU, the shape gate, geometry from
device facts, the lowering family, and the promotion record. Shapes in this file are test inputs; none appear in the
route code."""
import json, pathlib
import numpy as np
import pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.decode_kernels import (BLOCK_QUANT_LAYOUTS, BlockQuantGEMVRouteSpec, block_quant_gemv_geometry,
  emit_block_quant_gemv_kernel)
from tinygrad.llm.decode_routes import DECODE_ROUTE_SHAPE_PROBES
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec, execute_research_program
from tinygrad.llm.model_route_plan import (DecodeRouteShapeGate, DecodeRouteShapes, load_decode_q8_0_gemv_formats,
  load_decode_q8_0_gemv_promotion)
from tinygrad.llm.qk_layout import Q4_0, Q5_0, Q8_0, QUANT_FORMATS

def _q8_0_blocks(rows:int, k:int, seed:int) -> tuple[np.ndarray, np.ndarray]:
  """Random GGUF Q8_0 bytes as uint16 halfwords, and the numpy dequantized dense matrix."""
  rng = np.random.default_rng(seed); nb = rows * k // 32
  d = (rng.random(nb) * 0.02 + 0.001).astype(np.float16)
  q = rng.integers(-128, 128, size=(nb, 32), dtype=np.int8)
  raw = np.zeros((nb, 34), np.uint8); raw[:, :2] = d.view(np.uint8).reshape(nb, 2); raw[:, 2:] = q.view(np.uint8)
  dense = (d.astype(np.float32)[:, None] * q.astype(np.float32)).reshape(rows, k)
  return raw.reshape(-1).view(np.uint16), dense

def _random_blocks(fmt, rows:int, k:int, seed:int) -> tuple[np.ndarray, np.ndarray]:
  """Random bytes in any GGML block format with a sane fp16 scale; dense reference from gguf.ggml_data_to_tensor
  (an independent dequantizer: the ordinary model load path)."""
  rng = np.random.default_rng(seed); nb = rows * k // fmt.block_elems
  raw = rng.integers(0, 256, size=(nb, fmt.block_bytes), dtype=np.uint8)
  raw[:, :2] = (rng.random(nb) * 0.02 + 0.001).astype(np.float16).view(np.uint8).reshape(nb, 2)
  ggml_type = BLOCK_QUANT_LAYOUTS[fmt.name].ggml_type
  dense = ggml_data_to_tensor(Tensor(raw.reshape(-1).copy(), device="CPU"), rows * k, ggml_type).numpy().astype(np.float32).reshape(rows, k)
  return raw.reshape(-1).view(np.uint16), dense

def _run(spec:BlockQuantGEMVRouteSpec, halfs:np.ndarray, x:np.ndarray) -> np.ndarray:
  program = KernelProgram("bq_test", "bq_test.gemv", KernelProgramProvenance.RESEARCH_ONLY, emit_block_quant_gemv_kernel(spec),
                          output_spec=OutputSpec((spec.rows,), dtypes.float32))
  return execute_research_program(Tensor.empty(spec.rows, dtype=dtypes.float32), Tensor(halfs), Tensor(x), program=program).numpy()

def _close(got, dense, x):
  ref = dense.astype(np.float64) @ x.astype(np.float64)
  np.testing.assert_allclose(got, ref, rtol=0, atol=2e-6 * float(np.abs(ref).max()) + 1e-6)

# (rows, k): square; non-square wide; non-square tall with k % 512 != 0; tiny k with k_blocks < one warp step;
# and rows not a multiple of the warps per CTA.
@pytest.mark.parametrize("rows,k", [(64, 1024), (24, 5120), (48, 1056), (37, 96), (130, 2080)])
def test_q8_0_gemv_matches_numpy_dequantize_and_dot(rows, k):
  halfs, dense = _q8_0_blocks(rows, k, rows * 7919 + k)
  x = np.random.default_rng(k).normal(0, 1, k).astype(np.float32)
  _close(_run(block_quant_gemv_geometry(Q8_0, rows, k, wave_size=32, sm_count=2, max_threads=1024, lane_reduce="serial"), halfs, x), dense, x)

@pytest.mark.parametrize("fmt", [Q8_0, Q5_0, Q4_0], ids=lambda f: f.name)
@pytest.mark.parametrize("rows,k", [(40, 1024), (21, 1056), (9, 96)])
def test_block_quant_formats_match_the_ggml_dequantizer(fmt, rows, k):
  halfs, dense = _random_blocks(fmt, rows, k, rows + k)
  x = np.random.default_rng(rows).normal(0, 1, k).astype(np.float32)
  _close(_run(block_quant_gemv_geometry(fmt, rows, k, wave_size=32, sm_count=2, max_threads=1024, lane_reduce="serial"), halfs, x), dense, x)

@pytest.mark.parametrize("fmt", [Q8_0, Q5_0], ids=lambda f: f.name)
@pytest.mark.parametrize("lanes,warps", [(1, 1), (2, 3), (4, 2), (8, 1)])
def test_every_lane_partition_is_the_same_dot(fmt, lanes, warps):
  rows, k = 19, 1056  # k_blocks=33: ragged for every lanes_per_block > 1; rows ragged for warps 2..3
  halfs, dense = _random_blocks(fmt, rows, k, 3)
  x = np.random.default_rng(4).normal(0, 1, k).astype(np.float32)
  _close(_run(BlockQuantGEMVRouteSpec(rows, k, fmt, lanes_per_block=lanes, warps_per_cta=warps, lane_reduce="serial"), halfs, x), dense, x)

def test_q8_0_gemv_reads_fp16_activations():
  rows, k = 16, 512
  halfs, dense = _q8_0_blocks(rows, k, 5)
  x = np.random.default_rng(6).normal(0, 1, k).astype(np.float16)
  _close(_run(block_quant_gemv_geometry(Q8_0, rows, k, wave_size=32, sm_count=None, max_threads=None, lane_reduce="serial"), halfs, x), dense, x)

def test_validate_states_only_real_constraints():
  for rows, k in ((1, 32), (48, 5120), (151936, 1024), (7, 96)): BlockQuantGEMVRouteSpec(rows, k).validate()
  for rows, k in ((0, 32), (4, 0), (4, 16), (4, 1040 + 8)):
    with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(rows, k).validate()
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(4, 64, lanes_per_block=3).validate()   # 16 code units / 3
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(4, 64, Q5_0, lanes_per_block=16).validate()  # Q5_0 has 8 units
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(4, 64, wave_size=0).validate()         # wave not reported
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(4, 64, wave_size=64).validate()        # beyond the staged ladder
  with pytest.raises(ValueError): BlockQuantGEMVRouteSpec(4, 256, QUANT_FORMATS["Q4_K"]).validate()  # no layout: refused

def test_geometry_comes_from_format_shape_and_device_facts():
  wide = block_quant_gemv_geometry(Q8_0, 10240, 5120, wave_size=32, sm_count=170, max_threads=1024)
  assert (wide.lanes_per_block, wide.warps_per_cta, wide.wave_size) == (16, 2, 32)
  assert block_quant_gemv_geometry(Q5_0, 10240, 5120, wave_size=32, sm_count=170, max_threads=1024).lanes_per_block == 8
  # Too few rows for two CTAs per SM at 2 warps: one warp per CTA. Fewer SMs: two rows per CTA.
  assert block_quant_gemv_geometry(Q8_0, 48, 5120, wave_size=32, sm_count=170, max_threads=1024).warps_per_cta == 1
  assert block_quant_gemv_geometry(Q8_0, 48, 5120, wave_size=32, sm_count=8, max_threads=1024).warps_per_cta == 2
  # A device thread limit of one warp caps the CTA. Unknown SM count: one warp. Unknown wave: refused.
  assert block_quant_gemv_geometry(Q8_0, 10240, 5120, wave_size=32, sm_count=170, max_threads=32).warps_per_cta == 1
  assert block_quant_gemv_geometry(Q8_0, 10240, 5120, wave_size=32, sm_count=None, max_threads=None).warps_per_cta == 1
  with pytest.raises(ValueError):
    block_quant_gemv_geometry(Q8_0, 10240, 5120, wave_size=None, sm_count=170, max_threads=1024).validate()
  narrow = block_quant_gemv_geometry(Q8_0, 64, 64, wave_size=8, sm_count=None, max_threads=None); narrow.validate()
  assert narrow.lanes_per_block == 8

def _shapes(dim, hidden, heads, kv_heads, head_dim, vocab):
  return DecodeRouteShapes(dim=dim, hidden_dim=hidden, n_heads=heads, n_kv_heads=kv_heads, head_dim=head_dim, vocab=vocab,
                           eps=1e-6, target=("NV", "sm_120"))

@pytest.mark.parametrize("shapes", [_shapes(5120, 17408, 24, 4, 256, 248320), _shapes(1024, 3072, 16, 8, 128, 151936)],
                         ids=["5120-wide", "1024-wide"])
def test_shape_gate_admits_the_route(shapes):
  gate = DecodeRouteShapeGate(shapes, DECODE_ROUTE_SHAPE_PROBES)
  assert gate.admit("decode_q8_0_gemv", True) and gate.admitted("decode_q8_0_gemv")
  assert gate.report_lines() == []

def test_shape_gate_refuses_the_route_for_k_not_a_multiple_of_32():
  gate = DecodeRouteShapeGate(_shapes(1040, 3072, 16, 8, 65, 151936), DECODE_ROUTE_SHAPE_PROBES)
  assert gate.admit("decode_q8_0_gemv", True) is False
  lines = gate.report_lines()
  assert len(lines) == 1 and "decode_q8_0_gemv" in lines[0] and "multiple of 32" in lines[0]
  gate2 = DecodeRouteShapeGate(_shapes(1040, 3072, 16, 8, 65, 151936), DECODE_ROUTE_SHAPE_PROBES)
  assert gate2.admit("decode_q8_0_gemv", False) is False and gate2.report_lines() == []

def test_lowering_family_binds_the_spec_and_refuses_drift():
  from tinygrad.llm.boltbeam_authority import lower_authorized_candidate
  spec = block_quant_gemv_geometry(Q8_0, 256, 1024, wave_size=32, sm_count=170, max_threads=1024)
  candidate = {"family": "block_quant_gemv_route.v1", **spec.to_json(), "spec_binding": "s"}
  emitter, ticket = lower_authorized_candidate(candidate, (("decode_q8_0_gemv", "block_quant_gemv"),), lowering_bindings={"s": spec})
  assert callable(emitter) and len(ticket.tickets) == 1
  other = block_quant_gemv_geometry(Q8_0, 256, 2048, wave_size=32, sm_count=170, max_threads=1024)
  with pytest.raises(ValueError): lower_authorized_candidate(candidate, (("decode_q8_0_gemv", "block_quant_gemv"),), lowering_bindings={"s": other})

def test_promotion_record_names_nv_sm120_formats_and_measured_shapes():
  path = pathlib.Path(__file__).resolve().parents[2] / "tinygrad" / "llm" / "generated" / "decode-q8-0-gemv-route-policy.json"
  assert load_decode_q8_0_gemv_promotion(str(path)) == frozenset({("NV", "sm_120")})
  assert load_decode_q8_0_gemv_formats(str(path)) == frozenset({"Q8_0"})
  record = json.loads(path.read_text())
  shapes = {(s["rows"], s["k"]) for s in record["measured_shapes"]}
  assert len({k for _, k in shapes}) >= 3
  for rows, k in shapes: block_quant_gemv_geometry(Q8_0, rows, k, wave_size=32, sm_count=None, max_threads=None).validate()

def test_rollback_switch_closes_the_route():
  from tinygrad.llm.model_route_plan import decode_q8_0_gemv_promoted
  assert decode_q8_0_gemv_promoted(("NV", "sm_120"), getenv_fn=lambda key, default=0: default)
  assert not decode_q8_0_gemv_promoted(("NV", "sm_120"), getenv_fn=lambda key, default=0: 1 if key == "TINYGRAD_Q8_0_GEMV_DISABLE" else default)
  assert not decode_q8_0_gemv_promoted(("AMD", "gfx1100"), getenv_fn=lambda key, default=0: default)

def test_quant_formats():
  for name, elems, nbytes in (("Q8_0", 32, 34), ("Q5_0", 32, 22), ("Q4_0", 32, 18)):
    assert (QUANT_FORMATS[name].block_elems, QUANT_FORMATS[name].block_bytes) == (elems, nbytes)
    assert BLOCK_QUANT_LAYOUTS[name].quant is QUANT_FORMATS[name]
