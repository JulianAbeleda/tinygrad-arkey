from extra.llm_research.decode.nv_rmsnorm_scale_gateup_microgate import K, ROWS, cpu_gate, per_load_affine, rms_scale
import numpy as np
import pytest
from tinygrad import dtypes
from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel
from tinygrad.uop.ops import UOp

def test_production_shape_and_cpu_algebra_contract():
  row=cpu_gate()
  assert row["shape"] == {"K":K,"rows":ROWS}
  assert row["finite"] and row["roundpoint_exact"] and row["postdot_scalar_is_invalid"]
  assert row["topology"]["normalized_vector_store"] is False

def test_fp16_roundpoint_precedes_affine_weight():
  x=np.array([.3333,-.6665],np.float16); w=np.array([1.1,.9],np.float16); scale=rms_scale(x)
  got=per_load_affine(x,w,scale)
  expect=((x.astype(np.float32)*scale).astype(np.float16)*w).astype(np.float16)
  np.testing.assert_array_equal(got,expect)

def test_raw_load_consumer_has_explicit_scale_and_weight_inputs():
  sink=q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(12288,4096)(
    UOp.placeholder((12288,),dtypes.float32,0),UOp.placeholder((12288*16*36,),dtypes.uint32,1),
    UOp.placeholder((12288*16*36,),dtypes.uint32,2),UOp.placeholder((4096,),dtypes.float16,3),
    UOp.placeholder((4096,),dtypes.float16,4),UOp.placeholder((1,),dtypes.float32,5))
  assert sink.arg.name == "q4k_g3_lanemap_w1w3_rms_affine_12288_4096"
  assert len(sink.src) == 1

def test_raw_load_consumer_is_exact_shape_guarded():
  with pytest.raises(ValueError,match=r"requires \(12288,4096\)"):
    q4k_g3_lanemap_gemv_w1w3_rms_affine_kernel(4096,4096)
