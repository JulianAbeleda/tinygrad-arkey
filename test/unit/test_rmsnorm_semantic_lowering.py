"""Path 3 semantic RMSNorm tests (path3-semantic-rmsnorm-task-20260802.md): the marker is
created only when a closed promotion record opened the norm's route and the shape/dtype
admission admits it; the fail-closed lowering turns an admitted marker into ONE
scheduler-owned kernel (`rmsnorm_native_*`), and every unadmitted marker keeps the ordinary
graph unchanged. The M3 opaque emitter (`decode_rmsnorm_*`) stays byte-identical (pinned
key digest), and the native epilogue is a distinct kernel with its own pin."""
import hashlib

import pytest

from tinygrad import dtypes
from tinygrad.llm.decode_kernels import DecodeRMSNormSpec, emit_decode_rmsnorm_kernel
from tinygrad.uop.ops import Ops, RMSNormSpec, UOp


def _norm(flag=True, dim=4096, dtype=dtypes.float32, eps=1e-6, weight=True):
  from tinygrad import Tensor, nn
  norm = nn.RMSNorm(dim, eps=eps)
  norm._rmsnorm_native_promoted = flag
  if weight:
    norm.weight = Tensor.randn(dim, dtype=dtype).realize()
  else:
    norm.weight = None
  return norm


def _kernels(out):
  lin, _ = out.linear_with_vars()
  return [k.src[0].arg.name for k in lin.src]


# ── marker creation ──────────────────────────────────────────────────────────

def test_marker_created_only_when_promoted_flag_set():
  from tinygrad import Tensor
  x = Tensor.randn(1, 4096, dtype=dtypes.float32).realize()
  assert _norm(flag=True)(x).uop.op.name == "RMSNORM"
  assert _norm(flag=False)(x).uop.op.name != "RMSNORM"


def test_marker_requires_affine_weight():
  from tinygrad import Tensor
  x = Tensor.randn(1, 4096, dtype=dtypes.float32).realize()
  assert _norm(weight=False)(x).uop.op.name != "RMSNORM"


def test_marker_refuses_prefill_rows():
  from tinygrad import Tensor
  x = Tensor.randn(64, 4096, dtype=dtypes.float32).realize()
  assert _norm()(x).uop.op.name != "RMSNORM"


@pytest.mark.parametrize("shape,admitted", [
  ((1, 4096), True),
  ((32, 4096), True),
  ((1, 128), True),
  ((1, 32), True),
  ((33, 4096), False),
  ((1, 48), False),   # dim % 32 != 0
  ((1, 16), False),   # dim < 32
])
def test_marker_admission_shape(shape, admitted):
  from tinygrad import Tensor
  x = Tensor.randn(*shape, dtype=dtypes.float32).realize()
  got = _norm(dim=shape[-1])(x).uop.op.name == "RMSNORM"
  assert got is admitted


def test_marker_refuses_non_fp16_fp32_input():
  from tinygrad import Tensor
  x = Tensor.randn(1, 4096, dtype=dtypes.bfloat16).realize()
  assert _norm(dtype=dtypes.bfloat16)(x).uop.op.name != "RMSNORM"


# ── fail-closed lowering ─────────────────────────────────────────────────────

def test_admitted_marker_lowers_to_single_native_kernel_no_copies():
  from tinygrad import Tensor, nn
  norm = _norm()
  x = Tensor.randn(1, 4096, dtype=dtypes.float32).realize()
  out = norm(x)
  assert _kernels(out) == ["rmsnorm_native_1_4096"]
  xh = Tensor.randn(1, 4096, dtype=dtypes.float16).realize()
  outh = norm(xh)
  assert _kernels(outh) == ["rmsnorm_native_1_4096"]
  # weight must be realized in the same graph (no fill/copy kernel around the norm)
  assert all(k == "rmsnorm_native_1_4096" for k in _kernels(norm(Tensor.randn(1, 4096, dtype=dtypes.float32).realize())))


def test_unadmitted_marker_returns_ordinary_source():
  from tinygrad import Tensor
  # A marker over a prefill shape exists only if constructed directly; lowering must
  # return src[0] (the ordinary graph) and never emit the native kernel.
  x = Tensor.randn(64, 4096, dtype=dtypes.float32).realize()
  w = Tensor.randn(4096, dtype=dtypes.float32).realize()
  fallback = x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
  marker = Tensor(UOp(Ops.RMSNORM, fallback.dtype,
                      src=(fallback.uop, x.uop, w.uop),
                      arg=RMSNormSpec(4096, 1e-6, fallback.dtype, True)),
                  device=fallback.device)
  names = _kernels(marker)
  assert "rmsnorm_native_1_4096" not in names
  assert names


# ── emitter contracts and byte identity ──────────────────────────────────────

def test_native_kernel_name_distinct_from_m3_opaque():
  legacy = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16,
                             x_dtype=dtypes.float32, weight_dtype=dtypes.float16,
                             out_dtype=dtypes.float16, native=False)
  native = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16,
                             x_dtype=dtypes.float32, weight_dtype=dtypes.float16,
                             out_dtype=dtypes.float16, native=True)
  assert legacy.kernel_name == "decode_rmsnorm_1_4096"
  assert native.kernel_name == "rmsnorm_native_1_4096"
  assert legacy.kernel_name != native.kernel_name


def _emit(spec):
  out = UOp.placeholder((spec.rows * spec.dim,), spec.out_dtype, 0)
  x = UOp.placeholder((spec.rows * spec.dim,), spec.x_dtype, 1)
  w = UOp.placeholder((spec.dim,), spec.weight_dtype, 2)
  return emit_decode_rmsnorm_kernel(spec)(out, x, w)


def test_m3_opaque_emitter_key_digest_preserved():
  # Path 3 must not move M3's opaque emitter: pg3 pins the rendered source for
  # decode_rmsnorm_1_4096 (2f3b80f7b426...), this pins the uop key.
  legacy = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16,
                             x_dtype=dtypes.float32, weight_dtype=dtypes.float16,
                             out_dtype=dtypes.float16, native=False)
  digest = hashlib.sha256(repr(_emit(legacy).key).encode()).hexdigest()
  assert digest == "19f2d6a86892e28f25f73cef405ffed804c36be0c3d2a299135b8c110e6d4081"


def test_native_emitter_key_digest_pin():
  native = DecodeRMSNormSpec(rows=1, dim=4096, eps=1e-6, warps_per_row=16,
                             x_dtype=dtypes.float32, weight_dtype=dtypes.float16,
                             out_dtype=dtypes.float16, native=True)
  digest = hashlib.sha256(repr(_emit(native).key).encode()).hexdigest()
  assert digest == "533c1f052c67b09967b576a2752f442494227ece40b0b5e9b5bc8cae4f0b3b35"
  from dataclasses import replace as dreplace
  legacy = dreplace(native, native=False)
  assert digest != hashlib.sha256(repr(_emit(legacy).key).encode()).hexdigest()


def test_spec_validate_rejects_bad_contracts():
  bad = [
    (dict(rows=0, dim=4096, eps=1e-6), "rows>=1"),
    (dict(rows=1, dim=100, eps=1e-6), "dim >= lane_width"),
    (dict(rows=1, dim=4096, eps=1e-6, warps_per_row=7), r"dim % \(lane_width"),
    (dict(rows=1, dim=4096, eps=0.0), "eps>0"),
    (dict(rows=1, dim=4096, eps=1e-6, x_rank=4), r"x_rank in \(1, 2, 3\)"),
    (dict(rows=1, dim=4096, eps=1e-6, out_dtype=dtypes.float64), "out_dtype"),
  ]
  for kw, msg in bad:
    with pytest.raises(ValueError, match=msg):
      DecodeRMSNormSpec(**kw).validate()


# ── GPU value parity (skipped without CUDA) ─────────────────────────────────

def _cuda_available():
  from tinygrad.device import Device
  try:
    return str(Device.DEFAULT).startswith(("CUDA", "NV"))
  except Exception:
    return False


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype,tol", [(dtypes.float32, 1e-4), (dtypes.float16, 0.01)])
def test_native_lowering_value_parity_cuda(dtype, tol):
  import numpy as np
  from tinygrad import Tensor, nn
  from tinygrad.engine.realize import run_linear
  norm = _norm(dim=4096, dtype=dtype)
  x = Tensor.randn(1, 4096, dtype=dtype, device="CUDA").realize()
  w = Tensor.randn(4096, dtype=dtype, device="CUDA").realize()
  norm.weight = w
  out = norm(x)
  lin, var_vals = out.linear_with_vars()
  assert [k.src[0].arg.name for k in lin.src] == ["rmsnorm_native_1_4096"]
  run_linear(lin, var_vals)
  y = out.numpy().astype(np.float32)
  xr = x.numpy().astype(np.float32)
  wr = w.numpy().astype(np.float32)
  ref = (xr * np.reciprocal(np.sqrt(np.mean(np.square(xr), axis=-1, keepdims=True) + norm.eps))) * wr
  assert float(np.abs(y - ref).max()) < tol


@pytest.mark.skipif(not _cuda_available(), reason="requires CUDA")
def test_prefill_never_gets_marker_cuda():
  from tinygrad import Tensor
  x = Tensor.randn(64, 4096, dtype=dtypes.float32, device="CUDA").realize()
  assert _norm()(x).uop.op.name != "RMSNORM"
