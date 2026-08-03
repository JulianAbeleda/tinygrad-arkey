"""L1 P4 M4 unit tests (l1-decode-plumbing-fusion-design-20260802.md section 2.1):
Q4_K G3 GEMV epilogue absorption -- spec validation, admission wiring, fused kernel
render arms, and legacy byte-identity guarantees."""
import hashlib

import pytest

from tinygrad import dtypes
from tinygrad.helpers import Target
from tinygrad.llm.decode_kernels import (Q4KGEMVEpilogue, Q4KGateUpLaneMap,
  q4k_g3_lanemap_gemv_kernel, LanePartition)
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.llm.qk_layout import Q4_K_BLOCK_ELEMS, Q4K_WORDS_PER_BLOCK
from tinygrad.uop.ops import Ops, UOp


# ── spec validation ──────────────────────────────────────────────────────────

def test_empty_epilogue_is_valid():
  epi = Q4KGEMVEpilogue()
  assert epi.kind == ""
  assert epi.kernel_suffix == ""
  epi.validate(4096, 4096)
  epi.validate(1024, 4096)


def test_residual_add_epilogue_is_valid():
  epi = Q4KGEMVEpilogue("residual_add")
  assert epi.kernel_suffix == "_epi_resadd"
  epi.validate(4096, 4096)


def test_ffn_down_fused_epilogue_requires_rows_4096():
  epi = Q4KGEMVEpilogue("ffn_down_fused")
  assert epi.kernel_suffix == "_epi_ffndown"
  epi.validate(4096, 12288)
  with pytest.raises(ValueError, match="rows=4096"):
    epi.validate(1024, 4096)


def test_fp16_cast_epilogue_is_valid():
  epi = Q4KGEMVEpilogue("fp16_cast")
  assert epi.kernel_suffix == "_epi_f16cast"
  epi.validate(1024, 4096)


def test_unknown_epilogue_kind_rejected():
  with pytest.raises(ValueError, match="unsupported"):
    Q4KGEMVEpilogue("bogus").validate(4096, 4096)


# ── legacy byte-identity ─────────────────────────────────────────────────────

@pytest.mark.parametrize("rows,k,digest", [
  (4096, 4096, "10d8d359bca1f310b7b41940680cb1f7c0d84b3d6280b8e63636a6440f91be13"),
  (32, 1024, "b034e43c12561149fac0faa838142c07d48d1cd55ea2073dce9c25a16a64754f"),
])
def test_legacy_kernel_hash_preserved_with_default_epilogue(rows, k, digest):
  args = (UOp.placeholder((rows,), dtypes.float32, 0),
          UOp.placeholder((rows * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1),
          UOp.placeholder((k,), dtypes.float16, 2))
  promoted = q4k_g3_lanemap_gemv_kernel(rows, k)(*args)
  assert hashlib.sha256(repr(promoted.key).encode()).hexdigest() == digest
  # Explicit None epilogue is byte-identical
  promoted_none = q4k_g3_lanemap_gemv_kernel(rows, k, epilogue=None)(*args)
  assert hashlib.sha256(repr(promoted_none.key).encode()).hexdigest() == digest
  # Empty epilogue is byte-identical
  promoted_empty = q4k_g3_lanemap_gemv_kernel(rows, k, epilogue=Q4KGEMVEpilogue())(*args)
  assert hashlib.sha256(repr(promoted_empty.key).encode()).hexdigest() == digest


# ── fused kernel names ───────────────────────────────────────────────────────

def test_fused_kernel_names_are_distinct():
  for epi_kind, expected_suffix in [
    ("residual_add", "_epi_resadd"),
    ("ffn_down_fused", "_epi_ffndown"),
    ("fp16_cast", "_epi_f16cast"),
  ]:
    epi = Q4KGEMVEpilogue(epi_kind)
    rows, k = {"ffn_down_fused": (4096, 12288)}.get(epi_kind, (4096, 4096))
    kernel = q4k_g3_lanemap_gemv_kernel(rows, k, epilogue=epi)
    # Call the kernel with appropriate placeholders to get the name from arg
    from tinygrad.uop.ops import KernelInfo
    out = UOp.placeholder((rows,), dtypes.float32, 0)
    words = UOp.placeholder((rows * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
    if epi_kind == "ffn_down_fused":
      uop = kernel(out, words,
                   UOp.placeholder((k,), dtypes.float32, 2),
                   UOp.placeholder((k,), dtypes.float32, 3),
                   UOp.placeholder((rows,), dtypes.float32, 4))
    else:
      x = UOp.placeholder((k,), dtypes.float16, 2)
      extra = (UOp.placeholder((rows,), dtypes.float32, 3),) if epi_kind == "residual_add" else ()
      uop = kernel(out, words, x, *extra)
    assert expected_suffix in uop.arg.name, f"Expected {expected_suffix!r} in {uop.arg.name}"
    assert f"q4k_g3_lanemap_gemv{expected_suffix}_{rows}_{k}" == uop.arg.name


# ── render arms (HIP + CUDA, no GPU needed) ──────────────────────────────────

def _render_kernel(rows, k, epi_kind, ren):
  from tinygrad.codegen import to_program
  epi = Q4KGEMVEpilogue(epi_kind) if epi_kind else None
  kernel = q4k_g3_lanemap_gemv_kernel(rows, k, epilogue=epi)
  out = UOp.placeholder((rows,), dtypes.float32, 0)
  words = UOp.placeholder((rows * (k // Q4_K_BLOCK_ELEMS) * Q4K_WORDS_PER_BLOCK,), dtypes.uint32, 1)
  if epi_kind == "ffn_down_fused":
    uop = kernel(out, words,
                 UOp.placeholder((k,), dtypes.float32, 2),
                 UOp.placeholder((k,), dtypes.float32, 3),
                 UOp.placeholder((rows,), dtypes.float32, 4))
  elif epi_kind == "residual_add":
    uop = kernel(out, words,
                 UOp.placeholder((k,), dtypes.float16, 2),
                 UOp.placeholder((rows,), dtypes.float32, 3))
  elif epi_kind == "fp16_cast":
    uop = kernel(out, words, UOp.placeholder((k,), dtypes.float16, 2))
  else:
    uop = kernel(out, words, UOp.placeholder((k,), dtypes.float16, 2))
  src = next(u.arg for u in to_program(uop, ren).src if u.op is Ops.SOURCE)
  return src


def test_residual_add_renders_through_hip_and_cuda():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
              CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = _render_kernel(4096, 4096, "residual_add", ren)
    assert "__shfl_xor_sync" in src or "ds_bpermute" in src


def test_ffn_down_fused_renders_through_hip_and_cuda():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
              CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = _render_kernel(4096, 12288, "ffn_down_fused", ren)
    assert "__shfl_xor_sync" in src or "ds_bpermute" in src


def test_fp16_cast_renders_through_hip_and_cuda():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
              CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
    src = _render_kernel(1024, 4096, "fp16_cast", ren)
    assert "__shfl_xor_sync" in src or "ds_bpermute" in src


# ── admission wiring ─────────────────────────────────────────────────────────

def test_q4k_epilogue_admission_closed_default():
  cap = QKPrimitiveCapability(backend="NV", architecture="sm_120",
                              wave_size=32, supports_warp_shfl_xor=True)
  # M4's q4k variants have their OWN closed record: even when target and M2's Q6K
  # epilogue fusion are both promoted, q4k_epilogue_fusion_admitted stays False.
  adm = QKPrimitiveRouteAdmission(cap, target_promoted=True, epilogue_fusion_promoted=True)
  assert adm.admitted
  assert adm.fusion_admitted
  assert not adm.q4k_epilogue_fusion_admitted
  # Only the q4k record opens the q4k epilogue variants
  adm_open = QKPrimitiveRouteAdmission(cap, target_promoted=True, epilogue_fusion_promoted=True,
                                       q4k_epilogue_fusion_promoted=True)
  assert adm_open.admitted
  assert adm_open.q4k_epilogue_fusion_admitted
  # Neither capability nor promotion satisfied -> nothing works
  adm_none = QKPrimitiveRouteAdmission()
  assert not adm_none.admitted
  assert not adm_none.fusion_admitted
  assert not adm_none.q4k_epilogue_fusion_admitted


def test_legacy_q4k_kernel_name_unchanged():
  """The legacy kernel (no epilogue) keeps its original name with no suffix."""
  kernel = q4k_g3_lanemap_gemv_kernel(4096, 4096)
  out = UOp.placeholder((4096,), dtypes.float32, 0)
  words = UOp.placeholder((4096 * (4096 // 256) * 36,), dtypes.uint32, 1)
  x = UOp.placeholder((4096,), dtypes.float16, 2)
  uop = kernel(out, words, x)
  assert uop.arg.name == "q4k_g3_lanemap_gemv_4096_4096"
  assert "_epi_" not in uop.arg.name
