"""Fused decode kv-store chain gate tests (decode-kv-store-chain-fusion-scope-20260803.md): the promotion
record is CLOSED by default, the loader never infers promotion from a target string, the checked-in record
promotes NOTHING (no target has a same-session A/B record yet), the fused kernel is an additive elementwise
lowering named decode_kv_rope_store_<Hkv>_<Hd> that renders through both the HIP and CUDA renderers without
a GPU, and the route returns the cache-AFTER-store contract the legacy chain provides."""
import json, pathlib

import pytest

from tinygrad import dtypes, Tensor
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm.model_route_plan import (decode_kv_store_fusion_promoted,
  load_decode_kv_store_fusion_promotion, _DECODE_KV_STORE_FUSION_PROMOTED_TARGETS)
from tinygrad.llm.decode_kernels import decode_kv_rope_store_kernel
from tinygrad.llm.decode_routes import _kv_store_parts_view, decode_kv_store_route
from tinygrad.uop.ops import Ops, UOp


def _write_policy(path, *, targets="absent"):
  doc = {"schema": "boltbeam.route_policy.v1", "route": "decode_kv_store_fusion"}
  if targets != "absent": doc["promoted_targets"] = targets
  pathlib.Path(path).write_text(json.dumps(doc))
  return path


def test_closed_default_when_no_promoted_targets_key(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets="absent")
  assert load_decode_kv_store_fusion_promotion(p) == frozenset()


def test_closed_default_when_promoted_targets_empty(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[])
  assert load_decode_kv_store_fusion_promotion(p) == frozenset()


def test_loader_names_explicit_targets_only(tmp_path):
  p = _write_policy(tmp_path / "policy.json", targets=[{"backend": "NV", "architecture": "sm_120"}])
  promoted = load_decode_kv_store_fusion_promotion(p)
  assert ("NV", "sm_120") in promoted
  assert ("AMD", "gfx1100") not in promoted


def test_checked_in_record_promotes_nothing():
  # No same-session A/B record has landed yet: the record must promote zero targets and the authority
  # must return False for every concrete target (the gate then keeps the legacy chain everywhere).
  assert _DECODE_KV_STORE_FUSION_PROMOTED_TARGETS == frozenset()
  assert not decode_kv_store_fusion_promoted(("NV", "sm_120"))
  assert not decode_kv_store_fusion_promoted(("AMD", "gfx1100"))
  assert not decode_kv_store_fusion_promoted(("METAL", "apple8"))
  assert not decode_kv_store_fusion_promoted((None, None))


def test_kv_store_fused_kernel_name():
  kern = decode_kv_rope_store_kernel(8, 128, 2048)
  cache = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 0)
  k = UOp.placeholder((1024,), dtypes.float32, 1)
  v = UOp.placeholder((1024,), dtypes.float32, 2)
  freqs = UOp.placeholder((2048, 128), dtypes.float32, 3)
  assert kern(cache, k, v, freqs).arg.name == "decode_kv_rope_store_8_128"


def test_kv_store_fused_kernel_vparts_name_and_geometry():
  # VPART=4 names itself decode_kv_rope_store_8_128_v4 and consumes the RAW parts view (1024,4),
  # summing the four fp32 partials in-register (legacy left-to-right order) before the store cast.
  kern = decode_kv_rope_store_kernel(8, 128, 2048, VPART=4)
  cache = UOp.placeholder((2, 1, 8, 2048, 128), dtypes.float16, 0)
  k = UOp.placeholder((1024,), dtypes.float32, 1)
  v = UOp.placeholder((1024, 4), dtypes.float32, 2)
  freqs = UOp.placeholder((2048, 128), dtypes.float32, 3)
  assert kern(cache, k, v, freqs).arg.name == "decode_kv_rope_store_8_128_v4"
  with pytest.raises(ValueError, match="VPART>=1"):
    decode_kv_rope_store_kernel(8, 128, 2048, VPART=0)


def test_kv_store_geometry_rejects_unsupported_shapes():
  with pytest.raises(ValueError, match="Hkv>=1"):
    decode_kv_rope_store_kernel(0, 128, 2048)
  with pytest.raises(ValueError, match="even Hd>=2"):
    decode_kv_rope_store_kernel(8, 127, 2048)
  with pytest.raises(ValueError, match="even Hd>=2"):
    decode_kv_rope_store_kernel(8, 1, 2048)


def test_kv_store_fused_renders_through_hip_and_cuda_without_gpu():
  from tinygrad.renderer.cuda import CUDARenderer
  from tinygrad.renderer.cstyle import HIPRenderer
  for cache_dtype in (dtypes.float16, dtypes.float32):
    # fp32 is the NV cache dtype (the fused store must reproduce the legacy full-precision fp32 cache
    # bytes), fp16 the AMD one. Both must render through both renderers without a GPU.
    for vparts, v_size in ((1, 1024), (4, 4096)):
      kern = decode_kv_rope_store_kernel(8, 128, 2048, VPART=vparts)
      cache = UOp.placeholder((2, 1, 8, 2048, 128), cache_dtype, 0)
      k = UOp.placeholder((1024,), dtypes.float32, 1)
      v = UOp.placeholder((1024 * vparts // 4, vparts) if vparts > 1 else (1024,), dtypes.float32, 2)
      freqs = UOp.placeholder((2048, 128), dtypes.float32, 3)
      ast = kern(cache, k, v, freqs)
      for ren in (HIPRenderer(Target.parse("AMD:HIP:gfx1100")),
                  CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=True)):
        src = next(u.arg for u in to_program(ast, ren).src if u.op is Ops.SOURCE)
        # The kernel is elementwise and target-agnostic: no warp shuffles, no shared memory, no vendor
        # intrinsics -- the rope halves, the store casts and both cache stores appear in both sources.
        assert "decode_kv_rope_store_8_128" in src
        assert "start_pos" in src
        assert "__shfl_xor_sync" not in src and "ds_bpermute" not in src
        assert "__syncthreads" not in src
        if cache_dtype is dtypes.float16:
          assert "half" in src or "_Float16" in src


def test_kv_store_parts_view_walk_finds_after_reduce():
  # Runtime shape of the NV v chain: BUFFER(4096) -> RESHAPE(1024,4) -> AFTER -> REDUCE(ADD,(1,))
  # -> RESHAPE(1024,) -> RESHAPE(1,1,1024) -> MEMORY_SEMANTIC (the q4k decode GEMV emits 4 fp32
  # partials per row; the model's v is their axis-1 sum). The walk must return the AFTER parts view.
  from tinygrad.llm.memory_semantics import runtime_scratch
  parts = UOp.placeholder((4096,), dtypes.float32, 0)
  read = parts.reshape(1024, 4).after(UOp.sink(parts))
  v = runtime_scratch(Tensor(read.reduce(arg=(Ops.ADD, (1,)))).reshape(1, 1, 1024))
  view, vparts = _kv_store_parts_view(v)
  assert vparts == 4
  assert tuple(view.shape) == (1024, 4)
  assert view.uop.op is Ops.AFTER


def test_kv_store_parts_view_walk_keeps_flat_v_without_reduce():
  # A plain GEMV output (no parts reduce in the graph) must keep (v, 1) -- the reduce-less v is bound
  # exactly as before and no shape contract is invented.
  v = Tensor(UOp.placeholder((1024,), dtypes.float32, 0)).reshape(1, 1, 1024)
  view, vparts = _kv_store_parts_view(v)
  assert vparts == 1 and view is v


def test_kv_store_parts_view_walk_unwraps_q4k_after():
  # Runtime shape of the q4k v chain: BUFFER(1024) -> AFTER -> RESHAPE(1,1,1024) -> MEMORY_SEMANTIC
  # (the q4k GEMV already emits the reduced v). The walk must hand the route the raw AFTER so
  # custom_kernel binds the GEMV output buffer directly instead of materializing a copy.
  from tinygrad.llm.memory_semantics import runtime_scratch
  buf = UOp.placeholder((1024,), dtypes.float32, 0)
  v = runtime_scratch(Tensor(buf.after(UOp.sink(buf))).reshape(1, 1, 1024))
  view, vparts = _kv_store_parts_view(v)
  assert vparts == 1
  assert tuple(view.shape) == (1024,)
  assert view.uop.op is Ops.AFTER


def test_kv_store_parts_view_walk_rejects_other_reduce_axes():
  # An axis-0 reduce (or any non-(ADD,(1,)) reduce) is NOT the GEMV parts sum: keep (v, 1) so the
  # reduce materializes as a kernel exactly as the legacy chain did.
  parts = UOp.placeholder((4096,), dtypes.float32, 0)
  read = parts.reshape(4, 1024).after(UOp.sink(parts))
  v = Tensor(read.reduce(arg=(Ops.ADD, (0,)))).reshape(1, 1, 1024)
  view, vparts = _kv_store_parts_view(v)
  assert vparts == 1 and view is v


def test_kv_store_route_returns_cache_after_store_contract():
  # The route's returned tensor is the cache AFTER the store (same shape/dtype buffer view), the exact
  # contract the legacy `Tensor(cache.uop.after(store))` chain provides to the flash route. Building the
  # program requires no GPU and no realization.
  cache = Tensor.empty(2, 1, 8, 2048, 128, dtype=dtypes.float16)
  k = Tensor.empty(1, 1, 1024, dtype=dtypes.float32)
  v = Tensor.empty(1, 1, 1024, dtype=dtypes.float32)
  freqs = Tensor.empty(2048, 128, dtype=dtypes.float32)
  out = decode_kv_store_route(cache, k, v, freqs, 8, 128, 2048)
  assert out.shape == cache.shape and out.dtype == cache.dtype and out is not cache
  assert out.uop.op is Ops.AFTER
