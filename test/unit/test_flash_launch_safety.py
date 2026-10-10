"""K3/K4/K5: the flash decode kernels refuse a launch they cannot run, and never let an unwritten cache slot into a sum.

Reference inputs carry a real score spread (max scores across +-100, past where exp overflows in fp32), so a combine without max subtraction
overflows and fails, instead of passing on a nearly flat softmax."""
import numpy as np, pytest
from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.llm.decode_routes import _flash_combine_register_weights_admitted
from tinygrad.llm.flash_decode_attention import (COMBINE_REGISTER_WEIGHTS_MAX_SPLITS, FLASH_DECODE_G4, FLASH_DECODE_G5,
  describe_flash_decode_attention, flash_decode_route_for, flash_decode_route_refusal, flash_fused_gmax_combine_kernel,
  flash_vec_llama_score_pv_kernel)


# ---- K3: the tile's threads per block ----

def test_a_tile_wider_than_a_block_is_refused_and_says_why():
  for Hq in (48, 64):  # MQA: one KV head, so the default query group is every head: 32 x Hq threads
    assert flash_decode_route_for(Hq, 1, 128) is None
    assert "threads per block" in flash_decode_route_refusal(Hq, 1, 128)
  assert flash_decode_route_for(32, 1, 128) is not None  # exactly 1024 threads still launches


def test_the_device_thread_limit_lowers_the_portable_one():
  assert flash_decode_route_for(16, 1, 128) is not None
  assert flash_decode_route_for(16, 1, 128, max_threads=256) is None
  assert "256" in flash_decode_route_refusal(16, 1, 128, max_threads=256)
  assert flash_decode_route_for(16, 1, 128, max_threads=4096) is not None  # a larger fact never raises the cap
  assert flash_decode_route_for(48, 1, 128, max_threads=4096) is None


def test_the_measured_routes_are_unchanged():
  for tuned in (FLASH_DECODE_G4, FLASH_DECODE_G5):
    assert flash_decode_route_for(tuned.query_heads, tuned.kv_heads, tuned.head_dim) == tuned
    assert flash_decode_route_for(tuned.query_heads, tuned.kv_heads, tuned.head_dim, max_threads=1024) == tuned
  spec = describe_flash_decode_attention(32, 128, 8, 48, 48)
  assert spec.tile.threads == 128 and spec.tile.max_threads == 1024


# ---- K5: register combine weights hold one split per warp lane ----

def _partials(Hq, S, Hd, seed=0):
  rng = np.random.default_rng(seed)
  p = np.zeros((Hq, S, Hd + 2), np.float32)
  p[:, :, :Hd] = rng.standard_normal((Hq, S, Hd))
  p[:, :, Hd] = rng.uniform(1, 5, (Hq, S))         # each split's softmax denominator
  p[:, :, Hd + 1] = rng.uniform(-100, 100, (Hq, S))  # max scores past exp overflow (88): a real spread
  return p


def _combine_reference(p, Hd, weight_split=lambda s: s):
  S = p.shape[1]
  m = p[:, :, Hd + 1].max(axis=1, keepdims=True)
  w = np.exp(p[:, [weight_split(s) for s in range(S)], Hd + 1] - m)
  return (w[:, :, None] * p[:, :, :Hd]).sum(1) / (w * p[:, :, Hd]).sum(1)[:, None]


def test_the_combine_matches_a_reference_with_real_score_spread():
  Hd, Hq, S = 128, 4, 40
  p = _partials(Hq, S, Hd)
  with np.errstate(over="ignore", invalid="ignore"):  # without the max subtraction the weights overflow
    flat = (np.exp(p[:, :, Hd + 1])[:, :, None] * p[:, :, :Hd]).sum(1) / (np.exp(p[:, :, Hd + 1]) * p[:, :, Hd]).sum(1)[:, None]
  assert not np.allclose(flat, _combine_reference(p, Hd), atol=1e-3)
  out = Tensor.empty(Hq * Hd, dtype=dtypes.float32).uop_program(
    Tensor(p.reshape(-1)), fxn=flash_fused_gmax_combine_kernel(Hd, Hq, S))[0].numpy().reshape(Hq, Hd)
  np.testing.assert_allclose(out, _combine_reference(p, Hd), rtol=1e-4, atol=1e-5)


def test_register_weights_wrap_past_one_warp_so_they_are_refused():
  Hd, Hq, S = 128, 2, 34
  p = _partials(Hq, S, Hd, seed=1)
  # the numpy model of the register path: split s reads the weight held by warp lane s & 31
  wrapped = _combine_reference(p, Hd, weight_split=lambda s: s & 31)
  assert not np.allclose(wrapped, _combine_reference(p, Hd), atol=1e-3)
  flash_fused_gmax_combine_kernel(Hd, Hq, COMBINE_REGISTER_WEIGHTS_MAX_SPLITS, lane_width=128, register_weights=True)
  with pytest.raises(ValueError, match="S <= 32"):
    flash_fused_gmax_combine_kernel(Hd, Hq, S, lane_width=128, register_weights=True)


def test_register_weight_admission_falls_back_above_one_warp():
  off = lambda *_: 0
  assert _flash_combine_register_weights_admitted(True, {}, off, split_count=8)
  assert _flash_combine_register_weights_admitted(True, {}, off, split_count=32)
  assert not _flash_combine_register_weights_admitted(True, {}, off, split_count=34)
  assert not _flash_combine_register_weights_admitted(False, {"combine_register_weights": True}, off, split_count=34)


# ---- K4: the wide substrate never loads V from an unwritten slot ----

@pytest.mark.parametrize("v_pipeline_tail", [0, 1, 2])
def test_every_wide_v_load_is_gated_by_the_live_length(v_pipeline_tail):
  pout = UOp.placeholder((32 * 6 * 130,), dtypes.float32, 0)
  q = UOp.placeholder((32 * 64,), dtypes.uint32, 1)
  cache_k = UOp.placeholder((1, 1, 8, 768, 64), dtypes.uint32, 2)
  cache_v = UOp.placeholder((1, 1, 8, 768, 64), dtypes.uint32, 3)
  ast = flash_vec_llama_score_pv_kernel(128, 32, 8, 768, 6, UOp.variable("Tc", 1, 768), wide_kv=True, token_bound=768,
                                        separate_kv=True, v_pipeline_tail=v_pipeline_tail)(pout, q, cache_k, cache_v)
  v_param = cache_v.src[0] if cache_v.op is Ops.RESHAPE else cache_v
  v_loads = [u for u in ast.toposort() if u.op is Ops.LOAD and u.src[0].src[0] is v_param]
  assert v_loads, "the kernel must read V"
  # a gated load carries (index, alternative, gate); the gate must name the live length Tc
  for u in v_loads:
    assert len(u.src) == 3, "an ungated V load reads stale slots >= Tc (Tensor.empty cache: may be NaN)"
    assert any(x.op is Ops.DEFINE_VAR and x.arg[0] == "Tc" for x in u.src[2].toposort())
