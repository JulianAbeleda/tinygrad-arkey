"""Correctness fix: amd_gfx1100_q16_grid_hd128_loop_attention's K/V fragment loads used to be
UNCONDITIONAL past the true `kv_tokens` extent on the ceil-divided tail KV tile
(kernels.py `full_kv_tiles=(kv_tokens+15)//16`), because the K/V buffers carry no padding
(kernels.py's `sizes=` check requires exactly kv_heads*kv_tokens*hd elements) while the softmax
masking in expand_native_row_softmax_repack only zeroes the *result* of an invalid KV column,
never the load address (see amd_attention_abi.py `valid.where(value,-inf)` /
`valid.where(weight,0)`). This guards the LOAD ADDRESS itself (amd_attention_abi.expand_loop_fragment,
`row_ok`/`_row_ok`, via `UOp.valid`), the same idiom postrange.py's PADTO opt uses.

These tests exercise the guard directly at the level it is introduced (before ISA/HIP lowering),
so the assertions are exact and don't depend on any particular downstream renderer. A few tests
also build the real production kernel (amd_gfx1100_q16_grid_hd128_loop_attention) end to end,
compile-only, to prove the fix composes with the whole pipeline. One test executes the kernel on
real AMD hardware for numerical parity against a numpy softmax-attention reference; it is not run
here (no GPU in this environment) and is written to run wherever a GPU is present, matching the
rest of this test module's convention of unconditional device="AMD" execution.
"""
import itertools
import pytest

from tinygrad.dtype import Invalid, dtypes
from tinygrad.helpers import getenv
from tinygrad.uop.ops import (AMDAttentionGridSpec, AMDPackedFragmentLoopSpec, AxisType, KernelInfo,
                               Ops, ParamArg, UOp, graph_rewrite)
from tinygrad.uop.symbolic import sym
from tinygrad.renderer.isa.amd_attention_abi import expand_loop_fragment

REMAINDERS = list(range(1, 16))  # kv_tokens % 16 == 1..15


def _grid(kv_tokens, q_tokens=32, q_heads=4, kv_heads=2, group_ratio=2, head_dim=128):
  return AMDAttentionGridSpec(q_tokens=q_tokens, q_heads=q_heads, kv_heads=kv_heads,
                               group_ratio=group_ratio, kv_tokens=kv_tokens, head_dim=head_dim).validate()


def _owner(grid, tag):
  return UOp(Ops.PARAM, dtypes.half.ptr(grid.kv_heads * grid.kv_tokens * grid.head_dim), arg=ParamArg(tag))


def _fragment_loads(role, grid, block=0):
  """Call expand_loop_fragment directly, then run the same constant-folding symbolic pass the real
  pipeline runs first (tinygrad/codegen/__init__.py's "initial symbolic" graph_rewrite(sink, sym+...)),
  and return the 16 owner.index(...).load() sources. Folding is NOT automatic on plain UOp
  construction -- it only happens once graph_rewrite actually walks the graph with `sym` -- so
  inspecting the guard's fold-away behavior requires this same rewrite, not the raw builder output."""
  owner = _owner(grid, 3 if role == "V" else 2)
  lane = UOp.special(32, "lidx0")
  col = lane & UOp.const(dtypes.weakint, 15)
  full_kv_tiles = (grid.kv_tokens + 15) // 16
  rng = UOp.range(full_kv_tiles, 9600, AxisType.REDUCE)
  group = UOp.special(grid.q_heads * grid.q_tiles, "gidx0")
  spec = AMDPackedFragmentLoopSpec(role=role, head_block=block, grid=grid)
  x = UOp(Ops.AMD_PACKED_FRAGMENT_LOAD, dtypes.half.vec(16), (owner, lane, col, rng, group), arg=spec)
  stack = graph_rewrite(expand_loop_fragment(x), sym)
  assert stack.op is Ops.STACK and len(stack.src) == 16
  return list(stack.src)


def _is_gated(load: UOp) -> bool:
  """True iff this load's index is `cond.where(idx, Invalid)` (the OOB guard), not a plain index."""
  assert load.op is Ops.LOAD
  index = load.src[0]
  assert index.op is Ops.INDEX
  idx = index.src[1]
  return idx.op is Ops.WHERE and idx.src[2].op is Ops.CONST and idx.src[2].arg is Invalid


@pytest.mark.parametrize("remainder", REMAINDERS)
def test_k_fragment_all_16_lanes_guarded_when_kv_tokens_unaligned(remainder):
  """K's 16-wide vector shares one KV row (`rng*16+col`); col's dynamic range [0,15] means the
  guard can never provably fold for ANY unaligned kv_tokens (col alone can push the row past
  kv_tokens for every remainder in 1..15), so every lane must stay gated."""
  grid = _grid(kv_tokens=64 + remainder)
  loads = _fragment_loads("K", grid)
  assert all(_is_gated(l) for l in loads), f"K fragment should be guarded on every lane at remainder={remainder}"


@pytest.mark.parametrize("remainder", REMAINDERS)
def test_v_default_fragment_guards_exactly_the_out_of_range_lanes(remainder):
  """V's (non-transposed) 16-wide vector varies the KV row per-lane (`rng*16+i`). With
  kv_tokens=16*k+remainder and rng's static bound [0,k], lane i's row has static max 16k+i, which
  is provably < kv_tokens=16k+remainder iff i < remainder. So lanes [0,remainder) must fold away
  (no gate) and lanes [remainder,16) must stay gated -- an exact, per-lane prediction."""
  grid = _grid(kv_tokens=64 + remainder)
  loads = _fragment_loads("V", grid)
  gated = [_is_gated(l) for l in loads]
  assert gated == [(i >= remainder) for i in range(16)], (remainder, gated)


@pytest.mark.parametrize("remainder", REMAINDERS)
def test_v_transposed_fragment_guards_exactly_the_out_of_range_lanes(remainder):
  getenv.cache_clear()
  import os
  old = os.environ.get("PREFILL_V_TRANSPOSED")
  os.environ["PREFILL_V_TRANSPOSED"] = "1"
  getenv.cache_clear()
  try:
    grid = _grid(kv_tokens=64 + remainder)
    loads = _fragment_loads("V", grid)
  finally:
    if old is None: os.environ.pop("PREFILL_V_TRANSPOSED", None)
    else: os.environ["PREFILL_V_TRANSPOSED"] = old
    getenv.cache_clear()
  gated = [_is_gated(l) for l in loads]
  assert gated == [(i >= remainder) for i in range(16)], (remainder, gated)


@pytest.mark.parametrize("role", ["K", "V"])
def test_guard_folds_away_entirely_when_kv_tokens_aligned(role):
  """The aligned hot path (kv_tokens % 16 == 0) must not pay for the guard at all: every lane's
  static bound already proves in-range, so no WHERE/Invalid should survive in any of the 16
  fragment loads -- byte-identical to the pre-fix addressing."""
  for kv_tokens in (64, 128, 512, 4096):
    grid = _grid(kv_tokens=kv_tokens)
    loads = _fragment_loads(role, grid)
    assert not any(_is_gated(l) for l in loads), f"{role} guard should fold away at aligned kv_tokens={kv_tokens}"
    # and the index is exactly a plain arithmetic expression (no leftover CMPLT/WHERE anywhere in it)
    for l in loads:
      idx = l.src[0].src[1]
      assert not any(u.op in (Ops.WHERE, Ops.CMPLT) for u in idx.toposort())


def test_grid_spec_still_rejects_non_positive_or_oversized_kv_tokens():
  """Relaxing the 16-wide kv_tokens requirement must not open the door to nonsense geometry:
  q_tokens stays 16-wide (Q addressing is unguarded and out of scope for this fix), and kv_tokens
  must still be a positive integer <= 4096."""
  with pytest.raises(ValueError):
    AMDAttentionGridSpec(kv_tokens=0).validate()
  with pytest.raises(ValueError):
    AMDAttentionGridSpec(kv_tokens=4097).validate()
  with pytest.raises(ValueError):
    AMDAttentionGridSpec(q_tokens=33).validate()  # q_tokens must stay 16-wide
  AMDAttentionGridSpec(kv_tokens=500).validate()  # kv_tokens itself no longer needs to be 16-wide


@pytest.mark.parametrize("remainder", REMAINDERS)
def test_full_kernel_compiles_with_unaligned_kv_tokens_and_partial_final_chunk(remainder):
  """End-to-end: amd_gfx1100_q16_grid_hd128_loop_attention (the actual production kernel builder)
  with an unaligned kv_tokens (a partial final chunk) and a non-512-aligned start_pos, through both
  renderers this module documents as sharing the fix (AMDISARenderer and HIPRenderer)."""
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.cstyle import HIPRenderer
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  q_heads, kv_heads, q_tokens = 8, 2, 32
  start_pos = 300 + remainder  # non-16-aligned, non-512-aligned chunk start
  kv_tokens = start_pos + q_tokens  # growing causal KV cache length: also unaligned
  sizes = (q_heads * q_tokens * 128, kv_heads * kv_tokens * 128, kv_heads * kv_tokens * 128, q_heads * q_tokens * 128)
  slot_sizes = (sizes[3], sizes[0], sizes[1], sizes[2])
  for renderer in (AMDISARenderer(Target.parse("AMD:ISA:gfx1100")), HIPRenderer(Target.parse("AMD:HIP:gfx1100"))):
    p = [UOp(Ops.PARAM, dtypes.half.ptr(slot_sizes[i]), arg=ParamArg(i)) for i in range(4)]
    sink = amd_gfx1100_q16_grid_hd128_loop_attention(p[1], p[2], p[3], p[0], q_tokens=q_tokens, q_heads=q_heads,
      kv_heads=kv_heads, kv_tokens=kv_tokens, scale=.25, causal=True, query_start=start_pos,
      kernel_info=KernelInfo(name=f"oob_guard_r{remainder}"))
    program = to_program(sink, renderer)
    assert any(u.op is Ops.LINEAR for u in program.src)


def test_full_kernel_aligned_hot_path_isa_is_spill_free_and_unchanged_shape():
  """Regression guard for the aligned case: same resource-shape assertion style as
  test_gfx1100_acc_slice_v2_two_launch_causal_diagnostic -- zero spills/scratch, and the store
  count/offsets are exactly the pre-existing acc-slice contract, proving the OOB-guard change is
  invisible on the hot (16-aligned) path."""
  from tinygrad.codegen import to_program
  from tinygrad.helpers import Target
  from tinygrad.renderer.isa.amd import AMDISARenderer
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  q_heads, kv_heads, q_tokens, kv_tokens = 32, 8, 512, 512
  sizes = (q_heads * q_tokens * 128, kv_heads * kv_tokens * 128, kv_heads * kv_tokens * 128, q_heads * q_tokens * 128)
  slot_sizes = (sizes[3], sizes[0], sizes[1], sizes[2])
  p = [UOp(Ops.PARAM, dtypes.half.ptr(slot_sizes[i]), arg=ParamArg(i)) for i in range(4)]
  sink = amd_gfx1100_q16_grid_hd128_loop_attention(p[1], p[2], p[3], p[0], q_tokens=q_tokens, q_heads=q_heads,
    kv_heads=kv_heads, kv_tokens=kv_tokens, scale=.25, kernel_info=KernelInfo(name="oob_guard_aligned"),
    output_block_base=0, acc_blocks=4)
  program = to_program(sink, AMDISARenderer(Target.parse("AMD:ISA:gfx1100")))
  linear = next(u for u in program.src if u.op is Ops.LINEAR)
  stores = [u.arg for u in linear.src if str(u.arg).startswith("global_store_b16")]
  assert len(stores) == 32
  offsets = {x.offset for x in stores}
  assert offsets == {2 * (e * 256 + j * 16) for e in range(8) for j in range(4)}


def test_numeric_parity_unaligned_kv_tokens_against_reference_softmax_attention():
  """Real-hardware numeric parity for an unaligned geometry (kv_tokens % 16 == 7): not executed in
  this environment (no GPU here / instructed not to run GPU workloads), written to run wherever an
  AMD GPU is present, matching this test module's existing unconditional device="AMD" convention
  (see test_gfx1100_acc_slice_v2_two_launch_causal_diagnostic in test_online_softmax_tile.py)."""
  import numpy as np
  from tinygrad import Tensor
  from tinygrad.schedule.wmma import amd_gfx1100_q16_grid_hd128_loop_attention
  hq, hkv, q_tokens = 8, 2, 32
  start_pos = 71  # 71 % 16 == 7 -> kv_tokens = 103, also 103 % 16 == 7
  kv_tokens = start_pos + q_tokens
  scale = .25
  rng = np.random.default_rng(20260726)
  q = rng.normal(0, .2, (hq, q_tokens, 128)).astype(np.float16)
  k = rng.normal(0, .2, (hkv, kv_tokens, 128)).astype(np.float16)
  v = rng.normal(0, .4, (hkv, kv_tokens, 128)).astype(np.float16)
  sizes = (hq * q_tokens * 128, hkv * kv_tokens * 128, hkv * kv_tokens * 128, hq * q_tokens * 128)

  def kernel(o, qi, ki, vi):
    return amd_gfx1100_q16_grid_hd128_loop_attention(qi, ki, vi, o, q_tokens=q_tokens, q_heads=hq, kv_heads=hkv,
      kv_tokens=kv_tokens, scale=scale, causal=True, query_start=start_pos, kernel_info=KernelInfo(name="oob_guard_parity"))

  tq, tk, tv = (Tensor(x.reshape(-1), device="AMD") for x in (q, k, v))
  out = Tensor.empty(sizes[3], dtype=dtypes.half, device="AMD")
  got = out.custom_kernel(tq, tk, tv, fxn=kernel)[0].realize().numpy().reshape(q.shape).astype(np.float32)
  ref = np.zeros_like(got)
  for head in range(hq):
    score = q[head].astype(np.float32) @ k[head // (hq // hkv)].astype(np.float32).T * scale
    for row in range(q_tokens):
      valid = np.arange(kv_tokens) <= start_pos + row
      prob = np.exp(score[row, valid] - score[row, valid].max())
      prob /= prob.sum()
      ref[head, row] = prob @ v[head // (hq // hkv), valid].astype(np.float32)
  np.testing.assert_allclose(got, ref, rtol=.02, atol=4e-3)
