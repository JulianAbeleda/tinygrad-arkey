"""P1 vocab-head aux scatter-chain fusion gate (nv-vocab-aux-chain-fusion-scope-20260812.md).

The four top-1 aux kernels (E_1187_32_4, r_32_4_1187, r_128_16_8_1187, r_16_8) reduce the
151936-row logits to one token id.  The fused P1 design carries a packed u64 (max, index)
key per GEMV warp tile in the vocab_top1 epilogue, warp-reduces those keys in-kernel, and
finishes with one tiny u64 MAX over the per-tile keys.  These tests pin the tie semantics
of today's r_16_8 chain (first index wins), prove the ordinary-UOp fused arithmetic and the
emitted final-reduce kernel are bit-identical with the legacy argmax, and check the emitted
vocab_top1 epilogue contains no float->int CAST anywhere in the max+idx compare.
"""
import numpy as np
import pytest

from tinygrad import Tensor, dtypes
from tinygrad.codegen import to_program
from tinygrad.helpers import Target
from tinygrad.llm import decode_routes
from tinygrad.llm.decode_kernels import (Q6KGEMVRouteSpec, emit_q6k_gemv_kernel,
                                         q6k_spec_for_role)
from tinygrad.llm.qk_primitives import QKPrimitiveCapability, QKPrimitiveRouteAdmission
from tinygrad.llm.packed_argmax import (packed_argmax_from_tile_keys, packed_argmax_tile_keys_fp32,
                                        packed_argmax_tiles_fp32)
from tinygrad.renderer.cuda import CUDARenderer
from tinygrad.uop.ops import Ops, UOp

VOCAB_ROWS, VOCAB_K = 151936, 4096


def _tie_logits(n=VOCAB_ROWS) -> np.ndarray:
  """Max value tied across the head, middle, and tail, plus signed-zero ties."""
  a = np.full((1, n), -100.0, dtype=np.float32)
  a[0, 0], a[0, 1000], a[0, n - 1] = 3.0, 3.0, 3.0
  a[0, 5], a[0, 6] = -0.0, 0.0
  return a


def _fused_token_id(logits: Tensor) -> int:
  """The two-stage fused top-1 through the ordinary-UOp graph (per-tile keys + final MAX)."""
  return int(packed_argmax_tiles_fp32(logits, 2, keepdim=True).numpy().ravel()[0])


def test_vocab_top1_tie_semantics_first_index_wins_over_vocab_shape():
  """Pin today's r_16_8 behavior: over the 151936-row shape the argmax keeps the FIRST index."""
  x = Tensor(_tie_logits())
  legacy = x.argmax(-1, keepdim=True).numpy().ravel()
  assert legacy.tolist() == [0]  # 3.0 ties at row 0, 1000, n-1 -> first wins; signed zero -> row 5
  assert int(np.argmax(_tie_logits(), axis=-1).item()) == 0
  assert _fused_token_id(x) == 0
  keys = packed_argmax_tile_keys_fp32(x, 2)
  assert packed_argmax_from_tile_keys(keys, VOCAB_ROWS).numpy().ravel().tolist() == [0]


@pytest.mark.parametrize("kind", ["normal", "allzero", "ties"])
def test_vocab_top1_fused_matches_legacy_argmax(kind):
  rng = np.random.default_rng(1234)
  if kind == "normal":
    a = rng.standard_normal((1, VOCAB_ROWS)).astype(np.float32)
  elif kind == "allzero":
    a = np.zeros((1, VOCAB_ROWS), dtype=np.float32)
  else:
    a = _tie_logits()
  x = Tensor(a)
  legacy = x.argmax(-1, keepdim=True).numpy().ravel()
  assert _fused_token_id(x) == int(legacy[0])
  keys = packed_argmax_tile_keys_fp32(x, 2)
  assert packed_argmax_from_tile_keys(keys, VOCAB_ROWS).numpy().ravel().tolist() == legacy.tolist()


def _emit_vocab_top1_spec(rows=32, k=256):
  return q6k_spec_for_role(rows, k, row_tile=2, reduction="in_kernel", epilogue="vocab_top1")


def test_vocab_top1_coop_kernel_emits_and_render_is_integer_only():
  """The vocab_top1 epilogue emits the coop GEMV with a packed-key store and renders with a
  warp-wide MAX ladder over the u64 key; the max+idx compare is BITCAST + integer ops only."""
  spec = _emit_vocab_top1_spec()
  assert spec.kernel_name == "q6k_gen_coop_32_256_inkernel_epi_vocabtop1"
  kernel = emit_q6k_gemv_kernel(spec)(
    UOp.placeholder((spec.rows // spec.row_tile,), dtypes.uint64, 0),
    UOp.placeholder((spec.rows * spec.k_blocks * 105,), dtypes.uint16, 1),
    UOp.placeholder((spec.k,), dtypes.float16, 2))
  assert kernel.arg.name == spec.kernel_name
  # Structural walk: from the packed-key store value backwards, never descending into
  # float-typed nodes (the fp32 logit total and the signed-zero 0.0 are boundary markers;
  # the GEMV's own fp16->fp32 input casts live upstream of them).  Inside that integer-only
  # epilogue/reduce path any float->int Ops.CAST would let float rounding reorder ties, so
  # the only float conversion allowed is the fp32->u32 BITCAST of the logit total.
  stores = [u for u in kernel.toposort()
            if u.op is Ops.STORE and u.src[1].dtype == dtypes.uint64 and u.src[0].op is Ops.INDEX
            and u.src[0].src[0].op is Ops.PARAM and u.src[0].src[0].dtype.base == dtypes.uint64]
  assert len(stores) == 1
  seen, stack = set(), [stores[0].src[1]]
  while stack:
    u = stack.pop()
    if u in seen: continue
    seen.add(u)
    if not dtypes.is_float(u.dtype): stack.extend(u.src)
  float_casts = [u for u in seen if u.op is Ops.CAST and dtypes.is_float(u.src[0].dtype)]
  assert float_casts == [], f"float cast in max+idx compare: {float_casts}"
  assert any(u.op is Ops.BITCAST and u.dtype == dtypes.uint32 and u.src[0].dtype == dtypes.float32 for u in seen)
  assert any(u.op is Ops.MAX and u.dtype == dtypes.uint64 for u in seen)
  ren = CUDARenderer(Target("NV", arch="sm_120"), use_nvcc=False)
  src = next(u.arg for u in to_program(kernel, ren).src if u.op is Ops.SOURCE)
  assert "__shfl_xor_sync" in src
  i = src.rfind("tg_bitcast<unsigned int>")
  assert i >= 0 and "__shfl_xor_sync" in src[i:]  # epilogue then the warp MAX ladder


def test_vocab_top1_route_validation_fail_closed():
  with pytest.raises(ValueError, match="in_kernel"):
    Q6KGEMVRouteSpec(rows=32, k=256, row_tile=2, reduction="external_sum", epilogue="vocab_top1").validate()
  with pytest.raises(ValueError, match="q6k_partial"):
    Q6KGEMVRouteSpec(rows=32, k=256, row_tile=2, parts=1, reduction="in_kernel",
                     route_family="q6k_partial", epilogue="vocab_top1", pos_axis="reduce").validate()


def test_vocab_top1_call_finishes_with_ordinary_packed_reduce(monkeypatch):
  """The fused route emits one GEMV program and finishes the cross-tile reduce with
  packed_argmax_from_tile_keys, not the single-thread custom reduce kernel.  The old
  custom path serialized rows/row_tile keys in one thread and regressed the wall."""
  capability = QKPrimitiveCapability(backend="NV", architecture="sm_120",
                                     wave_size=32, supports_warp_shfl_xor=True)

  class _FakeQ6VocabLinear:
    def __init__(self):
      self.q6k_storage = type("_FakeQ6Storage", (), {"halfs": Tensor.zeros(8, dtype=dtypes.uint16)})()
      self.decode_enabled, self.bias = True, None
      self.in_features, self.out_features, self.parts = VOCAB_K, VOCAB_ROWS, 1
      self.opts, self.route_role = (), "lm_head"
      self.route_admission = QKPrimitiveRouteAdmission(capability, True)

  linear = _FakeQ6VocabLinear()
  captured = []
  emitted_keys = []
  winning_row = 1000

  def fake_execute_research_program(output, *inputs, program):
    captured.append(program)
    assert output.shape == (VOCAB_ROWS // 2,)
    assert output.dtype == dtypes.uint64
    keys = np.zeros(VOCAB_ROWS // 2, dtype=np.uint64)
    keys[winning_row // 2] = (np.uint64(1) << 32) | np.uint64(VOCAB_ROWS - 1 - winning_row)
    emitted_keys.append(Tensor(keys))
    return emitted_keys[-1]

  monkeypatch.setattr(decode_routes, "execute_research_program", fake_execute_research_program)
  x = Tensor.zeros((1, 1, VOCAB_K), dtype=dtypes.float16)
  token = decode_routes.q6k_vocab_top1_call(linear, x, True)
  assert token is not None and token.shape == (1, 1)
  assert int(token.numpy().ravel()[0]) == winning_row
  assert len(captured) == 1
  assert captured[0].program_id == f"{decode_routes.Q6K_DECODE_CANDIDATE.candidate_id}.vocab_top1_gemv"
  assert token.uop.buf_uop is not emitted_keys[0].uop.buf_uop


def test_vocab_top1_call_warms_l2_with_keys_copy_before_reduce(monkeypatch):
  """The fused route copies the packed keys out of the vocab GEMV output before the
  cross-tile reduce.  The 2026-08-17 measurement (nv-vocab-reduce-l2-mechanism) showed
  the single-block u64 reduce reads the GEMV's keys L2-cold after the ~510 MB weight
  stream (85 us vs 44 us L2-warm); the copy re-warms L2 (the legacy E_1187_32_4 role)
  and flips the fused tail from +25.8 us slower to ~-11 us faster."""
  capability = QKPrimitiveCapability(backend="NV", architecture="sm_120",
                                     wave_size=32, supports_warp_shfl_xor=True)

  class _FakeQ6VocabLinear:
    def __init__(self):
      self.q6k_storage = type("_FakeQ6Storage", (), {"halfs": Tensor.zeros(8, dtype=dtypes.uint16)})()
      self.decode_enabled, self.bias = True, None
      self.in_features, self.out_features, self.parts = VOCAB_K, VOCAB_ROWS, 1
      self.opts, self.route_role = (), "lm_head"
      self.route_admission = QKPrimitiveRouteAdmission(capability, True)

  linear = _FakeQ6VocabLinear()
  emitted_keys = []
  winning_row = 1000

  def fake_execute_research_program(output, *inputs, program):
    keys = np.zeros(VOCAB_ROWS // 2, dtype=np.uint64)
    keys[winning_row // 2] = (np.uint64(1) << 32) | np.uint64(VOCAB_ROWS - 1 - winning_row)
    emitted_keys.append(Tensor(keys))
    return emitted_keys[-1]

  monkeypatch.setattr(decode_routes, "execute_research_program", fake_execute_research_program)
  x = Tensor.zeros((1, 1, VOCAB_K), dtype=dtypes.float16)
  token = decode_routes.q6k_vocab_top1_call(linear, x, True)

  # The reduce must consume a copied buffer, not the raw vocab-GEMV key output: walk
  # the returned graph for the STORE that copies the emitted keys into a fresh buffer
  # (clone lowers to STORE(dst, COPY(keys)) with the value in slot 1).
  keys_uop = emitted_keys[0].uop
  copied_key_bufs = set()
  for u in token.uop.toposort():
    if u.op is Ops.STORE and u.src[1] is keys_uop:
      copied_key_bufs.add(u.buf_uop)
  assert len(copied_key_bufs) == 1, f"expected one keys warm-up copy store, got {len(copied_key_bufs)}"
  # The warm-up copy must land in a distinct buffer from the GEMV output.
  assert next(iter(copied_key_bufs)) is not keys_uop.buf_uop
  assert token is not None and int(token.numpy().ravel()[0]) == winning_row
