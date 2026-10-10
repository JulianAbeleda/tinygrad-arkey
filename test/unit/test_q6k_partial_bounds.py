"""K1/K2: the Q6_K partial route never reads past its weight or x buffer.

The PYTHON device raises on any out-of-bounds load, so it checks the loads themselves, not only the sums."""
import numpy as np, pytest
from tinygrad import Tensor, dtypes
from tinygrad.llm.decode_kernels import Q6KGEMVRouteSpec, _emit_q6k_partial
from extra.llm_research import decode_emitter_space as space


def _spec(rows, k, parts):
  return Q6KGEMVRouteSpec(rows=rows, k=k, role="test", route_family="q6k_partial", parts=parts, pos_axis="reduce")


@pytest.mark.parametrize("k_blocks,parts", [(5, 4), (10, 4), (3, 2), (4, 4)])
def test_ragged_parts_read_only_inside_the_buffers(k_blocks, parts):
  rows, k = 2, k_blocks * 256
  raw = space.fixture_bytes("Q6_K", rows, k)
  halfs = Tensor(raw.view(np.uint16), dtype=dtypes.uint16, device="PYTHON").realize()
  x = Tensor(space.fixture_activation(k), dtype=dtypes.float16, device="PYTHON").realize()
  out = Tensor.empty(rows, parts, dtype=dtypes.float32, device="PYTHON").uop_program(halfs, x, fxn=_emit_q6k_partial(_spec(rows, k, parts)))[0]
  got = out.numpy().sum(axis=1)  # an out-of-bounds load raises IndexError here
  want = space.reference("Q6_K", raw, rows * k).reshape(rows, k).astype(np.float32) @ space.fixture_activation(k).astype(np.float32)
  np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_more_parts_than_blocks_is_refused():
  _spec(1, 2 * 256, 2).validate()
  with pytest.raises(ValueError, match="parts"): _spec(1, 2 * 256, 3).validate()


def test_the_attn_v_default_never_asks_for_more_parts_than_blocks():
  from tinygrad.llm.model_route_plan import _default
  assert _default("blk.0.attn_v.weight", "Q6_K", "attn_v", 1024, 4096)[0] == 4
  for cols in (256, 512, 768):
    parts = _default("blk.0.attn_v.weight", "Q6_K", "attn_v", 64, cols)[0]
    assert parts == cols // 256
    _spec(64, cols, parts).validate()
