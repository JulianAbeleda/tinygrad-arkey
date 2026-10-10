"""_emit_q6k_partial covers every block when the parts do not divide the blocks (a ceiling, not a truncation)."""
import numpy as np
from tinygrad import Tensor, dtypes, Device
from tinygrad.llm.decode_kernels import Q6KGEMVRouteSpec, _emit_q6k_partial
from extra.llm_research import decode_emitter_space as space


def _run(rows, k, parts):
  spec = Q6KGEMVRouteSpec(rows=rows, k=k, role="test", route_family="q6k_partial", parts=parts, pos_axis="reduce")
  raw = space.fixture_bytes("Q6_K", rows, k)
  halfs = Tensor(raw.view(np.uint16), dtype=dtypes.uint16).realize()
  x = Tensor(space.fixture_activation(k), dtype=dtypes.float16).realize()
  out = Tensor.empty(rows, parts, dtype=dtypes.float32).uop_program(halfs, x, fxn=_emit_q6k_partial(spec))[0]
  got = out.numpy().sum(axis=1)
  want = space.reference("Q6_K", raw, rows * k).reshape(rows, k).astype(np.float32) @ space.fixture_activation(k).astype(np.float32)
  return got, want


def test_parts_that_do_not_divide_the_blocks_still_cover_all_of_k():
  for parts in (1, 2, 4):  # 5 blocks: 4 parts used to read 4 of them
    got, want = _run(2, 5 * 256, parts)
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)
