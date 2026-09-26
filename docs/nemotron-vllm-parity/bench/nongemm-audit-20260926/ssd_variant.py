"""Scratch copy of tinygrad/llm/nemotron_h_ssd.py:ssd_scan with MAT=1 materializing the decay-scaled operands.
Audit experiment only (not product code)."""
import os
from tinygrad import Tensor, dtypes
from tinygrad.llm.nemotron_h_ssd import _side, _dot, _segment_sums
import tinygrad.llm.nemotron_h_ssd as prod
MAT = int(os.environ.get("MAT", "0"))
def ssd_scan(x_dt: Tensor, b: Tensor, c: Tensor, log_a: Tensor, state: Tensor, *, chunk: int = 256,
             precision: str = "float") -> tuple[Tensor, Tensor]:
  """The recurrence ``h_t = exp(log_a_t) h_{t-1} + x_dt_t b_t^T``, ``y_t = h_t c_t``, over a whole piece.

  x_dt: ``(batch, L, heads, P)``, already scaled by dt; b, c: ``(batch, L, groups, N)`` (heads are
  grouped contiguously, ``head = group * (heads // groups) + r``); log_a: ``(batch, L, heads)``;
  state: ``(batch, heads, P, N)``. Returns y ``(batch, L, heads, P)`` float32 and the final state.
  """
  batch, length, heads, width = x_dt.shape
  groups, size = b.shape[2], b.shape[3]
  repeats = heads // groups
  chunk = min(chunk, length)
  chunks = -(-length // chunk)
  pad = chunks * chunk - length
  x_dt, b, c, log_a = x_dt.float(), b.float(), c.float(), log_a.float()
  if pad:  # zero steps: decay exp(0) = 1 and no input, so the final state is the state at `length`
    x_dt = x_dt.pad((None, (0, pad), None, None))
    b, c = b.pad((None, (0, pad), None, None)), c.pad((None, (0, pad), None, None))
    log_a = log_a.pad((None, (0, pad), None))

  # (batch, heads, chunks, chunk, P) and (batch, groups, chunks, chunk, N)
  xs = x_dt.reshape(batch, chunks, chunk, heads, width).permute(0, 3, 1, 2, 4)
  bs = b.reshape(batch, chunks, chunk, groups, size).permute(0, 3, 1, 2, 4)
  cs = c.reshape(batch, chunks, chunk, groups, size).permute(0, 3, 1, 2, 4)
  cumulative = log_a.reshape(batch, chunks, chunk, heads).permute(0, 3, 1, 2).cumsum(-1).contiguous()  # (batch, heads, chunks, chunk)

  def per_head(value: Tensor) -> Tensor:  # (batch, groups, ...) -> (batch, heads, ...)
    rest = value.shape[2:]
    return value.unsqueeze(2).expand(batch, groups, repeats, *rest).reshape(batch, heads, *rest)

  # 1. intra-chunk
  positions = Tensor.arange(chunk)
  causal = positions.reshape(chunk, 1) >= positions.reshape(1, chunk)
  decay = causal.where(cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2), float("-inf")).exp()
  x_side = _side(xs, precision, "hlh", -2)  # (batch, heads, chunks, Q', P), Q' = Q or 3Q
  c_side = _side(cs, precision, "hlh", -1)  # (batch, groups, chunks, Q, N')
  scores = _dot(c_side, _side(bs, precision, "hhl", -1).transpose(-1, -2)).contiguous()  # (batch, groups, chunks, Q, Q)
  weights = per_head(scores) * decay
  if MAT: weights = (weights.cast(dtypes.bfloat16) if precision == 'bf16' else weights).contiguous()
  diagonal = _dot(_side(weights, precision, "hhl", -1, materialize=False), x_side).contiguous()  # (batch, heads, chunks, Q, P)

  # 2. chunk states, all chunks at once
  last = cumulative[..., -1:]
  to_end = (last - cumulative).exp()  # (batch, heads, chunks, Q)
  scaled_b = per_head(bs) * to_end.unsqueeze(-1)
  if MAT: scaled_b = (scaled_b.cast(dtypes.bfloat16) if precision == 'bf16' else scaled_b).contiguous()
  local = _dot(x_side.transpose(-1, -2), _side(scaled_b, precision, "hhl", -2)).contiguous()
  # local: (batch, heads, chunks, P, N)

  # 3. state passing: entering[c] = sum_{j<c} exp(A[j+1..c-1]) local[j] + exp(A[0..c-1]) state
  totals = last.squeeze(-1)  # (batch, heads, chunks)
  # prepend the initial state as chunk -1 with zero decay, then segment sums over chunks + 1 entries
  padded_totals = totals.pad((None, None, (1, 0)))
  carry = _segment_sums(padded_totals).exp().contiguous()  # [c, j'] = exp(sum totals[j' .. c-1])
  # entering[c] = sum_{j'<=c} carry[c, j'] source[j'] for c in 0..chunks (c = chunks is the final state),
  # with source[0] the initial state and source[j'] = local[j'-1]
  sources = state.float().reshape(batch, heads, 1, width * size).cat(
    local.reshape(batch, heads, chunks, width * size), dim=2)  # (batch, heads, chunks + 1, P*N)
  entering = (carry.float() @ sources).reshape(batch, heads, chunks + 1, width, size).contiguous()
  final = entering[:, :, -1]
  entering = entering[:, :, :-1]  # (batch, heads, chunks, P, N)

  # 4. output from the entering state: C is per group, so one matmul per group against its heads' states
  # side by side, (Q, N) @ (N, repeats * P), instead of per head against an expanded C
  grouped = entering.reshape(batch, groups, repeats, chunks, width, size).permute(0, 1, 3, 5, 2, 4)
  grouped = grouped.reshape(batch, groups, chunks, size, repeats * width)
  previous = _dot(c_side, _side(grouped, precision, "hhl", -2)).contiguous().reshape(batch, groups, chunks, chunk, repeats, width)
  previous = previous.permute(0, 1, 4, 2, 3, 5).reshape(batch, heads, chunks, chunk, width)
  previous = previous * cumulative.exp().unsqueeze(-1)
  y = (diagonal + previous).permute(0, 2, 3, 1, 4).reshape(batch, chunks * chunk, heads, width)
  return y[:, :length], final



if MAT: prod.ssd_scan = ssd_scan
