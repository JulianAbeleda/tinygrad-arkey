"""Chunked state-space-duality (SSD) scan for Nemotron-H Mamba-2 prefill.

`NemotronHMamba2.cached` scans a prompt in `scan_chunk`-token pieces, each
piece's state feeding the next, so a piece of L tokens is L/64 dependent steps.
Here the whole piece is one fixed graph of batched matmuls (the Mamba-2 SSD
algorithm, as vLLM's `ssd_*` Triton kernels run it), with no loop over chunks:

1. intra-chunk: per group, ``C @ B^T`` over each Q-token chunk, masked by the
   per-head decay ``exp(cumA[t] - cumA[s])`` (s <= t), times ``x * dt``;
2. chunk states: ``(x * dt)^T @ (B * exp(cumA[end] - cumA[s]))``, all chunks at once;
3. state passing: the state entering chunk c is a decay-weighted sum of the
   earlier chunk states plus the carried initial state; with at most a few dozen
   chunks this is one ``(chunks x chunks) @ (chunks x P*N)`` matmul per head
   instead of a sequential loop;
4. output: ``C @ state_in^T * exp(cumA[t])`` plus the intra-chunk part.

Decays stay float32. The three large matmuls take an operand precision:
``"float"`` (float32, matches `cached` to float32 noise), ``"split"`` (each
float32 operand as bf16 hi + lo, three bf16 x bf16 -> float32 products, about
2**-16 relative, on the tensor core) or ``"bf16"`` (one rounded product).
"""
from __future__ import annotations

from tinygrad import Tensor, dtypes

PRECISIONS = ("float", "split", "bf16")


def _side(value: Tensor, precision: str, order: str, dim: int, materialize: bool = True) -> Tensor:
  """One matmul operand at the chosen precision, contracted along `dim`.

  "split" writes the operand as bf16 hi = bf16(v) and lo = bf16(v - hi), concatenated along the
  contracted axis in `order`: an "hlh" operand against an "hhl" one gives hi*hi + lo*hi + hi*lo,
  about 2**-16 relative, as one bf16 x bf16 -> float32 tensor-core matmul over a threefold K.
  Inputs (x, C) are "hlh" so one split copy serves both matmuls that read them.
  """
  if precision == "float":
    return value.float()
  hi = value.cast(dtypes.bfloat16)
  if precision == "bf16":
    return hi
  if precision != "split":
    raise ValueError(f"precision must be one of {PRECISIONS}")
  parts = {"h": hi, "l": (value.float() - hi.float()).cast(dtypes.bfloat16)}
  out = parts[order[0]].cat(*(parts[o] for o in order[1:]), dim=dim)
  return out.contiguous() if materialize else out


def _dot(a: Tensor, b: Tensor) -> Tensor:
  return a.matmul(b, dtype=dtypes.float)


def _segment_sums(values: Tensor) -> Tensor:
  """``out[..., i, j] = sum(values[..., j+1 : i+1])`` for j <= i, -inf above the diagonal.

  Summed directly, not as a difference of cumulative sums, so long segments keep full precision.
  """
  count = values.shape[-1]
  rows = Tensor.arange(count).reshape(count, 1)
  cols = Tensor.arange(count).reshape(1, count)
  spread = values.unsqueeze(-2).expand(*values.shape[:-1], count, count)  # [..., j, k] = values[k]
  kept = (cols > rows).where(spread, 0.0).cumsum(-1)
  return (cols <= rows).where(kept.transpose(-1, -2), float("-inf"))


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
  diagonal = _dot(_side(weights, precision, "hhl", -1, materialize=False), x_side).contiguous()  # (batch, heads, chunks, Q, P)

  # 2. chunk states, all chunks at once
  last = cumulative[..., -1:]
  to_end = (last - cumulative).exp()  # (batch, heads, chunks, Q)
  local = _dot(x_side.transpose(-1, -2), _side(per_head(bs) * to_end.unsqueeze(-1), precision, "hhl", -2)).contiguous()
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


def ssd_mixer(block, hidden: Tensor, cache: dict | None = None, *, chunk: int = 256,
              precision: str = "float", valid=None) -> tuple[Tensor, dict]:
  """`NemotronHMamba2.cached(hidden, cache, keep_graph=True)` with the scan as one SSD graph.

  `valid` (an int or bound variable, with a `cache`): only the first `valid` positions are the sequence, the rest
  padding. Padded positions get x*dt = 0 and a decay of exp(0) = 1, so the returned state is the state after
  `valid` tokens and the convolution tail ends at `valid`; outputs at positions before `valid` never read later
  positions (causal), so they do not depend on the padding either.
  """
  from tinygrad.llm.nemotron_h import NemotronHMamba2
  cfg = block.config
  batch, length, _ = hidden.shape
  projected = block.ssm_in(hidden)
  conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state
  gate, raw_xbc, dt = projected.split((cfg.ssm_inner, conv_dim, cfg.ssm_heads), dim=-1)
  previous_conv = None if cache is None else cache["conv"]
  combined = raw_xbc if previous_conv is None else previous_conv.cat(raw_xbc, dim=1)
  xbc = NemotronHMamba2._causal_conv(block, combined)[:, -length:]
  if valid is None:
    conv_tail = combined[:, -(cfg.conv_kernel - 1):]
  elif previous_conv is None:
    raise ValueError("a padded piece needs the carried convolution tail")
  else:  # combined row k-1+i is position i: the tail is rows [valid, valid + k - 1)
    conv_tail = combined[:, valid:valid + cfg.conv_kernel - 1]
  x, b, c = xbc.split((cfg.ssm_inner, cfg.ssm_groups * cfg.ssm_state, cfg.ssm_groups * cfg.ssm_state), dim=-1)
  width = cfg.ssm_inner // cfg.ssm_heads
  x = x.reshape(batch, length, cfg.ssm_heads, width).float()
  b = b.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
  c = c.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
  dt = (dt + block.ssm_dt["bias"]).softplus().float()
  log_a = dt * block.ssm_a.reshape(cfg.ssm_heads).float()
  x_dt = x * dt.unsqueeze(-1)
  if valid is not None:
    keep = Tensor.arange(length).reshape(1, length, 1) < valid
    x_dt, log_a = keep.unsqueeze(-1).where(x_dt, 0.0), keep.where(log_a, 0.0)
  state = cache["state"] if cache is not None else Tensor.zeros(batch, cfg.ssm_heads, width, cfg.ssm_state,
                                                                  device=hidden.device)
  y, state = ssd_scan(x_dt, b, c, log_a, state, chunk=chunk, precision=precision)
  y = y + x * block.ssm_d.reshape(1, 1, cfg.ssm_heads, 1)
  y = y.reshape(batch, length, cfg.ssm_groups, cfg.ssm_inner // cfg.ssm_groups)
  gated = y * gate.reshape(y.shape).silu()
  normalized = gated * (gated.square().mean(axis=-1, keepdim=True) + cfg.norm_eps).rsqrt()
  normalized = normalized.reshape(batch, length, cfg.ssm_inner) * block.ssm_norm.weight.reshape(cfg.ssm_inner)
  return block.ssm_out(normalized.cast(hidden.dtype)), {"conv": conv_tail, "state": state}


def ssd_cached(block, hidden: Tensor, cache: dict | None = None, *, chunk: int = 256,
               precision: str = "float", valid=None) -> tuple[Tensor, dict]:
  """A Mamba block's `cached(hidden, cache, keep_graph=True)`: pre-norm, SSD mixer, residual."""
  mixed, next_cache = ssd_mixer(block, block.attn_norm(hidden), cache, chunk=chunk, precision=precision, valid=valid)
  return hidden + mixed.cast(hidden.dtype), next_cache


__all__ = ["PRECISIONS", "ssd_cached", "ssd_mixer", "ssd_scan"]
