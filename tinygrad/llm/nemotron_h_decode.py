"""One-token Mamba-2 decode update for Nemotron-H.

``NemotronHBlock.cached`` is the general chunked path: for a single token it
still builds the cumsum, the causal mask and the chunk contractions, and the
sampler then realizes the new state and copies it into its fixed buffer.  At
batch 128 the fp32 state is about 82.6 MB per sequence, so every extra pass
over it costs more than all the rest of the Mamba block.

This module writes the recurrence for exactly one token per sequence:

  conv  = silu(sum_k window[k] * w[:, k] + bias),  window = (conv_tail, new)
  dt    = softplus(dt + dt_bias),  dA = exp(dt * A)
  state = state * dA + (dt * x) (outer) B        (assigned in place)
  y     = state . C + D * x

followed by the same gated group RMSNorm and ``ssm_out`` projection as
``cached``, with the same dtype policy (float32 scan, ``ssm_out`` through the
block's own Linear).  The residual add and the block's input norm are included,
so ``mamba_decode_step`` is a drop-in for ``block.cached(hidden, cache,
keep_graph=True)`` on a length-1 input.
"""
from __future__ import annotations

from tinygrad import Tensor


def mamba_decode_step(block, hidden: Tensor, conv: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """One token of one Nemotron-H Mamba block, batch B.

  hidden: (B, 1, dim) residual stream.  conv: (B, K-1, conv_dim) raw pre-conv
  tail, as ``cached`` stores it.  state: (B, heads, head_dim, state) float32.
  ``conv`` and ``state`` are updated in place through ``assign`` (they must be
  buffers); the returned tensors are those same, now-assigned, tensors.  The
  caller realizes ``out`` together with them (``Tensor.realize(out, conv, state)``).
  """
  cfg = block.config
  batch = hidden.shape[0]
  heads, head_dim, groups, width = cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads, cfg.ssm_groups, cfg.ssm_state
  conv_dim = cfg.ssm_inner + 2 * groups * width
  normalized = block.attn_norm(hidden)
  projected = block.ssm_in(normalized)
  gate, raw_xbc, dt = projected.split((cfg.ssm_inner, conv_dim, heads), dim=-1)

  # causal conv for one token: a K-tap dot over (tail, new), then SiLU
  window = conv.cat(raw_xbc.cast(conv.dtype), dim=1)                    # (B, K, conv_dim)
  xbc = ((window * block.ssm_conv1d.weight.T).sum(axis=1) + block.ssm_conv1d.bias).silu()  # (B, conv_dim)
  new_tail = window[:, 1:]

  x, b, c = xbc.split((cfg.ssm_inner, groups * width, groups * width), dim=-1)
  x = x.reshape(batch, heads, head_dim).float()
  repeats = heads // groups
  b = b.reshape(batch, groups, 1, width).expand(batch, groups, repeats, width).reshape(batch, heads, 1, width).float()
  c = c.reshape(batch, groups, 1, width).expand(batch, groups, repeats, width).reshape(batch, heads, 1, width).float()
  dt = (dt.reshape(batch, heads) + block.ssm_dt["bias"]).softplus().float()
  decay = (dt * block.ssm_a.reshape(heads).float()).exp().reshape(batch, heads, 1, 1)
  x_dt = (x * dt.unsqueeze(-1)).unsqueeze(-1)                            # (B, heads, head_dim, 1)

  # The state update must read only the state and one small packed operand.
  # The scheduler keeps (bufferizes) an assign source that reads more than
  # three buffers (`remove_bufferize` in schedule/rangeify.py), and a kept
  # source over the state costs a full temporary write plus a copy back.
  pack = x_dt.reshape(batch, heads, head_dim).cat(decay.reshape(batch, heads, 1), b.reshape(batch, heads, width),
                                                  dim=-1).contiguous()
  x_dt = pack[:, :, :head_dim].reshape(batch, heads, head_dim, 1)
  decay = pack[:, :, head_dim:head_dim + 1].reshape(batch, heads, 1, 1)
  b = pack[:, :, head_dim + 1:].reshape(batch, heads, 1, width)

  # the conv tail is read by `window` above, so stage it before overwriting
  conv.assign(new_tail.contiguous())
  state.assign(state * decay + x_dt * b)
  y = (state * c).sum(axis=-1)                                          # (B, heads, head_dim)
  y = y + x * block.ssm_d.reshape(1, heads, 1)

  y = y.reshape(batch, 1, groups, cfg.ssm_inner // groups)
  gated = y * gate.reshape(y.shape).silu()
  normed = gated * (gated.square().mean(axis=-1, keepdim=True) + cfg.norm_eps).rsqrt()
  normed = normed.reshape(batch, 1, cfg.ssm_inner) * block.ssm_norm.weight.reshape(cfg.ssm_inner)
  mixed = block.ssm_out(normed.cast(normalized.dtype))
  return hidden + mixed.cast(hidden.dtype), conv, state


def mamba_decode_buffers(block, hidden: Tensor, buffer: dict) -> Tensor:
  """Advance a sampler's fixed ``{"conv", "state"}`` buffers in place; returns the block output.

  Realizes the output and both writes in one schedule, so under TinyJit the
  step captures only kernels over the same fixed buffers.
  """
  out, conv, state = mamba_decode_step(block, hidden, buffer["conv"], buffer["state"])
  Tensor.realize(out, conv, state)
  return out


__all__ = ["mamba_decode_step", "mamba_decode_buffers"]
