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


def _token_inputs(block, hidden: Tensor, conv: Tensor):
  """Input norm, ssm_in and the one-token conv; returns per-token SSM operands (float32)."""
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
  x, b, c = xbc.split((cfg.ssm_inner, groups * width, groups * width), dim=-1)
  x = x.reshape(batch, heads, head_dim).float()
  dt = (dt.reshape(batch, heads) + block.ssm_dt["bias"]).softplus().float()
  log_decay = dt * block.ssm_a.reshape(heads).float()                   # (B, heads)
  return (normalized, gate, x, x * dt.unsqueeze(-1), log_decay,
          b.reshape(batch, groups, width).float(), c.reshape(batch, groups, width).float(), window[:, 1:])


def _heads(value: Tensor, heads: int) -> Tensor:
  """(B, groups, ...) -> (B, heads, ...), each group shared by heads // groups heads."""
  batch, groups, *rest = value.shape
  return value.reshape(batch, groups, 1, *rest).expand(batch, groups, heads // groups, *rest).reshape(batch, heads, *rest)


def _block_output(block, hidden: Tensor, normalized: Tensor, gate: Tensor, x: Tensor, y: Tensor) -> Tensor:
  """D skip, gated group RMSNorm, ssm_out and the residual add, exactly as ``cached``."""
  cfg = block.config
  batch, heads = x.shape[0], cfg.ssm_heads
  y = y + x * block.ssm_d.reshape(1, heads, 1)
  y = y.reshape(batch, 1, cfg.ssm_groups, cfg.ssm_inner // cfg.ssm_groups)
  gated = y * gate.reshape(y.shape).silu()
  normed = gated * (gated.square().mean(axis=-1, keepdim=True) + cfg.norm_eps).rsqrt()
  normed = normed.reshape(batch, 1, cfg.ssm_inner) * block.ssm_norm.weight.reshape(cfg.ssm_inner)
  mixed = block.ssm_out(normed.cast(normalized.dtype))
  return hidden + mixed.cast(hidden.dtype)


def _readout(state: Tensor, c: Tensor) -> Tensor:
  """state (B, heads, head_dim, N) . c (B, heads, N) -> (B, heads, head_dim).

  Read over flat (row, N) rows into its own small buffer. On the 4-D view, or
  fused with the D*x/gate epilogue, the reduce heuristic picks a split-reduce
  schedule at ~0.5 TB/s; flat and unfused it reads the state at ~1.5 TB/s (B=128).
  """
  batch, heads, head_dim, width = state.shape
  rows = batch * heads * head_dim
  y = (state.reshape(rows, width) * c.reshape(batch, heads, 1, width).expand(state.shape).reshape(rows, width)).sum(-1)
  return y.contiguous().reshape(batch, heads, head_dim)


def mamba_decode_step(block, hidden: Tensor, conv: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
  """One token of one Nemotron-H Mamba block, batch B.

  hidden: (B, 1, dim) residual stream.  conv: (B, K-1, conv_dim) raw pre-conv
  tail, as ``cached`` stores it.  state: (B, heads, head_dim, state) float32.
  ``conv`` and ``state`` are updated in place through ``assign`` (they must be
  buffers); the returned tensors are those same, now-assigned, tensors.  The
  caller realizes ``out`` together with them (``Tensor.realize(out, conv, state)``).
  """
  cfg = block.config
  batch, heads, head_dim, width = hidden.shape[0], cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads, cfg.ssm_state
  normalized, gate, x, x_dt, log_decay, b, c, new_tail = _token_inputs(block, hidden, conv)
  # The state update must read only the state and one small packed operand.
  # The scheduler keeps (bufferizes) an assign source that reads more than
  # three buffers (`remove_bufferize` in schedule/rangeify.py), and a kept
  # source over the state costs a full temporary write plus a copy back.
  pack = x_dt.cat(log_decay.exp().reshape(batch, heads, 1), _heads(b, heads), dim=-1).contiguous()
  x_dt = pack[:, :, :head_dim].reshape(batch, heads, head_dim, 1)
  decay = pack[:, :, head_dim:head_dim + 1].reshape(batch, heads, 1, 1)
  b = pack[:, :, head_dim + 1:].reshape(batch, heads, 1, width)
  # the conv tail is read by the conv above, so stage it before overwriting
  conv.assign(new_tail.contiguous())
  state.assign(state * decay + x_dt * b)
  y = _readout(state, _heads(c, heads))
  return _block_output(block, hidden, normalized, gate, x, y), conv, state


def mamba_decode_buffers(block, hidden: Tensor, buffer: dict) -> Tensor:
  """Advance a sampler's fixed ``{"conv", "state"}`` buffers in place; returns the block output.

  Realizes the output and both writes in one schedule, so under TinyJit the
  step captures only kernels over the same fixed buffers.
  """
  out, conv, state = mamba_decode_step(block, hidden, buffer["conv"], buffer["state"])
  Tensor.realize(out, conv, state)
  return out


# ---------------------------------------------------------------------------
# Replay decode (deferred state write-back), the analogue of vLLM's ReplaySSM.
#
# The one-pass step still reads the state twice and writes it once per token
# (update, then y), because the scheduler cannot store the new state and reduce
# y from it in one kernel.  Replay decode leaves the state as a checkpoint that
# is only *read* each step.  A ring of the last `ring` tokens' (x*dt, dt*A, B)
# carries the recent inputs, and with L_k = sum of dt*A since the checkpoint:
#
#   y_t = exp(L_t) (S_ckpt . C_t) + sum_{j<=t} exp(L_t - L_j) (x_j*dt_j) (B_j . C_t)
#
# Every `ring` tokens `mamba_replay_flush` folds the ring into the checkpoint:
#
#   S_ckpt <- exp(L_n) S_ckpt + sum_{j<n} exp(L_n - L_j) (x_j*dt_j) (outer) B_j
#
# Per token that is one state read plus a small ring read; the state read+write
# happens once per `ring` tokens.
# ---------------------------------------------------------------------------

def mamba_replay_buffers(block, conv: Tensor, state: Tensor, ring: int) -> dict:
  """Fresh replay buffers around an existing conv tail and checkpoint state (both copied)."""
  cfg = block.config
  batch, heads, head_dim, width = state.shape
  def fresh(value: Tensor) -> Tensor: return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()
  return {"conv": fresh(conv), "state": fresh(state),
          "ring_x": fresh(Tensor.zeros(batch, heads, head_dim, ring)),  # ring last: flat rows
          "ring_a": fresh(Tensor.zeros(batch, heads, ring)),
          "ring_b": fresh(Tensor.zeros(batch, cfg.ssm_groups, ring, width))}


def _ring_weights(ring_a: Tensor, count, extra: Tensor | None = None) -> tuple[Tensor, Tensor]:
  """Masked ring log decays -> (exp(L_n - L_j) per slot (B, heads, ring), exp(L_n) (B, heads)).

  Slots j < count are valid; ``extra`` (B, heads) is one more decay after them (the current token).
  """
  ring = ring_a.shape[-1]
  valid = Tensor.arange(ring) < count
  a = valid.reshape(1, 1, ring).where(ring_a, 0.0)
  cumulative = a.cumsum(-1)                                             # L_j, inclusive
  total = a.sum(-1) if extra is None else a.sum(-1) + extra             # L_n
  weights = valid.reshape(1, 1, ring).where((total.unsqueeze(-1) - cumulative).exp(), 0.0)
  return weights, total.exp()


def mamba_replay_step(block, hidden: Tensor, buffer: dict, slot) -> Tensor:
  """One token with a read-only checkpoint; writes ring slot `slot` (int or bound UOp variable).

  Slots < `slot` hold the tokens since the last flush. Returns the block output
  (unrealized); the caller realizes it with the buffers (``mamba_replay_buffers_step``).
  """
  cfg = block.config
  heads = cfg.ssm_heads
  normalized, gate, x, x_dt, log_decay, b, c, new_tail = _token_inputs(block, hidden, buffer["conv"])
  batch, _, head_dim = x_dt.shape
  weights, checkpoint_decay = _ring_weights(buffer["ring_a"], slot, log_decay)
  # recent tokens: sum_j w_j (x_j*dt_j) (B_j . C), plus the current token (weight 1)
  # Realize the small per-(sequence, head) coefficients first. Fused, the
  # B.C dots and the cumsum/exp chain are recomputed for every head_dim row.
  bc = _heads((buffer["ring_b"] * c.unsqueeze(2)).sum(-1), heads)        # (B, heads, ring)
  coef = (weights * bc).cat(_heads((b * c).sum(-1, keepdim=True), heads), checkpoint_decay.unsqueeze(-1),
                            dim=-1).contiguous()                        # (B, heads, ring + 2)
  ring = bc.shape[-1]
  rows = batch * heads * head_dim
  recent = (buffer["ring_x"].reshape(rows, ring) *
            coef[:, :, :ring].reshape(batch, heads, 1, ring).expand(batch, heads, head_dim, ring).reshape(rows, ring))
  recent = recent.sum(-1).reshape(batch, heads, head_dim)
  current = x_dt * coef[:, :, ring:ring + 1]
  checkpoint_decay = coef[:, :, ring + 1]
  # Realized before the gated norm: fused, the ring sum is recomputed inside the
  # low-parallelism group-norm reduce (~0.3 TB/s at B=128).
  y = (_readout(buffer["state"], _heads(c, heads)) * checkpoint_decay.unsqueeze(-1) + recent + current).contiguous()
  out = _block_output(block, hidden, normalized, gate, x, y)
  buffer["conv"].assign(new_tail.contiguous())
  buffer["ring_x"][:, :, :, slot:slot + 1].assign(x_dt.reshape(batch, heads, head_dim, 1))
  buffer["ring_a"][:, :, slot:slot + 1].assign(log_decay.reshape(batch, heads, 1))
  buffer["ring_b"][:, :, slot:slot + 1].assign(b.reshape(batch, cfg.ssm_groups, 1, cfg.ssm_state))
  return out


def mamba_replay_buffers_step(block, hidden: Tensor, buffer: dict, slot) -> Tensor:
  """``mamba_replay_step`` realized together with its conv and ring writes (TinyJit-safe)."""
  out = mamba_replay_step(block, hidden, buffer, slot)
  Tensor.realize(out, *(buffer[key] for key in ("conv", "ring_x", "ring_a", "ring_b")))
  return out


def mamba_replay_flush(block, buffer: dict, count) -> Tensor:
  """Fold the first `count` ring slots into the checkpoint state in place, and realize it."""
  heads = block.config.ssm_heads
  weights, decay = _ring_weights(buffer["ring_a"], count)
  weights = weights.contiguous()
  batch, _, head_dim, ring = buffer["ring_x"].shape
  # One packed (B, heads, ring*head_dim + 1) operand: decayed x*dt per slot, then exp(L_n).
  pack = (buffer["ring_x"] * weights.unsqueeze(2)).transpose(-1, -2).reshape(batch, heads, ring * head_dim).cat(
    decay.reshape(batch, heads, 1), dim=-1).contiguous()
  b = _heads(buffer["ring_b"], heads)                                   # (B, heads, ring, N)
  # The rank-`ring` update is unrolled into `ring` elementwise terms, not a
  # matmul: an assign source containing a reduce keeps a full-state temporary
  # plus a copy back (`remove_bufferize`), doubling the flush traffic.
  update = pack[:, :, ring * head_dim:].reshape(batch, heads, 1, 1) * buffer["state"]
  for j in range(ring):
    update = update + pack[:, :, j * head_dim:(j + 1) * head_dim].reshape(batch, heads, head_dim, 1) * b[:, :, j:j + 1]
  return buffer["state"].assign(update).realize()


__all__ = ["mamba_decode_step", "mamba_decode_buffers", "mamba_replay_buffers", "mamba_replay_step",
           "mamba_replay_buffers_step", "mamba_replay_flush"]
