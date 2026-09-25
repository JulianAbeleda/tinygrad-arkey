"""Batched, TinyJit-compiled Nemotron-H decoding for RL rollouts.

One compiled graph per token for a whole group of sequences sharing one
prompt: the prompt runs once and its state is broadcast; every Mamba block
carries fixed per-sequence convolution and scan state; the four attention
blocks write into fixed-capacity key/value buffers at a symbolic position.
Sampling is Gumbel-max on the GPU and each step returns only the sampled
token ids and their log probabilities, the values an RLOO update needs.

Mamba and MLP blocks step through the model's own `cached(..., keep_graph=True)`
path, the one verified exact for training; only attention is written here.
Ordering inside a step is explicit: a new key/value is stored before
attention reads the buffer, and recurrent state is read before it is stored.
"""
from __future__ import annotations

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.nemotron_h_decode import mamba_replay_buffers, mamba_replay_buffers_step, mamba_replay_flush


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHBatchSampler:
  def __init__(self, model, batch: int, capacity: int, bias=None, ring: int = 16):
    self.model, self.batch, self.capacity, self.ring = model, batch, capacity, ring
    config = model.config
    self.bias = _fresh(Tensor(bias) if bias is not None else Tensor.zeros(config.vocab_size))
    probe = model.token_embd(Tensor([[0] * config.conv_kernel])).float()
    self.buffers = []
    for block in model.blk:
      probe, cache = block.cached(probe, None)
      if block.block_type == "attention":
        _, heads, _, width = cache["k"].shape
        self.buffers.append({key: _fresh(Tensor.zeros(batch, heads, capacity, width, dtype=cache[key].dtype))
                             for key in ("k", "v")})
      elif block.block_type == "mamba":
        self.buffers.append(mamba_replay_buffers(block, *(Tensor.zeros(batch, *cache[key].shape[1:],
                                                   dtype=cache[key].dtype) for key in ("conv", "state")), ring))
      else:
        self.buffers.append(None)
    self.position = UOp.variable("position", 0, capacity - 1)
    self.slot, self.count = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
    self.step, self.flush = TinyJit(self._step), TinyJit(self._flush)

  def prime(self, prompt: list[int]) -> Tensor:
    """Run the prompt once and broadcast its state; returns the last position's hidden state."""
    if len(prompt) >= self.capacity:
      raise ValueError("prompt does not fit the decode capacity")
    hidden, caches = self.model.prefix(prompt, through=len(self.model.blk) - 1)
    for block, buffer, cache in zip(self.model.blk, self.buffers, caches):
      if block.block_type == "attention":
        for key in ("k", "v"):
          buffer[key][:, :, :len(prompt)] = cache[key].expand(self.batch, *cache[key].shape[1:])
          buffer[key].realize()
      elif block.block_type == "mamba":
        for key, value in cache.items():  # the ring starts empty: slot 0 masks every older entry
          buffer[key].assign(value.expand(self.batch, *value.shape[1:])).realize()
    return hidden[:, -1:].expand(self.batch, 1, hidden.shape[-1]).contiguous().realize()

  def _sample(self, hidden: Tensor, temperature: Tensor) -> tuple[Tensor, Tensor]:
    # contiguous: the vocab projection reads the whole embedding table; left lazy, every consumer of the logits
    # (max, sum, argmax, gather) recomputes it
    logits = (self.model.output(self.model.output_norm(hidden))[:, 0].float() + self.bias).contiguous()
    logprobs = (logits / temperature).log_softmax(-1)
    gumbel = -(-(Tensor.rand_like(logprobs).maximum(1e-12)).log()).log()
    token = (logprobs + gumbel).argmax(-1)
    chosen = logprobs.gather(-1, token.unsqueeze(-1))[:, 0]
    return token.cast(dtypes.int32).realize(), chosen.realize()

  def first(self, hidden: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    return self._sample(hidden, Tensor([temperature]))

  def _step(self, tokens: Tensor, position: UOp, slot: UOp, temperature: Tensor) -> tuple[Tensor, Tensor]:
    # Materialize the residual stream at every block boundary. Left lazy, a block's output feeds several kernels
    # (the next norm, the residual add) and each one recomputes the producing projection or embedding lookup.
    hidden = self.model.token_embd(tokens.reshape(self.batch, 1)).float().contiguous()
    for block, buffer in zip(self.model.blk, self.buffers):
      if block.block_type == "attention":
        normed = block.attn_norm(hidden)
        heads, kv_heads, width = block.n_heads, block.n_kv_heads, self.model.config.head_dim
        q = block.attn_q(normed).reshape(self.batch, 1, heads, width).transpose(1, 2)
        for key, layer in (("k", block.attn_k), ("v", block.attn_v)):
          new = layer(normed).reshape(self.batch, 1, kv_heads, width).transpose(1, 2)
          buffer[key][:, :, position:position + 1].assign(new.cast(buffer[key].dtype)).realize()
        allowed = (Tensor.arange(self.capacity) <= position).reshape(1, 1, 1, self.capacity)
        mask = allowed.where(0.0, float("-inf")).cast(hidden.dtype)
        attended = q.scaled_dot_product_attention(buffer["k"], buffer["v"], attn_mask=mask, enable_gqa=True)
        hidden = (hidden + block.attn_output(attended.transpose(1, 2).reshape(self.batch, 1, -1)).cast(hidden.dtype))
      elif block.block_type == "mamba":
        hidden = mamba_replay_buffers_step(block, hidden, buffer, slot)
      else:
        hidden, _ = block.cached(hidden, None, keep_graph=True)
      hidden = hidden.contiguous()
    return self._sample(hidden, temperature)

  def _flush(self, count: UOp) -> None:
    """Fold the ring's first `count` tokens into every Mamba checkpoint state."""
    for block, buffer in zip(self.model.blk, self.buffers):
      if block.block_type == "mamba":
        mamba_replay_flush(block, buffer, count)

  def generate(self, prompt: list[int], steps: int, temperature: float = 1.0, stop: set[int] | None = None):
    """Sample up to `steps` tokens for every sequence; returns (tokens, logprobs) as lists per sequence."""
    if len(prompt) + steps > self.capacity:
      raise ValueError("rollout exceeds the decode capacity")
    token, chosen = self.first(self.prime(prompt), temperature)
    rate = Tensor([temperature]).realize()
    tokens, logprobs = [token.numpy().tolist()], [chosen.numpy().tolist()]
    finished = [bool(stop) and t in stop for t in tokens[0]]
    for step in range(1, steps):
      if all(finished):
        break
      slot = (step - 1) % self.ring
      token, chosen = self.step(Tensor(tokens[-1], dtype=dtypes.int32).realize(),
                                self.position.bind(len(prompt) + step - 1), self.slot.bind(slot), rate)
      if slot == self.ring - 1:
        self.flush(self.count.bind(self.ring))
      tokens.append(token.numpy().tolist())
      logprobs.append(chosen.numpy().tolist())
      finished = [done or (bool(stop) and t in stop) for done, t in zip(finished, tokens[-1])]
    per_sequence = []
    for index in range(self.batch):
      ids, lps = [row[index] for row in tokens], [row[index] for row in logprobs]
      end = next((i for i, t in enumerate(ids) if stop and t in stop), len(ids) - 1)
      per_sequence.append((ids[:end + 1], lps[:end + 1]))
    return per_sequence


__all__ = ["NemotronHBatchSampler"]
