"""Batched, TinyJit-compiled Nemotron-H decoding for RL rollouts.

One compiled graph per token for a whole group of sequences sharing one
prompt: the prompt runs once and its state is broadcast; every Mamba block
carries fixed per-sequence convolution and scan state; the four attention
blocks keep the prompt's keys/values once, shared by the batch, and write each
sequence's generated keys/values into a per-sequence suffix at a symbolic row
(`nemotron_h_attention`). Attention reads the suffix only up to a power-of-two
length bucket, with one compiled graph per bucket.
Sampling is Gumbel-max on the GPU and each step returns only the sampled
token ids and their log probabilities, the values an RLOO update needs.

Mamba and MLP blocks step through the model's own `cached(..., keep_graph=True)`
path, the one verified exact for training; only attention is written here.
Ordering inside a step is explicit: a new key/value is stored before
attention reads the buffer, and recurrent state is read before it is stored.
"""
from __future__ import annotations

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.nemotron_h_attention import (decode_attention, load_prefix_from_prefill, shared_kv_for_model,
                                               suffix_bucket)
from tinygrad.llm.nemotron_h_decode import mamba_replay_buffers, mamba_replay_buffers_step, mamba_replay_flush


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHBatchSampler:
  def __init__(self, model, batch: int, capacity: int, bias=None, ring: int = 16, prefix_capacity: int | None = None,
               min_bucket: int = 64):
    """`capacity` bounds the generated tokens per sequence; `prefix_capacity` (default `capacity`) the prompt."""
    self.model, self.batch, self.capacity, self.ring = model, batch, capacity, ring
    self.prefix_capacity, self.min_bucket = prefix_capacity or capacity, min_bucket
    config = model.config
    self.bias = _fresh(Tensor(bias) if bias is not None else Tensor.zeros(config.vocab_size))
    probe = model.token_embd(Tensor([[0] * config.conv_kernel])).float()
    self.buffers = []
    for block in model.blk:
      probe, cache = block.cached(probe, None)
      if block.block_type == "mamba":
        self.buffers.append(mamba_replay_buffers(block, *(Tensor.zeros(batch, *cache[key].shape[1:],
                                                   dtype=cache[key].dtype) for key in ("conv", "state")), ring))
      else:
        self.buffers.append(None)
    self.attention = shared_kv_for_model(model, batch, self.prefix_capacity, capacity)
    self.prompt_length = 0
    self.prefix_length = UOp.variable("prefix_length", 1, self.prefix_capacity)
    self.position = UOp.variable("position", 0, capacity - 1)  # suffix row of the token being fed
    self.slot, self.count = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
    self.graphs: dict[int, TinyJit] = {}
    # sampled ids and log probabilities stay on the device; column c holds suffix row c (0 = the first sample)
    self.history = {"tokens": _fresh(Tensor.zeros(batch, capacity, dtype=dtypes.int32)),
                    "logprobs": _fresh(Tensor.zeros(batch, capacity))}
    self.flush = TinyJit(self._flush)

  def step(self, tokens: Tensor, index: int, temperature: Tensor) -> tuple[Tensor, Tensor]:
    """Feed each sequence's token at suffix row `index` (0 = first generated token); returns the next sample.

    The returned token tensor can be fed straight back as the next step's input: the graph reads its input tokens
    before it writes the new ones, so the host never waits on the device between steps.

    Mamba inputs go to ring slot `index % ring`; a full ring is folded into the checkpoint state right after."""
    bucket = suffix_bucket(index, self.capacity, self.min_bucket)
    graph = self.graphs.setdefault(bucket, TinyJit(self._step))
    slot = index % self.ring
    out = graph(tokens, self.prefix_length.bind(self.prompt_length), self.position.bind(index), self.slot.bind(slot),
                temperature, bucket)
    if slot == self.ring - 1:
      self.flush(self.count.bind(self.ring))
    return out

  def prime(self, prompt: list[int], prefill=None) -> Tensor:
    """Run the prompt once and share its state; returns the last position's hidden state.

    With a `NemotronHPrefill`, the prompt runs through its chunked graphs and
    its buffers are the source; otherwise through `model.prefix`.
    """
    if not 0 < len(prompt) <= self.prefix_capacity:
      raise ValueError("prompt does not fit the prefix capacity")
    if prefill is not None:
      hidden, caches = prefill(prompt), prefill.buffers
    else:
      hidden, caches = self.model.prefix(prompt, through=len(self.model.blk) - 1)
    self.prompt_length = len(prompt)
    load_prefix_from_prefill(self.attention, caches)
    for block, buffer, cache in zip(self.model.blk, self.buffers, caches):
      if block.block_type == "mamba":
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

  def _step(self, tokens: Tensor, prefix_length: UOp, position: UOp, slot: UOp, temperature: Tensor,
            bucket: int) -> tuple[Tensor, Tensor]:
    # Materialize the residual stream at every block boundary. Left lazy, a block's output feeds several kernels
    # (the next norm, the residual add) and each one recomputes the producing projection or embedding lookup.
    hidden = self.model.token_embd(tokens.reshape(self.batch, 1)).float().contiguous()
    for block, buffer, attention in zip(self.model.blk, self.buffers, self.attention):
      if block.block_type == "attention":
        mixed = decode_attention(block, block.attn_norm(hidden), attention, prefix_length, position, bucket)
        hidden = hidden + mixed.cast(hidden.dtype)
      elif block.block_type == "mamba":
        hidden = mamba_replay_buffers_step(block, hidden, buffer, slot)
      else:
        hidden, _ = block.cached(hidden, None, keep_graph=True)
      hidden = hidden.contiguous()
    token, chosen = self._sample(hidden, temperature)
    self._record(token, chosen, position + 1)
    return token, chosen

  def _record(self, token: Tensor, chosen: Tensor, column: UOp | int):
    for key, value in (("tokens", token), ("logprobs", chosen)):
      self.history[key][:, column:column + 1].assign(value.reshape(self.batch, 1)).realize()

  def _flush(self, count: UOp) -> None:
    """Fold the ring's first `count` tokens into every Mamba checkpoint state."""
    for block, buffer in zip(self.model.blk, self.buffers):
      if block.block_type == "mamba":
        mamba_replay_flush(block, buffer, count)

  def generate(self, prompt: list[int], steps: int, temperature: float = 1.0, stop: set[int] | None = None,
               prefill=None, check_every: int = 32):
    """Sample up to `steps` tokens for every sequence; returns (tokens, logprobs) as lists per sequence.

    Steps chain on the device (each step's sampled tokens are the next step's input); the host reads the sampled
    history back only every `check_every` steps, to stop early once every sequence has emitted a stop token.
    """
    if steps > self.capacity:
      raise ValueError("rollout exceeds the decode capacity")
    token, chosen = self.first(self.prime(prompt, prefill), temperature)
    self._record(token, chosen, 0)
    rate = Tensor([temperature]).realize()
    done = 1
    while done < steps:
      token, _ = self.step(token, done - 1, rate)
      done += 1
      if stop and (done % check_every == 0 or done == steps):
        ids = self.history["tokens"].numpy()[:, :done]
        if all(any(int(t) in stop for t in row) for row in ids):
          break
    tokens = self.history["tokens"].numpy()[:, :done].tolist()
    logprobs = self.history["logprobs"].numpy()[:, :done].tolist()
    per_sequence = []
    for ids, lps in zip(tokens, logprobs):
      end = next((i for i, t in enumerate(ids) if stop and t in stop), len(ids) - 1)
      per_sequence.append((ids[:end + 1], lps[:end + 1]))
    return per_sequence


__all__ = ["NemotronHBatchSampler"]
