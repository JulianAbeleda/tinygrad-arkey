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

import functools

import numpy as np

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.nemotron_h_attention import (decode_attention, load_prefix_from_prefill, rollout_kv_for_model,
                                               shared_kv_for_model, suffix_bucket)
from tinygrad.llm.nemotron_h_decode import mamba_replay_buffers, mamba_replay_buffers_step, mamba_replay_flush


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


ATTENTION_CHUNK = 1024  # nemotron_h_attention's default split-K chunk


class NemotronHBatchSampler:
  def __init__(self, model, batch: int, capacity: int, bias=None, ring: int = 16, prefix_capacity: int | None = None,
               min_bucket: int = 64):
    """`capacity` bounds the generated tokens per sequence; `prefix_capacity` (default `capacity`) the prompt."""
    if ring < 2:
      raise ValueError("the Mamba replay ring needs at least 2 slots")  # 1 degenerates the slot/count variables
    # Attention splits a key span into 1024-key chunks only when the chunk divides it; a span that does not
    # (P=10000) is one serial reduction per output, ~5x slower. Round long capacities up to whole chunks.
    def whole_chunks(n: int) -> int: return n if n <= ATTENTION_CHUNK else -(-n // ATTENTION_CHUNK) * ATTENTION_CHUNK
    capacity = whole_chunks(capacity)
    self.model, self.batch, self.capacity, self.ring = model, batch, capacity, ring
    self.prefix_capacity, self.min_bucket = whole_chunks(prefix_capacity or capacity), min_bucket
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
    self.attention = shared_kv_for_model(model, batch, self.prefix_capacity, capacity, chunk=ATTENTION_CHUNK)
    self.prompt_length = 0
    self.prefix_length = UOp.variable("prefix_length", 1, self.prefix_capacity)
    self.position = UOp.variable("position", 0, capacity - 1)  # suffix row of the token being fed
    self.slot, self.count = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
    self.graphs: dict[int, TinyJit] = {}
    # sampled ids and log probabilities stay on the device; column c holds suffix row c (0 = the first sample)
    self.history = {"tokens": _fresh(Tensor.zeros(batch, capacity, dtype=dtypes.int32)),
                    "logprobs": _fresh(Tensor.zeros(batch, capacity))}
    self.flush = TinyJit(self._flush)
    self.tokens = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))

  def step(self, tokens: Tensor, index: int, temperature: Tensor) -> tuple[Tensor, Tensor]:
    """Feed each sequence's token at suffix row `index` (0 = first generated token); returns the next sample.

    The returned token tensor can be fed straight back as the next step's input: the graph reads its input tokens
    before it writes the new ones, so the host never waits on the device between steps.

    Mamba inputs go to ring slot `index % ring`; a full ring is folded into the checkpoint state right after."""
    bucket = suffix_bucket(index, self.capacity, self.min_bucket)
    graph = self.graphs.setdefault(bucket, TinyJit(self._step))
    slot = index % self.ring
    # The graph reads its input tokens from `self.tokens` and writes the sample back there. A caller feeding the
    # returned tokens straight back passes that same buffer and nothing moves; handing a graph its own output as a
    # TinyJit input would force a shadow copy outside the graph every step.
    if tokens is not self.tokens:
      self.tokens.assign(tokens.cast(dtypes.int32)).realize()
    out = graph(self.prefix_length.bind(self.prompt_length), self.position.bind(index), self.slot.bind(slot),
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

  def _logprobs(self, hidden: Tensor, temperature: Tensor) -> Tensor:
    # contiguous: the vocab projection reads the whole embedding table; left lazy, every consumer of the logits
    # (max, sum, argmax, gather) recomputes it
    # only the real vocabulary: a tile-padded head's extra columns must never be sampled
    logits = self.model.output(self.model.output_norm(hidden))[:, 0, :self.model.config.vocab_size]
    logits = (logits.float() + self.bias).contiguous()
    # contiguous: one log_softmax kernel whose output the sampler's consumers (argmax, gather) and a recompute
    # (`tail_logprobs`) both read; fused into different consumers it rounds differently
    return (logits / temperature).log_softmax(-1).contiguous()

  def _sample(self, hidden: Tensor, temperature: Tensor) -> tuple[Tensor, Tensor]:
    logprobs = self._logprobs(hidden, temperature)
    gumbel = -(-(Tensor.rand_like(logprobs).maximum(1e-12)).log()).log()
    token = (logprobs + gumbel).argmax(-1)
    chosen = logprobs.gather(-1, token.unsqueeze(-1))[:, 0]
    return token.cast(dtypes.int32).realize(), chosen.realize()

  def first(self, hidden: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    return self._sample(hidden, Tensor([temperature]))

  def _step(self, prefix_length: UOp, position: UOp, slot: UOp, temperature: Tensor,
            bucket: int) -> tuple[Tensor, Tensor]:
    tokens = self.tokens
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
    self.tokens.assign(token).realize()
    return self.tokens, chosen

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


class NemotronHRolloutSampler:
  """Continuously batched rollouts over several prompts, each sampled `n` times (RLOO groups).

  The batch is `batch` lanes. `prompts` prompt slots each hold one prompt's attention prefix and Mamba state; a
  prompt stays resident until all of its rollouts have finished, and each of its running rollouts occupies one lane
  and one of the slot's `rows` rows. A group's longest rollout holds its slot long after its siblings have finished,
  so the default is two slots' worth of rows per lane (`2 * batch // rows + 2`): with RL-skewed lengths (median 1.2k,
  cap 4096, groups of 8) `batch // rows + 2` slots leave about a third of the lanes idle while prompts are still
  queued, and twice that keeps about 99% of them busy. Decode runs in windows of `window` steps (a multiple
  of the Mamba replay ring, so every window ends right after a ring flush); between windows the host reads the
  sampled history, retires lanes that hit a stop token or `max_new` tokens, and refills free lanes with the next
  rollouts, priming new prompts (through `prefill` when given) into free prompt slots. A slot holds its prompt
  without the last token; a lane starts from the slot's Mamba state with empty generated keys (length 0) and feeds
  the prompt's last token through the decode step, so every sampled token, the first included, comes from the same
  step graph. Prompts therefore need at least two tokens.

  `capture=k` also returns, for every sampled token, the float32 hidden state entering block `k` at the position
  that produced it: the buffer block `k` read in that decode step, bit for bit. Each window's captures are copied to
  the host (`[batch, window, dim]` on the device), so the device holds no per-token history. `replay` runs blocks
  `k..` and the sampling tail again from those captures on the same decode graph code, one rollout per lane.

  Every lane starts on a window boundary, so a rollout's position t always sits in Mamba ring slot `t % ring` and is
  followed by a flush when that slot is the last one. Its generated keys sit at ring rows `start + t` (mod
  `capacity`), where `start` (`stats["starts"]`) is the sampler step it began on. Attention reads only a bucket of
  the most recent ring rows, a power of two covering every running lane's length by the end of the window (at
  least `min_bucket`; the whole ring once it exceeds half of it), one step graph per bucket and wrap form, chosen
  per window (`stats["buckets"]`). Attention's reduction depends on where the keys sit and on the bucket, so a
  replay through an attention block takes both from the stats.
  """

  def __init__(self, model, batch: int, capacity: int, prefix_capacity: int, prompts: int | None = None, rows: int = 8,
               bias=None, ring: int = 16, window: int = 32, prefill=None, capture: int | None = None,
               min_bucket: int = 64):
    if ring < 2 or window % ring:
      raise ValueError("the refill window must be a whole number of replay rings of at least 2 slots")
    def whole_chunks(n: int) -> int: return n if n <= ATTENTION_CHUNK else -(-n // ATTENTION_CHUNK) * ATTENTION_CHUNK
    if capture is not None and not 0 <= capture < len(model.blk):
      raise ValueError("capture names the block whose input hidden state is kept")
    self.model, self.batch, self.ring, self.window, self.prefill = model, batch, ring, window, prefill
    self.capture, self.min_bucket = capture, min_bucket
    self.capacity, self.prefix_capacity = whole_chunks(capacity), whole_chunks(prefix_capacity)
    self.prompts, self.rows = prompts or 2 * batch // rows + 2, rows
    config = model.config
    self.bias = _fresh(Tensor(bias) if bias is not None else Tensor.zeros(config.vocab_size))
    probe = model.token_embd(Tensor([[0] * config.conv_kernel])).float()
    self.buffers, self.prompt_state = [], []
    for block in model.blk:
      probe, cache = block.cached(probe, None)
      if block.block_type == "mamba":
        self.buffers.append(mamba_replay_buffers(block, *(Tensor.zeros(batch, *cache[key].shape[1:],
                                                   dtype=cache[key].dtype) for key in ("conv", "state")), ring))
        self.prompt_state.append({key: _fresh(Tensor.zeros(self.prompts, *value.shape[1:], dtype=value.dtype))
                                  for key, value in cache.items()})
      else:
        self.buffers.append(None)
        self.prompt_state.append(None)
    self.attention = rollout_kv_for_model(model, batch, self.prompts, self.prefix_capacity, self.capacity,
                                          chunk=ATTENTION_CHUNK)
    self.prefix_lengths = _fresh(Tensor.zeros(self.prompts, dtype=dtypes.int32))
    self.lengths = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))
    self.lane_rows = _fresh(Tensor.zeros(self.prompts * rows, dtype=dtypes.int32))
    self.lane_slot = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))
    self.tokens = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))  # each step's input, overwritten by its sample
    self.history = {"tokens": _fresh(Tensor.zeros(batch, window, dtype=dtypes.int32)),
                    "logprobs": _fresh(Tensor.zeros(batch, window))}
    if capture is not None:
      self.history["hidden"] = _fresh(Tensor.zeros(batch, window, config.dim))
      # replay inputs: a window of captured hidden states and of the tokens whose log probabilities are read
      self.feed = {"hidden": _fresh(Tensor.zeros(batch, window, config.dim)),
                   "tokens": _fresh(Tensor.zeros(batch, window, dtype=dtypes.int32))}
    self.row = UOp.variable("ring_row", 0, self.capacity - 1)
    self.slot, self.count = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
    self.column = UOp.variable("column", 0, window - 1)
    self.lane, self.source = UOp.variable("lane", 0, batch - 1), UOp.variable("source", 0, self.prompts - 1)
    self.flush, self.load_lane, self.seed = TinyJit(self._flush), TinyJit(self._load_lane), TinyJit(self._seed)
    self.replay_flush = TinyJit(self._replay_flush)
    # priming copies a prompt's state into its slot in one graph; eager, its ~60 copies cost ~0.2 s of host time
    self.load_slot = TinyJit(self._load_slot)
    self.prompt_length = UOp.variable("prompt_length", 1, self.prefix_capacity)
    self.lows: dict[int, UOp] = {}  # bucket -> the first ring row it reads when it does not wrap
    self.graphs: dict[tuple[bool, int, bool], TinyJit] = {}  # (replay, bucket, wraps)

  def bucket(self, longest: int) -> int:
    """Ring rows attention reads for lanes of at most `longest` generated keys."""
    size = max(self.min_bucket, 1 << max(0, longest - 1).bit_length())
    return self.capacity if 2 * size > self.capacity else size

  def _run(self, replay: bool, bucket: int, step: int, column: int, temperature: Tensor) -> None:
    """Decode (or replay) step `step`, the sampler's global step count: ring row `step % capacity`."""
    row = step % self.capacity
    wraps = bucket < self.capacity and row < bucket - 1
    key = (replay, bucket, wraps)
    if key not in self.graphs:
      self.graphs[key] = TinyJit(functools.partial(self._replay_step if replay else self._step, bucket=bucket,
                                                   wraps=wraps))
    low = ()
    if bucket < self.capacity and not wraps:
      if bucket not in self.lows:
        self.lows[bucket] = UOp.variable(f"ring_low_{bucket}", 0, self.capacity - bucket)
      low = (self.lows[bucket].bind(row - bucket + 1),)
    self.graphs[key](self.row.bind(row), self.slot.bind(step % self.ring), self.column.bind(column), temperature, *low)

  # *** device graphs ***

  def _step(self, row: UOp, slot: UOp, column: UOp, temperature: Tensor, *low: UOp, bucket: int,
            wraps: bool) -> None:
    self.lengths.assign(self.lengths + 1).realize()
    hidden = self.model.token_embd(self.tokens.reshape(self.batch, 1)).float().contiguous()
    hidden = self._blocks(hidden, 0, row, slot, column, bucket, low[0] if low else None)
    token, chosen = self._sample(hidden, temperature)
    for key, value in (("tokens", token), ("logprobs", chosen)):
      self.history[key][:, column:column + 1].assign(value.reshape(self.batch, 1)).realize()
    self.tokens.assign(token).realize()

  def _replay_step(self, row: UOp, slot: UOp, column: UOp, temperature: Tensor, *low: UOp, bucket: int,
                   wraps: bool) -> None:
    """Blocks `capture..` and the tail from the fed hidden states; records the fed tokens' log probabilities."""
    self.lengths.assign(self.lengths + 1).realize()
    hidden = self.feed["hidden"][:, column:column + 1].contiguous()
    hidden = self._blocks(hidden, self.capture, row, slot, column, bucket, low[0] if low else None)
    chosen = self._logprobs(hidden, temperature).gather(-1, self.feed["tokens"][:, column:column + 1])
    self.history["logprobs"][:, column:column + 1].assign(chosen).realize()

  def _blocks(self, hidden: Tensor, first: int, row: UOp, slot: UOp, column: UOp, bucket: int,
              low: UOp | None) -> Tensor:
    """Blocks `first..` of one decode step; every block's input is a realized buffer."""
    for index, (block, buffer, attention) in enumerate(zip(self.model.blk, self.buffers, self.attention)):
      if index < first:
        continue
      if index == self.capture and first == 0:  # the buffer block `index` reads, bit for bit
        self.history["hidden"][:, column:column + 1].assign(hidden).realize()
      if block.block_type == "attention":
        normed = block.attn_norm(hidden)
        width = self.model.config.head_dim
        q = block.attn_q(normed).reshape(self.batch, 1, block.n_heads, width).transpose(1, 2)
        k = block.attn_k(normed).reshape(self.batch, 1, block.n_kv_heads, width).transpose(1, 2)
        v = block.attn_v(normed).reshape(self.batch, 1, block.n_kv_heads, width).transpose(1, 2)
        attention.write(k, v, row)
        attended = attention.attend(q, self.lane_rows, self.lane_slot, self.prefix_lengths, self.lengths, row,
                                    bucket if bucket < self.capacity else None, low)
        mixed = block.attn_output(attended.transpose(1, 2).reshape(self.batch, 1, -1))
        hidden = (hidden + mixed.cast(hidden.dtype)).contiguous()
      elif block.block_type == "mamba":
        hidden = mamba_replay_buffers_step(block, hidden, buffer, slot).contiguous()
      else:
        hidden = self._stateless(block, hidden)
    return hidden

  _sample, _logprobs = NemotronHBatchSampler._sample, NemotronHBatchSampler._logprobs

  @staticmethod
  def _stateless(block, hidden: Tensor) -> Tensor:
    return block.cached(hidden, None, keep_graph=True)[0].contiguous()

  def tail_logprobs(self, hidden: Tensor, temperature: float = 1.0) -> Tensor:
    """Log probabilities `[n, vocab]` from captured hidden states `[n, 1, dim]` (float32) through the same tail code
    the decode step runs: blocks `capture..` (stateless MLP blocks only), the output norm and head, bias and
    temperature. With `n == batch` the kernels are the step's, so the result is bit-identical to the sampled values."""
    if self.capture is None or any(b.block_type != "mlp" for b in self.model.blk[self.capture:]):
      raise ValueError("the tail from the captured block must be stateless MLP blocks")
    for block in self.model.blk[self.capture:]:
      hidden = self._stateless(block, hidden)
    return self._logprobs(hidden, Tensor([temperature])).realize()

  def _flush(self, count: UOp, first: int = 0) -> None:
    for block, buffer in zip(self.model.blk[first:], self.buffers[first:]):
      if block.block_type == "mamba":
        mamba_replay_flush(block, buffer, count)

  def _replay_flush(self, count: UOp) -> None:
    self._flush(count, self.capture)

  def _load_lane(self, lane: UOp, source: UOp) -> None:
    """Lane `lane` takes prompt slot `source`'s Mamba state; its generated keys restart empty.

    The lane's replay ring and generated-key ring are zeroed: masked entries get weight 0, and 0 times a previous
    occupant's non-finite value is NaN, so a lane that went NaN would otherwise poison the next rollout it serves."""
    for buffer, state in zip(self.buffers, self.prompt_state):
      if buffer is not None:
        for key in ("conv", "state"):
          buffer[key][lane:lane + 1].assign(state[key][source:source + 1]).realize()
        for key in ("ring_x", "ring_a", "ring_b"):
          buffer[key][lane:lane + 1].assign(Tensor.zeros(1, *buffer[key].shape[1:], dtype=buffer[key].dtype)).realize()
    for attention in self.attention:
      if attention is not None:
        for value in attention.suffix.values():
          value[lane:lane + 1].assign(Tensor.zeros(1, *value.shape[1:], dtype=value.dtype)).realize()
    self.lengths[lane:lane + 1].assign(Tensor.zeros(1, dtype=dtypes.int32)).realize()

  def _load_slot(self, source: UOp, length: UOp) -> None:
    """Prompt slot `source` takes the prefill's state: its first `length` key/value rows (the rest zeroed, since a
    masked row's value still meets a 0 weight) and every Mamba block's convolution tail and scan state."""
    for attention, state, cache in zip(self.attention, self.prompt_state, self.prefill.buffers):
      if attention is not None:
        span = attention.prefix_capacity
        for key, target in attention.prefix.items():
          value = cache[key][:, :, :min(cache[key].shape[2], span)]
          if value.shape[2] < span:
            value = value.pad((None, None, (0, span - value.shape[2]), None))
          keep = Tensor.arange(span).reshape(1, 1, span, 1) < length
          target[source:source + 1].assign(keep.where(value, 0).cast(target.dtype)).realize()
      elif state is not None:
        for key in ("conv", "state"):
          state[key][source:source + 1].assign(cache[key]).realize()

  def _seed(self, fresh: Tensor, last: Tensor) -> None:
    """Lanes flagged `fresh` feed their prompt's last token next."""
    self.tokens.assign(fresh.where(last, self.tokens)).realize()

  # *** host ***

  def _prime(self, slot: int, prompt: list[int]):
    if not 1 < len(prompt) <= self.prefix_capacity + 1:
      raise ValueError("a prompt needs at least two tokens and must fit the prefix capacity")
    prompt = prompt[:-1]  # the last token runs through the decode step
    if self.prefill is not None:
      self.prefill(prompt)
      self.load_slot(self.source.bind(slot), self.prompt_length.bind(len(prompt)))
      return
    caches = self.model.prefix(prompt, through=len(self.model.blk) - 1)[1]
    length = len(prompt)
    for attention, state, cache in zip(self.attention, self.prompt_state, caches):
      if attention is not None:
        # only the prompt's rows; the rest are zeroed, since a masked row's value still meets a 0 weight
        attention.load_prefix(slot, cache["k"][:, :, :length], cache["v"][:, :, :length])
        if length < attention.prefix_capacity:
          for value in attention.prefix.values():
            value[slot:slot + 1, :, length:].assign(Tensor.zeros(1, value.shape[1], attention.prefix_capacity - length,
                                                                  value.shape[3], dtype=value.dtype)).realize()
      elif state is not None:
        for key in ("conv", "state"):
          value = cache[key]
          if (short := state[key].shape[1] - value.shape[1]) > 0:  # a prompt shorter than the conv tail
            value = Tensor.zeros(1, short, *value.shape[2:], dtype=value.dtype).cat(value, dim=1)
          state[key][slot:slot + 1].assign(value).realize()

  def _bind(self, lanes: list[tuple[int, int] | None], prefix_lengths: list[int], started: list[int]):
    """Lane `b` serves row `lanes[b][1]` of prompt slot `lanes[b][0]`; lanes in `started` begin from their slot."""
    table = [0] * (self.prompts * self.rows)
    for lane, place in enumerate(lanes):
      if place is not None:
        table[place[0] * self.rows + place[1]] = lane
    self.lane_rows.assign(Tensor(table, dtype=dtypes.int32)).realize()
    self.lane_slot.assign(Tensor([0 if p is None else p[0] * self.rows + p[1] for p in lanes],
                                 dtype=dtypes.int32)).realize()
    lengths = [min(n, self.prefix_capacity) for n in prefix_lengths]
    self.prefix_lengths.assign(Tensor(lengths, dtype=dtypes.int32)).realize()
    for lane in started:
      self.load_lane(self.lane.bind(lane), self.source.bind(lanes[lane][0]))

  def replay(self, requests: list[tuple], temperature: float = 1.0, stats: dict | None = None
             ) -> list[list[np.ndarray]]:
    """Log probabilities of given tokens through blocks `capture..` and the tail, from captured inputs.

    `requests` is `[(prompt, [(tokens, hidden), ...]), ...]` with `hidden` the `[len(tokens), dim]` captures; returns
    float32 log probabilities per rollout, laid out like the request. Each rollout runs on its own lane through the
    decode step's code for those blocks: Mamba blocks from the prompt's state through the same replay ring and
    flushes, attention blocks against the prompt's keys and the keys the replay itself writes at the rollout's ring
    rows with the same ring buckets (`stats`, the dict `generate` filled: its `starts` and `buckets`; needed only when
    the window has an attention block). With the sampler's weights and temperature this gives the sampled log
    probabilities bit for bit.
    """
    if self.capture is None:
      raise ValueError("replay needs a sampler built with capture")
    attends = any(block.block_type == "attention" for block in self.model.blk[self.capture:])
    mambas = any(block.block_type == "mamba" for block in self.model.blk[self.capture:])
    if attends and not (stats and "starts" in stats and "buckets" in stats):
      raise ValueError("a window with an attention block needs generate's stats (each rollout's first ring row and "
                       "the ring buckets)")
    pending = []
    for r, (prompt, rollouts) in enumerate(requests):
      for i, (tokens, hidden) in enumerate(rollouts):
        if hidden.shape != (len(tokens), self.model.config.dim) or not 0 < len(tokens) <= self.capacity:
          raise ValueError("each rollout needs one captured row per token, within the capacity")
        pending.append((stats["starts"][r][i] if attends else 0, r, i, prompt, tokens, hidden))
    pending.sort(key=lambda item: item[0])
    out: list[list] = [[None] * len(rollouts) for _, rollouts in requests]
    rate = Tensor([temperature]).realize()
    while pending:
      start, taken, slots, lanes = pending[0][0], [], {}, []
      for item in pending:
        key = tuple(item[3])
        if item[0] != start or len(lanes) == self.batch:
          continue
        if key not in slots:
          if len(slots) == self.prompts:
            continue
          slots[key] = [len(slots), 0]
        if slots[key][1] == self.rows:
          continue
        lanes.append((slots[key][0], slots[key][1]))
        slots[key][1] += 1
        taken.append(item)
      pending = [item for item in pending if all(item is not t for t in taken)]
      for key, (g, _) in slots.items():
        self._prime(g, list(key))
      lengths = [0] * self.prompts
      for key, (g, _) in slots.items():
        lengths[g] = len(key) - 1
      self._bind(lanes + [None] * (self.batch - len(lanes)), lengths, list(range(len(lanes))))
      steps = max(len(item[4]) for item in taken)
      values = np.zeros((len(taken), steps), np.float32)
      for first in range(0, steps, self.window):
        width = min(self.window, steps - first)
        hidden = np.zeros((self.batch, self.window, self.model.config.dim), np.float32)
        tokens = np.zeros((self.batch, self.window), np.int32)
        for lane, item in enumerate(taken):
          part = item[5][first:first + width]
          hidden[lane, :len(part)] = part
          tokens[lane, :len(part)] = item[4][first:first + len(part)]
        self.feed["hidden"].assign(Tensor(hidden)).realize()
        self.feed["tokens"].assign(Tensor(tokens)).realize()
        for column in range(width):
          t = first + column
          bucket = stats["buckets"][(start + t) // self.window] if attends else self.capacity
          self._run(True, bucket, start + t, column, rate)
          if t % self.ring == self.ring - 1 and mambas:
            self.replay_flush(self.count.bind(self.ring))
        values[:, first:first + width] = self.history["logprobs"].numpy()[:len(taken), :width]
      for lane, item in enumerate(taken):
        out[item[1]][item[2]] = values[lane, :len(item[4])]
    return out

  def generate(self, requests: list[tuple], max_new: int, temperature: float = 1.0,
               stop: set[int] | None = None, stats: dict | None = None) -> list[list[tuple[list[int], list[float]]]]:
    """Sample `n` rollouts of up to `max_new` tokens for each `(prompt, n)`; returns per request its (tokens, logprobs).

    With `capture` set, each rollout is `(tokens, logprobs, hidden)`, hidden a float32 `[len(tokens), dim]` array.
    A request `(prompt, n, limits)` caps its i-th rollout at `min(limits[i], max_new)` tokens instead (a forced
    length, e.g. to replay a length mix). A rollout ends at (and includes) its first stop token. `stats`, when given,
    receives the decode step count, the lane-steps spent on rollouts that were still running (in total and while
    rollouts were still waiting to start, `steady_*`), the time spent priming prompts, `starts` (each rollout's first
    step, laid out like the result) and `buckets` (the ring rows attention read, per window): what `replay` needs
    for a window with attention.
    """
    import time
    if max_new > self.capacity:
      raise ValueError("max_new exceeds the generated-token ring")
    stop = stop or set()
    results: list[list] = [[] for _ in requests]
    starts: list[list[int]] = [[] for _ in requests]
    buckets: list[int] = []
    queue = []
    for index, (prompt, n, *limits) in enumerate(requests):
      limits = [min(int(x), max_new) for x in limits[0]] if limits else [max_new] * n
      if len(limits) != n or min(limits, default=1) < 1:
        raise ValueError("a request's limits must give each of its n rollouts at least one token")
      if n > 0:
        queue.append((index, prompt, limits))
    slots: list[dict | None] = [None] * self.prompts  # request, rollouts left to start, running rows
    lanes: list[dict | None] = [None] * self.batch
    rate = Tensor([temperature]).realize()
    steps = active_lane_steps = steady_steps = steady_active = 0
    prime_time = 0.0

    def refill() -> bool:
      nonlocal prime_time
      started = []
      for lane in range(self.batch):
        if lanes[lane] is not None:
          continue
        slot = next((g for g, s in enumerate(slots) if s is not None and s["left"] and None in s["rows"]), None)
        if slot is None and queue and None in slots:
          slot = slots.index(None)
          index, prompt, limits = queue.pop(0)
          start = time.perf_counter()
          self._prime(slot, prompt)
          prime_time += time.perf_counter() - start
          slots[slot] = {"request": index, "left": limits, "rows": [None] * self.rows, "length": len(prompt) - 1,
                         "last": prompt[-1]}
        if slot is None:
          continue
        s = slots[slot]
        j = s["rows"].index(None)
        s["rows"][j] = lane
        lanes[lane] = {"slot": slot, "row": j, "limit": s["left"].pop(0), "tokens": [], "logprobs": [], "hidden": [],
                       "start": steps}
        started.append(lane)
      if not started:
        return any(lane is not None for lane in lanes)
      self._bind([None if l is None else (l["slot"], l["row"]) for l in lanes],
                 [0 if s is None else s["length"] for s in slots], started)
      fresh = Tensor([lane in started for lane in range(self.batch)]).realize()
      last = Tensor([slots[l["slot"]]["last"] if l is not None else 0 for l in lanes], dtype=dtypes.int32).realize()
      self.seed(fresh, last)
      return True

    def retire():
      for lane, l in enumerate(lanes):
        if l is None:
          continue
        end = next((i for i, t in enumerate(l["tokens"][:l["limit"]]) if t in stop), None)
        if end is None and len(l["tokens"]) < l["limit"]:
          continue
        end = l["limit"] - 1 if end is None else end
        s = slots[l["slot"]]
        rollout = (l["tokens"][:end + 1], l["logprobs"][:end + 1])
        if self.capture is not None:
          rollout += (np.concatenate(l["hidden"])[:end + 1],)
        results[s["request"]].append(rollout)
        starts[s["request"]].append(l["start"])
        s["rows"][l["row"]], lanes[lane] = None, None
        if not s["left"] and all(r is None for r in s["rows"]):
          slots[l["slot"]] = None

    while refill() or any(lane is not None for lane in lanes):
      waiting = bool(queue) or any(s is not None and s["left"] for s in slots)
      window_active = 0
      bucket = self.bucket(max(len(l["tokens"]) for l in lanes if l is not None) + self.window)
      buckets.append(bucket)
      for i in range(self.window):
        self._run(False, bucket, steps, i, rate)
        steps += 1
        if steps % self.ring == 0:
          self.flush(self.count.bind(self.ring))
      ids, lps = self.history["tokens"].numpy(), self.history["logprobs"].numpy()
      vocab = self.model.config.vocab_size
      if (broken := (ids < 0) | (ids >= vocab) | ~np.isfinite(lps)).any():
        for lane, l in enumerate(lanes):  # an all-NaN row's argmax is `vocab` and its gathered log probability 0
          if l is None:
            continue
          kept = max(0, l["limit"] - len(l["tokens"]))  # samples past the limit or a stop token are discarded
          kept = next((i + 1 for i, t in enumerate(ids[lane][:kept].tolist()) if t in stop), kept)
          if (cols := np.flatnonzero(broken[lane][:kept])).size:
            raise FloatingPointError(f"non-finite logits: request {slots[l['slot']]['request']}, lane {lane}, rollout "
                                     f"position {len(l['tokens']) + int(cols[0])}: token {int(ids[lane, cols[0]])}, "
                                     f"log probability {float(lps[lane, cols[0]])}")
      ids, lps = ids.tolist(), lps.tolist()
      hidden = self.history["hidden"].numpy() if self.capture is not None else None
      for lane, l in enumerate(lanes):
        if l is not None:
          before = len(l["tokens"])
          l["tokens"] += ids[lane]
          l["logprobs"] += lps[lane]
          if hidden is not None and before < l["limit"]:
            l["hidden"].append(hidden[lane].copy())
          end = next((i for i, t in enumerate(l["tokens"][:l["limit"]]) if t in stop), l["limit"] - 1)
          window_active += max(0, min(end + 1, before + self.window) - before)
      active_lane_steps += window_active
      if waiting:
        steady_steps, steady_active = steady_steps + self.window, steady_active + window_active
      retire()
    if stats is not None:
      stats.update(steps=steps, lane_steps=steps * self.batch, active_lane_steps=active_lane_steps,
                   steady_lane_steps=steady_steps * self.batch, steady_active_lane_steps=steady_active, starts=starts,
                   buckets=buckets,
                   prime_s=prime_time)
    return results


__all__ = ["NemotronHBatchSampler", "NemotronHRolloutSampler"]
