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


class NemotronHRolloutSampler:
  """Continuously batched rollouts over several prompts, each sampled `n` times (RLOO groups).

  The batch is `batch` lanes. `prompts` prompt slots each hold one prompt's attention prefix, its Mamba state and
  its last hidden state; a prompt stays resident until all of its rollouts have finished, and each of its running
  rollouts occupies one lane and one of the slot's `rows` rows. Decode runs in windows of `window` steps (a multiple
  of the Mamba replay ring, so every window ends right after a ring flush); between windows the host reads the
  sampled history, retires lanes that hit a stop token or `max_new` tokens, and refills free lanes with the next
  rollouts, priming new prompts (through `prefill` when given) into free prompt slots. A lane's first token is
  sampled from its prompt's last hidden state; its Mamba state is copied from the prompt slot; its generated keys
  start empty (length 0) in the shared ring.
  """

  def __init__(self, model, batch: int, capacity: int, prefix_capacity: int, prompts: int | None = None, rows: int = 8,
               bias=None, ring: int = 16, window: int = 32, prefill=None):
    if window % ring:
      raise ValueError("the refill window must be a whole number of replay rings")
    def whole_chunks(n: int) -> int: return n if n <= ATTENTION_CHUNK else -(-n // ATTENTION_CHUNK) * ATTENTION_CHUNK
    self.model, self.batch, self.ring, self.window, self.prefill = model, batch, ring, window, prefill
    self.capacity, self.prefix_capacity = whole_chunks(capacity), whole_chunks(prefix_capacity)
    self.prompts, self.rows = prompts or batch // rows + 2, rows
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
    self.prompt_hidden = _fresh(Tensor.zeros(self.prompts, 1, config.dim))
    self.prefix_lengths = _fresh(Tensor.zeros(self.prompts, dtype=dtypes.int32))
    self.lengths = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))
    self.lane_rows = _fresh(Tensor.zeros(self.prompts * rows, dtype=dtypes.int32))
    self.lane_slot = _fresh(Tensor.zeros(batch, dtype=dtypes.int32))
    self.history = {"tokens": _fresh(Tensor.zeros(batch, window, dtype=dtypes.int32)),
                    "logprobs": _fresh(Tensor.zeros(batch, window))}
    self.row = UOp.variable("ring_row", 0, self.capacity - 1)
    self.slot, self.count = UOp.variable("slot", 0, ring - 1), UOp.variable("count", 1, ring)
    self.column = UOp.variable("column", 0, window - 1)
    self.lane, self.source = UOp.variable("lane", 0, batch - 1), UOp.variable("source", 0, self.prompts - 1)
    self.step, self.flush = TinyJit(self._step), TinyJit(self._flush)
    self.load_lane, self.start = TinyJit(self._load_lane), TinyJit(self._start)

  # *** device graphs ***

  def _step(self, tokens: Tensor, row: UOp, slot: UOp, column: UOp, temperature: Tensor) -> Tensor:
    self.lengths.assign(self.lengths + 1).realize()
    hidden = self.model.token_embd(tokens.reshape(self.batch, 1)).float().contiguous()
    for block, buffer, attention in zip(self.model.blk, self.buffers, self.attention):
      if block.block_type == "attention":
        normed = block.attn_norm(hidden)
        width = self.model.config.head_dim
        q = block.attn_q(normed).reshape(self.batch, 1, block.n_heads, width).transpose(1, 2)
        k = block.attn_k(normed).reshape(self.batch, 1, block.n_kv_heads, width).transpose(1, 2)
        v = block.attn_v(normed).reshape(self.batch, 1, block.n_kv_heads, width).transpose(1, 2)
        attention.write(k, v, row)
        attended = attention.attend(q, self.lane_rows, self.lane_slot, self.prefix_lengths, self.lengths, row)
        hidden = hidden + block.attn_output(attended.transpose(1, 2).reshape(self.batch, 1, -1)).cast(hidden.dtype)
      elif block.block_type == "mamba":
        hidden = mamba_replay_buffers_step(block, hidden, buffer, slot)
      else:
        hidden, _ = block.cached(hidden, None, keep_graph=True)
      hidden = hidden.contiguous()
    token, chosen = self._sample(hidden, temperature)
    for key, value in (("tokens", token), ("logprobs", chosen)):
      self.history[key][:, column:column + 1].assign(value.reshape(self.batch, 1)).realize()
    return token

  _sample = NemotronHBatchSampler._sample

  def _flush(self, count: UOp) -> None:
    for block, buffer in zip(self.model.blk, self.buffers):
      if block.block_type == "mamba":
        mamba_replay_flush(block, buffer, count)

  def _load_lane(self, lane: UOp, source: UOp) -> None:
    """Lane `lane` takes prompt slot `source`'s Mamba state; its generated keys restart empty."""
    for buffer, state in zip(self.buffers, self.prompt_state):
      if buffer is not None:
        for key in ("conv", "state"):
          buffer[key][lane:lane + 1].assign(state[key][source:source + 1]).realize()
    self.lengths[lane:lane + 1].assign(Tensor.zeros(1, dtype=dtypes.int32)).realize()

  def _start(self, tokens: Tensor, fresh: Tensor, lane_prompt: Tensor, temperature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """First tokens for every lane from its prompt's last hidden state; lanes flagged `fresh` switch to them."""
    token, chosen = self._sample(self.prompt_hidden[lane_prompt], temperature)
    return fresh.where(token, tokens).contiguous().realize(), token, chosen

  # *** host ***

  def _prime(self, slot: int, prompt: list[int]):
    if not 0 < len(prompt) <= self.prefix_capacity:
      raise ValueError("prompt does not fit the prefix capacity")
    if self.prefill is not None:
      hidden, caches = self.prefill(prompt), self.prefill.buffers
    else:
      hidden, caches = self.model.prefix(prompt, through=len(self.model.blk) - 1)
      hidden = hidden[:, -1:]
    for attention, state, cache in zip(self.attention, self.prompt_state, caches):
      if attention is not None:
        attention.load_prefix(slot, cache["k"], cache["v"])
      elif state is not None:
        for key in ("conv", "state"):
          value = cache[key]
          if (short := state[key].shape[1] - value.shape[1]) > 0:  # a prompt shorter than the conv tail
            value = Tensor.zeros(1, short, *value.shape[2:], dtype=value.dtype).cat(value, dim=1)
          state[key][slot:slot + 1].assign(value).realize()
    self.prompt_hidden[slot:slot + 1].assign(hidden.reshape(1, 1, -1).float()).realize()

  def generate(self, requests: list[tuple[list[int], int]], max_new: int, temperature: float = 1.0,
               stop: set[int] | None = None, stats: dict | None = None) -> list[list[tuple[list[int], list[float]]]]:
    """Sample `n` rollouts of up to `max_new` tokens for each `(prompt, n)`; returns per request its (tokens, logprobs).

    A rollout ends at (and includes) its first stop token. `stats`, when given, receives the decode step count,
    the lane-steps spent on rollouts that were still running, and the time spent priming prompts.
    """
    import time
    if max_new > self.capacity:
      raise ValueError("max_new exceeds the generated-token ring")
    stop = stop or set()
    results: list[list] = [[] for _ in requests]
    queue = [(index, prompt, n) for index, (prompt, n) in enumerate(requests) if n > 0]
    slots: list[dict | None] = [None] * self.prompts  # request, rollouts left to start, running rows
    lanes: list[dict | None] = [None] * self.batch
    rate = Tensor([temperature]).realize()
    tokens = _fresh(Tensor.zeros(self.batch, dtype=dtypes.int32))
    steps = active_lane_steps = 0
    prime_time = 0.0

    def refill() -> bool:
      nonlocal tokens, prime_time
      started = []
      for lane in range(self.batch):
        if lanes[lane] is not None:
          continue
        slot = next((g for g, s in enumerate(slots) if s is not None and s["left"] and None in s["rows"]), None)
        if slot is None and queue and None in slots:
          slot = slots.index(None)
          index, prompt, n = queue.pop(0)
          start = time.perf_counter()
          self._prime(slot, prompt)
          prime_time += time.perf_counter() - start
          slots[slot] = {"request": index, "left": n, "rows": [None] * self.rows, "length": len(prompt)}
        if slot is None:
          continue
        s = slots[slot]
        j = s["rows"].index(None)
        s["rows"][j], s["left"] = lane, s["left"] - 1
        lanes[lane] = {"slot": slot, "row": j, "tokens": [], "logprobs": []}
        started.append(lane)
      if not started:
        return any(lane is not None for lane in lanes)
      table = [0] * (self.prompts * self.rows)
      for g, s in enumerate(slots):
        for j, lane in enumerate(s["rows"] if s is not None else []):
          if lane is not None:
            table[g * self.rows + j] = lane
      self.lane_rows.assign(Tensor(table, dtype=dtypes.int32)).realize()
      self.lane_slot.assign(Tensor([0 if l is None else l["slot"] * self.rows + l["row"] for l in lanes],
                                   dtype=dtypes.int32)).realize()
      self.prefix_lengths.assign(Tensor([min(s["length"], self.prefix_capacity) if s is not None else 0
                                         for s in slots], dtype=dtypes.int32)).realize()
      for lane in started:
        self.load_lane(self.lane.bind(lane), self.source.bind(lanes[lane]["slot"]))
      fresh = Tensor([lane in started for lane in range(self.batch)]).realize()
      lane_prompt = Tensor([0 if l is None else l["slot"] for l in lanes], dtype=dtypes.int32).realize()
      tokens, first, chosen = self.start(tokens, fresh, lane_prompt, rate)
      first, chosen = first.numpy().tolist(), chosen.numpy().tolist()
      for lane in started:
        lanes[lane]["tokens"].append(int(first[lane]))
        lanes[lane]["logprobs"].append(float(chosen[lane]))
      retire()
      return True

    def retire():
      for lane, l in enumerate(lanes):
        if l is None:
          continue
        end = next((i for i, t in enumerate(l["tokens"][:max_new]) if t in stop), None)
        if end is None and len(l["tokens"]) < max_new:
          continue
        end = max_new - 1 if end is None else end
        s = slots[l["slot"]]
        results[s["request"]].append((l["tokens"][:end + 1], l["logprobs"][:end + 1]))
        s["rows"][l["row"]], lanes[lane] = None, None
        if not s["left"] and all(r is None for r in s["rows"]):
          slots[l["slot"]] = None

    while refill() or any(lane is not None for lane in lanes):
      for i in range(self.window):
        tokens = self.step(tokens, self.row.bind(steps % self.capacity), self.slot.bind(steps % self.ring),
                           self.column.bind(i), rate)
        steps += 1
        if steps % self.ring == 0:
          self.flush(self.count.bind(self.ring))
      ids, lps = self.history["tokens"].numpy().tolist(), self.history["logprobs"].numpy().tolist()
      for lane, l in enumerate(lanes):
        if l is not None:
          before = len(l["tokens"])
          l["tokens"] += ids[lane]
          l["logprobs"] += lps[lane]
          end = next((i for i, t in enumerate(l["tokens"][:max_new]) if t in stop), max_new - 1)
          active_lane_steps += max(0, min(end + 1, before + self.window) - before)
      retire()
    if stats is not None:
      stats.update(steps=steps, lane_steps=steps * self.batch, active_lane_steps=active_lane_steps,
                   prime_s=prime_time)
    return results


__all__ = ["NemotronHBatchSampler", "NemotronHRolloutSampler"]
