"""Chunked, TinyJit-compiled Nemotron-H prompt prefill.

`NemotronHModel.prefix` slices every scan and attention chunk at a new
offset, so each chunk is scheduled from scratch: about 11 ms of host time per
prompt token even with every kernel compiled, which made a 10k-token prompt
take many minutes. Here the prompt runs in fixed-size pieces through one
compiled graph per piece size, at a symbolic position, into fixed-capacity
key/value buffers and fixed-shape Mamba state. A length that is not a multiple
of the piece size finishes with power-of-two pieces, so a prompt uses at most
log2(piece) extra graphs, each compiled once and reused across prompts.

Mamba and MLP blocks step through the model's own `cached(..., keep_graph=True)`
path; only attention is written here, as in the batched sampler.
"""
from __future__ import annotations

from tinygrad import Tensor, TinyJit, UOp, dtypes


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHPrefill:
  def __init__(self, model, capacity: int, piece: int = 256):
    if piece & (piece - 1) or piece % model.config.scan_chunk:
      raise ValueError("piece must be a power of two and a multiple of the scan chunk")
    self.model, self.capacity, self.piece = model, capacity, piece
    config = model.config
    probe = model.token_embd(Tensor([[0] * config.conv_kernel])).float()
    self.buffers = []
    for block in model.blk:
      probe, cache = block.cached(probe, None)
      if block.block_type == "attention":
        _, heads, _, width = cache["k"].shape
        self.buffers.append({key: _fresh(Tensor.zeros(1, heads, capacity, width, dtype=cache[key].dtype))
                             for key in ("k", "v")})
      elif block.block_type == "mamba":
        self.buffers.append({key: _fresh(Tensor.zeros(*value.shape, dtype=value.dtype)) for key, value in cache.items()})
      else:
        self.buffers.append(None)
    self.position = UOp.variable("prefill_position", 0, capacity - 1)
    self.graphs: dict[int, TinyJit] = {}

  def _run(self, tokens: Tensor, position: UOp) -> Tensor:
    length = tokens.shape[1]
    hidden = self.model.token_embd(tokens).float()
    rows = (Tensor.arange(length) + position).reshape(length, 1)
    for block, buffer in zip(self.model.blk, self.buffers):
      if block.block_type == "attention":
        normed = block.attn_norm(hidden)
        heads, kv_heads, width = block.n_heads, block.n_kv_heads, self.model.config.head_dim
        q = block.attn_q(normed).reshape(1, length, heads, width).transpose(1, 2)
        for key, layer in (("k", block.attn_k), ("v", block.attn_v)):
          new = layer(normed).reshape(1, length, kv_heads, width).transpose(1, 2)
          buffer[key][:, :, position:position + length].assign(new.cast(buffer[key].dtype)).realize()
        allowed = (Tensor.arange(self.capacity).reshape(1, self.capacity) <= rows).reshape(1, 1, length, self.capacity)
        mask = allowed.where(0.0, float("-inf")).cast(hidden.dtype)
        attended = q.scaled_dot_product_attention(buffer["k"], buffer["v"], attn_mask=mask, enable_gqa=True)
        hidden = (hidden + block.attn_output(attended.transpose(1, 2).reshape(1, length, -1)).cast(hidden.dtype))
        hidden = hidden.contiguous().realize()
      elif block.block_type == "mamba":
        hidden, carried = block.cached(hidden, buffer, keep_graph=True)
        hidden, *states = (hidden.contiguous().realize(), *(carried[key].contiguous().realize() for key in buffer))
        for key, state in zip(buffer, states):
          buffer[key].assign(state).realize()
      else:
        hidden, _ = block.cached(hidden, None, keep_graph=True)
        hidden = hidden.contiguous().realize()
    return hidden[:, -1:].contiguous().realize()

  def pieces(self, length: int) -> list[int]:
    """Piece sizes covering `length`: full pieces, then the remainder's powers of two, largest first."""
    sizes, rest = [self.piece] * (length // self.piece), length % self.piece
    size = self.piece
    while rest:
      size //= 2
      if rest >= size:
        sizes.append(size)
        rest -= size
    return sizes

  def reset(self):
    for buffer in self.buffers:
      if buffer is not None:
        for value in buffer.values():
          value.assign(Tensor.zeros(*value.shape, dtype=value.dtype)).realize()

  def __call__(self, prompt: list[int]) -> Tensor:
    """Run the prompt from an empty state; returns the last position's hidden state, shape (1, 1, dim).

    Afterwards `buffers` holds the prompt's state: attention key/value rows
    [0, len(prompt)) and each Mamba block's convolution tail and scan state.
    """
    if not prompt or len(prompt) > self.capacity:
      raise ValueError("prompt must be nonempty and fit the prefill capacity")
    self.reset()
    start, hidden = 0, None
    for size in self.pieces(len(prompt)):
      graph = self.graphs.setdefault(size, TinyJit(self._run))
      tokens = Tensor([prompt[start:start + size]], dtype=dtypes.int32).realize()
      hidden = graph(tokens, self.position.bind(start))
      start += size
    return hidden


__all__ = ["NemotronHPrefill"]
