"""Chunked, TinyJit-compiled Nemotron-H prompt prefill.

`NemotronHModel.prefix` slices every scan and attention chunk at a new
offset, so each chunk is scheduled from scratch: about 11 ms of host time per
prompt token even with every kernel compiled, which made a 10k-token prompt
take many minutes. Here the prompt runs in fixed-size pieces through one
compiled graph per piece size, at a symbolic position, into fixed-capacity
key/value buffers and fixed-shape Mamba state. A length that is not a multiple
of the piece size finishes with power-of-two pieces, so a prompt uses at most
log2(piece) extra graphs, each compiled once and reused across prompts. Attention reads keys only up to
a power-of-two bound past the piece's last position (one graph per bound), not
the whole capacity.

Mamba and MLP blocks step through the model's own `cached(..., keep_graph=True)`
path; only attention is written here, as in the batched sampler.

Where the fused flash-prefill route is admitted (`nemotron_h_prefill_attention`),
attention runs on that generated kernel instead of SDPA: pieces are at most 512
tokens and graphs are keyed by the 512-token block the piece lies in (the
kernel's fixed geometry), not by a power-of-two key bound.
"""
from __future__ import annotations

import functools

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm import nemotron_h_prefill_attention as fused_attention


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHPrefill:
  def __init__(self, model, capacity: int, piece: int = 256, fused: bool | None = None):
    if piece & (piece - 1) or piece % model.config.scan_chunk:
      raise ValueError("piece must be a power of two and a multiple of the scan chunk")
    self.fused = fused_attention.fused_prefill_supported(model) if fused is None else fused
    if self.fused:
      piece = min(piece, fused_attention.BLOCK)
      capacity = -(-capacity // fused_attention.BLOCK) * fused_attention.BLOCK  # the kernel reads whole blocks
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
    self.tile = fused_attention.query_tile(config.head_counts[next(i for i, b in enumerate(model.blk)
                                                                     if b.block_type == "attention")],
                                           config.head_dim) if self.fused else None
    self.position = UOp.variable("prefill_position", 0, capacity - 1)
    self.graphs: dict[tuple[int, int], TinyJit] = {}
    self.flash_blocks: dict[int, bool] = {}

  def _run(self, tokens: Tensor, position: UOp, keys: int, flash_block: int | None = None) -> Tensor:
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
        if flash_block is not None:
          attended = fused_attention.fused_prefill_attention(q, buffer["k"], buffer["v"], self.tile, position, length,
                                                             flash_block)
          hidden = (hidden + block.attn_output(attended.transpose(1, 2).reshape(1, length, -1)).cast(hidden.dtype))
          hidden = hidden.contiguous().realize()
          continue
        if self.fused:  # a block the flash kernel does not admit: tensor-core matmuls over the key bound
          attended = fused_attention.grouped_prefill_attention(q, buffer["k"], buffer["v"], position, keys)
          hidden = (hidden + block.attn_output(attended.transpose(1, 2).reshape(1, length, -1)).cast(hidden.dtype))
          hidden = hidden.contiguous().realize()
          continue
        # keys: a power-of-two bound on the positions this piece can see, not the full capacity
        allowed = (Tensor.arange(keys).reshape(1, keys) <= rows).reshape(1, 1, length, keys)
        mask = allowed.where(0.0, float("-inf")).cast(hidden.dtype)
        attended = q.scaled_dot_product_attention(buffer["k"][:, :, :keys], buffer["v"][:, :, :keys], attn_mask=mask,
                                                  enable_gqa=True)
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

  def _flash_admitted(self, block: int) -> bool:
    if block not in self.flash_blocks:
      attention = next(b for b in self.model.blk if b.block_type == "attention")
      self.flash_blocks[block] = fused_attention.block_admitted(block, attention.n_heads, attention.n_kv_heads,
                                                                self.model.config.head_dim)
    return self.flash_blocks[block]

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
      # fused: keyed by the piece's 512-token block while the kernel admits it; SDPA: a power-of-two key bound
      block = start // fused_attention.BLOCK
      if self.fused and self._flash_admitted(block):
        key, run = (size, -1 - block), functools.partial(self._run, keys=0, flash_block=block)
      else:
        keys = min(self.capacity, max(self.piece, 1 << (start + size - 1).bit_length()))
        key, run = (size, keys), functools.partial(self._run, keys=keys)
      graph = self.graphs.get(key)
      if graph is None:
        graph = self.graphs[key] = TinyJit(run)
      tokens = Tensor([prompt[start:start + size]], dtype=dtypes.int32).realize()
      hidden = graph(tokens, self.position.bind(start))
      start += size
    return hidden


__all__ = ["NemotronHPrefill"]
