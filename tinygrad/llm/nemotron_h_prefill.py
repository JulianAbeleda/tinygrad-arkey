"""Chunked, TinyJit-compiled Nemotron-H prompt prefill.

`NemotronHModel.prefix` slices every scan and attention chunk at a new
offset, so each chunk is scheduled from scratch: about 11 ms of host time per
prompt token even with every kernel compiled, which made a 10k-token prompt
take many minutes. Here the prompt runs in fixed-size pieces through compiled
graphs at a symbolic position, into fixed-capacity key/value buffers and
fixed-shape Mamba state. Each piece runs one graph per segment (the blocks
between two attention blocks, plus the attention projections) and one
attention graph per attention block. Splitting at attention keeps a graph's
captured intermediates to a segment (a whole-model graph at a 2048-token piece
holds ~22 GB while capturing), and the segment graphs, which hold the big-M
projections, depend only on the piece size, not on where attention's key
bound or flash block falls. A length that is not a multiple
of the piece size finishes with power-of-two pieces, so a prompt uses at most
log2(piece) extra graphs, each compiled once and reused across prompts. Attention reads keys only up to
a power-of-two bound past the piece's last position (one graph per bound), not
the whole capacity.

MLP blocks step through the model's own `cached(..., keep_graph=True)` path.
Mamba blocks run the chunked SSD scan (`nemotron_h_ssd`, `mamba="ssd"`, the
default): the whole piece is one graph of batched matmuls, so large pieces
(2k-8k tokens) stay cheap; `mamba="scan"` keeps the model's sequential
`scan_chunk` loop. Attention is written here, as in the batched sampler.

Where the fused flash-prefill route is admitted (`nemotron_h_prefill_attention`),
attention runs on that generated kernel instead of SDPA, and graphs are keyed by
the 512-token block the piece starts in (the kernel's fixed geometry), not by a
power-of-two key bound. A piece larger than 512 tokens (512-aligned, as full
pieces are) runs its attention as one flash call per 512-token block, so the
projections, MLPs and Mamba scans still see the whole piece.
"""
from __future__ import annotations

import functools

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.llm.dense_candidate_gemm import route_scratch
from tinygrad.llm import nemotron_h_prefill_attention as fused_attention
from tinygrad.llm.nemotron_h_ssd import PRECISIONS, ssd_cached


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHPrefill:
  def __init__(self, model, capacity: int, piece: int = 256, fused: bool | None = None, mamba: str = "ssd",
               ssd_chunk: int = 256, ssd_precision: str = "float"):
    if piece & (piece - 1) or piece % model.config.scan_chunk:
      raise ValueError("piece must be a power of two and a multiple of the scan chunk")
    if mamba not in ("ssd", "scan") or ssd_precision not in PRECISIONS:
      raise ValueError(f"mamba must be 'ssd' or 'scan' and ssd_precision one of {PRECISIONS}")
    self.mamba, self.ssd_chunk, self.ssd_precision = mamba, ssd_chunk, ssd_precision
    self.fused = fused_attention.fused_prefill_supported(model) if fused is None else fused
    if self.fused:
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
    # segments end at each attention block: [bounds[i], bounds[i+1]) holds Mamba/MLP blocks, then one attention block
    attention = [i for i, b in enumerate(model.blk) if b.block_type == "attention"]
    self.bounds = [0, *(i + 1 for i in attention), len(model.blk)]
    self.graphs: dict[tuple[int, int], TinyJit] = {}  # attention, keyed (piece, key bound) or (piece, -1 - block)
    self.segment_graphs: dict[tuple[int, int], TinyJit] = {}  # (segment, piece)
    self.flash_blocks: dict[int, bool] = {}

  def _segment(self, index: int, inputs: Tensor, attended: Tensor | None, position: UOp) -> tuple[Tensor, ...]:
    """One segment: the previous attention block's output projection, the Mamba/MLP blocks up to the next
    attention block, and that block's projections (its key/value rows written into the cache).

    Returns the hidden state, then the next attention block's float32 query (1, heads, length, Hd); the last
    segment returns only the last position's hidden state.
    """
    first, stop = self.bounds[index], self.bounds[index + 1]
    if index == 0:
      hidden = self.model.token_embd(inputs).float()
    else:
      hidden, block = inputs, self.model.blk[first - 1]
      length = hidden.shape[1]
      hidden = hidden + block.attn_output(attended.transpose(1, 2).reshape(1, length, -1)).cast(hidden.dtype)
    length = hidden.shape[1]
    for block, buffer in zip(self.model.blk[first:stop], self.buffers[first:stop]):
      if block.block_type == "attention":  # the segment's last block: projections only
        normed = block.attn_norm(hidden)
        heads, kv_heads, width = block.n_heads, block.n_kv_heads, self.model.config.head_dim
        q = block.attn_q(normed).reshape(1, length, heads, width).transpose(1, 2)
        for key, layer in (("k", block.attn_k), ("v", block.attn_v)):
          new = layer(normed).reshape(1, length, kv_heads, width).transpose(1, 2)
          # rounded to the key projection dtype, as the model and the sampler store keys and values
          buffer[key][:, :, position:position + length].assign(new.cast(layer.weight.dtype).cast(buffer[key].dtype)).realize()
        return hidden.contiguous().realize(), q.contiguous().realize()
      if block.block_type == "mamba":
        if self.mamba == "ssd":
          hidden, carried = ssd_cached(block, hidden, buffer, chunk=self.ssd_chunk, precision=self.ssd_precision)
        else:
          hidden, carried = block.cached(hidden, buffer, keep_graph=True)
        hidden, *states = (hidden.contiguous().realize(), *(carried[key].contiguous().realize() for key in buffer))
        for key, state in zip(buffer, states):
          buffer[key].assign(state).realize()
      else:
        hidden, _ = block.cached(hidden, None, keep_graph=True)
        hidden = hidden.contiguous().realize()
    return (hidden[:, -1:].contiguous().realize(),)

  def _attend(self, q: Tensor, k: Tensor, v: Tensor, position: UOp, keys: int, flash_block: int | None = None) -> Tensor:
    """Causal attention of the piece's queries over the cache: float32 (1, heads, length, Hd)."""
    length = q.shape[2]
    if flash_block is not None:
      step = min(length, fused_attention.BLOCK)  # one flash call per 512-token block of the piece
      return Tensor.cat(*(fused_attention.fused_prefill_attention(
        q[:, :, offset:offset + step], k, v, self.tile, position + offset, step,
        flash_block + offset // fused_attention.BLOCK) for offset in range(0, length, step)), dim=2).contiguous().realize()
    if self.fused:  # a block the flash kernel does not admit: tensor-core matmuls over the key bound
      return fused_attention.grouped_prefill_attention(q, k, v, position, keys).contiguous().realize()
    # keys: a power-of-two bound on the positions this piece can see, not the full capacity
    rows = (Tensor.arange(length) + position).reshape(length, 1)
    allowed = (Tensor.arange(keys).reshape(1, keys) <= rows).reshape(1, 1, length, keys)
    mask = allowed.where(0.0, float("-inf"))
    return q.scaled_dot_product_attention(k[:, :, :keys], v[:, :, :keys], attn_mask=mask,
                                          enable_gqa=True).contiguous().realize()

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
      last = (start + size - 1) // fused_attention.BLOCK
      if self.fused and all(self._flash_admitted(b) for b in range(block, last + 1)):
        key, attend = (size, -1 - block), functools.partial(self._attend, keys=0, flash_block=block)
      else:
        keys = min(self.capacity, max(self.piece, 1 << (start + size - 1).bit_length()))
        key, attend = (size, keys), functools.partial(self._attend, keys=keys)
      if key not in self.graphs:
        self.graphs[key] = TinyJit(attend)
      position = self.position.bind(start)
      out = (Tensor([prompt[start:start + size]], dtype=dtypes.int32).realize(),)
      attended = None
      with route_scratch():
        for index in range(len(self.bounds) - 1):
          if (index, size) not in self.segment_graphs:
            self.segment_graphs[(index, size)] = TinyJit(functools.partial(self._segment, index))
          out = self.segment_graphs[(index, size)](out[0], attended, position)
          if len(out) == 2:
            buffer = self.buffers[self.bounds[index + 1] - 1]
            attended = self.graphs[key](out[1], buffer["k"], buffer["v"], position)
      hidden = out[0]
      start += size
    return hidden


__all__ = ["NemotronHPrefill"]
