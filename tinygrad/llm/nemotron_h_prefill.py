"""Chunked, TinyJit-compiled Nemotron-H prompt prefill.

`NemotronHModel.prefix` slices every scan and attention chunk at a new
offset, so each chunk is scheduled from scratch: about 11 ms of host time per
prompt token even with every kernel compiled, which made a 10k-token prompt
take many minutes. Here the prompt runs in fixed-size pieces through compiled
graphs at a symbolic position, into fixed-capacity key/value buffers and
fixed-shape Mamba state. Each piece runs one graph per block kind and weight
geometry (every Mamba block shares one, every MLP block one, and every
attention block one for its projections and one for its output projection),
with the block's weights passed in as graph inputs, plus one attention graph
per key bound or flash block. Per-block graphs keep captured intermediates to
one block (a whole-model graph at a 2048-token piece holds ~22 GB while
capturing; per-segment graphs ran out of VRAM at 1024-token pieces of a 10k
prompt), and hidden states alternate between two persistent buffers per piece
size. A routed projection's tile-padded weight lives in a binding outside the
state dict, so binding a template block's weights rebinds that too. A length that is not a multiple
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
from tinygrad.nn.state import get_state_dict
from tinygrad.llm.dense_candidate_gemm import CandidateBinding, route_scratch
from tinygrad.llm import nemotron_h_prefill_attention as fused_attention
from tinygrad.llm.nemotron_h_ssd import PRECISIONS, ssd_cached


def _fresh(value: Tensor) -> Tensor:
  """A new plain buffer; TinyJit replays only buffers, never lazy constants or views."""
  return Tensor.empty(*value.shape, dtype=value.dtype).assign(value).realize()


class NemotronHPrefill:
  def __init__(self, model, capacity: int, piece: int = 1024, fused: bool | None = None, mamba: str = "ssd",
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
    # the SSD path pads the last piece; its padded rows still need cache rows to write into (masked) and to attend
    self.rows = -(-capacity // piece) * piece if mamba == "ssd" else capacity
    config = model.config
    probe = model.token_embd(Tensor([[0] * config.conv_kernel])).float()
    self.buffers = []
    for block in model.blk:
      probe, cache = block.cached(probe, None)
      if block.block_type == "attention":
        _, heads, _, width = cache["k"].shape
        self.buffers.append({key: _fresh(Tensor.zeros(1, heads, self.rows, width, dtype=cache[key].dtype))
                             for key in ("k", "v")})
      elif block.block_type == "mamba":
        self.buffers.append({key: _fresh(Tensor.zeros(*value.shape, dtype=value.dtype)) for key, value in cache.items()})
      else:
        self.buffers.append(None)
    self.tile = fused_attention.query_tile(config.head_counts[next(i for i, b in enumerate(model.blk)
                                                                     if b.block_type == "attention")],
                                           config.head_dim) if self.fused else None
    self.position = UOp.variable("prefill_position", 0, self.rows - 1)
    self.valid: dict[int, UOp] = {}  # piece size -> its count of real (unpadded) positions
    # weights are graph inputs: one graph per block kind and weight geometry serves every block of that kind
    # (a weight may be a view of a buffer, and blocks may share buffers: the graphs take the distinct buffers)
    self.params = [sorted(get_state_dict(block).items()) for block in model.blk]
    Tensor.realize(*(value for params in self.params for _, value in params))
    self.bases, self.kinds = [], []
    for block, params in zip(model.blk, self.params):
      bases = list(dict.fromkeys(value.uop.base for _, value in params))
      self.bases.append(tuple(Tensor(base) for base in bases))
      self.kinds.append((block.block_type, tuple((name, value.shape, value.dtype, bases.index(value.uop.base))
                                                 for name, value in params),
                         tuple((base.shape, base.dtype) for base in bases)))
    self.templates = {kind: index for index, kind in reversed(list(enumerate(self.kinds)))}
    self.hidden = {}  # piece size -> two float32 (1, size, dim) buffers the block graphs alternate between
    self.queries = {}  # piece size -> the attention query buffer
    self.graphs: dict[tuple[int, int], TinyJit] = {}  # attention, keyed (piece, key bound) or (piece, -1 - block)
    self.block_graphs: dict[tuple, TinyJit] = {}  # (kind, piece) and ("embed", piece), ("attn_out", kind, piece)
    self.bindings = {}  # (template block, projection) -> its own route binding, restored after each graph
    self.flash_blocks: dict[int, bool] = {}
    self._reset = TinyJit(self._zero)
    self.pad_token = 0  # fills a padded piece; nothing the prefill keeps may depend on it

  def _bind(self, kind, bases: tuple[Tensor, ...] | None):
    """Point the template block's weights at the same views of `bases` (None: its own); returns the block."""
    index = self.templates[kind]
    block = self.model.blk[index]
    for (name, old), (_, _, _, slot) in zip(self.params[index], kind[1]):
      value = old if bases is None else Tensor(old.uop.substitute({old.uop.base: bases[slot].uop}))
      *path, leaf = name.split(".")
      owner = block
      for part in path:
        owner = owner[part] if isinstance(owner, dict) else getattr(owner, part)
      if isinstance(owner, dict): owner[leaf] = value
      else: setattr(owner, leaf, value)
    # a routed projection reads its tile-padded weight through a binding outside the state dict; `.weight` is a
    # prefix view of that buffer, so the binding follows the weight's buffer too
    for name, layer in vars(block).items():
      if (binding := getattr(layer, "_candidate", None)) is None: continue
      original = self.bindings.setdefault((index, name), binding)
      if bases is None:
        layer._candidate = original
        continue
      slot = next(slot for (key, _, _, slot) in kind[1] if key == f"{name}.weight")
      padded = Tensor(original.padded.uop.substitute({original.padded.uop.base: bases[slot].uop}))
      layer._candidate = CandidateBinding(original.role, padded)
    return block

  def _embed(self, tokens: Tensor, out: Tensor) -> None:
    out.assign(self.model.token_embd(tokens).float()).realize()

  def _block(self, kind, hidden: Tensor, out: Tensor, position: UOp, valid: UOp, *tensors: Tensor):
    """One Mamba or MLP block (hidden -> out), or an attention block's projections (hidden -> out: the query;
    keys and values written into the cache rows at `position`). `tensors`: the carried state (key/value
    cache or convolution tail and scan state; none for an MLP), then weights replacing the template block's.

    Only the piece's first `valid` positions are the prompt; the rest pad it to the graph's size. Padded positions
    never reach the carried state: their keys and values are not written (the rows keep their contents), and the
    Mamba update treats them as identity steps (`ssd_mixer`'s `valid`). Rows before `valid` are causal, so they
    never read the padding."""
    count = 0 if kind[0] == "mlp" else 2
    state, bases = tensors[:count], tensors[count:]
    block = self._bind(kind, bases)
    try:
      length = hidden.shape[1]
      if block.block_type == "attention":
        normed = block.attn_norm(hidden)
        heads, kv_heads, width = block.n_heads, block.n_kv_heads, self.model.config.head_dim
        out.assign(block.attn_q(normed).reshape(1, length, heads, width).transpose(1, 2)).realize()
        for cache, layer in zip(state, (block.attn_k, block.attn_v)):
          new = layer(normed).reshape(1, length, kv_heads, width).transpose(1, 2)
          # rounded to the key projection dtype, as the model and the sampler store keys and values
          new = new.cast(layer.weight.dtype).cast(cache.dtype)
          rows = cache[:, :, position:position + length]
          keep = Tensor.arange(length).reshape(1, 1, length, 1) < valid
          rows.assign(keep.where(new, rows)).realize()
      elif block.block_type == "mamba":
        carry = dict(zip(("conv", "state"), state))
        if self.mamba == "ssd":
          result, carried = ssd_cached(block, hidden, carry, chunk=self.ssd_chunk, precision=self.ssd_precision,
                                       valid=valid)
        else:
          result, carried = block.cached(hidden, carry, keep_graph=True)
        result, *states = (result.contiguous().realize(), *(carried[key].contiguous().realize() for key in carry))
        for buffer, value in zip(state, states):
          buffer.assign(value).realize()
        out.assign(result).realize()
      else:
        out.assign(block.cached(hidden, None, keep_graph=True)[0]).realize()
    finally:
      self._bind(kind, None)

  def _attn_out(self, kind, hidden: Tensor, attended: Tensor, out: Tensor, *bases: Tensor) -> None:
    block = self._bind(kind, bases)
    try:
      length = hidden.shape[1]
      out.assign(hidden + block.attn_output(attended.transpose(1, 2).reshape(1, length, -1)).cast(hidden.dtype)).realize()
    finally:
      self._bind(kind, None)

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

  def spans(self, length: int) -> list[tuple[int, int]]:
    """(piece size, real positions in it) covering `length`: full pieces, then the remainder.

    On the SSD path the remainder is one piece padded to a power of two of at least the scan chunk, so a prompt
    costs at most one partial piece whatever its length; the scan path splits it into its powers of two.
    """
    full, rest = [(self.piece, self.piece)] * (length // self.piece), length % self.piece
    if not rest:
      return full
    if self.mamba == "ssd":
      return full + [(max(self.model.config.scan_chunk, 1 << (rest - 1).bit_length()), rest)]
    return full + [(size, size) for size in self.pieces(rest)]

  def pieces(self, length: int) -> list[int]:
    """Piece sizes covering `length` unpadded: full pieces, then the remainder's powers of two, largest first."""
    sizes, rest = [self.piece] * (length // self.piece), length % self.piece
    size = self.piece
    while rest:
      size //= 2
      if rest >= size:
        sizes.append(size)
        rest -= size
    return sizes

  def _graph(self, key, function) -> TinyJit:
    if key not in self.block_graphs:
      self.block_graphs[key] = TinyJit(function)
    return self.block_graphs[key]

  def _zero(self):
    for buffer in self.buffers:
      if buffer is not None:
        for value in buffer.values():
          value.assign(Tensor.zeros(*value.shape, dtype=value.dtype)).realize()

  def reset(self):
    """Zero every cache buffer, in one graph (eager, its ~50 assigns cost ~0.1-0.2 s of host time per prompt)."""
    self._reset()

  def __call__(self, prompt: list[int]) -> Tensor:
    """Run the prompt from an empty state; returns the last position's hidden state, shape (1, 1, dim).

    Afterwards `buffers` holds the prompt's state: attention key/value rows
    [0, len(prompt)) and each Mamba block's convolution tail and scan state.
    """
    if not prompt or len(prompt) > self.capacity:
      raise ValueError("prompt must be nonempty and fit the prefill capacity")
    self.reset()
    start, hidden = 0, None
    for size, count in self.spans(len(prompt)):
      # fused: keyed by the piece's 512-token block while the kernel admits it; SDPA: a power-of-two key bound
      block = start // fused_attention.BLOCK
      last = (start + size - 1) // fused_attention.BLOCK
      if self.fused and all(self._flash_admitted(b) for b in range(block, last + 1)):
        key, attend = (size, -1 - block), functools.partial(self._attend, keys=0, flash_block=block)
      else:
        keys = min(self.rows, max(self.piece, 1 << (start + size - 1).bit_length()))
        key, attend = (size, keys), functools.partial(self._attend, keys=keys)
      if key not in self.graphs:
        self.graphs[key] = TinyJit(attend)
      position = self.position.bind(start)
      if size not in self.valid:
        self.valid[size] = UOp.variable(f"prefill_valid_{size}", 1, size)
      valid = self.valid[size].bind(count)
      if size not in self.hidden:
        self.hidden[size] = [Tensor.empty(1, size, self.model.config.dim).contiguous().realize() for _ in range(2)]
      current, spare = self.hidden[size]
      tokens = Tensor([prompt[start:start + count] + [self.pad_token] * (size - count)], dtype=dtypes.int32).realize()
      with route_scratch():
        self._graph(("embed", size), self._embed)(tokens, current)
        for index, (block, kind) in enumerate(zip(self.model.blk, self.kinds)):
          weights = self.bases[index]
          graph = self._graph((kind, size), functools.partial(self._block, kind))
          if block.block_type == "attention":
            k, v = self.buffers[index]["k"], self.buffers[index]["v"]
            if size not in self.queries:
              self.queries[size] = Tensor.empty(1, block.n_heads, size, self.model.config.head_dim).contiguous().realize()
            graph(current, self.queries[size], position, valid, k, v, *weights)
            attended = self.graphs[key](self.queries[size], k, v, position)
            self._graph(("attn_out", kind, size), functools.partial(self._attn_out, kind))(current, attended, spare, *weights)
          elif block.block_type == "mamba":
            graph(current, spare, position, valid, self.buffers[index]["conv"], self.buffers[index]["state"], *weights)
          else:
            graph(current, spare, position, valid, *weights)
          current, spare = spare, current
      hidden = current[:, count - 1:count].contiguous().realize()
      start += count
    return hidden


__all__ = ["NemotronHPrefill"]
