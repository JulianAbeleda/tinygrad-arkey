"""tinygrad forward model for NVIDIA Nemotron-H hybrid GGUF checkpoints.

Nemotron-H schedules separate Mamba-2, attention, and MLP residual blocks.  It
cannot be represented by tinygrad's ordinary ``Transformer`` surface, where
every block contains attention followed by an FFN.  This module owns that model
family boundary while exposing the same ``token_embd``/``blk``/``output`` names
used by DayCare's model-neutral completion-loss path. Ported from DayCare
(daycare/model/nemotron_h.py) so tinygrad-arkey can sample and train it.

The Mamba-2 scan is exact but chunked.  Within a chunk it contracts the
state-space recurrence like causal attention; between chunks it carries one
state tensor.  This avoids a Python operation per token and avoids the unstable
``cumprod``/division form of a linear recurrence.
"""
from __future__ import annotations

from dataclasses import dataclass
import pathlib

from tinygrad import Tensor, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.nn.state import load_state_dict


@dataclass(frozen=True)
class NemotronHConfig:
    num_blocks: int
    dim: int
    vocab_size: int
    norm_eps: float
    max_context: int
    block_types: tuple[str, ...]
    head_counts: tuple[int, ...]
    kv_head_counts: tuple[int, ...]
    ffn_dims: tuple[int, ...]
    head_dim: int
    ssm_inner: int
    ssm_state: int
    ssm_groups: int
    ssm_heads: int
    conv_kernel: int
    scan_chunk: int = 64


def _per_layer(value, count: int, default: int = 0) -> tuple[int, ...]:
    if value is None:
        return (default,) * count
    if isinstance(value, list):
        if len(value) != count:
            raise ValueError(f"per-layer metadata has {len(value)} values, expected {count}")
        return tuple(int(item) for item in value)
    return (int(value),) * count


def config_from_gguf(metadata: dict, *, max_context: int | None = None,
                     scan_chunk: int = 64) -> NemotronHConfig:
    arch = str(metadata.get("general.architecture", ""))
    if arch != "nemotron_h":
        raise ValueError(f"Nemotron-H loader requires 'nemotron_h', got {arch!r}")
    count = int(metadata[f"{arch}.block_count"])
    heads = _per_layer(metadata.get(f"{arch}.attention.head_count"), count)
    kv_heads = _per_layer(metadata.get(f"{arch}.attention.head_count_kv"), count)
    ffn_dims = _per_layer(metadata.get(f"{arch}.feed_forward_length"), count)
    block_types = tuple(
        "mamba" if kv == 0 and ffn == 0 else "mlp" if ffn > 0 else "attention"
        for kv, ffn in zip(kv_heads, ffn_dims)
    )
    context = int(metadata[f"{arch}.context_length"])
    if max_context is not None:
        context = min(context, int(max_context))
    return NemotronHConfig(
        num_blocks=count,
        dim=int(metadata[f"{arch}.embedding_length"]),
        vocab_size=len(metadata["tokenizer.ggml.tokens"]),
        norm_eps=float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"]),
        max_context=context,
        block_types=block_types,
        head_counts=heads,
        kv_head_counts=kv_heads,
        ffn_dims=ffn_dims,
        head_dim=int(metadata[f"{arch}.attention.key_length"]),
        ssm_inner=int(metadata[f"{arch}.ssm.inner_size"]),
        ssm_state=int(metadata[f"{arch}.ssm.state_size"]),
        ssm_groups=int(metadata[f"{arch}.ssm.group_count"]),
        ssm_heads=int(metadata[f"{arch}.ssm.time_step_rank"]),
        conv_kernel=int(metadata[f"{arch}.ssm.conv_kernel"]),
        scan_chunk=scan_chunk,
    )


class _WeightBias:
    def __init__(self, weight_shape: tuple[int, ...], bias_shape: tuple[int, ...] | None = None):
        self.weight = Tensor.zeros(*weight_shape)
        if bias_shape is not None:
            self.bias = Tensor.zeros(*bias_shape)


class NemotronHMamba2:
    def __init__(self, config: NemotronHConfig):
        conv_dim = config.ssm_inner + 2 * config.ssm_groups * config.ssm_state
        projection = config.ssm_inner + conv_dim + config.ssm_heads
        self.config = config
        self.ssm_in = nn.Linear(config.dim, projection, bias=False)
        self.ssm_conv1d = _WeightBias((conv_dim, config.conv_kernel), (conv_dim,))
        self.ssm_dt = {"bias": Tensor.zeros(config.ssm_heads)}
        self.ssm_a = Tensor.zeros(config.ssm_heads, 1)
        self.ssm_d = Tensor.zeros(config.ssm_heads, 1)
        self.ssm_norm = _WeightBias((config.ssm_groups, config.ssm_inner // config.ssm_groups))
        self.ssm_out = nn.Linear(config.ssm_inner, config.dim, bias=False)

    def _causal_conv(self, values: Tensor) -> Tensor:
        batch, length, channels = values.shape
        terms = []
        for lag in range(self.config.conv_kernel):
            if lag >= length:
                shifted = Tensor.zeros(batch, length, channels, device=values.device, dtype=values.dtype)
            elif lag:
                prefix = Tensor.zeros(batch, lag, channels, device=values.device, dtype=values.dtype)
                shifted = prefix.cat(values[:, : length - lag], dim=1)
            else:
                shifted = values
            weight_index = self.config.conv_kernel - 1 - lag
            terms.append(shifted * self.ssm_conv1d.weight[:, weight_index])
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return (result + self.ssm_conv1d.bias).silu()

    def _scan_chunk(self, x: Tensor, b: Tensor, c: Tensor, log_a: Tensor,
                    state: Tensor) -> tuple[Tensor, Tensor]:
        # Inputs use (batch, time, heads, feature/state); state is (batch, heads, feature, state).
        _, length, heads, _ = x.shape
        cumulative = log_a.cumsum(axis=1).transpose(1, 2)  # (batch, heads, time)
        positions = Tensor.arange(length).to(x.device)
        causal = (positions.reshape(length, 1) >= positions.reshape(1, length)).reshape(1, 1, length, length)
        # Mask before exp: strongly negative A can make the unused upper
        # triangle strongly positive.  ``exp(value) * 0`` would become NaN.
        decay_log = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
        decay = causal.where(decay_log, float("-inf")).exp()

        bh_c = c.transpose(1, 2)  # (batch, heads, time, state)
        bh_b = b.transpose(1, 2)
        bh_x = x.transpose(1, 2)  # (batch, heads, time, feature)
        weights = (bh_c @ bh_b.transpose(-1, -2)) * decay
        diagonal = weights @ bh_x

        previous = (bh_c @ state.transpose(-1, -2)) * cumulative.exp().unsqueeze(-1)
        output = (diagonal + previous).transpose(1, 2)

        end_decay = (cumulative[:, :, -1:] - cumulative).exp()
        contribution = bh_x.transpose(-1, -2) @ (bh_b * end_decay.unsqueeze(-1))
        next_state = state * cumulative[:, :, -1].exp().unsqueeze(-1).unsqueeze(-1) + contribution
        return output, next_state

    def __call__(self, hidden: Tensor, *, retain_graph: bool = True) -> Tensor:
        cfg = self.config
        batch, length, _ = hidden.shape
        projected = self.ssm_in(hidden)
        conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state
        gate, xbc, dt = projected.split((cfg.ssm_inner, conv_dim, cfg.ssm_heads), dim=-1)
        xbc = NemotronHMamba2._causal_conv(self, xbc)
        x, b, c = xbc.split(
            (cfg.ssm_inner, cfg.ssm_groups * cfg.ssm_state, cfg.ssm_groups * cfg.ssm_state),
            dim=-1,
        )
        x = x.reshape(batch, length, cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads).float()
        b = b.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
        c = c.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
        repeats = cfg.ssm_heads // cfg.ssm_groups
        b = b.unsqueeze(3).expand(batch, length, cfg.ssm_groups, repeats, cfg.ssm_state).reshape(
            batch, length, cfg.ssm_heads, cfg.ssm_state
        )
        c = c.unsqueeze(3).expand(batch, length, cfg.ssm_groups, repeats, cfg.ssm_state).reshape(
            batch, length, cfg.ssm_heads, cfg.ssm_state
        )
        dt = (dt + self.ssm_dt["bias"]).softplus().float()
        # GGUF conversion stores the continuous decay directly as
        # ``A = -exp(A_log)`` (not the source checkpoint's A_log parameter).
        log_a = dt * self.ssm_a.reshape(cfg.ssm_heads).float()
        x_dt = x * dt.unsqueeze(-1)
        if not retain_graph:
            gate, x, b, c, log_a, x_dt = (
                value.realize() for value in (gate, x, b, c, log_a, x_dt)
            )
        state = Tensor.zeros(
            batch, cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads, cfg.ssm_state,
            device=hidden.device,
        )
        outputs = []
        for start in range(0, length, cfg.scan_chunk):
            stop = min(start + cfg.scan_chunk, length)
            output, state = NemotronHMamba2._scan_chunk(
                self,
                x_dt[:, start:stop], b[:, start:stop], c[:, start:stop],
                log_a[:, start:stop], state,
            )
            if not retain_graph:
                output, state = output.realize(), state.realize()
            outputs.append(output)
        y = outputs[0]
        for output in outputs[1:]:
            y = y.cat(output, dim=1)
        y = y + x * self.ssm_d.reshape(1, 1, cfg.ssm_heads, 1)
        y = y.reshape(batch, length, cfg.ssm_groups, cfg.ssm_inner // cfg.ssm_groups)
        gated = y * gate.reshape(y.shape).silu()
        normalized = gated * (gated.square().mean(axis=-1, keepdim=True) + cfg.norm_eps).rsqrt()
        normalized = normalized.reshape(batch, length, cfg.ssm_inner) * self.ssm_norm.weight.reshape(cfg.ssm_inner)
        return self.ssm_out(normalized.cast(hidden.dtype))

    def cached(self, hidden: Tensor, cache: dict | None = None, *,
               keep_graph: bool = False) -> tuple[Tensor, dict]:
        """Run only new tokens and carry convolution plus recurrent state.

        `keep_graph` skips every realize so gradients reach the incoming state
        and the weights (segment-wise training); realize severs autograd.
        """
        cfg = self.config
        batch, length, _ = hidden.shape
        projected = self.ssm_in(hidden)
        conv_dim = cfg.ssm_inner + 2 * cfg.ssm_groups * cfg.ssm_state
        gate, raw_xbc, dt = projected.split((cfg.ssm_inner, conv_dim, cfg.ssm_heads), dim=-1)
        previous_conv = None if cache is None else cache["conv"]
        combined = raw_xbc if previous_conv is None else previous_conv.cat(raw_xbc, dim=1)
        xbc = NemotronHMamba2._causal_conv(self, combined)[:, -length:]
        conv_tail = combined[:, -(cfg.conv_kernel - 1):]
        if not keep_graph:
            # contiguous: a realized slice is a view that keeps the whole
            # prefix-length buffer alive (about 14 GB over a 10k-token prefix).
            conv_tail = conv_tail.contiguous().realize()
        x, b, c = xbc.split(
            (cfg.ssm_inner, cfg.ssm_groups * cfg.ssm_state, cfg.ssm_groups * cfg.ssm_state), dim=-1
        )
        x = x.reshape(batch, length, cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads).float()
        b = b.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
        c = c.reshape(batch, length, cfg.ssm_groups, cfg.ssm_state).float()
        repeats = cfg.ssm_heads // cfg.ssm_groups
        b = b.unsqueeze(3).expand(batch, length, cfg.ssm_groups, repeats, cfg.ssm_state).reshape(
            batch, length, cfg.ssm_heads, cfg.ssm_state
        )
        c = c.unsqueeze(3).expand(batch, length, cfg.ssm_groups, repeats, cfg.ssm_state).reshape(
            batch, length, cfg.ssm_heads, cfg.ssm_state
        )
        dt = (dt + self.ssm_dt["bias"]).softplus().float()
        log_a = dt * self.ssm_a.reshape(cfg.ssm_heads).float()
        x_dt = x * dt.unsqueeze(-1)
        # The prefix projection spans ~10k tokens. Materialize it before the
        # scan so tinygrad does not fuse that full graph into every small
        # recurrent chunk's CUDA kernel.
        if not keep_graph:
            gate, x, b, c, log_a, x_dt = (
                value.realize() for value in (gate, x, b, c, log_a, x_dt)
            )
        state = cache["state"] if cache is not None else Tensor.zeros(
            batch, cfg.ssm_heads, cfg.ssm_inner // cfg.ssm_heads, cfg.ssm_state,
            device=hidden.device,
        )
        outputs = []
        for start in range(0, length, cfg.scan_chunk):
            stop = min(start + cfg.scan_chunk, length)
            output, state = NemotronHMamba2._scan_chunk(
                self, x_dt[:, start:stop], b[:, start:stop], c[:, start:stop],
                log_a[:, start:stop], state,
            )
            if not keep_graph:
                output, state = output.realize(), state.realize()
            outputs.append(output)
        y = outputs[0]
        for output in outputs[1:]:
            y = y.cat(output, dim=1)
        y = y + x * self.ssm_d.reshape(1, 1, cfg.ssm_heads, 1)
        y = y.reshape(batch, length, cfg.ssm_groups, cfg.ssm_inner // cfg.ssm_groups)
        gated = y * gate.reshape(y.shape).silu()
        normalized = gated * (gated.square().mean(axis=-1, keepdim=True) + cfg.norm_eps).rsqrt()
        normalized = normalized.reshape(batch, length, cfg.ssm_inner) * self.ssm_norm.weight.reshape(cfg.ssm_inner)
        return self.ssm_out(normalized.cast(hidden.dtype)), {"conv": conv_tail, "state": state}


class NemotronHAttention:
    def __init__(self, config: NemotronHConfig, layer: int):
        self.config = config
        self.n_heads = config.head_counts[layer]
        self.n_kv_heads = config.kv_head_counts[layer]
        self.attn_q = nn.Linear(config.dim, self.n_heads * config.head_dim, bias=False)
        self.attn_k = nn.Linear(config.dim, self.n_kv_heads * config.head_dim, bias=False)
        self.attn_v = nn.Linear(config.dim, self.n_kv_heads * config.head_dim, bias=False)
        self.attn_output = nn.Linear(self.n_heads * config.head_dim, config.dim, bias=False)

    def __call__(self, hidden: Tensor, *, retain_graph: bool = True) -> Tensor:
        batch, length, _ = hidden.shape
        q = self.attn_q(hidden).reshape(batch, length, self.n_heads, -1).transpose(1, 2)
        k = self.attn_k(hidden).reshape(batch, length, self.n_kv_heads, -1).transpose(1, 2)
        v = self.attn_v(hidden).reshape(batch, length, self.n_kv_heads, -1).transpose(1, 2)
        if not retain_graph:
            q, k, v = q.realize(), k.realize(), v.realize()
        # A 10k-token GameTerm envelope would materialize a multi-gigabyte
        # score matrix. Query chunks preserve exact causal attention while
        # bounding the temporary to chunk x prefix. Keys and values remain
        # shared, so this changes memory shape rather than model semantics.
        chunks = []
        query_chunk = max(128, self.config.scan_chunk * 2)
        for start in range(0, length, query_chunk):
            stop = min(start + query_chunk, length)
            q_chunk, k_prefix, v_prefix = q[:, :, start:stop], k[:, :, :stop], v[:, :, :stop]
            q_pos = Tensor.arange(start, stop).to(hidden.device).reshape(-1, 1)
            k_pos = Tensor.arange(stop).to(hidden.device).reshape(1, -1)
            allowed = (q_pos >= k_pos).reshape(1, 1, stop - start, stop)
            mask = allowed.where(0.0, float("-inf")).cast(hidden.dtype)
            attended = q_chunk.scaled_dot_product_attention(
                k_prefix, v_prefix, attn_mask=mask, enable_gqa=True
            )
            chunks.append(attended if retain_graph else attended.realize())
        attended = chunks[0]
        for chunk in chunks[1:]:
            attended = attended.cat(chunk, dim=2)
        return self.attn_output(attended.transpose(1, 2).reshape(batch, length, -1))

    def cached(self, hidden: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        batch, length, _ = hidden.shape
        q = self.attn_q(hidden).reshape(batch, length, self.n_heads, -1).transpose(1, 2).realize()
        new_k = self.attn_k(hidden).reshape(batch, length, self.n_kv_heads, -1).transpose(1, 2).realize()
        new_v = self.attn_v(hidden).reshape(batch, length, self.n_kv_heads, -1).transpose(1, 2).realize()
        prefix = 0 if cache is None else cache["k"].shape[2]
        k = new_k if cache is None else cache["k"].cat(new_k, dim=2).realize()
        v = new_v if cache is None else cache["v"].cat(new_v, dim=2).realize()
        chunks = []
        query_chunk = max(128, self.config.scan_chunk * 2)
        for start in range(0, length, query_chunk):
            stop = min(start + query_chunk, length)
            key_stop = prefix + stop
            q_chunk = q[:, :, start:stop]
            q_pos = Tensor.arange(prefix + start, prefix + stop).to(hidden.device).reshape(-1, 1)
            k_pos = Tensor.arange(key_stop).to(hidden.device).reshape(1, -1)
            allowed = (q_pos >= k_pos).reshape(1, 1, stop - start, key_stop)
            mask = allowed.where(0.0, float("-inf")).cast(hidden.dtype)
            chunks.append(q_chunk.scaled_dot_product_attention(
                k[:, :, :key_stop], v[:, :, :key_stop], attn_mask=mask, enable_gqa=True
            ).realize())
        attended = chunks[0]
        for chunk in chunks[1:]:
            attended = attended.cat(chunk, dim=2)
        output = self.attn_output(attended.transpose(1, 2).reshape(batch, length, -1))
        return output, {"k": k, "v": v}


class NemotronHMLP:
    def __init__(self, config: NemotronHConfig, layer: int):
        hidden = config.ffn_dims[layer]
        self.ffn_up = nn.Linear(config.dim, hidden, bias=False)
        self.ffn_down = nn.Linear(hidden, config.dim, bias=False)

    def __call__(self, hidden: Tensor, *, retain_graph: bool = True) -> Tensor:
        return self.ffn_down(self.ffn_up(hidden).relu().square())


class NemotronHBlock:
    def __init__(self, config: NemotronHConfig, layer: int):
        self.config = config
        self.block_type = config.block_types[layer]
        self.attn_norm = nn.RMSNorm(config.dim, config.norm_eps)
        if self.block_type == "mamba":
            conv_dim = config.ssm_inner + 2 * config.ssm_groups * config.ssm_state
            projection = config.ssm_inner + conv_dim + config.ssm_heads
            self.ssm_in = nn.Linear(config.dim, projection, bias=False)
            self.ssm_conv1d = _WeightBias((conv_dim, config.conv_kernel), (conv_dim,))
            self.ssm_dt = {"bias": Tensor.zeros(config.ssm_heads)}
            self.ssm_a = Tensor.zeros(config.ssm_heads, 1)
            self.ssm_d = Tensor.zeros(config.ssm_heads, 1)
            self.ssm_norm = _WeightBias((config.ssm_groups, config.ssm_inner // config.ssm_groups))
            self.ssm_out = nn.Linear(config.ssm_inner, config.dim, bias=False)
        elif self.block_type == "attention":
            self.n_heads = config.head_counts[layer]
            self.n_kv_heads = config.kv_head_counts[layer]
            self.attn_q = nn.Linear(config.dim, self.n_heads * config.head_dim, bias=False)
            self.attn_k = nn.Linear(config.dim, self.n_kv_heads * config.head_dim, bias=False)
            self.attn_v = nn.Linear(config.dim, self.n_kv_heads * config.head_dim, bias=False)
            self.attn_output = nn.Linear(self.n_heads * config.head_dim, config.dim, bias=False)
        else:
            hidden = config.ffn_dims[layer]
            self.ffn_up = nn.Linear(config.dim, hidden, bias=False)
            self.ffn_down = nn.Linear(hidden, config.dim, bias=False)

    def __call__(self, hidden: Tensor, *, retain_graph: bool = True) -> Tensor:
        normalized = self.attn_norm(hidden)
        if self.block_type == "mamba":
            mixed = NemotronHMamba2.__call__(self, normalized, retain_graph=retain_graph)
        elif self.block_type == "attention":
            mixed = NemotronHAttention.__call__(self, normalized, retain_graph=retain_graph)
        else:
            mixed = NemotronHMLP.__call__(self, normalized, retain_graph=retain_graph)
        return hidden + mixed.cast(hidden.dtype)

    def cached(self, hidden: Tensor, cache: dict | None = None, *,
               keep_graph: bool = False) -> tuple[Tensor, dict | None]:
        normalized = self.attn_norm(hidden)
        if self.block_type == "mamba":
            mixed, next_cache = NemotronHMamba2.cached(self, normalized, cache, keep_graph=keep_graph)
        elif self.block_type == "attention":
            if keep_graph:
                raise ValueError("segment-wise gradients are implemented for Mamba and MLP blocks only")
            mixed, next_cache = NemotronHAttention.cached(self, normalized, cache)
        else:
            mixed, next_cache = NemotronHMLP.__call__(self, normalized), None
        out = hidden + mixed.cast(hidden.dtype)
        return (out if keep_graph else out.realize()), next_cache


class NemotronHModel:
    def __init__(self, config: NemotronHConfig):
        self.config = config
        self.blk = [NemotronHBlock(config, layer) for layer in range(config.num_blocks)]
        self.token_embd = nn.Embedding(config.vocab_size, config.dim)
        self.output_norm = nn.RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def prefix(self, ids: list[int], *, through: int) -> tuple[Tensor, list[dict | None]]:
        """Run one shared causal prefix once: its states after `through`, and the caches."""
        # float32 residual stream, as in the whole-sequence path (bf16 drifted ~0.3% per layer)
        hidden = self.token_embd(Tensor([ids])).float().realize()
        caches = []
        for block in self.blk[:through + 1]:
            hidden, cache = block.cached(hidden)
            caches.append(cache)
        return hidden, caches

    def prime(self, ids: list[int], *, through: int) -> list[dict | None]:
        """Cache one shared causal prefix through the requested frozen layer."""
        return self.prefix(ids, through=through)[1]

    def suffix(self, ids: list[int], caches: list[dict | None], *, through: int) -> Tensor:
        """Propagate a short suffix against immutable shared-prefix caches."""
        if len(caches) != through + 1:
            raise ValueError("prefix cache depth does not match requested frozen stack")
        return self.advance(ids, caches, through=through)[0]

    def advance(self, ids: list[int], caches: list[dict | None], *, through: int):
        """Advance immutable frozen-prefix state and return the new token states."""
        if not ids or len(caches) != through + 1:
            raise ValueError("nonempty tokens and matching cache depth required")
        # float32 residual stream, as in the whole-sequence path (bf16 drifted ~0.3% per layer)
        hidden = self.token_embd(Tensor([ids])).float().realize()
        next_caches = []
        for index, block in enumerate(self.blk[:through + 1]):
            hidden, cache = block.cached(hidden, caches[index])
            next_caches.append(cache)
        return hidden, next_caches


def load_state(metadata: dict, state: dict[str, Tensor], *, max_context: int | None = None,
               scan_chunk: int = 64) -> NemotronHModel:
    config = config_from_gguf(metadata, max_context=max_context, scan_chunk=scan_chunk)
    model = NemotronHModel(config)
    # Keep NVIDIA's training release at its source BF16 precision. Quantized
    # GGUF tensors dequantize to float and are narrowed for the memory-bounded
    # compatibility/evaluation path.
    if int(metadata.get("general.file_type", -1)) != 32:  # GGML_FTYPE_MOSTLY_BF16
        state = {name: value.cast("float16") for name, value in state.items()}
    if "output.weight" not in state:
        state["output.weight"] = state["token_embd.weight"]
    load_state_dict(model, state, verbose=False, consume=True, realize=False)
    return model


def load(path: str | pathlib.Path, *, max_context: int | None = None,
         scan_chunk: int = 64) -> tuple[NemotronHModel, dict]:
    metadata, state = gguf_load(path)
    model = load_state(metadata, state, max_context=max_context, scan_chunk=scan_chunk)
    return model, metadata


__all__ = [
    "NemotronHBlock", "NemotronHConfig", "NemotronHModel", "config_from_gguf", "load", "load_state",
]
