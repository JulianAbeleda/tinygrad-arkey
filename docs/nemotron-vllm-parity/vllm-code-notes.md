# vLLM 0.30.0 code notes — Nemotron-3-Nano-4B (NemotronHForCausalLM)

All paths relative to `vllm/` inside
`/home/ubuntu/vllm-bench/.venv/lib/python3.12/site-packages/vllm`.

## 1. model_executor/models/nemotron_h.py

Layer classes, dispatched by `ALL_DECODER_LAYER_TYPES` (nemotron_h.py:542-547)
keyed off `config.hybrid_override_pattern` chars: `"M"`→`NemotronHMambaDecoderLayer`,
`"-"`→`NemotronHMLPDecoderLayer`, `"*"`→`NemotronHAttentionDecoderLayer`,
`"E"`→`NemotronHMoEDecoderLayer`. `NemotronHModel.get_layer` (nemotron_h.py:587-603)
builds each layer from the pattern string; `make_layers` (nemotron_h.py:605-607)
assembles `self.layers`. This 4B model has no MoE layers (has_moe False), so only
Mamba/MLP/Attention layer types matter.

**Attention layer** — `NemotronHAttention` (nemotron_h.py:420-494):
- `qkv_proj`: `QKVParallelLinear` (nemotron_h.py:454-462) — QKV **is merged** into
  one linear (single weight matrix, split via `qkv.split(...)` at nemotron_h.py:491
  into q/k/v by size). GQA config: `total_num_heads=40`, `total_num_kv_heads=8`
  (from `config.num_attention_heads` / `config.num_key_value_heads`), head_dim=128
  (from `config.head_dim` if present, nemotron_h.py:446-449). No RoPE applied in
  this attention module — no rotary embedding call anywhere in
  `NemotronHAttention.forward` (nemotron_h.py:485-494); matches "no RoPE" model spec.
- `o_proj`: `RowParallelLinear` (nemotron_h.py:463-469).
- Uses `vllm.model_executor.layers.attention.Attention` (imported nemotron_h.py:38)
  as `self.attn` (nemotron_h.py:474-483), constructed with `num_kv_heads`,
  `scaling=head_dim**-0.5`, optional `per_layer_sliding_window`.
- `hf_to_vllm_mapper.orig_to_new_stacked` (nemotron_h.py:730-734) maps HF's
  separate `q_proj`/`k_proj`/`v_proj` checkpoint weights into the fused
  `qkv_proj` at load time; `packed_modules_mapping` (nemotron_h.py:737-743)
  records this for LoRA.

**MLP layer** — `NemotronHMLP` (nemotron_h.py:92-129): "up_proj"
(`ColumnParallelLinear`, nemotron_h.py:106-113) → `ReLUSquaredActivation`
(`self.act_fn`, nemotron_h.py:123, from `vllm.model_executor.layers.activation`)
→ "down_proj" (`RowParallelLinear`, nemotron_h.py:114-122). **relu2** activation
= `ReLUSquaredActivation` applied via `maybe_fused_act_quant(self.act_fn, x,
self.down_proj)` (nemotron_h.py:127) which fuses the activation with a possible
quantized-input cast for `down_proj`. This is a plain (non-gated) MLP — only
up_proj/down_proj, no separate gate_proj (matches "no gate" in the model spec);
`is_non_gated_moe = True` is also set at the model level (nemotron_h.py:725) for
the (unused, for this checkpoint) MoE path.

**Mamba layer** — `NemotronHMambaDecoderLayer` (nemotron_h.py:371-417): wraps
`MambaMixer2` (imported from `mamba_mixer2.py`, nemotron_h.py:55) constructed
with `hidden_size`, `ssm_state_size`, `conv_kernel_size=config.conv_kernel`,
`intermediate_size=mamba_num_heads*mamba_head_dim`, `n_groups`, `num_heads`,
`head_dim`, `activation=config.mamba_hidden_act`.

**Residual + RMSNorm fusion**: every decoder layer's `forward` (e.g.
nemotron_h.py:404-417, 312-325, 355-368, 525-539) follows the same pattern:
```
if residual is None:
    residual = hidden_states
    hidden_states = self.norm(hidden_states)          # first layer: single-input
else:
    hidden_states, residual = self.norm(hidden_states, residual)  # fused add+norm
```
`self.norm` is `vllm.model_executor.layers.layernorm.RMSNorm`, whose
`forward(x, residual)` two-argument overload does the "add x to residual, then
RMSNorm, return (normed, new_residual)" fusion (add_rms_norm pattern) — this is
the standard vLLM `RMSNorm.forward_cuda`/`forward_native` residual-fusion path
(not re-defined in nemotron_h.py; NemotronH just calls the two-arg form).
Residual is carried in the **same dtype as hidden_states** (no separate fp32
residual promotion visible in nemotron_h.py — residual is just whatever
`hidden_states`'s dtype is, i.e. bf16 for this checkpoint); this matches vLLM's
default `RMSNorm` residual-fusion contract which keeps residual in model dtype
unless the RMSNorm CustomOp internally upcasts for the variance computation only.
`NemotronHModel.forward` (nemotron_h.py:617-656) threads `hidden_states,
residual` through every layer and does a final `self.norm_f(hidden_states,
residual)` (nemotron_h.py:652) before returning.

**torch.compile**: `NemotronHModel` is decorated
`@support_torch_compile(dynamic_arg_dims={...})` (nemotron_h.py:550-558) — the
whole backbone (embedding → layer stack → final norm) is one compiled graph;
`input_ids`, `positions`, `intermediate_tensors`, `inputs_embeds` are the
dynamic-shape args. `NemotronHForCausalLM` itself (the outer
`nn.Module`/`HasInnerState`/... class, nemotron_h.py:711-910) is *not*
decorated — only the inner backbone.

**lm_head / logits**: `NemotronHForCausalLM.__init__` builds `self.lm_head =
ParallelLMHead(...)` (nemotron_h.py:833-838) and `self.logits_processor =
LogitsProcessor(config.vocab_size)` (nemotron_h.py:840). `compute_logits`
(nemotron_h.py:901-906) calls `self.logits_processor(self.lm_head,
hidden_states)` — standard vLLM logits processor (handles TP all-gather of
vocab-parallel logits internally, not custom to NemotronH).

---

## 2. model_executor/layers/mamba/mamba_mixer2.py — MambaMixer2

**in_proj**: `MergedColumnParallelLinear` (mamba_mixer2.py:348-360, taken when
`n_groups % tp_size == 0`, true for TP=1) with `output_sizes=[intermediate_size,
intermediate_size, groups_ssm_state_size, groups_ssm_state_size, num_heads]` —
i.e. **merged as [z (gate) | x | B | C | dt]** in one matmul (5-way merge, not
just z|xBC|dt as three pieces — it's actually gate, hidden(x), B, C, dt all
fused into one linear). `conv1d` is a separate `MergedColumnParallelLinear`
over `[intermediate_size, groups_ssm_state_size, groups_ssm_state_size]`
(x|B|C only, no z/dt — matches causal-conv1d only operating on x,B,C)
(mamba_mixer2.py:336-346).

**Top-level `forward`** (mamba_mixer2.py:567-606):
1. `projected_states, _ = self.in_proj(hidden_states)` (line 573).
2. Allocates `ssm_output` buffer (lines 578-585).
3. Calls **custom op** `torch.ops.vllm.mamba_mixer2(projected_states,
   ssm_output, layer_name)` (lines 590-594) — this is registered via
   `direct_register_custom_op(op_name="mamba_mixer2", op_func=mamba_mixer2,
   mutates_args=["output"])` (mamba_mixer2.py:1281-1285) and
   `@PluggableLayer.register("mamba_mixer2")` on the class (line 253). Splitting
   into a custom op is explicitly so the `hidden_states_B_C`/`dt` split happens
   *inside* the op and isn't treated as an intermediate tensor by torch.compile
   (comment at lines 588-589) — i.e. **yes, the mixer body is deliberately
   carved out of the compiled graph** via the custom-op boundary (the op itself
   dispatches to `self.conv_ssm_forward`, mamba_mixer2.py:1270-1278, looked up
   from `forward_context.no_compile_layers[layer_name]` — the "no_compile"
   naming confirms this runs outside torch.compile's traced region).
4. `gate = projected_states[..., :tped_intermediate_size]` then
   `self.norm(ssm_output, gate)` — gated RMSNorm (`Mixer2RMSNormGated`).
5. `self.out_proj` (`RowParallelLinear`, mamba_mixer2.py:486-493).

**conv_ssm_forward** (mamba_mixer2.py:706-1167) — the real prefill/decode
logic, run inside the custom op:
- Reads `Mamba2AttentionMetadata` for this layer prefix from
  `forward_context.attn_metadata` (lines 717-763).
- Splits `hidden_states_B_C` and `dt` into **decode** and **prefill** slices
  along the token dim by `[num_decode_tokens, num_prefill_tokens]`
  (mamba_mixer2.py:781-791) — decode tokens come first, prefill tokens after,
  in every mixed batch (standard vLLM V1 token ordering).
- **Prefill path** (`if has_prefill:`, lines 839-1013):
  - `causal_conv1d_fn(...)` (mamba_mixer2.py:856-871) — Triton varlen conv,
    writes `conv_state` cache in place, supports chunked-prefill-aware cache
    alignment args (`block_idx_first_scheduled_token`,
    `block_idx_last_scheduled_token`, `block_size_to_align=mamba_block_size`).
  - `mamba_chunk_scan_combined_varlen(...)` (mamba_mixer2.py:894-916) — the
    Triton SSD chunked-scan kernel pipeline (see §3) — writes into
    `preallocated_ssm_out_p` in place and returns `varlen_states` (per-sequence
    final/intermediate SSM states).
  - If `mamba_cache_mode == "all"` (`is_mamba_cache_all`, line 793), an
    elaborate per-sequence Python loop (lines 940-999, with a host sync via
    `gpu_sync_allowed()` at line 933) writes intermediate chunk states into
    `ssm_state` at block-aligned cache slots (this is the deprecated "all"
    mode's full-checkpoint-every-block behavior). Otherwise (the common
    "align"/"none" path, lines 1001-1013) only the final `varlen_states` is
    written to `ssm_state[state_indices_tensor_p]`.
- **Decode path** (`if has_decode:`, lines 1016-1167):
  - `causal_conv1d_update(...)` (mamba_mixer2.py:1048-1060) — Triton
    single-step conv update kernel.
  - Reshapes `A`, `dt`, `dt_bias`, `D`, `B`, `C`, `hidden_states_d` to the
    per-head/per-state shapes the SSU kernel expects (lines 1067-1089).
  - Dispatch to one of three decode-update paths depending on
    `self.use_replayssm` and `mamba_config.backend`:
    - ReplaySSM + FlashInfer backend →
      `selective_state_update_replayssm_flashinfer` (lines 1096-1122).
    - ReplaySSM + non-FlashInfer (Triton) →
      `selective_state_update_replayssm_output_only` (lines 1124-1149).
    - Default (no ReplaySSM, this config's default) →
      `selective_state_update(...)` (lines 1151-1167) — the `ssu_dispatch`
      unified entry point (see §3), Triton backend by default.

Prefill/decode split within one forward call is thus **entirely inside
`conv_ssm_forward`**, driven by `attn_metadata.num_prefills`/`num_decodes`
counts computed once per model forward pass by the Mamba2 attention-metadata
builder (§5) and shared across all Mamba layers via `forward_context`.

**Gated RMSNorm** — `Mixer2RMSNormGated` (mamba_mixer2.py:79-188):
`forward_cuda` (161-181) calls the Triton `rms_norm_gated` kernel
(`ops/layernorm_gated.py`) when `n_groups==1` and TP divides groups; falls back
to `forward_native` (110-159, plain PyTorch silu-gate + variance-based RMSNorm,
handling TP all-reduce/all-gather cases) otherwise.

---

## 3. model_executor/layers/mamba/ops/ — kernel implementations

All of the following are **pure Triton** (`@triton.jit`), not CUDA `_C` ops,
except the CPU fallback paths:

- **causal_conv1d.py** (1289 lines): `_causal_conv1d_fwd_kernel`
  (`@triton.jit`, line 16) → driven by `causal_conv1d_fn` (line 481, prefill/
  varlen conv). `_causal_conv1d_update_kernel` (`@triton.jit`, line 762) →
  driven by `causal_conv1d_update` (line 1096, decode single-step conv).
  Both are Triton kernels with continuous-batching / cache-index / chunked-
  prefill-alignment support baked in (comments at causal_conv1d.py:158-159 etc).

- **mamba_ssm.py** (857 lines): `_selective_scan_update_kernel` (`@triton.jit`,
  line 241) is the decode-step SSU Triton kernel, invoked by module-level
  `selective_state_update` (line 497) — this is the function imported as
  `_triton_selective_state_update` by `TritonSSUBackend` in `ssu_dispatch.py`.
  Also has launch-config tuning helpers `get_ssm_configs`/
  `_get_default_ssm_launch_config`/`try_get_optimal_ssm_config` (lines 67-197).
  A separate `selective_scan_fn` (line 704) exists for the legacy/Mamba1-style
  full scan (not used by Mamba2 prefill, which goes through `ssd_combined.py`
  instead).

- **ssu_dispatch.py** (572 lines): unified dispatcher. `MambaSSUBackend` ABC
  (lines 121-151) with three concrete backends registered in
  `_BACKEND_REGISTRY` (lines 352-356): `TritonSSUBackend` (154-209, wraps
  `mamba_ssm.selective_state_update`), `FlashInferSSUBackend` (212-284, wraps
  `flashinfer.mamba.selective_state_update`, requires flashinfer≥0.6.4),
  `CPUSSUBackend` (287-349, wraps compiled C++
  `torch.ops._C.selective_state_update_cpu` with AVX2/VSX SIMD + OpenMP).
  **Default backend = `MambaBackendEnum.TRITON`** (config/mamba.py:39,
  `MambaConfig.backend: MambaBackendEnum = MambaBackendEnum.TRITON`) — CPU-only
  platforms silently override to `CPU` (ssu_dispatch.py:481-489). Selection
  happens once in `initialize_mamba_ssu_backend(mamba_config, kv_cache_config,
  use_replayssm=...)` (lines 463-517), called at engine-core init; the module
  global `_mamba_ssu_backend` is then used by every `MambaMixer2.forward`'s
  call to `selective_state_update(...)` (the free function at line 530, a thin
  dispatch to `get_mamba_ssu_backend()(...)`). Also hosts the ReplaySSM ring-
  tracker Triton kernel `_update_replayssm_ring_trackers_kernel` (line 32) and
  `selective_state_update_replayssm_flashinfer` (line 376, only reached when
  `use_replayssm=True` and backend is FlashInfer).

- **replayssm_config.py** (51 lines) and
  **selective_state_update_replayssm_output_only.py** (720 lines) — **ReplaySSM**:
  an alternative Mamba2 *decode* algorithm that avoids the standard "write the
  full (headdim×dstate) SSM state to the cache on every decode step" cost. It
  instead keeps a small ring buffer (`replayssm_buffer_len`, default 16,
  config/cache.py:193-196) of the last B raw inputs (`x_cache`, `dt_cache`,
  `B_cache` — see kv_cache tuple slots 2-4 in `MambaMixer2.__init__`,
  mamba_mixer2.py:534-536: `_n_state = 5 if use_replayssm else 2`) and only
  materializes/writes the full recurrent state on a "flush" (checkpoint),
  amortizing the expensive full-state write over B steps ("cache recent SSM
  inputs and skip the per-step full-state store, writing the checkpoint back
  only on flush", config/cache.py:198-199). `replayssm_config.py` is a
  hand-tuned Triton launch-config table (`_mamba2_output_only`, lines 32-37)
  keyed by Blackwell-vs-not and dstate-tile sizes, with an `override`
  mechanism for benchmarking. **Not used by default** (`use_replayssm=False`
  default, config/cache.py:197); would require an explicit `--use-replayssm`
  flag plus `mamba_cache_mode` "none" or "align" and Triton/FlashInfer backend.

- **layernorm_gated.py** (172 lines): `_layer_norm_fwd_1pass_kernel`
  (`@triton.jit`, line 14) → `_layer_norm_fwd` (line 77) → public entry
  `rms_norm_gated` (line 145), used by `Mixer2RMSNormGated.forward_cuda`.

- **SSD chunked-scan prefill pipeline** — `ssd_combined.py`
  (`_mamba_chunk_scan_combined_fwd`, lines 27-154; public entry
  `mamba_chunk_scan_combined_varlen`, lines 157-227) runs **5 Triton
  sub-kernels in sequence** (comment block at ssd_combined.py:81-88 cites the
  Mamba2-part3-algorithm blog for the derivation):
  1. `_chunk_cumsum_fwd` (ssd_chunk_state.py:303, kernel
     `_chunk_cumsum_fwd_kernel` @triton.jit line 28) — chunked cumsum of
     `A*dt` (with dt softplus + bias).
  2. `_chunk_state_fwd` (ssd_chunk_state.py:353, kernel
     `_chunk_state_fwd_kernel` @triton.jit line 195) — per-chunk local SSM
     state (the "B terms" / right factor of the off-diagonal low-rank blocks).
  3. `_state_passing_fwd` (ssd_state_passing.py:102, kernel
     `_state_passing_fwd_kernel` @triton.jit line 26) — inter-chunk recurrence
     across chunk boundaries (the "A terms" / middle factor), parallelized per
     sequence via `last_chunk_indices`.
  4. `_bmm_chunk_fwd` (ssd_bmm.py:148, kernel `_bmm_chunk_fwd_kernel`
     @triton.jit line 64) — batched `Cᵀ·B` matmul per chunk (`CB` matrix).
  5. `_chunk_scan_fwd` (ssd_chunk_scan.py:418, kernel `_chunk_scan_fwd_kernel`
     @triton.jit line 147) — final causal intra-chunk scan combining `CB`,
     `x`, `dt`, `dA_cumsum`, `C`, inter-chunk `states`, `D`, optional `z`, into
     the in-place output tensor.
  On CPU, `ssd_combined.py:230-235` monkey-patches
  `_mamba_chunk_scan_combined_fwd` to a C++/vec CPU implementation
  (`ops/cpu/mamba_ssm.py`) — irrelevant here (CUDA platform).

---

## 4. Hybrid KV cache: Mamba state + attention blocks

**MambaSpec** (`v1/kv_cache_interface.py:993-1057+`): subclass of
`KVCacheSpec` describing Mamba's per-block state. Fields: `shapes` (tuple of
per-tensor shapes, from `MambaStateShapeCalculator.mamba2_state_shape`),
`dtypes`, `page_size_padded` (nullable override so the mamba page can be padded
to match the attention page — see below), `mamba_cache_mode`,
`num_speculative_blocks`, `num_prefill_checkpoint_blocks` (align-mode only).
`page_size_bytes` (line 1020) = sum of `prod(shape)*dtype_size` over all state
tensors, padded up to `page_size_padded` if set. `max_memory_usage_bytes`
(lines 1034-1045) branches on `mamba_cache_mode`: `"all"` → one block per
`block_size`-token chunk of `max_model_len` (+ spec blocks); `"align"` → fixed
`2 + num_speculative_blocks + num_prefill_checkpoint_blocks` blocks per
sequence (checkpoint model, not proportional to length); else (`"none"`) →
`1 + num_speculative_blocks` (single resident state per sequence, reused
in-place every step — matches this config's default since prefix caching
determines the mode, see below).

**MambaManager** (`v1/core/single_type_kv_cache_manager.py:1444`+): the
`SingleTypeKVCacheManager` subclass for Mamba groups. Overrides `block_size`
back to the raw `kv_cache_spec.block_size` (undoing DCP/PCP scaling that the
base class applies to sharded attention groups — line 1457-1460, "Mamba layers
use TP instead of DCP, so each rank holds the full recurrent state").
`has_positionally_stable_blocks` (1447-1451) is False only in `"align"` mode
(align mode can null/relocate interior state blocks). In `"align"` mode it
additionally tracks `last_state_block_idx`, `_checkpoints`,
`_producer_partial_tail_reqs` per request (1470-1484) to support prefix-cache
checkpointing and CoW hand-off of "partial tail" state between sibling
requests (e.g. EAGLE/MTP). `find_longest_cache_hit` (1486+) implements the
fine-grained (sub-block-size) prefix-hash lookup that Mamba's align mode
supports (`supports_fine_grained_hash_lookup: ClassVar[bool] = True`, line
1445) — this is what lets prefix caching share *some* prefix state even when
the shared prefix isn't a full `mamba_block_size` multiple.

**Block-size alignment** (attention block raised to match mamba page):
`platforms/interface.py:Platform.update_block_size_for_backend` (lines
608-651) runs after model construction. Phase 1 (628-640) picks the attention
backend's preferred block size. Phase 2 (642-645): if `model_config.is_hybrid`,
calls `_align_hybrid_block_size` (lines 778+), which: computes
`attn_page_size_1_token` (one token's KV-cache bytes in the chosen attention
backend's layout, via `backend_cls.customize_spec(FullAttentionSpec(...))`,
lines 812-863) and `mamba_page_size` (via
`model_cls.get_mamba_state_shape_from_config`/`get_mamba_state_dtype_from_config`
wrapped in a `MambaSpec`, lines 865-881) and then (code continues past what
was read, but per the docstring at interface.py:783-786) "ensures the
attention page size is >= the mamba page size, and pads the mamba page size to
match" — i.e. the **attention `block_size` (tokens per block) is bumped up**
until `block_size * attn_page_size_1_token >= mamba_page_size`, and
`cache_config.mamba_page_size_padded` is set so the (typically much larger,
single-block) mamba page is padded to exactly align with the attention block
pool, letting both KV-cache groups share one uniform block-pool page size.

**mamba_ssm_cache_dtype / mamba_cache_dtype**: generic fields in
`config/cache.py:170-177` (`"auto"` default, resolved per-model via
`MambaStateDtypeCalculator.mamba2_state_dtype(model_dtype, mamba_cache_dtype,
mamba_ssm_cache_dtype)`, called from both `NemotronHMambaDecoderLayer`'s
mixer indirectly and `NemotronHForCausalLM.get_mamba_state_dtype_from_config`,
nemotron_h.py:754-770). **NemotronH-specific override**:
`NemotronHForCausalLMConfig.update_mamba_ssm_cache_dtype`
(model_executor/models/config.py:664-680): if `mamba_ssm_cache_dtype=="auto"`,
sets it to `hf_config.mamba_ssm_cache_dtype` if the HF config specifies one,
else to `cls.DEFAULT_MAMBA_SSM_CACHE_DTYPE = "float32"`
(model_executor/models/config.py:661-662) — **the SSM state cache defaults to
fp32 for NemotronH models specifically** ("Only `float32` is known to have no
accuracy issues by default", line 662), called from
`NemotronHForCausalLMConfig.verify_and_update_config`
(model_executor/models/config.py:683-687), which runs at engine-config-verify
time for any NemotronH-architecture model, so it applies to this 4B checkpoint
unless the checkpoint's HF config already pins a `mamba_ssm_cache_dtype`.
`mamba_cache_dtype` (conv state) has no NemotronH-specific override and stays
whatever `mamba2_state_dtype` resolves it to from `"auto"` (typically the
model compute dtype, bf16 here).

**mamba_cache_mode semantics** (`config/cache.py:178-186`): `"none"` — set
automatically when prefix caching is disabled; single resident state,
overwritten in place every step. `"all"` (deprecated, warns at
`config/cache.py:321-330`) — caches state at every `block_size`-aligned
position, full positional recompute-free resume from any block boundary;
requires `model_config.supports_mamba_prefix_caching`
(`SupportsMambaPrefixCaching` mixin, which `NemotronHForCausalLM` does declare,
nemotron_h.py:721) or vLLM silently falls back to `"align"`
(model_executor/models/config.py:631-640). `"align"` — **the default when
prefix caching is enabled** (model_executor/models/config.py:622-630: "Mamba
cache mode is set to 'align' ... by default when prefix caching is enabled");
only checkpoints state at the last token of each scheduler step and at
`block_size`-aligned positions, not every block-aligned position mid-step;
requires chunked prefill enabled (assert at model_executor/models/config.py:
641-644).

**Is prefix caching enabled by default for hybrid models in 0.30.0?** The
logic in `MambaModelConfig.verify_and_update_config`
(model_executor/models/config.py:609-657) is conditioned entirely on
`cache_config.enable_prefix_caching` — it does not itself force prefix caching
on for hybrid/mamba models; it only *reacts* to whatever
`enable_prefix_caching` resolves to elsewhere (vLLM's general engine-args
default, which is **enabled by default in V1** for regular decoder models).
Given no evidence in these files of NemotronH/hybrid models being singled out
to *disable* prefix caching, the general V1 default (prefix caching on)
applies, and consequently `mamba_cache_mode` defaults to `"align"` for this
model unless the user explicitly passes `--no-enable-prefix-caching`, in which
case `mamba_cache_mode` is forced to `"none"`
(model_executor/models/config.py:650-655) and `mamba_block_size` defaults to
`max_model_len` (line 656-657) — i.e. one giant single block per sequence, no
alignment concerns.

**"Prompt tail" flush field** (`config/cache.py:187-192`,
`enable_mamba_fine_grained_prefix_cache`, default `False`): "Also register a
Mamba 'align' checkpoint at the shared-prefix junction -- where an EAGLE/MTP
sibling was observed to resume -- instead of only at the prompt tail." In
plain terms: in `"align"` mode, vLLM by default only writes a reusable Mamba
checkpoint at the very end of the prompt ("prompt tail") plus scheduler-step
boundaries; enabling this flag additionally checkpoints at the exact token
position where an EAGLE/MTP speculative sibling request diverged/resumed, so
that finer-grained sub-prompt prefixes can also be shared/hit in the cache
(only meaningful with EAGLE on the Mamba group and a prefix-match unit smaller
than `mamba_block_size`).

**n>1 (parallel sampling)**: `v1/engine/parallel_sampling.py:ParentRequest`
(class def line 13+) — when `sampling_params.n > 1`, the frontend/engine
creates **n independent child `EngineCoreRequest`s**, each with its own
`request_id` (see `_get_child_sampling_params`, line 52+) but the *same*
prompt token ids. There is no single shared in-scheduler "prefill once, branch
n ways" mechanism visible here — parallel sampling is implemented at the
request-fan-out level, not the KV-cache/compute level. Consequently: **with
prefix caching enabled** (the model's default, `mamba_cache_mode="align"`),
the 2nd..nth child requests can still get a full attention-KV and Mamba-state
cache **hit** on the identical prompt prefix (via `MambaManager.
find_longest_cache_hit` + the attention manager's ordinary prefix-hash match),
so in practice the prompt is only *computed* once and the other n-1 requests
reuse the cached blocks/checkpoint — this is exactly what `MambaManager`'s
fine-grained hash lookup and checkpoint retention exist to support for
multi-child (EAGLE/parallel-sampling-like) fan-out. **With
`mamba_cache_mode="none"`** (prefix caching off), there is no cache to hit, so
each of the n child requests must redo the full prefill through the Mamba SSD
kernels independently — no sharing is possible, confirming the question's
premise that "none" mode forecloses any Mamba-side prefix reuse across
parallel-sampling children (attention-side KV reuse would likewise be off,
since prefix caching is a single global on/off switch in this config).

---

## 5. Attention backend selection (sm_120) and Mamba2 CUDA-graph mode

**Backend priority for sm_120 (device_capability.major==12, GQA, non-MLA)**:
`platforms/cuda.py`'s backend-priority builder (the function around lines
100-179) branches: MLA models take one path (100-152); non-MLA models take the
`else` branch at line 167 whenever `device_capability.major` is **not** 9 or
10 (sm_120's major is 12, so it lands here) — **priority order: `FLASH_ATTN`,
`FLASHINFER`, `TRITON_ATTN`, `FLEX_ATTENTION`, `TURBOQUANT`**
(cuda.py:174-178). `CudaPlatformBase.get_valid_backends`
(cuda.py:387-427) then filters this list by calling each backend class's
`validate_configuration(device_capability=..., **attn_selector_config)` and
keeps only those that don't return invalidity reasons;
`get_attn_backend_cls` (cuda.py:449-548) picks the **highest-priority valid**
backend. Whether `FLASH_ATTN` (FlashAttention 2/3) is actually valid on
sm_120/head_dim=128/no-RoPE/GQA-40:8 in this installed vLLM+flash-attn build
is determined by `FlashAttentionBackend.validate_configuration` at runtime —
not directly re-derived from these notes; if it's invalid (e.g. the installed
flash-attn wheel lacks sm_120/Blackwell-consumer kernels),
`FLASHINFER` is the next candidate, then `TRITON_ATTN`. No model-specific
override forces a particular backend for NemotronH in
`model_executor/models/config.py`'s NemotronH-related classes (only
`mamba_ssm_cache_dtype` is special-cased there), so the general CUDA priority
list governs. (Confirming exact live selection requires running
`get_attn_backend_cls`/checking logs on the actual box — this file-level
analysis establishes the *candidate order and mechanism*, not the final
runtime pick, per the "verify before asserting" convention.)

**Mamba2 attention-metadata builder** —
`v1/attention/backends/mamba2_attn.py`: `Mamba2AttentionBackend.is_ssm()`
returns True (lines 88-99, marks it as an SSM/non-KV attention backend group).
`Mamba2AttentionMetadataBuilder` (line 114+) extends
`BaseMambaAttentionMetadataBuilder` (`v1/attention/backends/mamba_attn.py:89`)
and reads `chunk_size` from `vllm_config.model_config.get_mamba_chunk_size()`
(mamba2_attn.py:130-133; this is `config.chunk_size=256` for this model). Its
`build()` (lines 136-172) computes common decode/prefill split metadata via
`self._compute_common_metadata` (base class), then for prefill batches
(`common.num_prefills > 0`) additionally computes `prep_initial_states` (does
any prefill sequence have a non-empty initial state, i.e. is a continuation of
a chunked-prefill/prefix-cache-hit sequence) and chunk-aligned varlen metadata
(`cu_chunk_seqlen_p`, `seq_idx_p`, `last_chunk_indices_p`) via
`_build_chunk_metadata_tensors`/`compute_varlen_chunk_metadata`
(mamba2_attn.py:22-89) — this is exactly the metadata consumed by
`mamba_chunk_scan_combined_varlen` in `MambaMixer2.conv_ssm_forward`.

**CUDA-graph support mode**: `BaseMambaAttentionMetadataBuilder` sets
`_cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.
UNIFORM_BATCH` (`v1/attention/backends/mamba_attn.py:93`). `AttentionCGSupport`
(`v1/attention/backend.py:559-573`) enum: `ALWAYS=3` (mixed prefill+decode OK),
`UNIFORM_BATCH=2` ("batches that only contain query lengths that are the same
... i.e. 'decodes' are 1+num_speculative_tokens" — this is Mamba's level),
`UNIFORM_SINGLE_TOKEN_DECODE=1`, `NEVER=0`. So Mamba layers support full CUDA
graphs **only for uniform-length batches** (pure decode steps, or uniform
spec-decode draft batches), not for mixed prefill/decode batches. Separately,
`MambaModelConfig.verify_and_update_config`
(model_executor/models/config.py:609-618 docstring) "Enable FULL_AND_PIECEWISE
cuda graph mode by default (required to get good performance for mamba layers
in V1)" — `CUDAGraphMode.FULL_AND_PIECEWISE = (FULL, PIECEWISE)`
(config/compilation.py:63), meaning: **decode-only batches get a FULL CUDA
graph** (the entire forward pass, including the Mamba mixer's custom op and
its Triton SSU kernel, is captured whole — consistent with `UNIFORM_BATCH`
support since a pure-decode batch has uniform query length 1), while
prefill/mixed batches fall back to **PIECEWISE** graphs (only the
torch.compile-covered, non-mamba-mixer pieces of the graph are captured per
compiled subgraph, since the mixer's custom-op body containing varlen/chunked
Triton kernels can't be captured in a single static-shape CUDA graph for
variable-length prefills). So: **yes, decode gets a FULL CUDA graph including
the Mamba mixer**, as long as the decode batch is uniform (which
`HybridAttentionMambaModelConfig.verify_and_update_config`, model_executor/
models/config.py:445-459, ensures applies to NemotronH by delegating to
`MambaModelConfig.verify_and_update_config`).

---

## 6. Sampling (temperature=1, top_k=0, top_p=1, logprobs=1)

**Default `logprobs_mode`**: `Sampler.__init__` default parameter
`logprobs_mode: LogprobsMode = "raw_logprobs"` (v1/sample/sampler.py:64).

**Which sampling path runs**: `TopKTopPSampler.__init__`
(v1/sample/ops/topk_topp_sampler.py:101-118) on CUDA picks
`self.forward = self.forward_cuda if can_use_flashinfer else
self.forward_native`, where `can_use_flashinfer = logprobs_mode not in
PROCESSED_LOGPROBS_MODES and flashinfer_sampler_supported()`
(lines 112-115). Since the default `logprobs_mode="raw_logprobs"` is not in
`PROCESSED_LOGPROBS_MODES` (that set covers `"processed_logits"`/
`"processed_logprobs"`, which need post-top-k/top-p logits that FlashInfer's
sampling kernels don't expose, per the comment at lines 110-111), **FlashInfer
sampling (`forward_cuda`, line 174) is used by default**, provided
`flashinfer_sampler_supported()` (line 44) returns True for this
install/GPU (checks flashinfer is importable and its backend is usable — not
independently re-verified here). `forward_cuda`
(topk_topp_sampler.py:174-201) calls `flashinfer_sample(logits.contiguous(),
k, p, generators)` (line 201) → `flashinfer.sampling.top_k_top_p_sampling_from_logits`
et al. (lines 489-527, exact function chosen by which of k/p are set — with
`top_k=0` meaning "no filtering" and `top_p=1.0` meaning "no filtering",
these are effectively pass-through/full-distribution sampling calls). If
FlashInfer isn't available, it falls back to `forward_native`
(lines 150-172): `apply_top_k_top_p(logits, k, p)` (a no-op here since k=0/p=1
disable filtering, per `apply_top_k_top_p`/`apply_top_k_only`,
topk_topp_sampler.py:368-447) → `logits.softmax(...)` → `random_sample(probs,
generators, use_fp64_gumbel)` (line 170) → torch **Gumbel-max / exponential-
noise sampling**: `random_sample` (topk_topp_sampler.py:464-486) draws
`q = empty_exponential_noise_like(probs, ...); q.exponential_(...)` then
`sample_with_exponential_noise(probs, q)` (line 455+, effectively
`argmax(probs / q)`, the standard Gumbel-max trick implemented via
exponential-noise division rather than explicit Gumbel sampling) — this is the
"torch exponential/Gumbel `random_sample`" path referenced in the task, used
only as the non-FlashInfer fallback.

Before either path, `Sampler.sample()` (v1/sample/sampler.py:244-303) always:
applies `apply_temperature` (line 277, divides logits by temperature=1.0,
effectively a no-op here) before calling `self.topk_topp_sampler(...)`
(line 287) — since `temperature=1 >= _SAMPLING_EPS`, this is `all_random`
(not greedy), so `greedy_sample` is skipped and `random_sampled` from the
top-k/top-p sampler is returned directly (lines 258-259, 294-295).

**Logprobs computation**: with `logprobs=1` →
`sampling_metadata.max_num_logprobs = 1` (not None, not -1), so
`Sampler.forward` (v1/sample/sampler.py:73-150) takes the
`num_logprobs is not None` branch (line 87) and, since
`logprobs_mode == "raw_logprobs"` (the default here), computes
`raw_logprobs = self.compute_logprobs(logits)` (line 89) **before** any
temperature/top-k/top-p processing — i.e. on the raw model logits. `compute_logprobs`
(v1/sample/sampler.py:305-307) is `logits.log_softmax(dim=-1, dtype=torch.float32)`
— a **full log_softmax over the entire vocabulary** (not a top-k-restricted
approximation). Then `self.gather_logprobs(raw_logprobs, num_logprobs=1,
token_ids=sampled)` (line 130) gathers the top-`num_logprobs` entries plus the
sampled token's rank/logprob out of that full-vocab log-softmax tensor into
the compact `LogprobsTensors` returned to the API layer. Because
`logprobs_mode="raw_logprobs"` (not `"processed_logprobs"`/`"processed_logits"`),
the reported logprobs reflect the *pre-sampling, pre-top-k/p, pre-temperature-
adjusted* distribution (this particular request's temperature=1/top_k=0/top_p=1
happen to make raw and processed logits numerically closer than in the general
case, but the code path is still "raw" by default regardless).
