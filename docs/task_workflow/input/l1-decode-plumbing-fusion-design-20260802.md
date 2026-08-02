# L1 decode plumbing fusion - design (SUBSTRATE, P4)

Date: 2026-08-02

Status: design, not implemented. Answers the L1 shape question of
`decode-gap-per-target-lever-scope-20260802.md` (section 4, open question 2 in section
7). Design doc for the SUBSTRATE plumbing-fusion lever (P4 in the scope's section 6
sequencing). Branch boundary: tinygrad `nvidia-bringup-20260731`. Does not authorize
promotion to `dev`/`master`.

Bans for this scope: docs only - no tinygrad code changes, no GPU use. No
`prefill_routes.py`, no dtype cleanup, no commits to `master`/`dev`/`exp`, and never
commit the untracked scratchpads (`scratchpad/t6_metal_admission_probe.py`,
`extra/llm_research/microbench/dp4a_peak_cuda*`).

Evidence sources: `nv-performance-campaign-scope-20260801.md` section 14 (exhaustive
per-kernel attribution), `decode-gap-per-target-lever-scope-20260802.md` sections 1, 4,
5, 8 (L1 lever, controls, corrected budget), the flash-decode graph census and DEBUG=2
per-kernel trace behind them (`/tmp/census_sdpa_decode.py`, `/tmp/census_sdpa_decode.log`,
`/tmp/debug_decode_probe.log`, all d512, same machine and llama build as the campaign
doc), and the Qwen3-8B-Q4_K_M GGUF tensor table (`gguf_load_metadata`, local model
file).

---

## 1. Problem statement - the measured evidence

Per-token node-sum at d512 (flash decode graph, prime token, 1021 programs; campaign doc
14.1/14.3 and the decode-gap scope section 1): tinygrad runs **695 plumbing kernels (510
E_ + 185 r_) at 1.6-3.9us each = 1.56ms**; llama runs **327 kernels at 1.3-3.4us =
0.51ms** for its norms/rope/attention-aux class. The class delta is the single largest
gap in the 1.84ms node-sum attribution (+1.05ms of "norms / rope / adds / aux"). The
L1 recovery claim is 0.9-1.0ms node-sum, restated in the corrected budget
(`decode-gap-per-target-lever-scope-20260802.md` section 8.2) as **~45-50% of the
1.98-2.18ms realistic total - the largest single lever**.

Same-session context that licenses the number: 6.12ms wall, 5.83ms replay busy (95%),
node-sum 6.61ms (the ~8% over-count is documented in the decode-gap scope section 1; all
recovery numbers here are node-sum-derived upper bounds until re-measured in place).

Why the 18-count classes are the Q6_K partial-route merge chain (verified against the
GGUF, not assumed). The flash-decode census has exactly three 18-count plumbing classes:
`r_8_8_16_2_4` (18), `r_32_32_4_2_8` (18), `E_8_8_16_2` (18). The GGUF tensor table shows
`attn_v` and `ffn_down` are Q6_K on the **same 18 layers**
([0, 1, 2, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 31, 32, 33, 34, 35]) and `attn_k` is
Q4_K on all 36, so the counts match 18 Q6_K v-projections and 18 Q6_K down-projections
exactly. The mechanism is one line of shared code: `_Q6KDecodeCandidate.execute` ends
with `partial.sum(axis=1)` (`decode_routes.py:124`), which lowers to the generic reduce
chain:

| producer | partial shape | generic merge (the 18-count classes) |
| --- | --- | --- |
| `q6k_gen_partial_1024_4096_4` (18, attn_v on Q6_K layers) | `(1024, 4)` | `r_8_8_16_2_4` (18) |
| `q6k_gen_coop_4096_12288` (18, ffn_down on Q6_K layers) | `(4096, 16)` lane partials | `r_32_32_4_2_8` (18) + `E_8_8_16_2` (18) |

Both are the same generic `partial.sum(axis=1)`; the two r_ signatures differ only in
shape (parts=4 vs the coop kernel's 16-lane partial axis). The DEBUG=2 trace places
`r_8_8_16_2_4` between the v projection and the flash tile, and `r_32_32_4_2_8` +
`E_8_8_16_2` immediately after the down coop kernel, per Q6_K layer.

Why SUBSTRATE, per the scope doc's classification: the chain is ordinary generic JIT
lowering shared by every target (the same classes appear in both the flash and SDPA
decode graphs, so they are model plumbing, not flash-specific; the plumbing is model
code shared with AMD and Metal). llama's graph shows the ceiling: its GEMVs carry the
epilogue (no separate add/silu kernels; `w1+w3` fused into one 12288-row kernel) and its
norms run as one kernel each (`rms_norm_f32`, 145 nodes). The per-kernel 1.6-3.9us times
are launch-floor-bound; the chain also round-trips fp32 intermediates (campaign doc
14.4 item 5: 5.04 GB/token vs llama's ~4.7 GB; the extra ~0.3-0.9 GB is the E_/r_
traffic).

A correction to the campaign doc's "the E_/r_ sets are IDENTICAL in the flash and SDPA
decode graphs": identical at the per-layer plumbing class level (all 18/36/54/72/73-count
classes match), but the attention-context classes differ. SDPA adds three
symbolic-context reduce classes (`r_2_(start_pos+1)_8_4_4_16`,
`r_(start_pos+1)_8_4_(start_pos+1)_(start_pos+1)`, `r_4_2_8_16_4_(start_pos+1)`, 36 each)
and an `E_8_8_16_4` (36) where flash has `E_16_32_4_2` (36), and flash has one extra
`E_32_32_4` hash (36). The claim that the plumbing is model-level, not flash-specific,
survives at the class level; the census diff is not byte-identical. This does not change
any recovery number.

---

## 2. Candidate shape (a) - epilogue absorption into the custom emitters

The three decode emitters (`q4k_g3_lanemap_gemv_kernel`, `decode_kernels.py:110`;
`emit_q6k_gemv_kernel` and `q6k_spec_for_role`, `decode_kernels.py:200,240`;
`flash_block_tiled_xlane_score_pv_tile_whole_cache_kernel`, `flash_decode_attention.py:91`)
each gain optional epilogue parameters. Default None means the emitted UOps are
byte-identical to today, which is the pg3 guarantee. Fused variants get NEW kernel names
(e.g. `q4k_g3_lanemap_gemv_epi_*`, `q6k_gen_partial_*_inkernel`) so legacy names and
hashes are untouched.

### 2.1 What changes (files, emitters, call sites)

- `tinygrad/llm/decode_kernels.py`: `q4k_g3_lanemap_gemv_kernel` gains an epilogue spec
  (residual add, optional input reads) defaulting to None; `Q6KGEMVRouteSpec`
  (`decode_kernels.py:163`) gains `reduction: str = "external_sum"` with a new admitted
  value `"in_kernel"` (today `validate()` rejects anything else - that rejection is the
  legacy boundary, not a design constraint); `_emit_q6k_coop` / `_emit_q6k_partial` gain
  the in-kernel reduce over the partial axis (precedent: `emit_q6k_vocab_scalar_reduce_kernel`
  already does exactly this reduce for the vocab head) and the epilogue parameters.
- `tinygrad/llm/decode_routes.py`: `_Q4KDecodeCandidate.execute` and
  `_Q6KDecodeCandidate.execute` (`decode_routes.py:69,111-124`) select the fused variant
  when admitted and drop the generic `partial.sum(axis=1)` on that path; the epilogue
  inputs (residual x, gate/up activations, norm weights) are threaded from the call
  site. `_LinearDecodeBinding` carries the epilogue spec as data.
- `tinygrad/llm/flash_decode_attention.py`: `flash_decode_live_split_block_tile`
  (`flash_decode_attention.py:478`) and the combine emitter absorb the post-combine
  attention-output normalization (the E_32_32_4_0a5 kernel, 36x) so the o-projection
  GEMV reads the combine output directly. The flash tile itself is unchanged unless the
  rope question (section 9) chooses in-tile rope.
- `tinygrad/llm/model.py`: `TransformerBlock._run` / `_attention` / `_feed_forward`
  (`model.py:497-511`, 520-570, 448-483) - the decode branch threads the epilogue
  inputs under the capability gate: residual x into `attn_output`, `normed_h` + gate/up
  outputs into `ffn_down`, fp16 write for k/v. The `_prefill` branches are untouched;
  the legacy graph construction stays live as the gate-off fallback.
- `tinygrad/llm/qk_primitives.py`: `Q4KPrimitiveLinear.__call__` /
  `Q6KPrimitiveLinear.__call__` (`qk_primitives.py:192,204`) accept optional epilogue
  inputs, gated.
- The fused norm (section 6, the largest single block): a decode RMSNorm emitter
  (reduce + epilogue in one kernel, one UOp builder) added to `decode_kernels.py`,
  selected from the model's norm call sites under the same gate. This is a new decode
  emitter family; see section 9 question 2 for the campaign-discipline check.

### 2.2 What does NOT change

Legacy emitter signatures and generated source (epilogue defaults off, pg3 legacy hashes
must not move); the generic JIT/lowering machinery (`tinygrad/engine/jit.py`,
`codegen/` passes - zero changes); renderers; dtypes; packed storage layouts (Q4_K
36-word blocks, Q6_K 210 bytes/block) and the dequant math inside the GEMVs; the KV
cache layout; the prefill path; the SDPA path; `LanePartition`, `Q4KGateUpLaneMap`, and
the vocab scalar-reduce kernel (L4's scope).

### 2.3 Risks

1. Numeric parity: the epilogue ops are fp32 elementwise, so adds/muls/casts are
   bit-identical by construction. The norm epilogue uses rsqrt - the fused norm must
   reuse the exact lowering today's graph uses, or the decode sha256 moves. The
   in-kernel partial reduces change summation ORDER relative to the generic reduce:
   fp32 addition is not associative, so the fused merge must preserve the generic
   reduce's order (sequential over parts in the same order) or the sha pin moves and
   the delta needs review (section 7).
2. Register/LDS pressure: carrying residual + gate/up activations into a GEMV raises
   register demand and can change occupancy. Per-kernel bandwidth may regress if the
   fused kernel loses occupancy - llama proves the shape is viable, and per-target
   tuning rows (values) are the per-target knob, but the fused kernel must at least not
   lose more than it gains.
3. Model forward restructure: the epilogue inputs must be graph-stable JIT buffers.
   The gate-off fallback must stay exercised so the legacy construction is not dead
   code.
4. Flag composition: kv_quant (int8+scale store), rope-at-read, and ring decode change
   the epilogue contract (k/v store dtype, freqs inputs). The fused variants are
   admitted per route config; a flag combination the fused variant does not serve must
   fall back to the legacy route, not silently mix.

### 2.4 Mapping to the repo's additive/data-driven pattern

Every fused kernel is a new emitter variant with a new name/hash, selected by data
(route admission + spec row), never by a per-target branch. The legacy routes stay
byte-identical, which is exactly the pg3 render-equality contract. This is the same
shape as the Q4K direct_out variant and the vocab scalar-reduce admission:
additive variant, per-target gate, legacy untouched.

---

## 3. Candidate shape (b) - graph-level fusion pass at JIT lowering

A pass over the captured JIT linear that merges adjacent call roots before per-program
compilation, keyed by the same per-target fact. Two distinct powers, which must not be
conflated:

- (b1) Generic chain fusion: merge adjacent generic call roots where the intermediate
  buffer has exactly one consumer (e.g. norm reduce + epilogue into one kernel; the
  partial-sum chain into fewer kernels). No emitter changes.
- (b2) Graph-level dispatch to fused emitters: recognize "custom kernel + single-
  consumer epilogue chain" and reroute to a registered fused emitter. Requires the
  fused emitters to exist - this is a routing mechanism on top of shape (a)'s kernels,
  not an alternative to them.

### 3.1 What changes

- `tinygrad/engine/jit.py`: `jit_lower` (`jit.py:326`) - insert the pass between
  `memory_plan_rewrite` (line 331) and `compile_linear` (line 332), gated by the
  per-target fact threaded from the model; or a hook consulted there. The captured
  linear's call-root list is the fusion surface; `graph_split_rewrite` (line 333)
  already proves the repo rewrites call roots at this level.
- A new `tinygrad/llm` module owning the pattern registry and the single-consumer /
  memory-semantics analysis (the repo's callify metadata and memory_plan_rewrite
  already track the required buffer lifetimes).

### 3.2 What does NOT change

The emitters, `decode_routes.py` execute paths, and `model.py` forward - for (b1).
(b2) additionally leaves the generic lowering untouched but depends on the section 2
emitter work existing.

### 3.3 Risks

1. `jit_lower` serves every JIT in the repo: prefill, decode, tests, non-LLM
   workloads. A pass there must be provably inert when the gate is off (unit-pinned by
   the existing JIT tests), and when on, it changes the kernel mix of every admitted
   graph - including prefill graphs, whose census rows are pinned.
2. (b1) cannot deliver the memory-traffic win. A fused generic kernel still reads and
   writes the same intermediates; the extra ~0.3-0.9 GB/token round-trip stays. The
   win is launch-cost only, which the scope doc already downgraded ("the variable is
   per-kernel host cost, not kernel count" for prefill; decode is graph-replayed, so
   launch overhead is partially amortized - 5.83ms busy of 6.12ms wall).
3. The partial-route merge chain reduced through (b1) still round-trips the
   `(N, parts)` intermediate; the in-kernel reduce (shape a) removes it.
4. The registry for (b2) duplicates admission authority that `decode_routes.py` and
   `FlashDecodeAdmission` already own; two routing layers for one decision.

### 3.4 Mapping to the repo's pattern

(b1) is the weakest fit: it is a global lowering-behavior change protected only by a
per-target gate, and it delivers the smaller half of the win. It is additive in the
"new pass, off by default" sense but not in the "new route variant, legacy untouched"
sense - when on, every admitted graph's census changes. (b2) is additive but is
selection machinery over shape (a), so it adds blast radius without adding kernels.

---

## 4. Comparison and recommendation

| axis | (a) epilogue absorption | (b1) generic chain fusion | (b2) graph dispatch |
| --- | --- | --- | --- |
| delivers llama's fused-GEMV shape | yes | no (generic kernels only) | yes, but needs (a)'s kernels |
| removes fp32 intermediate traffic | yes | no | yes, via (a) |
| covers the Q6K partial merges | yes (in-kernel reduce) | partially (fewer generic kernels, intermediate stays) | yes, via (a) |
| covers the norm reduce+epilogue pair | yes (fused norm emitter) | yes (generic) | via (a) |
| blast radius | decode route family + model decode branch | every JIT user | jit_lower + registry |
| additive/legacy-byte-identical | new kernel names, legacy untouched | all admitted graphs change when on | new routing, legacy untouched |
| matches repo migration pattern | strongest | weakest | redundant with (a) |

**Recommendation: shape (a) is the design.** It is the only shape that delivers both
halves of the L1 claim (launch-count reduction AND the fp32-intermediate traffic
removal), it keeps the blast radius inside the decode route family, and it maps cleanly
to the repo's additive-variant pattern. The scope doc's open question framed (b) as the
shape that "also covers the generic partial.sum(axis=1) merges without touching the
custom emitters" - that is (b1) at its best, and it is strictly weaker than the
in-kernel reduce (shape a, section 6, class 9) because the intermediate round-trip
survives. A graph-level pass is not recommended for this lever; if review wants the
norm pair-fusion through the generic lowering instead of a fused norm emitter, that
narrow (b1) slice is the only piece worth revisiting (section 9 question 1).

---

## 5. Where the capability gate lives

Three candidates, one recommendation: **a decode route fact row with a closed default**,
consumed through the existing admission chain. Not a `DeviceCapabilities` field.

- `DeviceCapabilities` (`device_facts.py:44`) is a hardware-facts record: every field
  is "copied verbatim from the opened renderer, never inferred from a target string"
  (the comments on `wave_size`, `supports_warp_shfl_xor`, `supports_tensor_cores`,
  `supports_fp16`). Decode epilogue fusion is not a property of the opened renderer:
  once the fused emitter variants exist, every target's renderer compiles them - they
  are the same UOp machinery. Two mechanical costs on top of the category error:
  `canonical_hardware()` serializes `capabilities` and hashes it into
  `canonical_hardware_identity`, so a new field changes the identity of every machine
  and invalidates the exact-fact cache key; and it would conflate the capability axis
  with the promotion axis that TG3/TG7 explicitly split (`QKPrimitiveRouteAdmission`
  in `qk_primitives.py`, `FlashDecodeAdmission` in `flash_decode_attention.py:428`).
- Decode route fact row (recommended): a promotion record in the
  `boltbeam.route_policy.v1` schema family - the same channel as
  `load_qk_target_promotion` (`model_route_plan.py`) and the fused-prefill-attention
  record (`model.py:70-95`), e.g. `decode_epilogue_fusion` with `promoted_targets`.
  Default CLOSED (no record -> off), following the TG8 closed-default precedent
  (`_custom_kernel_prefill_attn_promoted`, model.py:70-95) and deviating deliberately
  from TG3's "no record -> open": the scope doc requires AMD/Metal admitted routes to
  stay byte-identical until each target opts in, and only a closed default guarantees
  that. Consumption: extend `QKPrimitiveRouteAdmission` and `FlashDecodeAdmission`
  each with a `epilogue_fusion_promoted` answer (or one shared `DecodeFusionAdmission`
  consulted by all three bind paths), so `arch_ok` (`decode_routes.py:62,101`) becomes
  capability AND promotion AND fusion-admitted per route. Resolve once at model load
  (the "resolve once, read many times" pattern of
  `_flash_decode_capability_and_target_for_device`), never per call.
- `FlashDecodeRouteConfig`-style row (`flash_decode_attention.py:451`): the right
  pattern for per-route GEOMETRY - G4/G5 already carry `split_size` /
  `query_group_size` / `stage_width` as data, and once the mechanism exists, flash-
  specific knobs (e.g. the combine absorbing the output normalization) can be G4/G5 row
  fields. But this lever spans three emitters (q4k, q6k, flash); a single cross-route
  record is the container, and three per-route flags would drift. The route-spec rows
  (`Q6KGEMVRouteSpec.target` already carries `target: str = "amd_gfx1100"` as
  provenance data, `decode_kernels.py:163`) hold the per-kernel tuning values once the
  gate admits the variant.

The gate is the ONLY per-target piece. The fusion itself is one shared emitter change;
the per-target question is the record's default, and the record is a data file whose
`git diff` is the promotion review artifact (the fused-prefill-attention precedent).

---

## 6. Census classification - absorbable vs must-stay

The full E_/r_ census of the flash decode graph (d512, prime token; counts, average
per-kernel time, and class totals from `/tmp/debug_decode_probe.log`; class names are
the schedule signatures, `postrange.py:117-120`). "Absorbable" means the work moves
into an existing custom emitter's prelude/epilogue (shape a); "pair-fused" means the
class merges with its sibling into one kernel (the norm family); "must-stay" means it
remains a separate kernel in the end-state graph.

| class | count | avg us | total ms | role (basis) | classification |
| --- | ---: | ---: | ---: | --- | --- |
| `r_16_256` | 73 | 3.93 | 0.287 | RMSNorm mean/var reduce, 2 per layer + final norm (count 72+1; position: before q-proj and before gate/up in every layer window) | must-stay as a kernel (cross-element reduce; llama keeps `rms_norm_f32`), pair-fused with its epilogue |
| `E_32_32_4_f14a...` | 72 | 2.2 | 0.160 | attn/ffn norm epilogue (follows each `r_16_256`) | pair-fused with `r_16_256` -> one norm kernel per norm (llama's 2.12us shape) |
| `r_2_8_4_4_16` | 36 | 2.45 | 0.088 | q qk_norm reduce, per-head 128 (Qwen3 config `qk_norm == head_dim`, GGUF `key_length` 128; 32 heads x 128) | must-stay as a kernel, pair-fused with its epilogue (llama runs qk norms as `rms_norm_f32` too) |
| `r_8_16_8` | 36 | 2.27 | 0.082 | k qk_norm reduce (8 heads x 128) | must-stay as a kernel, pair-fused |
| `E_4_2_8_16_4`, `E_2_8_16_4`, `E_2_8_16_4_4`, `E_8_2_16_4` | 36 each | 1.7-2.2 | 0.271 | qk_norm epilogues (4 per layer; exact per-hash split pinned in D0) | pair-fused with their reduces (one kernel per qk norm) |
| `E_128_32_3` (2 hashes) | 72 | 1.92 | 0.138 | ffn activation: silu(gate) and silu(gate)*up, 12288-wide (shape 128x32x3; position between up and down) | absorbable into the ffn_down GEMV prelude (down takes gate/up as extra inputs - llama's fused w1w3 shape) |
| `E_32_32_4_02a...` | 54 | 1.7 | 0.090 | attention residual add x+attn_out (36) plus an 18-count Q6_K-layer companion; exact op pinned in D0 (54 = 36 + 18 is unexplained by the census alone) | absorbable into the o-proj GEMV epilogue (o takes x as an extra input - llama's GEMV-epilogue residual) |
| `E_32_32_4_fab...` | 72 | 1.5 | 0.107 | h+ffn_out residual add + the layer-output contiguous (2 per layer after the down merge) | absorbable into the down GEMV epilogue (writes h+ffn_out directly; the contiguous disappears with the intermediate) |
| `E_32_32_4_0a5...` | 36 | 1.5 | 0.055 | attention-output normalization after the flash combine (view-materializing cast; combine output spec is fp32 `(Hq*Hd,)`, `flash_decode_attention.py:503`) | absorbable into the combine epilogue or the o-proj GEMV prelude |
| `E_16_32_4_2` | 36 | 2.46 | 0.088 | k/v fp32->fp16 cast for the KV cache write (position: last attention-input kernel before the flash tile; the cache store is fp16) | absorbable into the k/v GEMV epilogues (write fp16 in-kernel; deterministic fp32->fp16 round-trip) |
| `r_8_8_16_2_4` | 18 | 2.09 | 0.038 | Q6_K v partial merge: `partial.sum(axis=1)` over parts=4 (`decode_routes.py:124`) | absorbable as an in-kernel reduce in `_emit_q6k_partial` (new `reduction="in_kernel"` variant; legacy `external_sum` untouched) |
| `r_32_32_4_2_8` | 18 | 2.08 | 0.037 | Q6_K down coop lane-partial merge (16-lane axis) | absorbable as an in-kernel reduce in `_emit_q6k_coop` |
| `E_8_8_16_2` | 18 | 2.41 | 0.043 | Q6_K down merge-chain companion (present only on the 18 Q6_K down layers; exact op pinned in D0) | absorbable into the same down-path fusion |
| vocab one-offs (`E_1187_32_4` x2, `E_`, `E_2`, `E_16_4_2_8_16_2_4_4`, `E_32_32_4_c6fe`, `r_32_32_4_32_4`, `r_32_4_1187`, `r_128_16_8_1187`, `r_16_8`) | 10 | - | 0.072 | lm_head gather/scatter chain (1187 = 151936/128 chunks) + final norm epilogue | must-stay OUT of L1; L4's substrate variant owns this chain (overlap boundary named, not absorbed here) |
| SDPA-only (`r_2_(start_pos+1)_8_4_4_16`, `r_(start_pos+1)_8_4_(start_pos+1)_(start_pos+1)`, `r_4_2_8_16_4_(start_pos+1)`, `E_8_8_16_4`) | 36 each | - | - | context-shaped SDPA attention reduces over the KV cache | must-stay (not on the flash decode path; the flash tile owns context reads in-kernel) |

Totals: 510 E_ + 185 r_ = 695 kernels, 1.56ms. The absorbable/pair-fusable mass sums to
~1.48ms of pre-fusion census time (0.89 norm + 0.60 gemv/flash epilogue classes), of
which the net savings are bounded by the 1.05ms class delta versus llama's 0.51ms
plumbing class - the 0.9-1.0ms claim is the delta-capped, honest number. The norm
family is the largest single block: 361 kernels / 0.888ms today (73 reduce + 72
epilogue + 36 + 36 reduces + 144 qk epilogues) fuses to 145 kernels at llama's ~2.12us
shape = ~0.31ms, i.e. ~0.58ms of the claim sits in the norm pair-fusion alone; the
GEMV epilogue absorption (residual adds, silu*mul, k/v cast, Q6K merges) is the
remaining ~0.4ms of the claim (its pre-fusion mass, ~0.60ms, exceeds its claim share
after the delta cap). If the norm family is left as-is, L1 delivers ~0.4ms, not
0.9-1.0ms - the norm story is not optional for the claim.

Boundaries that keep things separate:

- Cross-layer dependencies: none of the E_/r_ classes cross layers (counts are
  18/36/72/73). The residual stream x is a graph input to every layer - under fusion it
  becomes an additional epilogue INPUT to the o-proj and down GEMVs, not a kernel
  dependency. The layer boundary stays a kernel boundary.
- KV gather: there is no KV-gather kernel in the flash graph (the flash tile reads the
  cache in-kernel via `make_kv_element_loader`, `flash_decode_attention.py:69`; llama's
  `k_get_rows`/`k_bin_bcast` have no tinygrad counterpart). The only KV-adjacent
  plumbing is the write-side fp16 cast (absorbable). The flash tile's context reads are
  untouched.
- Quantization boundaries: packed storage layouts and the in-GEMV dequant are
  untouched; the epilogue math operates on fp32 accumulators only. The k/v fp16 cast is
  the only dtype boundary adjacent to the quant path and is safe to absorb (it is
  exactly the cache-store contract). The kv_quant int8 path has a different store
  contract (int8 + per-(K|V,head,token) scale) - the fused k/v epilogue admits only the
  measured fp16-cache route config; kv_quant falls back to the legacy route.
- The final norm (the 73rd `r_16_256`) and the vocab chain belong to L4's scope; L1
  covers the per-layer plumbing only.

---

## 7. Controls and verification

- pg3 decode render equality (scope section 5.1): the legacy 10-kernel HIP baseline is
  already pinned (commit `9bd37e4e2`; `scratchpad/pg3_decode_rendered_source_equality.py`).
  Every fused variant is a NEW kernel name and gets NEW pinned rows; the legacy rows
  must be byte-identical at every step - that is the proof AMD and Metal admitted
  routes do not change while the gate is off. The Metal arm runs on the macOS box
  (MetalRenderer cannot instantiate on this Linux NV box); the AMD arm runs on either.
  The fused norm emitter and the q6k in-kernel variants are rendered through both
  renderers like every other decode kernel.
- NV pins (scope section 5.2): first-token digits, decode sha256
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, bench census row
  `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d...`. The decode census row
  (1021 programs/token) is the L1-specific pin: the E_/r_ count must fall from 695
  toward the llama-shaped end-state (~145 norm + ~72 rope + 72 flash + gemv family),
  and the bytes/token metric (5.04 GB -> ~4.7 GB, campaign 14.4 item 5) is the
  secondary signal.
- Reduce-order discipline: the in-kernel partial merges must preserve the generic
  reduce's summation order, or fp32 non-associativity moves the sha. If a digit delta
  appears, STOP and report the exact diff per the standing rule; a reviewed, documented
  pin change is possible but is not the default outcome.
- Perf protocol (scope section 5.3): fixed-depth W decode at d512/d2048/d4096,
  same-session llama `tg10 @ d`, 5 reps median, 5.83ms busy baseline recorded per step,
  DEBUG=2 per-kernel trace before/after.
- AMD runtime promotion gate later: the HIP and Metal render arms prove the change is
  shared now; a measured AMD runtime number is the promotion gate when AMD opts in,
  not the landing gate here (scope section 5.3).

---

## 8. Migration ordering, rollback, and the honest recovery estimate

Ordering (each step one commit with one owning prefix, pushed to
`nvidia-bringup-20260731` only):

1. D0 (no code): per-target diagnostic - count E_/r_ classes in the AMD and Metal
   decode graphs (the scope doc's diagnostic-first step); pin the `E_32_32_4_02a`
   54-count anomaly and the exact per-hash ops from DEBUG=2 UOp dumps; re-run all
   pins.
2. M1 (additive, zero behavior change): the gate + admission (closed default, section
   5) with unit tests; pg3 and NV pins re-run - all unchanged.
3. M2: Q6K in-kernel partial merges (smallest emitter change; `q6k_vocab_scalar_reduce`
   precedent) as new route variants; legacy `external_sum` untouched.
4. M3: the fused norm emitter (attn/ffn + qk norms, one kernel per norm) - the largest
   single block (~0.58ms).
5. M4: q4k epilogue absorption (o-proj residual, down silu*mul prelude + residual
   epilogue, k/v fp16 cast) plus the model.py threading.
6. M5: flash combine output normalization; rope decision (section 9) lands here if
   in-tile rope is chosen.

Rollback story: every step is additive - new kernel names, legacy routes byte-identical,
gate closed by default. Rollback is flipping the promotion record off, or reverting the
single step commit; no step rewrites legacy behavior, so there is no partial-revert
state. M1 is the pivot: everything after it is invisible until a promotion record names
a target.

Honest recovery: L1 is **0.9-1.0ms of the ~1.98-2.18ms realistic total (~45-50%), the
largest single lever**. All numbers are node-sum-derived and inherit the ~8% over-count
(decode-gap scope section 1). After the scope's 60-80% haircut, the expected WALL
recovery is ~0.55-0.8ms, landing the stack at the scope section 8.2 end-state of
~4.37-4.93ms vs llama's 4.07ms (~1.07-1.21x). The claim requires BOTH halves: the norm
pair-fusion (~0.58ms node-sum) and the GEMV/flash epilogue absorption (~0.4ms). A
values-only partial landing (epilogue absorption without the fused norm) delivers
~0.4ms, and the doc says so rather than letting the headline claim absorb it.

---

## 9. Open questions for review

1. Shape verdict: is shape (a) with a fused norm emitter the right design, or should
   the norm pair-fusion go through a narrow generic (b1) chain-fusion pass in
   `jit_lower` (global blast radius, no new emitter)?
2. Is a new decode RMSNorm emitter inside the campaign's "no new kernel" discipline -
   which was scoped to the prefill GEMM verdicts - or does "no new kernel" forbid it
   and force the norm pair through the generic lowering?
3. Gate: confirm the decode route fact row with a CLOSED default (TG8 precedent, not
   TG3's open default). One cross-route record, or per-route fields?
4. Reduce-order parity: must the in-kernel partial merges preserve the generic sum's
   order so the decode sha256 cannot move, or is a reviewed, documented delta
   acceptable when the fused reduce is measurably faster?
5. Rope: keep it as a separate kernel at llama parity (llama runs `rope_neox`, 72
   kernels, 0.127ms), or absorb the q rope into the flash tile prelude (the tile
   already has rope-at-read machinery for K)?
6. The `E_32_32_4_02a` 54-count anomaly (36 + 18): D0 pins the exact op; does any
   reviewer already know what the 18-count companion is?
7. The campaign doc's "E_/r_ sets IDENTICAL in flash and SDPA" is corrected here to
   "identical per-layer classes, differing attention-context classes" - confirm the
   correction (it changes no recovery number).
8. Ownership boundary with L2: the k/v fp16 cast and the q6k v-path in-kernel merge
   touch the same emitters L2 reworks. Should the v-path merge land in L1 (this
   design) or wait for L2's spec-table work?
9. The final-norm + vocab scatter chain stays with L4. Confirm the boundary so L1's
   census target (695 -> ~450 kernels) is reviewed against L4's scope, not silently
   expanded.

HARD STOP after this section. Nothing beyond this design without review.
