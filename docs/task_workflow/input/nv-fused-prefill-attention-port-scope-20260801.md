# NV fused prefill attention - port scope

Date: 2026-08-01

Status: scoped, not implemented. Branch boundary: tinygrad `nvidia-bringup-20260731`. Does not
authorize promotion to `dev`/`master`.

---

## 1. Why - the measurement that motivates this

Measured this campaign, same session, on the RTX 5090 (sm_120), Qwen3-8B-Q4_K_M through
`extra/llm/bench/model_e2e_bench.py` (pp512 / decode 96 / warmup-skip 16):

| metric | value |
| ---: | ---: |
| decode | 158.62 tok/s |
| prefill pp512 | 81.9 tok/s |
| fused attention census | `custom_kernel_attention_trace: {"dispatches": 0}` |
| prefill promotion census | `prefill_overlay_promotion: "no-promoted-candidate"` |

The fused prefill attention kernel (`custom_kernel_attention` -> `FlashPrefillAttentionSpec` ->
`amd_gfx1100_q16_grid_hd128_loop_attention`) **dispatches zero times on NV**. Prefill on the 5090
runs entirely on SDPA plus the dense-GEMM generated routes; the fused attention path is
AMD-gated by design (see section 3.5) and has never been compiled, let alone measured, on NVIDIA.

The dense prefill side is already NV-working (sm120 candidate set minted and measured:
`bench/models/qwen/data/nv-rtx5090/qwen3-8b.json` records prefill 87.4 tok/s with
`prefill_v2=True`, `prefill_concrete_kv=True`). Attention is the remaining prefill surface on NV.

**Caveat carried forward:** this scope does not claim the fused kernel is *faster* on NV than the
current SDPA/dense path. It claims the kernel can be made correct and admitted on NV through the
same mechanism decode used, and that it can then be measured head-to-head. P5 measures; it does
not assume.

---

## 2. The design question, answered: can this be generic like flash decode?

Short answer: **yes, but the generic mechanism is different from decode's, and the difference is
the whole shape of this work.**

### 2.1 Why decode's port was cheap

Flash decode is *scalar-shaped*. Its QK score is a per-lane half2 dot (`fdot2`), PV is scalar FMA,
the row reduction is a shuffle ladder. There is no tensor-core fragment in the graph; the only
per-target leaves are string spellings, and the repo already had the exact mechanism for that:
tagged CUSTOMI builders resolved per renderer through `ctx` (`codegen/late/warp_reduce.py:29`,
`codegen/late/flash_decode_intrinsics.py:23`, providers on HIP/Metal/CUDA). The NV port was "add
two CUDA providers" (`renderer/cuda.py:58` `__shfl_xor_sync`, `:59` two-FMA fdot2) plus a compile
pin (`dcc1bc778`). Decode is generic because it **renounced tensor cores** - its tile is per-thread
element ownership, which is target-neutral by construction.

### 2.2 Why fused prefill is not that shape

The fused prefill kernel is *fragment-shaped*. `kernels.py:265` builds

```python
qk = UOp(Ops.WMMA, dtypes.float.vec(8), (fr(q,"Q",b), fr(k,"K",b), qk),
         warg=("WMMA_16_16_16_half_float", (16,16,16), dtypes.half, dtypes.float,
               "AMD:gfx1100", 32, axes, ()))
```

Everything downstream is derived from AMD's m16n16k16 wave32 fragment ownership: the fragment
loads (`col = lane & 15`, `block*16 + i` element addressing, `amd_attention_abi.py:181`), the
softmax repack (`row = 2*e + halfwave`, `:243`), the output drain (`(2e+halfwave)*c_half`,
`cstyle.py:155`), and the raw `"bpermute"` strings (`:232`/`:246`; `schedule/wmma/softmax.py`).

CUDA sm_120 has **no m16n16k16**. `tc.get_cuda("sm_120")` (>= 89) returns cuda_81616
(dims (8,16,16), ept (8,4,4), `str(tc) == "WMMA_8_16_16_half_float"`) and cuda_8168_f16
(dims (8,16,8), ept (4,2,4)) - m16n8k16 and m16n8k8, whose C fragments are **4 floats per lane**
in a different lane->element map (PTX ISA fragment tables). A string provider cannot absorb this;
the *graph structure* differs. That is the structural difference from decode.

### 2.3 The generic form: tile-as-contract, fragment-as-data

The algorithm is genuinely tile-shaped: 16 q-tokens x 16 kv-tokens x 16 head-dim, online softmax
state, causal masking. Promote the **tile** to the contract and demote the **fragment
decomposition** to data.

- The builder (`kernels.py`) keeps writing tile-level math: one 16x16 QK score, one 16x16 PV
  accumulate per head block, flash state, causal boundary.
- A per-target **fragment model** (data, derived from `tc`: dims, elements_per_thread, swizzle,
  plus the documented C-fragment lane map where swizzle does not encode it) answers: how many
  WMMA calls compose the 16x16 tile (AMD: 1 m16n16k16; NV: 2 m16n8k16), each call's warg, each
  operand's lane->element map, and the C carrier width (AMD vec(8), NV vec(4) per call).
- The expansions in `amd_attention_abi.py` and `cstyle.py` consume that model instead of
  `col = lane & 15` / `row = 2*e + halfwave` / `(2e+halfwave)*c_half` literals. The softmax and
  drain then see the same *tile*; only the lane decomposition differs.
- The renderer binding is a registration, not a rewrite: the same C-style expansions HIP binds
  under `if target.arch.split(":")[0] == "gfx1100"` (`cstyle.py:630`) get an NVCC analogue, and
  the raw `"bpermute"` strings become the tagged `warp_shfl_xor` builder (decode's exact fix).

So the honest summary: **the architecture is decode's (one graph, per-target leaves), but the
per-target leaf is a fragment-ownership model derived from `tc`, not a string provider.** That is
the difference between "two provider lines" and a real (bounded) geometry-derivation package.

### 2.4 Why this is the right cut for this codebase

- The **HIP path already consumes the same expansions** as the AMD ISA path
  (`cstyle.py:630`, `native_repack_matcher` / `native_state_lane_matcher` /
  `native_loop_fragment_matcher`) - a second renderer consuming the same lowering is already the
  architecture, proven once.
- The **dense-prefill NV port already did the fragment-derivation half of this exact problem**:
  `948b26318` generalized the precontract fragment builder from RDNA3's `lane%16` to
  `derive_wmma_operand_lane_layout(tc)` (`kernel_lds.py:129`), with NV's split contracts derived
  from explicit (element_bit, axis_bit) term tuples, keeping the pg2 six AMD hashes unmoved. This
  scope's P3 is the attention analogue of that commit, reusing its helper family where possible.
- The **promotion mechanism** is the established BoltBeam record pattern (`model_route_plan.py:139`
  `load_qk_target_promotion`, `boltbeam.route_policy.v1` + `promoted_targets`; the fp16 overlay's
  checked-in compact artifact in `tinygrad/llm/generated/`). P4 mints the attention record through
  the same channel.

---

## 3. Established state (audited 2026-08-01 at dcc1bc778)

### 3.1 What already exists

- **Target-keyed emitter seam**: `_PREFILL_EMITTERS = {"amd_gfx1100": lambda spec, **kw: spec.emit(**kw)}`
  (`fused_attention.py:77`), resolved by `spec.target`; module comment: *"a second GPU is a new
  dict entry + a per-target emitter, a modular add rather than a rewrite of this routing code."*
- **The lowering is UOp-based**: `expand_loop_fragment(x: UOp) -> UOp` and the six typed Ops in
  `renderer/isa/amd_attention_abi.py` lower descriptor -> ordinary UOps *before* instruction
  selection; `AMDISARenderer` and `HIPRenderer` both consume them today.
- **Grid admission is model-shaped**: `ADMITTED_GRIDS = {(32,8,512), (40,8,512)}`
  (`fused_attention.py:68`) - Qwen3-8B and 14B head counts, shared by AMD and NV identically.
- **The AMD attention control exists and passes** (`scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py`):

  | grid | sha256 | v_wmma | instructions |
  | --- | --- | ---: | ---: |
  | (32,8,512) | `19829976aa55` | 16 | 1752 |
  | (40,8,512) | `7efc22cdda57` | 16 | 1753 |

  These hashes are the acceptance gate for every later package (see section 6.1). The control builds the
  AST through the production seam (`FlashPrefillAttentionSpec.emit()`), renders with
  `AMDISARenderer` (`AMD:ISA:gfx1100`), pure-Python assemble, no GPU.
- **Decode's portability machinery is in place and NV-proven**: tagged `warp_shfl_xor`
  (`warp_reduce.py:29`) with CUDA `__shfl_xor_sync(0xffffffffu, ...)` provider
  (`renderer/cuda.py:58`); `fdot2`/`exp2f` providers (`:59`); NVCC compile pin
  (`test_nv_kernel_source_has_no_amd_builtin_and_compiles_with_nvcc`). The NV decode tile is
  admitted and measured (decode 158.62 tok/s baseline above).
- **The dense-prefill NV fragment derivation exists**: `derive_wmma_operand_lane_layout`
  (`kernel_lds.py:129`), used by both fragment paths; sm120 candidate set minted
  (`bench/prefill-pure-full-kernel/multirole-buffer2-candidate-set-sm120-v1/`).
- **The fused-causal ternary CUSTOMI** `(({1}<={2})?{0}:-INFINITY)` is renderer-neutral C and
  works verbatim on NVCC.
- **Descriptor data for NV already exists**: `cuda_81616` / `cuda_8168_f16` in
  `codegen/opt/tc.py:116` / `:122`, with swizzle and `str(tc)` naming that the CUDA renderer
  already consumes (`renderer/cuda.py` `render_kernel` mma.sync emission).

### 3.2 What is welded - the AMD surface, itemized

1. **WMMA geometry literal**: `("WMMA_16_16_16_half_float", (16,16,16), half, float, "AMD:gfx1100", 32, ...)`
   at `kernels.py:265` (and the same literal at `:36/:61/:109/:166/:225/:346/:368` for the other
   kernel families; only :265 is on the `FlashPrefillAttentionSpec` emit path).
2. **Fragment load layout**: `expand_loop_fragment` (`amd_attention_abi.py:105`) addresses Q/K/V
   with `col = lane & 15`, `block*16 + i`, `row*16` - AMD's m16n16k16 A/B fragment ownership,
   hardcoded.
3. **Opaque-fragment matcher width**: `_opaque_exact_fragment_inputs` requires
   `dtypes.half.vec(16)` STACKs with 16 srcs and the `amd_gfx1100_fragment_load_hd128_loop_v1`
   tag family (`amd_attention_abi.py:47-76`).
4. **Softmax row map**: `row = 2*e + halfwave` with `halfwave = lane >> 4`, `col = lane & 15`
   (`amd_attention_abi.py:243`), and the repack requires `score.dtype == dtypes.float.vec(8)`
   (`softmax.py:56`, `amd_attention_abi.py:206`).
5. **Raw `"bpermute"` CUSTOMI strings**: `expand_native_row_softmax_repack` (`:232`, `:246`) and
   `amd_gfx1100_broadcast_row_state` (`schedule/wmma/softmax.py`). The fix already exists: tagged
   `warp_shfl_xor` (`warp_reduce.py:29`).
6. **Drain lane convention**: `(2e+halfwave)*c_half` in `AMDAttentionOutputDrainSpec.drain_lane_coeffs`
   (`uop/ops.py:1830`) and the HIP expansion `_hip_expand_attention_output_drain` (`cstyle.py:155-176`),
   including `range(8)` and `col = lane & 15` / `half = lane >> 4`.
7. **Renderer binding gate**: HIP binds all native matchers inside
   `if target.arch.split(":")[0] == "gfx1100":` (`cstyle.py:630`), including the
   `dtypes.weakint -> "int"` type_map entry. NVCC has no analogue.
8. **Identity pins**: `AMDAttentionGridSpec.validate()` hard-raises unless
   `native_abi == "amd_gfx1100_attention_grid_hd128_v1"` (`uop/ops.py:1743`); the spec field
   `FlashPrefillAttentionSpec.target = "amd_gfx1100"` default; identity string
   `f"amd_gfx1100_q16_grid_hd128_loop_attention:role=attention_tile,..."`
   (`fused_attention.py:205`).
9. **rangeify raw-gfx1100 path**: `rangeify.py:228` `_lower_shaped_wmma(raw_gfx1100_c=True)`
   requires `dims == (16,16,16)` and `device in ("AMD", "AMD:gfx1100")`. NOTE: this is the
   `SHAPED_WMMA` scheduler path (ROW_SOFTMAX_REPACK consumer rewrite), NOT the
   custom-kernel-injection path - the fused kernel constructs `Ops.WMMA` directly. Verify in P0
   whether custom-kernel injection ever reaches it; if not, it is out of scope and must be
   recorded as such, not silently changed.
10. **Target allowlist**: `_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset({("AMD","gfx1100")})`
    (`model.py:83`), default-closed by design with a long module comment (section 3.5).

### 3.3 The six typed Ops and their target-dependence

From `amd_attention_abi.py` (and the metal scope doc's section 3.2):

| Op | role | target-dependence after P1-P3 |
| --- | --- | --- |
| `AMD_PACKED_FRAGMENT_LOAD` | Q/K/V fragment addressing | **fragment geometry** (P3) |
| `AMD_ROW_SOFTMAX_REPACK` | QK-C -> P -> PV-A bridge | **lane layout** (P3) |
| `AMD_ROW_SOFTMAX_SLOT` | projection of the above | **lane layout** (P3) |
| `AMD_PV_C_LANE` | PV accumulator lane view | **lane layout** (P3) |
| `AMD_ATTENTION_LOOP_STATE` | loop-carried m/l/acc | neutral (flash state) |
| `AMD_ATTENTION_OUTPUT_DRAIN` / `AMD_ATTENTION_STATS_DRAIN` | final drain | **lane layout** (P3) |

The audit answers the metal scope's FA0 as **PARTIAL with a narrow weld**: the lowering output is
renderer-neutral UOps (HIP consumes the same expansions), and the AMD-ness lives in (a) the
fragment-lane literals, (b) the raw bpermute strings, (c) the WMMA warg literal, (d) the cstyle
binding gate. None of these requires a new emitter; all four are data-or-registration changes.

### 3.4 What is NOT on this path (out of scope, must not be touched)

- `rangeify.py:228` raw-gfx1100 path, unless P0 proves custom-kernel injection reaches it.
- The `amd_gfx1100_q32_hq4_hkv2_kv64_hd128_loop_v1` / `qk_stats_stage` / `pv_slice_stage`
  builder families (not on `FlashPrefillAttentionSpec.emit()`).
- `AMDISARenderer` itself (renders AMD ISA; keeps its name and its AMD-only binding).
- `PREFILL_V_TRANSPOSED` (a runtime env, not a weld; must keep working byte-identically on AMD).
- The dense-GEMM prefill path and its candidate set (already NV-working).

### 3.5 The promotion question, resolved

`model.py:70-88` currently documents the fused route as injecting "an already-CAPTURED,
hand-authored AMD gfx1100 machine-code program" and therefore default-closed. **That comment is
stale**: `FlashPrefillAttentionSpec` (P4a, `schedule/wmma/flash_prefill.py`) turned the route into
a UOp builder lowered per renderer through the emitter seam. The kernel is no longer opaque ISA
injection - it is renderer-lowered UOps like everything else.

The default-closed posture still stands for a *different, still-true* reason: the fragment math is
AMD-shaped until P3 lands, and admission must not run unproven geometry on a target. So:

- Keep default-closed semantics for this route (unlike TG3's `None -> open` default): no promotion
  record -> closed.
- Source the promoted-target set from a **BoltBeam-sourced record** in the exact shape of
  `load_qk_target_promotion` (`model_route_plan.py:139`: `boltbeam.route_policy.v1` +
  `promoted_targets` list of `{backend, architecture}`), minted for this route, containing
  `("AMD","gfx1100")` AND `("NV","sm_120")` once P5 proves NV.
- Replace the hardcoded `_CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS` frozenset with a load of
  that record; rewrite the stale `model.py:70` comment to name the real reason (unproven fragment
  geometry) instead of the retired "captured program" claim.

This is the house pattern (Q4K/Q6K promotion via `load_qk_target_promotion`; fp16 overlay via the
checked-in `ARTIFACT` in `prefill_candidate_runtime.py:24`), and it is how the user's
"use boltbeam for promotion" instruction is satisfied: the record is a BoltBeam mint, the
consumer is the explicit-path JSON loader, and no target string is hardcoded in production code.

---

## 4. Work packages

### P0 - Compile-only NVCC probe. First. Prerequisite: none.

Build the 8B AST via the production seam (`FlashPrefillAttentionSpec(Hq=32, Hkv=8, Hd=128,
q_tokens=512, kv_tokens=512, causal=True, ...).emit()` with PARAM placeholders in slots 0-3) and
run it through `NVCCRenderer(Target.parse("NV:NVCC:sm_120"))` via `to_program`, mirroring
`test/unit/test_flash_decode_intrinsics_renderer_lowering.py`'s `_tile_ast`/`_rendered_source`.

Deliverable: a catalog of the exact failure points (expected: WMMA warg invalid on sm_120, vec(8)
C width, bpermute provider missing on CUDA, native matcher not bound, weakint mapping). This
converts the audit's "should fail at X" into "fails at X, Y, Z" and pins the order P1-P3 must
land. Also check the section 3.2.9 rangeify question.

### P1 - Genericize the raw bpermute strings. Prerequisite: P0.

Replace the raw `"bpermute"` CUSTOMI in `expand_native_row_softmax_repack` and
`amd_gfx1100_broadcast_row_state` with the tagged `warp_shfl_xor` builder (`warp_reduce.py:29`),
whose HIP provider (`cstyle.py` `__builtin_amdgcn_ds_bpermute`) is byte-identical to the current
literal and whose CUDA provider already exists. Pure decode-machinery reuse; no geometry change.

Gate: AMD control hashes unchanged (both grids), plus the `PREFILL_SOFTMAX_REDUCE_FUSE` fmaxf
peer-matching rule (`_hip_native_bpermute_max`, `cstyle.py:139`) still matches the tagged form.

### P2 - NVCC native matcher bindings. Prerequisite: P1.

Give NVCCRenderer the analogue of the gfx1100 block (`cstyle.py:630`): bind
`native_repack_matcher`, `native_state_lane_matcher`, `native_loop_fragment_matcher`, the drain
expansions, and the `weakint -> int` type_map - but **parameterized by the fragment model** (P3's
data), not by copied literals. This is the package where the "same expansions, per-target lane
model" architecture becomes real: the expansions stop reading `lane & 15` and start reading the
model.

Gate: AMD control unchanged; NVCC render now produces *a* program for the 8B grid (compile may
still fail on the warg; see P3).

### P3 - NV WMMA geometry from `tc`, emitter entry, ABI generalization. Prerequisite: P2.

1. **Derive the warg from `tc`**: for target NV, resolve `tc.get_cuda("sm_120")` and build each
   WMMA call's warg from the descriptor (dims, `str(tc)` name, dtype pair, threads, upcast axes).
   The 16x16 tile becomes two m16n8k16 QK calls (C vec(4) each) and two PV calls; the builder
   composes the tile from the fragment model instead of one m16n16k16.
2. **Generalize `expand_loop_fragment`** to address operands from the fragment model's
   lane->element map (reuse `derive_wmma_operand_lane_layout`'s term-tuple representation from
   `kernel_lds.py:129` where the operand shapes match; extend where C-fragment/softmax layout is
   needed - PTX ISA fragment tables are the declared source for the parts swizzle does not
   encode).
3. **Generalize the softmax row map and drain lanes** to the model (row/col ownership per lane,
   C width vec(8)/vec(4)).
4. **Emitter entry**: `_PREFILL_EMITTERS["nv_sm120"]` -> a spec target `"nv_sm120"` whose emit
   routes through the same builder with the NV fragment model.
5. **Identity generalization**: `AMDAttentionGridSpec.validate()`'s `native_abi` pin becomes
   parameterized (grid spec already carries shape; add the ABI/fragment identity), the identity
   string in `fused_attention.py:205` derives target/geometry instead of the literal
   `amd_gfx1100_q16_grid_hd128_loop_attention` prefix, and `FlashPrefillAttentionSpec.target`
   resolves via the emitter registry.
6. **Rename what becomes shared, in the same commit that shares it** (metal scope FA2 rule 6):
   names that describe geometry or flash-attention concepts rather than AMD hardware lose the
   `AMD_` prefix once a second target consumes them (`Ops.AMD_ATTENTION_LOOP_STATE` -> the
   loop-state op; `AMDAttentionGridSpec` -> grid spec). `AMDISARenderer` keeps its name. Renames
   happen only for the Ops/names P3 actually shares, in the same commit, with the AMD control
   re-run after.
7. **No `if backend == "NV"`**: every branch is a data look-up on the fragment model or the
   renderer's declared facts.

Gate: AMD control unchanged; NVCC compile-only render succeeds for both admitted grids with no
`m16n16k16` text in the source; `wmma_args` counts per grid recorded.

### P4 - BoltBeam promotion record. Prerequisite: P3 + NV compile gate.

Mint a `boltbeam.route_policy.v1` record for the fused prefill attention route with
`promoted_targets = [{"backend": "AMD", "architecture": "gfx1100"}]` initially (byte-identical
AMD behavior), checked in under `tinygrad/llm/generated/` (fp16 overlay precedent,
`prefill_candidate_runtime.py:24`). Replace the hardcoded frozenset in `model.py:83` with the
record load; rewrite the stale comment (section 3.5). **The NV entry is added to the record only after
P5's e2e token parity passes** - the record is the promotion gate, so AMD behavior never flips
and NV admission is a one-line data change gated on evidence.

Gate: AMD path census unchanged (still dispatches), NV census still
`no-promoted-candidate` until the record gains NV; `test_route_admission_consistency`-style
policy tests updated to load the record.

### P5 - Verify and measure. Prerequisite: P4.

1. **AMD control**: both grid hashes byte-identical to section 3.1 after every commit in P1-P4.
2. **NVCC compile**: both admitted grids render through NVCCRenderer and `ren.compiler.compile`
   succeeds (the `dcc1bc778` pattern).
3. **Correctness on three axes, reported separately** (metal scope section 6.2): `max_abs_error` vs the
   SDPA reference, write coverage of the output, determinism across >= 3 runs. Fixed-seed
   first-token parity on the 5090 through `model_e2e_bench.py`; AMD e2e digits unchanged.
4. **Then measure**: pp512 prefill tok/s on the 5090 vs the 81.9 baseline; report the census row
   (`custom_kernel_attention_trace` dispatches > 0) and the promotion record's effect.
5. Only after (3) and (4): add `("NV","sm_120")` to the promotion record, re-run the e2e, and
   report the final pair.

---

## 5. Promotion mechanism detail (P4)

- **Consumer**: `_custom_kernel_prefill_attn_promoted` reads the record through an explicit-path
  loader identical in shape to `load_qk_target_promotion` (`model_route_plan.py:139`) - JSON,
  `schema == "boltbeam.route_policy.v1"`, `promoted_targets` list, each entry
  `{backend, architecture}`. No `extra/llm_research` import (Boundary Rule).
- **Semantics**: no record loaded -> CLOSED (this route's deliberate deviation from TG3's open
  default; reason recorded in the rewritten comment: unproven fragment geometry, not capability).
  Record loaded -> enforced set.
- **Mint**: BoltBeam emits the record (its route-policy emit path, `boltbeam/policy/emit.py`),
  checked into tinygrad under `tinygrad/llm/generated/`. The minting is evidence-gated by P5; the
  checked-in JSON is the single source of truth, and `git diff` on that file is the promotion
  review artifact.

---

## 6. Evidence contract

1. **AMD non-regression is structural and mandatory.** The section 3.1 control hashes (`19829976aa55`,
   `7efc22cdda57`) are byte-identical after every commit in P1-P4. No AMD hardware here; rendered
   source equality through the production seam is the strongest available guarantee, and it
   covers both admitted grids.
2. **Correctness on three axes, reported separately** - `max_abs_error`, write coverage,
   determinism across >= 3 rounds. Collapsing them hid a two-bug structure for a day (metal
   scope section 6.2).
3. **One defect per commit**, measured between, with a predicted signature stated in advance.
   P1-P3 each have an explicit gate in section 4; if a gate fails, that package is not done.
4. **Every number from a command actually run.** Three conclusions have been retracted in this
   campaign; the retraction ledger is the reason.
5. **No backend branches.** A `backend == "NV"` literal anywhere in the attention lowering is a
   review failure, not a style point.
6. **Rename-in-the-same-commit** for anything that becomes shared (P3.6), with the AMD control
   re-run immediately after the rename commit.
7. **GPU work serialised**; the 5090 is shared.
8. Test identity sets, not counts: `test/unit` failing IDs recorded per commit if any.

---

## 7. Non-goals

- Promotion to `dev`/`master`; this branch is `nvidia-bringup-20260731`.
- The rangeify raw-gfx1100 scheduler path (unless P0 proves the custom kernel reaches it).
- The `kv64`/`stats_stage`/`pv_slice` builder families (section 3.4).
- Rewriting the dense-GEMM prefill path or its sm120 candidate set.
- The Metal fused-attention port (separate scope, `metal-fused-attention-port-scope-20260731.md`);
  this scope's fragment model is designed so Metal (tc 8x8x8) is a future third consumer, but no
  Metal work is authorized here.
- Any claim that fused prefill is faster than SDPA on NV before P5 measures it.
- `PREFILL_SOFTMAX_REDUCE_FUSE` semantics changes (P1 must preserve the fmaxf peer rule exactly).

---

## 8. Known limitations

- **No AMD hardware.** AMD non-regression is structural only (rendered source + instruction
  counts), never an execution result.
- **No PTX/ISA fragment tables reproduced in this doc.** P3's fragment model needs the CUDA
  m16n8k16 C-fragment lane map; the PTX ISA (and `tc.py`'s swizzle data) is the declared source.
  If the derived map is wrong, P5's max_abs_error axis catches it - that axis is why it exists.
- **The m16n8k16 tile is two WMMA calls per 16x16 tile on NV.** This may be slower than one
  m16n16k16 on AMD; P5 measures head-to-head and the scope makes no speed promise.
- **The `_opaque_exact_fragment_inputs` matcher and the repack's `float.vec(8)` requirement**
  (amd_attention_abi.py:47, softmax.py:56) are welded to the AMD C width; P3 changes them, and
  the AMD control is what proves the change is invisible.
- `model.py:70`'s "captured program" comment is stale; P4 rewrites it as part of the promotion
  record change, and the rewrite is reviewable in the same commit.
- Pre-existing test failures on this box are unrelated (4 Metal unit tests without libSystem,
  `test_route_admission_consistency.py::test_candidate_quant_matches_the_manifest` fails at
  baseline 57e25b662) and must not be "fixed" as part of this scope.

---

## 9. One-line job

Derive the fused prefill attention fragment decomposition from `tc` per target, bind the shared
C-style expansions on NVCC, mint a BoltBeam promotion record for AMD+NV, and prove the 5090 e2e
parity while keeping the AMD rendered-source hashes byte-identical.

---

## 10. Review findings (Claude, 2026-08-01)

Every cited line in sections 1-9 was checked and is where the doc says it is: `kernels.py:265`'s warg
literal, `NVCCRenderer` (`renderer/cuda.py:117`, real class), `derive_wmma_operand_lane_layout`
(`kernel_lds.py:129`) and commit `948b26318`, `_PREFILL_EMITTERS`/`ADMITTED_GRIDS`
(`fused_attention.py:68,77`), `model.py:66-88`, and `boltbeam/policy/emit.py`. Section 2's genericity
argument -- scalar-shaped versus fragment-shaped, string providers versus a fragment model -- is correct
and is the right framing for the whole scope. Four findings.

### 10.1 The acceptance gate does not cover the code P1 and P2 change (blocking)

`scratchpad/fa_ctrl_amd_attention_rendered_source_equality.py` renders **only** through `AMDISARenderer`;
its own docstring at line 24 says so explicitly (*"RENDERER: AMDISARenderer (`AMD:ISA:gfx1100`), NOT
`HIPRenderer`"*), and it is the only attention control in `scratchpad/`.

But P1 edits `_hip_native_bpermute_max` (`cstyle.py:139`), P2 binds matchers and drain expansions inside
the gfx1100 block (`cstyle.py:630`), and the drain expansion at `cstyle.py:155-176` is HIP-path code.
Section 6.1 calls the two ISA hashes byte-identical-after-every-commit and names that structural AMD
non-regression -- but a HIP-path regression renders identically under `AMD:ISA` and the gate stays green.

The doc half-knows this: P1's gate requires the `PREFILL_SOFTMAX_REDUCE_FUSE` fmaxf peer rule to "still
match the tagged form" without saying what verifies it.

**Fix, in P0, before anything moves:** add a `HIPRenderer` arm to the control and pin both renderers' hashes.
Until that exists, section 6.1 overstates its own coverage.

**Also settle in P0:** the control's docstring asserts attention goes through the ISA renderer rather than
the C-style one. If that is true for production AMD, say so in section 3.1. If HIP is also a live consumer
(section 2.4 says it consumes the same expansions), HIP coverage is mandatory, not optional.

### 10.2 The stale-comment finding is right, but under-scoped

Confirmed stale -- and only partly, and in more places than the one named:

- Still **true**: injection is via `Tensor.uop_program` (`fused_attention.py:146`, `kernel_program.py:71`).
- Now **false**: "an already-CAPTURED, hand-authored AMD gfx1100 machine-code program (`.hip.cpp`/`.amdisa.s`
  produced by `generate_shared_attention_captures`)". The route builds `Ops.WMMA` UOps through `spec.emit()`.
- The same stale claim also lives at **`fused_attention.py:18`** ("already-proven captured program") and
  **`fused_attention.py:43`** ("produced by extra/llm_research/generate_shared_attention_captures").

P4 must rewrite all three, and must keep the accurate clause. The *consequence* half changes too: the risk
is no longer "inject raw AMD ISA on a non-AMD renderer -> crash" but "render AMD fragment math on a target
that cannot express it -> compile failure or wrong numbers". Same closed default, different failure mode;
the new comment should state the real one.

### 10.3 The PTX fragment table may not need to exist -- prove that in P0

Section 2.3 and P3.2 derive the fragment model from `tc` "plus the documented C-fragment lane map where
swizzle does not encode it", and section 8 admits the derived map might be wrong with `max_abs_error` as the
backstop. That hand-transcribed PTX table is the one piece of this scope that would be hand-authored data on
the production path, which is what the machine-search purity contract restricts.

It may be unnecessary. `cuda_81616` already carries a full `swizzle` spec, and `derive_wmma_operand_lane_layout`
already derives lane layout from swizzle for the dense path -- that is what `948b26318` did.

**Make this a P0 deliverable alongside the failure catalog:** does the swizzle encode the C-fragment map, or
not? If it does, the PTX table is never written and P3 is pure reuse of the existing helper family. If it does
not, the table is justified -- but it must be *declared with a citation and labelled as the one hand-authored
input*, never presented as "derived". This decides whether P3 is a derivation package or a transcription
package, which is a real difference in both risk and how it should be reviewed.

### 10.4 The P2/P3 dependency graph does not close

P2 is specified as "parameterized by the fragment model (P3's data)" but lists P1 as its only prerequisite.
P2 cannot bind expansions to a model P3 has not defined yet. Resolve one of two ways:

- shrink P2 to the binding-and-`weakint` half with the lane literals still in place (expansions move to the
  model in P3), or
- move the fragment-model definition to the front of P3 and make P2 depend on it.

Either is fine; as written the ordering is unbuildable.

### 10.5 Endorsed as-is

- **P0-first.** Cataloguing exact failures before ordering the fixes is the right instinct for a port whose
  audit is necessarily partial.
- **P4's promotion structure.** Mint the record AMD-only, add NV only after P5 parity, so the checked-in JSON
  diff *is* the promotion review artifact. This is the strongest part of the doc.
- **The no-`if backend == "NV"` rule and rename-in-the-same-commit** (section 6.5, 6.6).
- **The refusal to promise a speedup before P5 measures** (section 1 caveat, section 7).
