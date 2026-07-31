# Metal fused prefill attention — port scope

Date: 2026-07-31

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to
`dev`/`master`.

---

## 1. Why — the measurement that motivates this

Measured 2026-07-31, same session, GPU-serialized, Qwen3-8B-Q4_K_M on Apple M4:

| depth | ours | llama Metal | ours as % | gap |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 54.20 | 216.68 ± 6.21 | 25.0% | 4.00× |
| 1024 | 50.79 | 207.11 ± 2.36 | 24.5% | 4.08× |
| 2048 | 41.10 | 203.25 ± 1.41 | 20.2% | 4.95× |
| 4096 | **26.49** | **195.48 ± 1.17** | **13.6%** | **7.38×** |

**Ours decays −51.1%; llama decays −9.8%.** Per-chunk cost grows **3.59×** with start position
(9,446 ms → 33,881 ms). Artifacts: `bench/prefill-whole-synced/metal-depth-decay-20260731.json`,
`bench/llama-metal-depth-20260731/llama-bench-depth.json`.

`bench/prefill-whole-synced/t2-metal-pp512.json` reports
`custom_kernel_attention_trace: {"dispatches": 0}` — **Metal dispatches no fused attention kernel.**

AMD, with a fused path, moves the other way: its margin over llama *improves* with depth on 14B
(+5.6% at pp512 → +8.8% at pp4096).

**This is now the largest measured lever on Metal.** The precontract kernel completed today is 3.4× in
isolation but lives in the depth-flat half; it cannot explain a −51.1% decay when llama on identical
silicon decays −9.8%.

**Caveat carried forward:** the per-kernel attribution attempted earlier today was **retracted**
(`718d5717d`) — its `DEBUG=2` capture covered ~7% of wall clock. So *"attention causes the decay"* is a
reading of the depth curve plus the `dispatches: 0` fact, **not** a validated attribution. FA0 below
must establish it before FA2 builds anything.

---

## 2. What already exists, and it is more than the precontract port had

- **A target-keyed emitter seam, built for exactly this.** `tinygrad/llm/fused_attention.py:77`:
  ```python
  _PREFILL_EMITTERS = {"amd_gfx1100": lambda spec, **kw: spec.emit(**kw)}
  ```
  resolved at `:201` via `emitter = _PREFILL_EMITTERS[spec.target]`, with a comment stating *"a second
  GPU is a new dict entry + a per-target emitter, a modular add rather than a rewrite of this routing
  code."*
- **The lowering is UOp-based**, not hand-written assembly: `expand_loop_fragment(x:UOp) -> UOp`.
- **Grid admission is model-shaped, not target-shaped**: `ADMITTED_GRIDS = {(32,8,512), (40,8,512)}`
  — Qwen3-8B and 14B head counts.
- **`wave_size == 32` matches Metal's subgroup width exactly.**
- The file has already survived one generalization pass — `head_dim` de-literalized from `!= 128` to
  "any positive 16-wide", and a `kv_tokens` 16-alignment requirement removed once it was found to be
  masking an out-of-bounds read rather than expressing a real ABI constraint.

---

## 3. What is welded — and one of these is structural, not a literal

### 3.1 Literals of the class already solved today

- `AMDAttentionGridSpec.validate()` hard-raises unless
  `native_abi == "amd_gfx1100_attention_grid_hd128_v1"`.
- **16-wide fragment granularity**, which the code comments call *"our fragment granularity, hardware,
  and stays literal."* That 16 is AMD's WMMA `dims`. **Metal's is 8.** This is the identical literal
  found three times in the precontract path today (`tc.dims[0]`, `elements_per_thread[2]`, and a missing
  `k_groups` that equals 1 only on AMD).

### 3.2 The structural one: six bespoke Ops bound to an ISA renderer

`tinygrad/renderer/isa/amd_attention_abi.py` (425 lines) lowers **six renderer-specific Ops** whose
meaning is fixed by typed descriptors in `tinygrad/uop/ops.py`:

| Op | descriptor | role | likely target-dependence |
| --- | --- | --- | --- |
| `AMD_PACKED_FRAGMENT_LOAD` | `AMDPackedFragmentLoopSpec` | Q/K/V fragment addressing | **fragment geometry** |
| `AMD_ROW_SOFTMAX_REPACK` | `AMDRowSoftmaxRepackSpec` | QK-C → P → PV-A bridge | **lane layout** |
| `AMD_ROW_SOFTMAX_SLOT` | — | projection of the above | **lane layout** |
| `AMD_PV_C_LANE` | `AMDPVCLaneSpec` | PV accumulator lane view | **lane layout** |
| `AMD_ATTENTION_LOOP_STATE` | `AMDLoopStateSpec` | loop-carried m/l/acc | likely neutral (flash state) |
| StateHandle phase publication | — | generic | neutral |

The module's own docstring is explicit about what this is:

> This module is the whole lowering surface for them: **descriptor → ordinary UOps, before instruction
> selection sees anything AMD-specific.** It exists as its own module because these Ops are a *second
> system* living beside the generic RDNA3 renderer.

**Consumers: `AMDISARenderer` binds these as its `native_*` pattern matchers.** Metal has no ISA
renderer — `MetalRenderer` is a C-style source renderer. **There is no equivalent binding surface**, and
the precontract port never faced this.

---

## 4. The question the whole port turns on

**Do the six Ops lower to renderer-neutral UOps, or to something only `AMDISARenderer` can consume?**

The docstring claims *"descriptor → ordinary UOps, before instruction selection sees anything
AMD-specific."* If true, the AMD-ness is in the **descriptors** (fragment geometry — derivable, §3.1)
and the **renderer binding** (a registration, not a rewrite), and the port is the same shape as the
precontract work.

If false — if the lowered form assumes AMD ISA semantics — then this is a new Metal attention emitter
behind the `_PREFILL_EMITTERS` seam, which is a materially larger and different piece of work.

**FA0 answers this and nothing else. Do not start FA2 before it reports.**

---

## 5. Work packages

### FA-CTRL — Establish an AMD attention control. **Before anything else.**

Prerequisite: none. Compile-only.

`scratchpad/pg2_amd_all_routes_rendered_source_equality.py` covers the six `PACKED_WMMA_ROUTES`
rows — **it does not cover the attention path.** Without an equivalent, there is no way to prove AMD
non-regression on shared attention code, and there is no AMD hardware here.

Build the analogue: render the fused attention kernel for both admitted grids
(`(32,8,512)` = 8B, `(40,8,512)` = 14B), hash the rendered source, record `__WMMA` counts. One command,
rerunnable. **This is the acceptance gate for every later packet.**

Stop condition: if the attention path cannot be rendered compile-only the way the packed routes can,
report that — it changes the safety model for everything downstream.

### FA0 — Answer §4. Compile-only.

Determine whether the six Ops' lowering output is renderer-neutral. Concretely: take the lowered UOp
graph after `amd_attention_abi`'s matchers run, and establish whether anything in it is only meaningful
to `AMDISARenderer` — custom Ops that survive lowering, ISA-specific intrinsics, register/lane
assumptions that a source renderer cannot express.

Report **NEUTRAL** (port is a descriptor + registration job), **WELDED** (a new Metal emitter is
required), or **PARTIAL** with the specific Ops in each category.

### FA1 — Validate the attribution. GPU.

`718d5717d` retracted the per-kernel attribution because its capture covered ~7% of wall clock.
**Redo it with a method that reconciles against wall clock before drawing any conclusion.** Sum of
captured kernel times must account for the measured chunk time, or the capture is not representative
and must be reported as such rather than analysed.

Deliverable: what fraction of the 33,881 ms chunk at `start_pos=3584` is attention. If attention is not
the dominant term, this scope is aimed at the wrong thing and should stop.

### FA2 — The port. Prerequisite: FA-CTRL, FA0, FA1.

Shape depends on FA0. Apply the method that removed thirteen AMD couplings today:

1. **Derive from the target's descriptor** — the 16-wide granularity from `tc.dims`, not a literal.
2. **Declare what cannot be derived** as a `Renderer` field, with the polarity rule: optimizations
   default off, correctness defaults **on**, and only an explicit declaration citing the hardware
   property may skip a correctness requirement.
3. **Replay the working path rather than reimplementing it**, so there is one source of truth.
4. **Fail closed** on any descriptor family the code cannot resolve.
5. **No `if backend == "METAL"`.** Thirteen couplings were removed today with zero backend branches.

### FA3 — Measure. Prerequisite: FA2.

Whole-model Metal prefill at 512/1024/2048/4096 through the same harness that produced the baseline,
paired same-session against `llama-bench` as §1 was. The target is the **decay**, not the pp512 number:
−51.1% is the defect; llama's −9.8% is the reference.

---

## 6. Evidence contract

1. **AMD non-regression is structural and mandatory.** Both controls byte-identical after every change:
   the six packed-WMMA rows (`0e4c2e9218a7 8e01063e3c8f ce03d94bb58a 5ced48b9fa7c b0df79b8bb58
   349a2c8c521f`) **and** FA-CTRL's attention hashes. No AMD hardware here; byte-identical rendered
   source is the strongest available guarantee and it is only as strong as its coverage.
2. **Correctness on three axes, reported separately** — `max_abs_error`, write coverage, determinism
   across ≥3 rounds. Collapsing them hid a two-bug structure for a day.
3. **Any per-kernel capture must reconcile against wall clock** before conclusions are drawn. This is
   the specific failure that produced `718d5717d`.
4. `test/unit` failing-test-id **sets** (111 unique ids), never counts.
5. **One defect per commit**, measured between, with a predicted signature stated in advance.
6. Every number from a command actually run. Three conclusions have been retracted in this campaign.
7. GPU work serialised.

---

## 7. Use the oracle

llama.cpp's Metal flash-attention kernel is on disk at
`/Users/julianabeleda/env/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`, alongside the
`kernel_mul_mm` that resolved two precontract defects in one comparison after five hypotheses had
failed.

**Read it for structure — barrier placement, staging shape, how fragments reach
`simdgroup_multiply_accumulate`, how the KV loop is tiled. Never for geometry**: the corpus has the
measured proof that transplanting a tuned schedule between configurations ran **31% slower**.

---

## 8. Non-goals

- Promotion to `dev`/`master`.
- The precontract lifecycle work (QUALIFY, POLICY) — independent, separately scoped.
- Fixing the two known adjacent hardcodes (`kernel_pipeline.py:181`'s `vec(8)`,
  `_candidate_context`'s `stride=80`) — documented in the campaign doc, out of scope here.
- Any whole-model tok/s claim before FA3 measures one.

---

## 9. Known limitations

- **No AMD hardware.** AMD non-regression is structural only.
- **The attribution is unvalidated** (§1 caveat). FA1 exists to fix this and may redirect the scope.
- `amd_attention_abi.py` lives inside the hand-written AMD ISA backend rather than the generic
  renderer. The precontract port did not face an equivalent, and FA0 exists because the size of that
  difference is currently unknown.
- The precontract port took a full day and thirteen couplings starting from a *broken* path. This one
  starts from a working path, which is how that one also started.
