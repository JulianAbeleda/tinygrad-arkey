# Prefill qualification: one canonical path for all targets

Date: 2026-08-01

Status: scoped, not implemented. Reviewed 2026-08-01 (Claude); §3.1, §3.2, §3.3, §4, §5, §6 amended
with the review's findings. Written for review before code.

Branch boundary: all work begins on `nvidia-bringup-20260731` per the NVIDIA campaign rule (branch ->
selective merge to `exp` -> promote to `dev` only after NVIDIA is proven end-to-end). This scope does not
authorize promotion by itself.

## 1. The problem: one mechanism, two divergent qualification paths

The packed-WMMA precontract path is target-neutral in intent, but the qualification machinery that proves a
route correct has split into two designs with different authority shapes:

### Path A (AMD's, the original)

1. Admission resolves the tensor-core descriptor from a **frozen AMD capability lattice**:
   `FullKernelCapability` in `extra/llm_research/runtime_specs.py` defaults to backend `AMD`, arch `gfx1100`,
   `wmma_f32_16x16x16_f16`, `rdna3_wmma_f32_16x16x16_f16_lds2_static`, and
   `full_kernel_candidate_capability(payload)` returns one of the four `GFX1100_*` rows regardless of the
   payload's own target. `admit_full_kernel_candidate` then rejects any payload whose target is not exactly
   `{"backend":"AMD","arch":"gfx1100","wave_size":32}` with `capability_target: target is outside frozen
   capability`.
2. Compile evidence is produced by `prepare_current_prefill_compile` in
   `extra/llm_research/prefill/current_prefill_execution_adapter.py`, which **unconditionally disassembles
   the AMD ELF** (`disassemble_amdgpu`, `parse_amdgpu_metadata`, `analyze_final_isa`, operand attribution,
   descriptor register counts) and hardcodes `target = "gfx1100"`. This path requires a ROCm
   `llvm-objdump`/`llvm-readelf` toolchain and is AMD-only by construction.
3. Execution is delegated to `run_canary` (`packed_wmma_correctness_canary.py`), device-parameterized since
   M1b but built on the AMD-only compile-evidence path above.

### Path B (Metal's, the generalization)

1. Admission is device-aware: T6 (`7c03d821e`) threaded `device` through `admit_current_prefill` and resolved
   the tensor-core descriptor via a per-backend data table `_tensor_core_family_by_device` (`AMD` ->
   `tc.get_amd(arch)`, `METAL` -> `tc.metal`) -- the same declared facts the renderers already use. **The
   capability lattice itself stayed AMD-frozen**, so Metal probes still carry AMD-shaped payloads
   (`qwen3_8b_q4k_m_gfx1100`) and only vary the execution device.
2. Compile evidence is minimal: `metal_precontract_lane.py::_child_run` calls
   `compile_current_prefill_program` directly and builds `{"passed": True, "binary_sha256": ...}` rather than
   the AMD ISA manifest. The docstring says why: `prepare_current_prefill_compile` "unconditionally
   AMD-ELF-disassembles the compiled binary regardless of device" and the Mac has no ROCm toolchain.
3. Execution is the guarded lane: spawn-isolated, `synchronize` before/after every round, full-array readback
   capture, and `_summarize` computes `max_abs_error`, write coverage, and determinism -- the three-axis
   correctness proof the bring-up doc's phase 5 requires.

### The divergence, named

Both paths share the same compile primitive (`compile_current_prefill_program`, device-parameterized) and the
same guarded-execution primitive (`run_guarded_execution`). They differ in **two load-bearing places**:

| concern | Path A (AMD) | Path B (Metal lane) |
| --- | --- | --- |
| capability resolution | frozen AMD rows, payload target must equal AMD/gfx1100 | frozen AMD rows (same), device only affects tc resolution |
| compile evidence | AMD ELF disassembly + operand attribution + resource summary (ROCm toolchain required) | minimal `{passed, binary_sha256}` |
| correctness axes | `max_abs_error` only | `max_abs_error` + write coverage + determinism |
| entry point | `run_canary` / `verify_production_row` | `run_precontract_probe(ProbeConfig(device=...))` |

The result: a new target (NV, and originally Metal) cannot be qualified through Path A's admission, and Path B
exists as a workaround lane for the machinery Path A should already have provided. Two designs for one
mechanism means the AMD path is not the canonical one -- it is the first one.

## 2. End goal

**One-sentence reduction:** *the packed-WMMA precontract path has one qualification path -- device-aware
admission resolving a declared per-target capability (every target's row declared from measured facts, AMD
included), one compile-evidence producer that is device-parameterized with the AMD ISA manifest as an AMD
enrichment rather than the gate, and one guarded probe lane that measures all three correctness axes -- and
AMD's own promotion evidence is produced by running that same lane on AMD hardware, not by a parallel
AMD-only path or a frozen default.*

The AMD ISA manifest is real, valuable evidence (disassembly-verified instruction count, operand ownership,
resource summary). Nothing in this scope removes it. What changes is its **role**: it becomes the AMD-specific
enrichment inside a device-parameterized evidence producer, so the path is the same and the evidence is deeper
where the toolchain exists.

Two consequences follow from "AMD gets it by measured facts too":

1. **AMD's capability row is a declaration, not a default.** Its values equal today's (admission does not
   move), but the authority changes: the row is expressed as facts with citations (the target's own
   tensor-core descriptor, the declared wave size, the ISA's LDS budget) in exactly the same shape NV's row
   takes. Nothing in the row says "AMD is the default, everything else is outside."
2. **AMD runs the lane.** AMD's six promoted rows are re-verified through `run_precontract_probe` with
   `device="AMD"` on AMD hardware, producing the same three-axis evidence (error, coverage, determinism) NV
   and Metal produce. The re-verification lives on this branch/exp -- it does not touch the recorded
   `canary_max_abs_error` in `PACKED_WMMA_ROUTES` on `master` -- and it is a named gate, not an assumption:
   this machine has no AMD GPU, so the measured AMD lane run happens where AMD hardware exists and is
   recorded in the campaign evidence before the path is declared done.

   **The gate constrains the code, not just the evidence** (review finding, §3.3/C4). An open gate that
   only means "we have not measured yet" would leave AMD's promotion verifier *swapped to an untested
   implementation* on a box that cannot exercise it. So the swap itself is gated: `run_canary` stays the
   default verifier until the measured AMD lane run exists, and the lane path lands opt-in beside it.

## 3. Current vs proposed, aligned with the repo's coding principles

### 3.1 Capability resolution: frozen rows -> declared per-target rows

Current:

```python
GFX1100_SINGLE_BUFFER_CAPABILITY = FullKernelCapability()   # defaults say AMD/gfx1100
...
def full_kernel_candidate_capability(payload):
  ... return GFX1100_TWO_BUFFER_STAGE1_CAPABILITY if pipeline == (2,1) else GFX1100_SINGLE_BUFFER_CAPABILITY
```

`admit_full_kernel_candidate` then checks `target != {"backend": capability.backend, ...}` and raises
`capability_target` for anything not AMD/gfx1100.

Proposed: capability rows become a declared table and `full_kernel_candidate_capability(payload)` resolves
from the payload's own `workload.target` instead of ignoring it.

**Table key** (review finding). "One row per target" is the wrong shape: there are four AMD rows today and
the target does not distinguish them. `GFX1100_SINGLE_BUFFER_CAPABILITY` and
`GFX1100_TWO_BUFFER_STAGE1_CAPABILITY` differ only by `pipeline.(buffer_count, stage_count)`;
`GFX1100_REGISTER_RESIDENT_CAPABILITY` and `GFX1100_Q4K_Q8_FIVE_BUFFER_CAPABILITY` are selected by storage
kind and schedule family. The real key is therefore **`(backend, arch)` x schedule shape** -- the target
selects the hardware facts, the schedule selects the pipeline/transport row within that target. The two
AMD-specific schedule shapes (register-resident, five-buffer) stay AMD-only.

`wave_size` is *not* part of the key: it is a function of arch, and the repo already derives it that way
(`cstyle.py:598`, `64 if is_cdna(arch) else 32`). It stays a derived field that the existing three-way
`capability_target` equality check still asserts.

Every row is declared from the same kind of source, with a citation attached:

| target | row facts | source of the facts |
| --- | --- | --- |
| `amd_gfx1100` | `wmma_f32_16x16x16_f16`, `rdna3_wmma_f32_16x16x16_f16_lds2_static`, wave 32, vector bytes 16, max LDS 65536 | `tc.get_amd("gfx1100")` (the descriptor `HIPRenderer` uses), arch-derived `wave_size`, `HIPRenderer.shared_max = 65536` (`cstyle.py:583`) -- values identical to today's |
| `nvidia_sm120` | fp16/fp32 `mma` family as the descriptor declares it (tinygrad `dims=(8,16,16)`, `elements_per_thread=(8,4,4)`), wave 32, vector bytes 16, max LDS **49152** | `tc.get_cuda("sm_120")` -> `cuda_sm89`; declared `CUDARenderer.wave_size = 32` (`cuda.py:25`); `CUDARenderer.shared_max = 49152` (`cuda.py:12`); the instruction lowered and ran in the sm_120 microbench |
| `apple_m4_10c` | `simdgroup_matrix` 8x8x8 family, wave 32, max LDS **32768** | `tc.metal` (the descriptor `MetalRenderer` uses), declared subgroup width, `MetalRenderer.shared_max = 32768` (`cstyle.py:481`) |

Two review corrections are folded into that table:

- **`max_lds_bytes` is load-bearing and was missing from the non-AMD rows.** `runtime_specs.py:475` rejects
  `active_lds > capability.max_lds_bytes`. A row that omits it inherits AMD's 65536 default, so an
  AMD-shaped schedule above the real device budget would *admit* and then fail to compile. The fact is
  already declared per-target in tree as `Renderer.shared_max`, and the three targets genuinely disagree
  (65536 / 49152 / 32768) -- exactly the kind of divergence a frozen default hides.
- **The NV instruction-family label must come from the descriptor's own dims.** `tc.get_cuda("sm_120")`
  returns `cuda_sm89`, whose fp16-in/fp32-out entry is `dims=(8,16,16)`. Naming the row
  `mma_f32_m16n8k16_f16` (the PTX spelling) would not match what a derive-at-load check reads back. The
  measured `R = 255.4 TF` number is also the wrong citation for a *family* row -- throughput is not
  identity; the citation is that the instruction exists, lowered, and executed on sm_120.

AMD's values are byte-identical to today's, so AMD admission outcomes do not move -- but the row is now a
declaration that could be falsified, with the same shape as the other two, instead of the implicit default
that made every other target "outside frozen capability." This follows the §7a pattern: the difference rests
on a declared hardware fact, one line to flip, citing what it rests on.

**Derived vs declared fields** (resolves old review question 5). The two identity strings are not the same
kind of thing and should not be justified the same way:

- `instruction_family` is a hardware fact recoverable from the descriptor's dims and dtypes -- **derive it
  at load** from `tc.get_*`. A literal here is the frozen default wearing a new name.
- `fragment_layout` (`rdna3_wmma_f32_16x16x16_f16_lds2_static`) names *this repo's emitter contract*, not
  hardware. It is not derivable from any descriptor and stays a **declared literal**, cited to the emitter
  that implements it.

### 3.2 Compile evidence: AMD-only producer -> device-parameterized producer

Current: `prepare_current_prefill_compile` unconditionally disassembles AMD ELF and hardcodes `target =
"gfx1100"`, `target_id="amd_gfx1100"`, `wavefront_size=32`.

Proposed: the function gains a device-aware branch:

- device is AMD: produce the full manifest exactly as today -- disassembly, operand attribution, resource
  summary, packed-WMMA compile gate. Byte-identical output.
- device is any other declared target: produce the same minimal evidence the lane already builds
  (`{passed, binary_sha256}` plus source sha and target facts), and skip the AMD-only stages. The
  `final_isa_manifest`/`resource_summary` fields are then *present but AMD-enriched*, not load-bearing for
  non-AMD.

Two review findings shape that branch:

- **Branch on `device` only** -- not on "or the compiled binary is an AMD ELF", as an earlier draft of this
  section proposed. Sniffing the binary reintroduces exactly the implicit behaviour this scope removes, and
  it can route a non-AMD box into `disassemble_amdgpu`, which is the ROCm dependency the lane's workaround
  exists to avoid. This is also what makes deleting that workaround safe (old review question 4): the
  ROCm-requiring imports (`mmq_compile_evidence`, `renderer.amd.elf`) are already function-local, so a
  device branch taken before them never reaches a missing toolchain.
- **Record the absence, do not omit the key** (resolves old review question 2). Non-AMD evidence stays
  minimal -- no `wmma_families` fabricated from a SASS parse until a real CUDA disassembler exists -- but it
  emits `final_isa_manifest: null` with a named reason rather than dropping the field. That is the pattern
  the AMD path already uses for rows it cannot trace (`unknown_row_count` / `missing_evidence`
  discriminators). "Not applicable on this target" and "nobody looked" must not render identically.

The lane's `_child_run` workaround (`compile_current_prefill_program` directly + hand-built evidence) becomes
unnecessary; it can call the same producer and the workaround note is deleted. `run_canary` (Path A's entry
point) routes through the same producer, so AMD's own qualification evidence comes out of the same function
the lane uses.

### 3.3 Correctness proof: one probe lane for all targets

Current: Path A measures `max_abs_error` only (guarded outcome). Path B measures all three axes.

Proposed: `run_precontract_probe` is the canonical qualification entry for every target. `ProbeConfig.device`
is the only target-specific input, and it becomes a **required field with no default** (resolves old review
question 3): defaulting it to `"METAL"` is the historical accident, and re-defaulting it to `"AMD"` would
recreate the first-target-is-the-default defect one target later. Existing Metal callers pass it explicitly,
and the module is renamed off `metal_precontract_lane.py` in the same slice.

`verify_production_row` / `install_production_qualification_verifier` gain the lane as their qualification
path, so AMD's six promoted rows are re-verified through the same lane NV and Metal use and the three-axis
proof becomes the promotion evidence shape for every target.

**Sequencing (review finding).** Those two functions are already device-parameterized
(`packed_wmma_production_canary.py:35,51`, `device: str = "AMD"`) and today route through `run_canary` --
which is AMD's *only working promotion verifier*. Re-pointing them at the lane on this box replaces a
verified implementation with an unexercised one and leaves it that way for as long as the hardware gate
stays open. So the lane lands **beside** `run_canary`, opt-in, and `run_canary` remains the default until the
measured AMD lane run exists on AMD hardware; that run is what flips the default, and it is recorded as
campaign evidence. NV's qualification needs C1-C3 and C5 and does not depend on this flip at all.

### 3.4 What does NOT change

- The `PACKED_WMMA_ROUTES` production table, the model-forward attachment, `gate_combo`, and the runtime
  execution path in `tinygrad/llm/packed_wmma_prefill.py`.
- The four `GFX1100_*` capability rows and every AMD test that pins their identity.
- The AMD ISA manifest content and its authority.
- No new flags, no new env switches, no new subsystem, no changes to `dev`/`master`.

## 4. What the canonical path makes possible for NV

Once the path is canonical:

1. `_tensor_core_family_by_device` gains a `CUDA` row (`tc.get_cuda`), and the capability table gains the NV
   row (3.1). The NV mint's `CUDA:sm120:wave32` payload then clears the frozen AMD gate, which was the only
   *structural* blocker.

   **But admission succeeding is not the schedule being the same** (review correction to an earlier
   "passes by construction" claim). `_resolve_tensor_core` hands NV a `dims=(8,16,16)` descriptor where AMD
   gets `(16,16,16)`, and `derive_precontract_shape_factors` (`kernel_lds.py:404-409`) computes `sm`, `sn`
   and `ks` directly from `tc.dims`. The same payload therefore decomposes into a *different* per-wave
   subtile on NV. C2's verification records the resolved `PrecontractFactors` for the NV row; it does not
   just assert that admission failed to raise.
2. The lane runs with `device="CUDA"` on the 5090 against the AMD-shaped schedule first (mechanism test, same
   discipline as PG2/M1e): does the precontract path lower, execute, and produce correct output on NV
   hardware? Measure `max_abs_error`, coverage, determinism.
3. If it fails, read AMD's emitted kernel against NV's side by side (the M1f move), before any geometry
   search.
4. Only then: search NV geometry (BubbleBeam proposes legal dims from declared NV facts, FutureSight rejects,
   BoltBeam measures/ranks), qualify through the same lane, and let the minted candidate set become the
   promoted artifact.

This is Metal's playbook in the bring-up doc's own order, with the difference that AMD now runs the same lane
instead of a bespoke one.

## 5. Slices with verification

Each slice keeps the NV e2e bench green (honest ratchet: always one runnable artifact with a real number) and
adds unit tests at module boundaries.

| slice | change | verification |
| --- | --- | --- |
| C1 | capability table: declared rows keyed by `(backend, arch)` x schedule shape; `max_lds_bytes` per row from `Renderer.shared_max`; `instruction_family` derived at load, `fragment_layout` declared | existing `test_runtime_specs.py` pins AMD identities; new test: NV row resolves with `max_lds_bytes == 49152` and Metal with `32768`; all four AMD rows still resolve by schedule shape; AMD admission outcomes identical before/after |
| C2 | `_tensor_core_family_by_device` gains CUDA row | admission with `device="CUDA"` resolves `tc.get_cuda` and admits the NV mint payload's geometry; **resolved `PrecontractFactors` for the NV row recorded and asserted** (they differ from AMD's -- `dims=(8,16,16)` vs `(16,16,16)`); AMD/METAL rows unchanged |
| C3 | `prepare_current_prefill_compile` branches on `device` only; AMD branch byte-identical, non-AMD branch minimal with `final_isa_manifest: null` + named reason | AMD rendered-source equality (existing scratchpad); lane workaround deleted; `run_canary` uses the same producer; non-AMD branch exercised on a box with no ROCm toolchain |
| C4 | `ProbeConfig.device` becomes required; lane added **beside** `run_canary` as an opt-in qualification path for `verify_production_row`/`install_production_qualification_verifier`; `run_canary` stays the default | unit test: AMD row verifiable via the `run_precontract_probe(device="AMD")` path; default path unchanged and still `run_canary`; NV e2e decode ratchet unchanged (~156 tok/s, digits `50994`); flipping the default is gated on the measured AMD lane run (this box has no AMD GPU) |
| C5 | NV probe on the 5090 through the canonical lane | `max_abs_error`/coverage/determinism recorded for the AMD-shaped schedule on CUDA; result decides next step (fix lowering via M1f-style diff, or proceed to geometry search) |

Slices C1-C4 are code changes on `nvidia-bringup-20260731` with the repo's commit prefixes
(`[codegen]`, `[nn]`, `[test]`, `[docs]`). C5 is a scratchpad run, not a commit.

## 6. Review resolutions

The 2026-08-01 review closed all six original questions. Recorded here with their answers rather than
deleted, so the reasoning survives with the decision.

1. **Table granularity** -- `(backend, arch)` x schedule shape, not one row per target: four AMD rows exist
   and the target does not distinguish them. `wave_size` is derived from arch (`cstyle.py:598` already does
   this), not part of the key; the three-way `capability_target` equality check stays as the assertion.
   Folded into §3.1.
2. **Non-AMD compile evidence** -- stay minimal; no `wmma_families` fabricated from a SASS parse until a
   real CUDA disassembler exists. But emit `final_isa_manifest: null` with a named reason instead of
   dropping the field, matching the AMD path's existing explicit-unknown discriminators. Folded into §3.2.
3. **`ProbeConfig.device`** -- no default at all. `"AMD"` would recreate the defect this scope removes one
   target later. Callers pass it; the module is renamed off `metal_precontract_lane.py`. Folded into §3.3.
4. **Deleting the lane's `_child_run` workaround** -- safe, provided the producer branches on `device`
   alone and never on "the binary looks like an AMD ELF"; the ROCm-requiring imports are already
   function-local, so the non-AMD branch never reaches a missing toolchain. Folded into §3.2.
5. **Derive vs declare** -- split by kind: `instruction_family` is a hardware fact and is derived at load
   from `tc.get_*`; `fragment_layout` names this repo's emitter contract, is not derivable from any
   descriptor, and stays a declared literal. Folded into §3.1.
6. **The measured AMD lane run** -- happens where AMD hardware exists, and NV's qualification proceeds past
   it. The correction is that the gate must constrain the *code*, not only the evidence: C4 lands the lane
   opt-in beside `run_canary` rather than replacing it, so a box that cannot run AMD never ships an
   unexercised AMD promotion verifier. Folded into §2, §3.3 and C4.

Two findings from the same review were corrections rather than answers, and are folded in at their sites:
the missing `max_lds_bytes` on the non-AMD capability rows (§3.1 -- load-bearing at `runtime_specs.py:475`,
and the three targets genuinely disagree: 65536 / 49152 / 32768), and the overstated "NV passes admission by
construction" claim (§4 -- NV's `(8,16,16)` descriptor yields different `PrecontractFactors` than AMD's
`(16,16,16)`, so C2 must record them, not just assert no exception).

## 7. Open questions

None blocking C1. The remaining unknown is empirical, not design: what the NV probe in C5 measures.
