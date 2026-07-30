# Metal prefill loop-body decomposition — M theories scope

Date: 2026-07-30

Status: scoped, not implemented. Branch boundary: tinygrad `exp`. Does not authorize promotion to `dev`/`master`.

Deliberate structural copy of the AMD prefill method: `docs/prefill-needle-theories-20260724.md` established
where the time is *not*, `docs/prefill-R-theories-scope-20260724.md` decomposed the loop body by instruction
group and named theories against the dominant groups. **This scope reproduces that method for Metal. It does not
invent a new one.**

## 1. Why this and not a schedule search

An earlier draft of the Metal work proposed a BubbleBeam population search over upcast/unroll factors. That is
the right tool for *picking* a schedule once the bottleneck is known. It is not how the AMD result was reached.

The AMD sequence was:

1. build a **compile-only** probe that classifies every instruction in the real loop body;
2. publish the table, which established *"loads are only 15.1% of the loop body"* and closed the load-side;
3. observe *"R -- everything that is not a load and not a WMMA -- is 792 of 952 instructions (83%). It has never
   been attacked"*;
4. name theories against the dominant groups, each with claim / evidence / location / lever;
5. test them one at a time.

Step 4 is only possible because step 2 exists. **No Metal theory may be named before the Metal table exists.**

## 2. The AMD reference table (the shape of the deliverable)

From `prefill-R-theories-scope-20260724.md`, `amd_gfx1100_q16_grid_hd128_loop_attention`, per KV tile:

| group | instrs | share |
| --- | ---: | ---: |
| sync/sched (`s_waitcnt`, `s_delay_alu`, `s_clause`) | 213 | 22.4% |
| other VALU math | 175 | 18.4% |
| max/min | 135 | 14.2% |
| global loads | 144 | 15.1% |
| mask (`v_cndmask`, `v_cmp`) | 103 | 10.8% |
| cross-lane reduce (`ds_bpermute_b32`) | 96 | 10.1% |
| transcendental | 32 | 3.4% |
| other SALU/branch | 28 | 2.9% |
| LDS P repack | 10 | 1.1% |
| **WMMA (the only useful work)** | **16** | **1.7%** |
| total | 952 | |

Produced by `scratchpad/kv_tile_amortization_probe.py`: compiles the production emitter, disassembles the real
gfx1100 output via `disassemble_amdgpu`, locates the loop body, and classifies each instruction against an
`INST_CLASS` regex table. **No GPU execution.**

## 3. The Metal constraint, stated up front

`xcrun metal` is present but fails: *"cannot execute tool 'metal' due to missing Metal Toolchain; use:
`xcodebuild -downloadComponent MetalToolchain`"*. `metal-objdump` and `metal-nm` are not found. So the AMD
probe's foundation -- real ISA disassembly -- is **not available on this machine today**.

Two consequences:

- **MP0 classifies generated MSL source**, not ISA. MSL makes loads, `simdgroup_matrix` ops,
  `threadgroup_barrier`, `simd_shuffle_*`, selects and arithmetic explicit, so most AMD groups have a direct
  analogue. What it cannot see is **compiler-inserted scaffolding** -- the analogue of AMD's 213 `s_waitcnt`/
  `s_delay_alu` instructions, which was AMD's single largest group at 22.4%.
- Every MP0 conclusion must therefore be stated as **"of the MSL statements we can see"**, never as a claim about
  the executed instruction mix. A group that is invisible cannot be declared small.

**MP0b (optional, unblocks the rest):** installing the Metal Toolchain via
`xcodebuild -downloadComponent MetalToolchain` would allow AIR-level disassembly and close that gap. This scope
does not require it; it records it as the upgrade that would make the Metal table as trustworthy as the AMD one.

## 4. Architectural boundaries

### 4.1 One authority per concern

| Concern | Authority |
| --- | --- |
| kernel construction | the existing production prefill path (`tinygrad/llm/model.py`, prefill-v2) |
| rendering without execution | `tinygrad.codegen.to_program` + `MetalRenderer` (the TG1 technique) |
| classification table | one regex/AST table in the probe, mirroring `INST_CLASS` |
| measurement, when a theory is tested | `extra/llm_research/decode/kernel_log_diff.py` |

### 4.2 Required reuse

- Reuse `scratchpad/kv_tile_amortization_probe.py` as the structural template. Same shape: compile, locate the
  loop body, classify, emit a table. Do not invent a different output format -- the AMD table's columns are the
  contract.
- Reuse the render-without-execution technique already proven in
  `test/unit/test_warp_shfl_xor_renderer_lowering.py` (`Target.parse`, `to_program`, extract `Ops.SOURCE`).
- Reuse the production kernel. Do not decompose a synthetic proxy unless the production AST proves
  irreproducible, and say so explicitly if it does.

## 5. Evidence contract

1. **Compile-only.** MP0 runs no GPU workload. Another agent may hold the GPU lane.
2. **Reproducible.** The probe is committed and rerunnable, and its output is deterministic for a fixed commit.
3. **Honest coverage.** State what fraction of the emitted MSL the classifier accounted for. An
   `unclassified` bucket is mandatory and must be reported, not silently dropped.
4. **No theory without a table.** MP1 may not name a theory whose target group is not in MP0's output.
5. Any later theory test follows the AMD form: **claim / evidence that makes it more than a guess / exact
   location / lever**, and reports a measured result including negatives.

## 6. Work packages

### MP0 — Build the Metal loop-body decomposition probe

Prerequisite: none. Compile-only.

- Identify the real production prefill kernel. The depth-512 profile names it `r_16_256_8_16_4_3_16_4_2_8_4`
  (36.6% of prefill time); the second and third are `r_16_64_8_16_4_4_48_2_2_2_16_2` (30.5%) and
  `r_16_64_8_16_4_4_16_4_2_16_2` (11.7%). Decompose at least the first; all three if the AST is reachable.
- Render its MSL via `to_program` + `MetalRenderer` without executing.
- Classify every statement into groups mirroring the AMD table: global load, threadgroup load/store, barrier,
  simdgroup-matrix (the useful work), cross-lane shuffle, transcendental, select/compare (mask), other
  arithmetic, index/address math, **unclassified**.
- Emit the AMD-shaped table: group, count, share.

Deliverable: the probe under `scratchpad/`, plus the table recorded in a dated output doc.

Stop condition: if the production AST cannot be reached, fall back to the closest synthetic that reproduces the
kernel-name prefix (the technique PR1 used successfully), and label the table a proxy.

### MP1 — Name the Metal theories

Prerequisite: MP0's table.

- State the Metal analogue of *"R is 83% and has never been attacked"* -- whatever the table actually shows.
- Name at most three theories against the largest non-useful-work groups. Each gets: claim, the evidence that
  makes it more than a guess, exact file/pass location, and the lever.
- Explicitly record which groups are **closed** by the table (the analogue of AMD closing the load side at 15.1%).

**Do not carry AMD's theories over.** T4 (waitcnt) targets an AMD ISA pass that has no Metal equivalent and was
itself found DEAD (`e8a5fe4bf`: *"THEORY 4 waitcnt insertion: DEAD -- irreducible, and the theory targeted the
wrong compiler"*). T5/T6 target AMD instruction groups. Metal's dominant groups are unknown until MP0 runs.

### MP2+ — Test theories one at a time

Prerequisite: MP1. Each theory is its own packet, GPU-serialised, full section 5 evidence, negatives recorded.

## 7. Non-goals

- A BubbleBeam population search. That is `metal-prefill-schedule-search-scope-20260730.md`, and it is
  downstream of knowing where the time goes.
- Porting AMD theories or the AMD emitter.
- Installing the Metal Toolchain as a blocking prerequisite (see 3, MP0b).
- Promotion to `dev`/`master`.

## 8. Known limitations

- **MSL is above the compiler's scaffolding.** AMD's largest group (22.4% sync/sched) has no MSL-visible
  analogue. The Metal table will therefore understate total instruction count and cannot be compared
  instruction-for-instruction with the AMD table.
- **No AMD hardware**, so nothing here may claim AMD non-regression by execution.
- `test/unit` carries ~114 pre-existing failures. Diff failing-test-id **sets**, not counts.
