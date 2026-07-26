# 14B Decode Context-128 Recovery and Depth-Decay Scope

Date: 2026-07-26

Branch: `feature/14b-decode-ctx128-and-depth-decay`

Worktree: `/home/ubuntu/worktrees/14b-decode-ctx128-and-depth-decay`

Hardware: AMD RX 7900 XTX, gfx1100, wave32, `xccs == 1`

Model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`

Status: input

## 1. Objective

Solve two distinct 14B decode problems in order:

1. Restore real 14B prompt processing and decode at context 128, including ordinary short user prompts.
2. Remove the 14B-specific decode throughput decay from context 512 to context 4096 without regressing shallow-context throughput.

The first track is correctness and availability. The second track is performance. Do not accept a benchmark exclusion, a fail-loud guard, or a faster invalid route as completion of the first track.

### 1.1 Mandatory investigation order

This task is reference-first. Do not begin by tuning tinygrad or searching its schedule space.

Required order:

```text
llama.cpp source map
-> llama.cpp bounded ROCm trace at ctx128/512/4096
-> tinygrad equivalent trace and compile capture
-> first-principles path comparison
-> named mechanism and candidate contract
-> BoltBeam constrained search
-> tinygrad correctness and authority promotion
```

llama.cpp is a structural and measured reference, not code to vendor. BoltBeam must search reusable tinygrad candidates derived from demonstrated differences rather than blindly copying a llama kernel or exhaustively sweeping unrelated axes.

## 2. Current production state

### 2.1 Working published path

Current `master` runs 14B decode successfully at the published authority checkpoints:

| Context | Latest one-checkpoint process | Recorded baseline | Route |
|---:|---:|---:|---|
| 512 | 68.84 tok/s | 68.39 tok/s | flash |
| 1024 | 66.52 tok/s | not previously recorded | flash |
| 4096 | 59.52 tok/s | 59.41 tok/s | flash |

Commit `fd7cb8d1f` restored model-aware benchmark defaults:

- 8B defaults to `128,512,1024,4096`.
- 14B defaults to `512,1024,4096`.
- Explicit `--decode-ckpts 128` remains available and fails visibly.

That commit repaired the canonical benchmark entry point. It did not repair context-128 inference.

### 2.2 Context-128 failure

The exact minimal 14B context-128 authority fails deterministically during prompt setup:

```text
_warm_depth -> _prefill -> next(gen) -> Transformer.generate -> compile_linear
```

Timed decode never begins. AMD COMGR rejects emitted HIP resembling:

```c
make_float32(val39, ..., val40) = make_float32(alu60, ..., buf0[15]);
```

The left side is a constructed vector value, not an assignable location.

Confirmed discriminators:

- 14B context 128 fails at both `bece3963e` and `c437427db` with the same line and compiler error.
- 8B context 128 passes at both commits.
- 14B context 512 passes on current code.
- `FLASH_DECODE_THRESHOLD=0` does not fix context 128 because failure occurs before timed decode routing.
- The large `DEV=AMD:ISA` support-chain deletion in `c437427db` is not the cause.

### 2.3 Possible relationship to the unsafe fallback

The explicit 14B rollback:

```text
TINYGRAD_PREFILL_PACKED_WMMA=0
```

selects direct-packed prefill and has produced GPU MMU faults. Commit `0ec28fbe2` now rejects that explicit 14B configuration before model construction.

The packed-WMMA route also declines ungated or unknown shapes and falls through to direct-packed behavior. A short prompt may therefore reach the same fallback family implicitly even when the flag remains enabled. This is plausible but unproven.

The current fail-loud guard checks the environment flag. It does not prove that implicit per-shape fallback is impossible.

Do not call these the same defect until route attribution shows context 128 actually selects the direct-packed fallback. The symptoms differ: explicit rollback has produced a GPU fault, while context 128 currently fails during compilation.

## 3. Depth-decay defect

The 14B model is G=5 (`Hq=40`, `Hkv=8`, 160 threads/workgroup). The 8B model is G=4 (`Hq=32`, `Hkv=8`, 128 threads/workgroup).

Current full-model behavior:

| Configuration | ctx512 | ctx4096 | Achieved HBM trend |
|---|---:|---:|---:|
| 8B tinygrad, G=4 | 113.86 tok/s | 102.57 tok/s | 581 -> 578 GB/s |
| 14B tinygrad, G=5 | 68.39 tok/s | 59.41 tok/s | 621 -> 575 GB/s |
| 14B llama.cpp | 66.58 tok/s | 62.87 tok/s | 527 -> 545 GB/s |

Tinygrad leads llama.cpp at 14B context 512 and loses by approximately 5.5% at context 4096.

Target:

- Required: match or beat same-session llama.cpp at context 4096 without regressing context 512 beyond the measured noise floor.
- Stretch: hold approximately 620 GB/s at depth, corresponding to approximately 64 tok/s at context 4096.

## 4. Evidence already established

Do not rerun these theories without new contradictory evidence.

### 4.1 Excluded depth-decay theories

- K/V reread per query-head group is excluded. Staging is workgroup-cooperative into LDS and read once regardless of G.
- A partial fifth wave is excluded. `WARPS=G`, so G=5 launches five full wave32 waves.
- Split size mistuning is excluded by the existing sweep. `S=48` is already optimal for G=5 at shallow and deep contexts.
- Route or dispatch-count growth is excluded by retained artifacts.
- General VRAM pressure is not a mechanism. Higher-volume variants have run without the corresponding fault behavior.
- `K_ONLY` is correct in the tested G5 route but did not improve full-model throughput.
- Reducing G5 to four compute waves is not a safe parameter change. Query-head ownership, cooperative staging, and barriers are coupled to `WARPS=G`; serializing the fifth head or adding a second pass duplicates work or traffic.

### 4.2 Existing G5 authority

The focused authority reproduced a genuine deep G5 cliff:

| Configuration | Median | Normalized cost |
|---|---:|---:|
| G4 Hq32 Tc512 | 0.083436 ms | 217.28 ns/tile |
| G5 Hq40 Tc512 | 0.085240 ms | 222.00 ns/tile |
| G4 Hq32 Tc4096 | 0.127941 ms | 55.53 ns/tile |
| G5 Hq40 Tc4096 | 0.169178 ms | 73.43 ns/tile |

Ratios:

- Shallow G5/G4: `1.0217x`
- Deep G5/G4: `1.3223x`

The remaining lead is compiler allocation, lifetime, occupancy, or a G5-specific long-loop interaction. It is not yet proven to be register spilling.

### 4.3 Required llama-first reference investigation

The phases below precede Track A Phase A0 and Track B Phase B0. They are shared evidence for both tracks.

#### Phase L0: establish tracing authority

Record the exact reference environment before collecting data:

- llama.cpp repository path, Git commit, dirty state, build command, build backend, and binary SHA256.
- Proof that the installed `llama-bench` uses the ROCm/HIP backend on gfx1100 rather than Vulkan, CPU, or another device path.
- tinygrad commit and worktree identity.
- GGUF path, byte size, modification time, and identity digest.
- ROCm version, kernel driver version, GPU identity, active compute partition, power state, and available tracing tools.
- Positive control showing one known llama kernel and one known tinygrad kernel in a bounded trace.

Check availability explicitly for the installed ROCm generation:

```text
rocprofv3
rocprof
rocprof-compute or omniperf
rocm-smi
rocminfo
ROCtx marker support
```

Do not assume a tool exists because ROCm is installed. Do not install or replace system ROCm packages inside this task without operator approval.

Timing and counter collection are separate regimes:

- Use ordinary authority runs for publishable wall time.
- Use bounded trace runs for kernel sequence and attribution.
- Use focused counter runs only after a kernel family is named.
- Never publish profiler-instrumented wall time as ordinary throughput.

Deliverable: `bench/14b-decode-ctx128-depth-decay-20260726/l0-environment.json`.

#### Phase L1: map llama.cpp from CLI to hardware

Trace the source path before tracing performance. Produce a call and ownership map with exact file/function references for:

- `llama-bench` argument parsing and prompt/decode benchmark loops.
- Model load, tokenizer/BOS handling, prompt construction, and fixed-depth KV population.
- `llama_decode` or current equivalent graph construction.
- GGML graph nodes for Q4_K linear layers, normalization, RoPE, attention, softmax, KV write/read, and output sampling.
- ROCm backend graph scheduling, buffer placement, stream/queue ownership, and kernel dispatch.
- Prompt-path GEMM/MMQ selection by M, N, K, quant type, batch size, and ubatch size.
- Decode-path GEMV/MMQ selection for Q4_K and Q6_K roles.
- Flash-attention admission and its context/head geometry.
- GQA ownership for `Hq=40`, `Hkv=8`, and G=5.
- KV-cache layout, datatype, write ownership, and read/coalescing pattern.
- Any split-KV, reduction, combine, or multi-pass attention lifecycle.

For each decision, record whether it is:

- Algorithmic.
- Shape-policy driven.
- Backend-policy driven.
- Compile-time generated.
- Runtime selected.
- Hardware-specific to gfx1100.

Do not infer the path from kernel names alone. Connect source selection to a positively observed dispatch.

Deliverable: `docs/14b-llama-rocm-path-map-20260726.md`.

#### Phase L2: collect bounded llama.cpp ROCm traces

Use the same GGUF and explicit benchmark settings. Run context/prompt lengths in separate processes:

```text
128
512
4096
```

Make batch, ubatch, flash-attention, generation length, repetition count, and GPU-offload settings explicit. Retain the exact generated command. Start with one warmup and one measured repetition for traces; collect ordinary five-repetition timing separately.

The trace must distinguish:

- Model load and first-use compilation/setup.
- Prompt processing.
- First-token transition.
- Steady decode token.
- Synchronization and host-visible token extraction.

Use ROCtx markers if the installed llama binary or a bounded wrapper can provide them without changing kernel selection. Otherwise derive phase boundaries from positively identified dispatch sequences and document the inference.

For every dispatched kernel retain or derive:

- Kernel name and semantic family.
- Dispatch order and count.
- Grid and workgroup dimensions.
- Wave size and waves/workgroup.
- Kernel duration distribution.
- Code-object identity.
- VGPR, SGPR, LDS, private/scratch, and spill metadata.
- Global-read/write byte model.
- KV bytes versus weight bytes.
- WMMA/MFMA, VALU, VMEM, LDS, barrier, and wait instruction counts where disassembly permits.

After the dispatch map is stable, run focused counters for only the kernel families that explain material wall time or depth slope. Candidate counters include:

- Wave occupancy and active waves.
- VALU and matrix instruction utilization.
- VMEM/LDS instruction and busy cycles.
- L0/L1/L2 hit behavior.
- HBM read/write traffic.
- Stall and wait reasons supported by the installed gfx1100 tooling.
- Scratch/spill traffic.

Counter availability and semantics must be recorded from the installed ROCm tool. Do not invent gfx1100 counter meanings or compare counters with different normalization bases.

Required llama artifacts:

- Ordinary timing JSON for ctx128/512/4096.
- One bounded dispatch trace per context.
- One kernel/resource ledger.
- Focused counter artifacts only for named owners.
- A trace-to-source attribution table.

#### Phase L3: collect the equivalent tinygrad evidence

Use identical model geometry and, where possible, identical token IDs and prompt lengths. If tokenizer behavior differs, create and retain a token-ID fixture consumed by both paths or document the exact unavoidable difference.

For tinygrad context 512 and 4096 collect the same categories as llama:

- Prompt setup and steady decode separated.
- Dispatch sequence and kernel-family ownership.
- Grid/workgroup/wave geometry.
- Code-object and resource metadata.
- Modeled weight and KV bytes.
- Focused counters using the same definitions and normalization as llama.

For tinygrad context 128:

- Retain the compile failure and semantic UOp/HIP slice.
- Record every successful dispatch before the compiler failure.
- Do not fabricate hardware counters for a kernel that never compiled or ran.
- Use llama context 128 to establish the working reference lifecycle and tinygrad 8B context 128 as the same-runtime control.

Run each context in a separate process. A reset or fault invalidates later traces until a fresh recovery boundary is established.

#### Phase L4: first-principles llama versus tinygrad comparison

Build one normalized comparison table for ctx128, ctx512, and ctx4096. It must cover:

| Dimension | Required comparison |
|---|---|
| Algorithm | Attention, softmax, quant matmul/GEMV, KV lifecycle, sampling |
| Shape policy | Prompt M buckets, batch/ubatch, G=5 ownership, split size |
| Dispatch | Kernel count, fusion boundaries, launch order, synchronization |
| Traffic | Weight bytes, KV bytes, transient bytes, duplicate reads/writes |
| Compute | Scalar/vector/matrix instruction work and useful operations |
| Resources | Threads, waves, VGPR, SGPR, LDS, scratch, occupancy limits |
| Locality | Coalescing, LDS staging, cache hit behavior, reuse ownership |
| Depth scaling | Which terms grow with context and their measured slopes |
| Correctness | Bounds, masking, causal semantics, token parity |

Use an explicit latency model:

```text
t_token(ctx) = t_weight_path + t_kv_path(ctx) + t_combine(ctx)
             + t_launch + t_sync + t_host + t_other
```

For each term state:

- How bytes or operations are calculated from model geometry.
- Which trace kernels contribute.
- What is measured versus inferred.
- Whether llama and tinygrad perform equivalent useful work.
- Whether the term can explain `t(4096) - t(512)`.

The comparison must answer two separate questions:

1. Why does llama complete 14B context-128 prompt setup while tinygrad emits an invalid destination STORE?
2. Why does llama's achieved bandwidth stay flat or improve with depth while tinygrad falls from approximately 621 to 575 GB/s?

Do not treat a different kernel topology as automatically superior. Quantify its work, traffic, resource use, and measured contribution.

Deliverables:

- `docs/14b-llama-vs-tinygrad-first-principles-20260726.md`
- `bench/14b-decode-ctx128-depth-decay-20260726/path-comparison.json`

#### Phase L5: construct the BoltBeam search contract

BoltBeam begins only after L4 names a mechanism and a bounded candidate family.

The BoltBeam input must include:

- Immutable model, device, route, and workload facts.
- Reference llama structural facts separated from tinygrad implementation facts.
- Candidate axes justified by a measured or source-proven difference.
- Correctness, bounds, route-identity, resource, and performance constraints.
- A compile-only rejection stage before GPU launch.
- A cheap kernel-level timing stage before whole-model authority.
- A production promotion gate and a dead-probe cleanup list.

BoltBeam must not:

- Search raw HIP or vendor llama source into tinygrad.
- Treat llama's implementation as a correctness oracle without output parity.
- Search unrelated schedule axes merely because they are available.
- Optimize profiler-instrumented wall time.
- Promote a local win that cannot explain the full-model deficit.
- Bypass the context-128 defect by changing benchmark coverage.

The search output must identify reusable tinygrad compiler or route primitives, not a one-shape opaque kernel unless the scope explicitly justifies that production tradeoff.

Deliverable: `docs/task_workflow/in_progress/14b-decode-boltbeam-search-contract-20260726.md`.

## 5. Operational rules

1. Work only in `/home/ubuntu/worktrees/14b-decode-ctx128-and-depth-decay`.
2. Do not checkout this feature branch in `/home/ubuntu/tinygrad-arkey`.
3. Acquire `/tmp/gpu-bench.lock` for every GPU command, including tests and probes.
4. Record power state before and after every measured process. Expected state is `auto` unless the authority explicitly pins clocks.
5. Do not use `timeout`, `pkill`, or hard termination on a live AMD kernel.
6. Run one checkpoint per process for fault containment and artifact identity.
7. Use explicit artifact and log paths under one task-owned directory.
8. Treat empty route, trace, compiler, or journal output as a broken probe until a positive control proves otherwise.
9. Delete task-specific probes after their evidence is promoted.
10. Do not publish timing collected after a GPU reset as clean authority without a fresh process and recorded recovery boundary.

## 6. Track A: restore context 128

Track A must complete before Track B can promote. Track B may collect compile-only evidence in parallel, but no performance candidate should be promoted while ordinary short prompts remain broken.

### Phase A0: Re-establish the exact failure

Use a fresh process and retained output:

```bash
flock /tmp/gpu-bench.lock \
  env -u TINYGRAD_PREFILL_PACKED_WMMA PYTHONPATH=. \
  python3 extra/qk/decode/decode_runtime_overhead.py \
    --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
    --ckpts 128 --max-context 4608 --nmeas 1 --reps 1 \
    --warmup-decode 2 --chunk-size 32 \
    --out bench/14b-decode-ctx128-depth-decay-20260726/a0-ctx128.json
```

Required positive controls:

- Exact commit and clean worktree identity.
- Model path, size, and identity digest.
- Failure stage before timed decode.
- Failing HIP function name and full source retained once.
- Failing UOp program retained once with op, dtype, address space, lane count, and source indices.
- A same-process-policy 8B context-128 control.
- A separate-process 14B context-512 control.

Do not collect repeated failures after deterministic reproduction.

### Phase A1: Map the real short-prompt boundary

Test prompt/context lengths in separate processes:

```text
2, 8, 16, 32, 64, 96, 127, 128, 129, 192, 256, 384, 511, 512
```

Start with compile/setup smoke only. Do not run full decode repetitions for failing shapes.

For each length record:

- Prompt length and chunk size.
- Physical token extent seen by each prefill call.
- Start positions and final prompt chunk shape.
- Selected prefill linear route.
- Packed-WMMA candidate accept/decline reason.
- Direct-packed fallback selection, if any.
- Prompt attention route.
- First token/decode route if setup succeeds.
- Compile result, token result, GPU fault result, and artifact path.

Include real user prompts such as `"hi"` and a short sentence. Synthetic repeated-token prompts alone are insufficient.

### Phase A2: Decide whether this is the fallback family

Instrument the actual route-selection boundary, not only the environment gate.

Required discriminator:

- If context 128 reaches `route_direct_packed_prefill`, identify the exact packed candidate that declined and its shape guard.
- If context 128 never reaches direct-packed prefill, remove the fallback theory from the task and identify the semantic owner of the failing `r_toks_*` kernel.

The probe must positively show at least one known packed-WMMA selection at context 512 and one known direct-packed selection from an allowed non-14B control. An empty selection log is invalid.

If implicit unsafe fallback is confirmed, add an immediate route-bound fail-loud guard so it cannot silently launch. That guard is an interim safety measure, not Track A completion.

### Phase A3: Localize the invalid vector store

Compare three generated programs:

- 14B failing context 128.
- 14B passing context 512.
- 8B passing context 128.

Trace the invalid STORE backward through:

- Renderer `Ops.STORE` input.
- INDEX/STACK or vector construction used as the destination.
- Register bufferization and lane ownership.
- Scheduler or late-devectorizer transformation that created the destination.
- Original semantic state update or cache write.

Answer before editing:

- Is the destination supposed to update scalar registers, a register vector, local memory, or global memory?
- Is the STORE dead, incorrectly vectorized, or missing a materialized register destination?
- Would naming the vector merely compile while losing the intended lane updates?
- Is the defect generic or specific to the 14B shape and route?

Do not apply `EXPAND_SSA=1` or name the constructed vector as a fix unless semantic updates and output parity prove it correct. Making invalid C compile is not sufficient.

### Phase A4: Build valid candidates

Candidate priority:

1. Correct the generic compiler transformation if it creates an illegal or semantically impossible STORE.
2. Add a valid small-shape packed candidate if the compiler is correct and route coverage is missing.
3. Use a bucketed or padded proven path only if masking, KV positions, causal semantics, and first-token parity are exact.
4. Keep fail-loud behavior only as a temporary safety boundary while no valid candidate exists.

BoltBeam may search prompt chunk size, physical M bucket, packed candidate geometry, and schedule choices only after candidates pass compile and correctness gates.

Search must not optimize around the defect by excluding ordinary prompts.

### Phase A5: Context-128 acceptance

Required correctness:

- `"hi"`, one short sentence, and exact lengths 32, 64, 128, 256, and 512 complete prompt setup and at least eight decode tokens.
- Output is finite and token-identical to the accepted reference for deterministic temperature zero.
- KV/cache bounds and route identity are retained.
- 8B behavior is unchanged.

Required reliability:

- Three clean, separate-process context-128 runs.
- No compiler error, MMU fault, reset, `memory_lost`, or silent fallback.

Required performance:

- Report TTFT/prompt setup and steady decode separately.
- Context 128 must not use a path whose latency is accidentally dominated by a 512-token padded workload without labeling and justification.
- Do not set a promotion floor until a correct candidate exists and same-session noise is measured.

Track A completion requires successful short-prompt inference. A benchmark-default exclusion or clear error message is not completion.

## 7. Track B: remove depth decay

### Phase B0: Freeze clean baselines

Run one checkpoint per process at `512,1024,2048,4096` using the canonical authority, with five repetitions and the existing measurement count unless a retained current artifact already satisfies every identity requirement.

Collect same-session llama.cpp at context 512 and 4096 if the retained comparator is not from the same power and thermal session.

Record:

- W and D timing.
- Route sequence and programs per token.
- Model, commit, power, temperature if available, and lock ownership.
- Weight and KV bytes used for achieved-bandwidth conversion.
- Noise floor from repeated same-configuration measurements.

Do not combine pre-Track-A and post-Track-A baselines if the Track A fix changes any 512+ code object or route.

### Phase B1: Identify the slope owner

Use the cheapest discriminator first.

Required evidence:

- Per-kernel or per-family wall contribution at context 512 and 4096.
- Exact G5 flash tile and combine code-object identity.
- Static VGPR, SGPR, LDS, scratch/spill, waves/workgroup, and occupancy metadata.
- Loop-trip and instruction-count scaling.
- Allocation and launch geometry held constant or explicitly accounted for.

Answer:

- Does the slope live in the flash tile, combine, weight GEMMs, runtime synchronization, or another family?
- Is G5 losing issue occupancy, spilling, extending live ranges, or serializing a fifth-wave resource?
- Does the responsible kernel's share grow enough to explain the full-model loss?

Do not start with a broad PMC campaign. Use PMC only after timing and static metadata name a kernel and a mechanism that counters can distinguish.

### Phase B2: Compiler lifetime and allocation audit

If the G5 flash tile remains the owner, compare G4 and G5 at shallow and deep contexts:

- Register allocation and live intervals.
- Loop-carried state and address temporaries.
- LDS staging ownership versus query-head ownership.
- Barrier and wait placement.
- Scratch use and hidden spills in code-object metadata and disassembly.
- Wave occupancy limits from VGPR, SGPR, LDS, and 160-thread workgroups.

The existing four-loader-wave staging idea is allowed only as compile-only evidence first. Its theoretical staging slack is approximately 1.5%, so reject it unless it materially changes resources without changing traffic, output ownership, grid, or barriers.

### Phase B3: Candidate construction and BoltBeam search

Search only axes supported by the B1/B2 mechanism.

Potential candidate families:

- G5-specific lifetime shortening or address recomputation tradeoff.
- G5-specific register allocation or schedule ordering.
- Hq-aware reuse of a proven G4 compiler primitive without changing G5 query ownership.
- A revised G5 route whose staging and compute ownership remain explicit and verifier-clean.

Forbidden shortcuts:

- Serializing the fifth query head without accounting for duplicated work.
- Duplicating K/V traffic in a second workgroup pass.
- Reducing workgroup size while leaving ownership/barrier assumptions unchanged.
- Reopening split-size, K_ONLY, or K/V reread theories without new contradictory evidence.
- Promoting a kernel-local win whose full-model Amdahl contribution cannot close the measured gap.

Every BoltBeam candidate must pass compile, static-resource, numerical, route-identity, and bounded timing gates before whole-model authority.

### Phase B4: Depth-decay acceptance

Required correctness:

- Deterministic token parity at contexts 512, 1024, 2048, and 4096.
- Route identity and admitted G5 geometry unchanged or explicitly re-promoted.
- No 8B code-object or performance regression unless the change intentionally touches shared code and passes full 8B gates.

Required performance:

- Context 4096 matches or beats same-session llama.cpp.
- Context 512 does not regress beyond the measured noise floor.
- Throughput or achieved bandwidth no longer shows the current 621 -> 575 GB/s decay.
- Stretch result is approximately 64 tok/s at context 4096.
- Three same-session interleaved baseline/candidate pairs support promotion.

Required safety:

- No MMU fault, reset, `memory_lost`, or failed quiesce during authority.
- Power profile is restored after every run.

## 8. Interaction between tracks

- Track A and Track B must retain separate hypotheses, artifacts, and verdicts.
- A Track A compiler fix touching shared lowering requires 8B and 14B decode code-object/resource comparison before B baselines are reused.
- A Track B candidate must not bypass or hide a Track A failure.
- If Track A changes the 512+ route or code object, discard old B performance baselines and recollect them.
- If Track B reveals a generic compiler defect that also causes Track A, consolidate implementation only after both causal chains are demonstrated.

## 9. Probe and artifact policy

- Store retained evidence under `bench/14b-decode-ctx128-depth-decay-20260726/`.
- Use one summary ledger for accepted numeric evidence.
- Delete temporary monkeypatch wrappers, source-capture scripts, and `/tmp` probes after promotion.
- Keep a diagnostic only if it has a second plausible use, a positive control, an input contract, and an owner.
- Record deleted probes and recovery commits in the final findings.
- Do not commit enormous raw compiler dumps when a hash, failing region, semantic UOp slice, and reproduction command are sufficient.

## 10. Required deliverables

- `docs/14b-llama-rocm-path-map-20260726.md`
- `docs/14b-llama-vs-tinygrad-first-principles-20260726.md`
- A machine-readable llama/tinygrad dispatch, resource, traffic, and depth-scaling comparison.
- A bounded BoltBeam search contract derived from the first-principles comparison.
- `docs/14b-decode-ctx128-recovery-findings-20260726.md`
- `docs/14b-decode-depth-decay-findings-20260726.md`
- A machine-readable prompt-length/route/failure matrix.
- A retained failing semantic UOp slice and HIP excerpt for Track A.
- Track A token-parity and reliability artifacts.
- Track B baseline/candidate authority artifacts.
- Same-session llama comparison for the promoted depth candidate.
- Updated decode current-state numbers.
- Updated route/provenance records for any new or changed candidate.
- Probe cleanup ledger.

## 11. Stop conditions

Stop and request review if:

- The GPU lock is owned by another process.
- Source files change during measurement.
- A failure follows a GPU reset and cannot be separated into a fresh process/boot boundary.
- Route instrumentation lacks a positive control.
- Context 128 selects a different semantic model path than the task assumes.
- A proposed renderer fix only makes C compile without preserving the intended state update.
- A BoltBeam search begins before correctness and route identity pass.
- Track B cannot identify a kernel family whose contribution can explain the full-model slope.
- Promotion would require hiding context 128 or weakening a correctness gate.

## 12. Completion criteria

The task is complete only when both are true:

### Reference and first-principles authority

- llama.cpp's selected ROCm path is mapped from source decision to observed dispatch at context 128, 512, and 4096.
- llama and tinygrad have comparable bounded traces with phase, route, kernel, resource, and traffic attribution.
- The first-principles comparison explains which mechanisms can and cannot account for the context-128 failure and depth slope.
- BoltBeam searches only the bounded candidate families justified by that comparison.

### Context-128 recovery

- Real short prompts and exact context 128 run successfully on 14B.
- The compiler defect or missing candidate is causally identified and repaired.
- Any relationship to implicit direct-packed fallback is confirmed or refuted with route evidence.
- Token parity, route identity, reliability, and 8B non-regression pass.

### Depth-decay recovery

- The responsible G5 mechanism is demonstrated.
- Context-4096 14B decode matches or beats same-session llama.cpp.
- Context-512 throughput remains within noise of baseline.
- The performance claim survives three interleaved authority pairs.
- Dead probes and superseded candidates are removed.

If Track A completes but no honest Track B candidate closes the gap, publish Track A independently and close Track B with a precise mechanism, measured ceiling, and bounded next task. Do not withhold a correctness repair because the performance target remains open.

## 13. Branch retirement

This feature branch is temporary. After accepted Track A and Track B work has been promoted to the production pipeline:

1. Confirm every useful code change, regression test, authority artifact, findings document, and cleanup ledger exists on the promoted branch.
2. Confirm no uncommitted or unpushed work remains in `/home/ubuntu/worktrees/14b-decode-ctx128-and-depth-decay`.
3. Remove the feature worktree.
4. Delete the local `feature/14b-decode-ctx128-and-depth-decay` branch.
5. Delete the remote feature branch after its promoted commit is reachable from the retained production or integration branch.
6. Record branch retirement in the final findings.

Do not retain the branch as a research archive. Rejected implementations remain recoverable from Git history, while durable conclusions belong in the findings documents and ledgers.

## 14. Luna low-agent execution contract

This section is normative for low-reasoning Luna agents. An agent receives one task card, not the entire open problem. The orchestrator owns dependency ordering, merges, GPU serialization, and final claims.

### 14.1 Agent isolation

- Every code-writing agent gets a child branch from `feature/14b-decode-ctx128-and-depth-decay` and a dedicated worktree under `/home/ubuntu/worktrees/`.
- Read-only source-map agents may share the same commit but must not edit the same output file concurrently.
- No agent checks out its branch in `/home/ubuntu/tinygrad-arkey` or another actor's worktree.
- Only the orchestrator merges child commits into the feature branch.
- No agent pushes to `master` or deletes the feature branch.
- A child branch is removed after its useful commit is merged and its artifacts are promoted.

Suggested child naming:

```text
agent/14b-luna-000-safety
agent/14b-luna-010-llama-cli-map
agent/14b-luna-021-llama-ctx128-trace
agent/14b-luna-031-tinygrad-ctx128-capture
agent/14b-luna-053-ctx128-candidate
agent/14b-luna-063-depth-candidate
```

### 14.2 Agent task discipline

Each agent must:

1. Read this scope and only the files named by its task card.
2. Record its starting commit and worktree before doing work.
3. Refuse to use results from another commit without an explicit artifact identity match.
4. Use the task-owned artifact directory.
5. Produce the required positive control before interpreting empty or negative output.
6. Return one of the allowed verdicts.
7. Delete temporary probes or list them explicitly for cleanup.
8. Commit only durable source, tests, manifests, or findings assigned by the task card.
9. Report every file changed and every command that used the GPU.
10. Stop at the task boundary instead of starting the next card independently.

Allowed verdicts:

```text
PASS
SUPPORTED
REFUTED
INCONCLUSIVE
TOOL_FAILURE
CODE_FAILURE
GPU_FAULT
NOT_RUN
```

`INCONCLUSIVE`, `TOOL_FAILURE`, `GPU_FAULT`, and `NOT_RUN` never satisfy a downstream dependency. A zero exit code with a missing positive control is `TOOL_FAILURE`, not `PASS`.

### 14.3 GPU ownership

- Only one GPU task card may be `in_progress` at a time.
- The agent must acquire `/tmp/gpu-bench.lock` around the complete child process lifecycle.
- The agent must record the lock file, PID, boot ID, power before/after, process start/end, and exit code.
- The agent must not use `timeout`, `pkill`, `kill -9`, or Ctrl-C on an executing AMD kernel.
- After `memory_lost=1`, the agent stops. The orchestrator establishes a fresh recovery boundary before another GPU card begins.
- A source-map, parser, or comparison task may run concurrently only when it does not touch the GPU or mutate the measured worktree.

### 14.4 Commit and handoff format

Every completed card returns:

```text
task_id:
verdict:
start_commit:
end_commit:
child_branch:
worktree:
files_changed:
artifacts:
positive_controls:
gpu_commands:
temporary_files_deleted:
remaining_cleanup:
finding:
next_unblocked_task_ids:
```

The finding must distinguish measured facts, source-proven facts, and inference.

## 15. Machine-readable task state

The orchestrator creates and owns:

```text
docs/task_workflow/in_progress/14b-decode-luna-state-20260726.json
```

Required top-level fields:

```json
{
  "schema": "14b-decode-luna-state.v1",
  "feature_branch": "feature/14b-decode-ctx128-and-depth-decay",
  "base_commit": "<commit>",
  "updated_unix_ns": 0,
  "gpu_owner_task": null,
  "tasks": []
}
```

Each task row contains:

```text
id
title
status: pending|in_progress|completed|failed
dependencies
agent
child_branch
start_commit
end_commit
verdict
artifact_paths
finding_path
cleanup_complete
```

At most one task with `uses_gpu=true` may be `in_progress`. At most one task that edits a given production file may be `in_progress`.

The state file tracks execution. It does not replace evidence artifacts or findings documents.

## 16. Run artifact schema

Every GPU or compiler run writes a sidecar manifest. Required fields:

```text
schema
task_id
created_unix_ns
branch
commit
worktree
git_dirty_paths
command_argv
environment_overrides
model_path
model_size_bytes
model_mtime_ns
model_identity_sha256
backend
device
architecture
boot_id
lock_path
lock_owner_pid
power_before
power_after
start_time
end_time
exit_code
classification
positive_control
stdout_path
stderr_path
primary_artifact_path
kernel_or_route_identity
notes
```

Do not record secrets or the complete ambient environment. Record only overrides that can change route, compiler, GPU, or benchmark behavior.

Raw ROCm traces remain under `bench/14b-decode-ctx128-depth-decay-20260726/` unless they are small enough and stable enough to promote. Findings commit trace hashes, summaries, commands, and source attribution rather than uncontrolled multi-gigabyte output.

## 17. Dependency waves

| Wave | Task IDs | Parallelism | Gate to advance |
|---|---|---|---|
| 0 | LUNA-000 through LUNA-003 | CPU/static parallel | Environment, binaries, schema, and lock positive controls complete |
| 1 | LUNA-010 through LUNA-014 | CPU/static parallel | Source maps connect selection decisions to expected semantic families |
| 2 | LUNA-020 | CPU-only | Token fixture and exact commands accepted |
| 3 | LUNA-021 through LUNA-023 | GPU serial | Bounded llama traces for 128/512/4096 complete |
| 4 | LUNA-024 through LUNA-026 | CPU analysis; focused GPU counters serial | llama dispatch/resource/counter ledger complete |
| 5 | LUNA-030 through LUNA-034 | GPU/compiler serial | tinygrad control, failure, traces, and fallback attribution complete |
| 6 | LUNA-040 through LUNA-044 | CPU analysis parallel, then synthesis | First-principles comparison accepted |
| 7 | LUNA-050 | CPU-only | BoltBeam contract accepted |
| 8A | LUNA-051 through LUNA-055 | Sequential by dependency | Context-128 correctness and reliability complete |
| 8B | LUNA-060 through LUNA-064 | Sequential by dependency | Depth-decay mechanism and authority complete |
| 9 | LUNA-070 through LUNA-074 | Mostly CPU; final authority already retained | Regression, docs, cleanup, promotion, retirement complete |

Wave 8B may begin static analysis after Wave 6, but it may not promote before Track A passes. If Track A changes any 512+ route or code object, rerun LUNA-032, LUNA-033, LUNA-060, and all downstream B tasks.

## 18. Detailed Luna task cards

### LUNA-000: safety and ownership census

Uses GPU: no.

Dependencies: none.

Actions:

- Record feature commit, remote status, branch, worktree list, dirty files, stashes, active agents, GPU processes, lock state, boot ID, and power state.
- Identify every active worktree owner. Do not modify them.
- Confirm task artifact and in-progress directories can be created without overwriting retained evidence.

Outputs:

- `docs/task_workflow/in_progress/14b-decode-luna-state-20260726.json`
- `bench/14b-decode-ctx128-depth-decay-20260726/luna-000-safety.json`

Acceptance:

- Positive control records the feature branch and exact worktree.
- No ownership ambiguity remains for paths used by this task.

Stop:

- Existing branch/worktree collision, unclear unpushed ownership, or unlocked active GPU process.

### LUNA-001: ROCm tool capability census

Uses GPU: no dispatch; device queries only.

Dependencies: LUNA-000.

Actions:

- Resolve exact paths and versions for `rocprofv3`, `rocprof`, `rocprof-compute` or `omniperf`, `rocm-smi`, `rocminfo`, and ROCtx support.
- Capture help output relevant to kernel tracing, HIP API tracing, counters, output formats, kernel filters, and process scope.
- Record which tools are absent.
- Do not install anything.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/luna-001-rocm-tools.json`
- `docs/task_workflow/in_progress/14b-rocm-command-recipes-20260726.md`

Acceptance:

- At least one bounded kernel-dispatch trace method is positively identified.
- Counter collection is either positively available with gfx1100 semantics or explicitly unavailable.

Stop:

- Installed tooling cannot trace the actual llama backend. Return `TOOL_FAILURE` with exact missing capability.

### LUNA-002: llama and model identity

Uses GPU: no.

Dependencies: LUNA-000.

Actions:

- Locate the exact llama.cpp source tree that built the installed binary.
- Record Git commit, dirty state, binary path, SHA256, linked ROCm/HIP libraries, build flags, and backend list.
- Record GGUF identity and relevant Qwen3-14B metadata.
- Prove the benchmark selects AMD ROCm/HIP for `-ngl 99`.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/luna-002-llama-identity.json`

Acceptance:

- Source commit maps to binary build or the mismatch is explicitly documented.
- Backend identity has a positive runtime or linkage control.

Stop:

- Binary provenance cannot be connected to a source tree. Do not source-map a different revision silently.

### LUNA-003: run-manifest collector

Uses GPU: no.

Dependencies: LUNA-000.

Actions:

- Implement the smallest reusable collector for Section 16 fields, or extend an existing owned collector.
- Add focused tests for dirty-path capture, boot ID, power, command argv, and missing positive controls.
- The collector must remain side-effect free until explicitly invoked.

Outputs:

- Reusable collector in the existing canonical audit/benchmark utility location.
- Focused unit tests.

Acceptance:

- Tests prove a known command, branch, dirty path, and worktree are recorded.
- Missing required fields fail loudly.

Stop:

- An existing canonical collector already satisfies the contract. Reuse it and return `PASS` without duplication.

### LUNA-010: llama benchmark and request lifecycle source map

Uses GPU: no.

Dependencies: LUNA-002.

Read boundary:

- llama-bench entry point and benchmark loop.
- Request/token construction and decode invocation files reached directly from that loop.

Actions:

- Map CLI flags to prompt tokens, generation tokens, repetitions, batch, ubatch, flash attention, and GPU offload.
- Map prompt processing, first token, steady decode, and synchronization boundaries.
- Record exact file/function references.

Outputs:

- `docs/task_workflow/in_progress/luna-010-llama-request-lifecycle.md`

Acceptance:

- Every benchmark phase has a source owner and an observable trace boundary proposal.

### LUNA-011: llama quant linear source map

Uses GPU: no.

Dependencies: LUNA-002.

Actions:

- Map Q4_K/Q6_K prompt GEMM and decode GEMV/MMQ selection from graph op to ROCm kernel.
- Record shape predicates, M buckets, tile ownership, quant decode algebra, activation quantization if used, and fallback rules.
- Separate prompt, first-token, and steady-decode paths.

Outputs:

- `docs/task_workflow/in_progress/luna-011-llama-quant-path.md`

Acceptance:

- Each expected kernel family is connected to a source selection predicate.

### LUNA-012: llama attention and KV source map

Uses GPU: no.

Dependencies: LUNA-002.

Actions:

- Map G=5 attention ownership, flash admission, KV layout, KV writes, KV reads, split/combine behavior, masking, and causal bounds.
- Calculate expected KV bytes per token and per context from model geometry.
- Record source references and assumptions.

Outputs:

- `docs/task_workflow/in_progress/luna-012-llama-attention-kv.md`

Acceptance:

- Expected dispatch families and byte formulas exist for contexts 128/512/4096.

### LUNA-013: llama ROCm backend source map

Uses GPU: no.

Dependencies: LUNA-002.

Actions:

- Map graph scheduling, buffer allocation, HIP stream/queue use, kernel launch, synchronization, and code-object compilation/loading.
- Identify where grid, block, LDS, and kernel names are formed.
- Identify the source-to-dispatch join keys usable by trace parsing.

Outputs:

- `docs/task_workflow/in_progress/luna-013-llama-rocm-backend.md`

Acceptance:

- At least one known kernel can be predicted from source and later confirmed in LUNA-021.

### LUNA-014: tinygrad equivalent source map

Uses GPU: no.

Dependencies: LUNA-000.

Actions:

- Map the canonical fixed-depth decode harness through model prompt processing, prefill linear routing, attention routing, KV updates, first token, flash decode, combine, and token extraction.
- Record where context 128 versus 512 can select different paths.
- Record the explicit and implicit direct-packed fallback boundaries.

Outputs:

- `docs/task_workflow/in_progress/luna-014-tinygrad-request-map.md`

Acceptance:

- Every route decision needed by LUNA-034 has a named function and observable positive control.

### LUNA-020: comparable workload and token fixture

Uses GPU: no.

Dependencies: LUNA-010, LUNA-014.

Actions:

- Define exact prompt token IDs for lengths 128, 512, and 4096.
- Establish BOS/EOS handling and deterministic temperature-zero expectations.
- Define explicit llama and tinygrad commands using the same GGUF and comparable KV depths.
- Set trace runs to one bounded repetition and authority runs to the retained standard.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/token-fixture.json`
- `docs/task_workflow/in_progress/luna-020-command-matrix.md`

Acceptance:

- Both runtimes consume the same token fixture or the exact semantic difference is quantified.
- Commands explicitly set batch, ubatch, flash, offload, context, generation length, and output paths.

Stop:

- The installed llama entry point cannot accept or reproduce the fixture. Define a bounded wrapper rather than claiming equivalence.

### LUNA-021: llama context-128 bounded trace

Uses GPU: yes.

Dependencies: LUNA-001, LUNA-002, LUNA-003, LUNA-020.

Actions:

- Run one clean ordinary smoke proving context-128 output.
- Run one bounded ROCm dispatch trace.
- Separate prompt, transition, and one steady decode token.
- Record exact kernel order, grid/block, durations, and source join keys.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx128/`

Acceptance:

- Known kernel positive control from LUNA-013 appears.
- Context 128 completes without fault and produces deterministic output.

### LUNA-022: llama context-512 bounded trace

Uses GPU: yes.

Dependencies: LUNA-021.

Actions and acceptance: identical to LUNA-021 at context 512.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx512/`

### LUNA-023: llama context-4096 bounded trace

Uses GPU: yes.

Dependencies: LUNA-022.

Actions and acceptance: identical to LUNA-021 at context 4096, with raw trace size bounded by filtering to the measured request.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama/ctx4096/`

### LUNA-024: llama dispatch normalization

Uses GPU: no.

Dependencies: LUNA-021, LUNA-022, LUNA-023.

Actions:

- Parse traces into semantic kernel families.
- Separate load/setup, prompt, transition, and steady decode.
- Compute counts and wall contribution by family and context.
- Link every material family to LUNA-010 through LUNA-013 source references.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama-dispatch-ledger.json`

Acceptance:

- At least 95% of measured device wall is attributed or the unattributed remainder is explicitly listed.

### LUNA-025: llama resources and disassembly

Uses GPU: no after code objects are retained.

Dependencies: LUNA-024.

Actions:

- Extract VGPR, SGPR, LDS, scratch/spill, wave, grid, block, and code-object identity for material families.
- Count relevant VMEM, LDS, VALU, matrix, barrier, and wait instructions.
- Normalize counts per token, tile, or loop trip as appropriate.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama-resource-ledger.json`

Acceptance:

- Every family contributing at least 5% of steady decode wall has a resource row.

### LUNA-026: llama focused counters

Uses GPU: yes.

Dependencies: LUNA-001, LUNA-024, LUNA-025.

Actions:

- Select counters only for material families and named hypotheses.
- Collect the smallest number of passes required by the installed tool.
- Record counter definitions and normalization.
- Compare 512 versus 4096 without using profiler wall time as authority.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/llama-counter-ledger.json`

Acceptance:

- Every collected counter distinguishes a stated mechanism.

Stop:

- Counter tooling lacks gfx1100 support or requires unsafe system changes. Return `TOOL_FAILURE`; LUNA-040 may proceed using trace/resource evidence with the limitation stated.

### LUNA-030: tinygrad 8B context-128 control

Uses GPU: yes.

Dependencies: LUNA-003, LUNA-020.

Actions:

- Run the minimal context-128 setup and one decode token in a fresh process.
- Capture route/kernel positive controls and ordinary timing.
- Retain compiler and dispatch identity needed for comparison with 14B.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/tinygrad/8b-ctx128/`

Acceptance:

- Setup and decode pass with positively identified routes.

### LUNA-031: tinygrad 14B context-128 failure capture

Uses GPU: compiler/setup may dispatch; lock required.

Dependencies: LUNA-003, LUNA-020, LUNA-030.

Actions:

- Reproduce once.
- Retain failing HIP source, function name, semantic UOp slice, full STORE ancestry, and successful dispatch prefix.
- Classify failure before timed decode.
- Do not repeat a deterministic failure.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/tinygrad/14b-ctx128/`

Acceptance:

- Invalid destination STORE is connected to a semantic operation, not merely a C line.

### LUNA-032: tinygrad 14B context-512 trace

Uses GPU: yes.

Dependencies: LUNA-003, LUNA-020, LUNA-031.

Actions:

- Collect ordinary timing and one bounded trace using the same schema as llama.
- Retain route, dispatch, code-object, and resource identity.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/tinygrad/14b-ctx512/`

Acceptance:

- Context 512 passes through the expected production route.

### LUNA-033: tinygrad 14B context-4096 trace

Uses GPU: yes.

Dependencies: LUNA-032.

Actions and acceptance: identical to LUNA-032 at context 4096.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/tinygrad/14b-ctx4096/`

### LUNA-034: implicit fallback attribution

Uses GPU: no new GPU run unless existing route evidence lacks a positive control.

Dependencies: LUNA-014, LUNA-031, LUNA-032.

Actions:

- Prove or refute direct-packed fallback selection at context 128.
- Record the packed candidate accept/decline decision and exact shape.
- Compare with a known packed selection at 512 and a known allowed direct-packed control.
- Audit whether the environment-only fail-loud guard misses implicit selection.

Outputs:

- `docs/task_workflow/in_progress/luna-034-fallback-verdict.md`

Acceptance:

- Verdict is `SUPPORTED` or `REFUTED`; empty selection logs are invalid.

### LUNA-040: algorithm and lifecycle comparison

Uses GPU: no.

Dependencies: LUNA-024, LUNA-030, LUNA-031, LUNA-032, LUNA-033.

Actions:

- Compare prompt, transition, attention, KV, quant linear, combine, sampling, and synchronization algorithms.
- Identify equivalent and non-equivalent useful work.

Outputs:

- `docs/task_workflow/in_progress/luna-040-algorithm-comparison.md`

### LUNA-041: traffic and compute model

Uses GPU: no.

Dependencies: LUNA-012, LUNA-024, LUNA-025, LUNA-032, LUNA-033.

Actions:

- Calculate weight, KV, transient, and duplicate traffic.
- Calculate useful operations and instruction-normalized work.
- Reconcile formulas with measured counters when available.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/traffic-compute-model.json`

Acceptance:

- Formula inputs derive from model geometry and trace counts.

### LUNA-042: depth-slope decomposition

Uses GPU: no.

Dependencies: LUNA-024, LUNA-025, LUNA-026 or recorded counter limitation, LUNA-041.

Actions:

- Fit the explicit latency model for llama and tinygrad.
- Attribute `t(4096)-t(512)` by semantic family.
- Reject families whose maximum contribution cannot explain the gap.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/depth-slope-model.json`

Acceptance:

- Named owner or bounded owner set explains the measured slope within stated residual error.

### LUNA-043: context-128 causal comparison

Uses GPU: no.

Dependencies: LUNA-021, LUNA-030, LUNA-031, LUNA-034, LUNA-040.

Actions:

- Compare llama 14B working context 128, tinygrad 8B working context 128, and tinygrad 14B failing context 128.
- Identify the earliest semantic divergence that creates or avoids the illegal STORE.
- Separate missing route coverage from generic lowering failure.

Outputs:

- `docs/task_workflow/in_progress/luna-043-ctx128-causal-comparison.md`

Acceptance:

- One causal owner and invariant set is named before candidate design.

### LUNA-044: first-principles synthesis

Uses GPU: no.

Dependencies: LUNA-040, LUNA-041, LUNA-042, LUNA-043.

Actions:

- Merge the source, trace, resource, traffic, and causal findings.
- Mark every claim measured, source-proven, inferred, refuted, or unresolved.
- Produce the required Section L4 comparison.

Outputs:

- `docs/14b-llama-vs-tinygrad-first-principles-20260726.md`
- `bench/14b-decode-ctx128-depth-decay-20260726/path-comparison.json`

Acceptance:

- Independent facts support the context-128 owner and depth-slope owner.
- No open-ended candidate axis remains in the proposed search.

### LUNA-050: BoltBeam search contract

Uses GPU: no.

Dependencies: LUNA-044.

Actions:

- Locate and record the actual BoltBeam entry point and schema.
- Encode immutable facts, candidate axes, constraints, evaluation stages, and promotion rules from LUNA-044.
- Define separate candidate families for Track A and Track B.
- Add compile-only and correctness rejection before timing.

Outputs:

- `docs/task_workflow/in_progress/14b-decode-boltbeam-search-contract-20260726.md`

Acceptance:

- Every axis cites a measured or source-proven difference.
- Search cannot hide context 128, change useful work silently, or promote profiler timing.

### LUNA-051: context-128 STORE semantic repair design

Uses GPU: no.

Dependencies: LUNA-043, LUNA-050.

Actions:

- Specify intended destination semantics, invalid transformation, ownership boundary, and minimal repair layer.
- Define invariants and focused tests before code.
- Decide whether route coverage and compiler lowering require separate candidates.

Outputs:

- `docs/task_workflow/in_progress/luna-051-ctx128-repair-design.md`

Acceptance:

- Design explains why it changes underlying state correctly, not merely why HIP compiles.

### LUNA-052: context-128 candidate review gate

Uses GPU: no.

Dependencies: LUNA-051.

Actions:

- Review design against UOp semantics, vector lane ownership, cache/KV bounds, genericity, 8B impact, and fallback policy.
- Enumerate exact compile and numerical tests.

Outputs:

- `docs/task_workflow/in_progress/luna-052-ctx128-design-review.md`

Acceptance:

- Verdict `PASS` is required before implementation.

### LUNA-053: implement context-128 candidate

Uses GPU: compile-only first, then bounded GPU.

Dependencies: LUNA-052 PASS.

Actions:

- Implement one reviewed candidate.
- Add focused regression tests reproducing the illegal destination structure.
- Run compile-only 8B/14B shape matrix.
- Run deterministic token parity only after compile passes.

Outputs:

- Candidate code and focused tests on a child branch.
- `bench/14b-decode-ctx128-depth-decay-20260726/candidates/ctx128/<candidate-id>/`

Acceptance:

- Context 128 compiles and updates intended state correctly.
- 8B and 512+ code paths are unchanged or explicitly validated.

Failure loop:

- One agent implements one candidate only.
- A failed candidate receives a verdict and cleanup commit.
- LUNA-051 may be reopened with new evidence for the next bounded candidate.
- After three mechanistically distinct failures, stop blind implementation and require a revised LUNA-044 synthesis.

### LUNA-054: real short-prompt matrix

Uses GPU: yes, serial separate processes.

Dependencies: LUNA-053 PASS.

Actions:

- Run lengths `2,8,16,32,64,96,127,128,129,192,256,384,511,512`.
- Run `"hi"` and one short sentence.
- Produce at least eight deterministic decode tokens per admitted prompt.
- Record prompt route, first-token route, token parity, bounds, and failure state.

Outputs:

- `bench/14b-decode-ctx128-depth-decay-20260726/prompt-matrix.json`

Acceptance:

- Every ordinary prompt succeeds or has an explicitly justified unsupported boundary that does not include `"hi"` or context 128.

### LUNA-055: context-128 reliability and performance gate

Uses GPU: yes.

Dependencies: LUNA-054.

Actions:

- Run three clean context-128 processes.
- Measure prompt setup, TTFT, and steady decode separately.
- Verify route identity, token parity, no fallback ambiguity, and no GPU fault.
- Compare with llama context 128 using ordinary timing.

Outputs:

- `docs/14b-decode-ctx128-recovery-findings-20260726.md`
- Track A authority artifacts.

Acceptance:

- Track A completion criteria pass.

### LUNA-060: depth-slope owner confirmation

Uses GPU: focused measurement only if retained evidence is insufficient.

Dependencies: LUNA-042, LUNA-044, and LUNA-055 if Track A changed 512+ code.

Actions:

- Confirm the named kernel/family and mechanism on the final Track A commit.
- Recollect 512/4096 baselines if code objects changed.
- Bound maximum Amdahl contribution.

Outputs:

- `docs/task_workflow/in_progress/luna-060-depth-owner.md`

Acceptance:

- Owner can explain enough wall time to meet the required performance target.

### LUNA-061: G5 resource and lifetime design

Uses GPU: no.

Dependencies: LUNA-060.

Actions:

- Compare G4/G5 live ranges, allocation, staging, waits, barriers, loop state, and occupancy at shallow/deep contexts.
- Propose only mechanism-linked compiler or route primitives.
- Specify resource and traffic invariants.

Outputs:

- `docs/task_workflow/in_progress/luna-061-g5-design.md`

Acceptance:

- Candidate predicts a measurable resource or slope change and preserves useful work.

### LUNA-062: BoltBeam depth candidate search

Uses GPU: compile gates plus serialized bounded timings.

Dependencies: LUNA-050, LUNA-061.

Actions:

- Run compile-only rejection across the bounded axes.
- Reject spill, bounds, resource, route, and output failures before timing.
- Time surviving kernels at shallow/deep geometry.
- Rank by predicted whole-model contribution, not local percentage alone.

Outputs:

- Machine-readable candidate ledger with every attempted axis and verdict.
- Cleanup list for generated probes.

Acceptance:

- At least one candidate meets correctness and has enough predicted contribution, or the bounded search space is completely resolved.

Stop:

- No candidate has adequate Amdahl contribution. Return to LUNA-061 only with new LUNA-042 evidence; do not expand axes arbitrarily.

### LUNA-063: implement depth candidate

Uses GPU: compile-only first, then bounded GPU.

Dependencies: LUNA-062 qualifying candidate.

Actions:

- Promote one candidate through the normal tinygrad ownership layer.
- Add resource, route, code-object, and correctness regression coverage.
- Delete search-only adapters not required by production.

Outputs:

- Candidate code and tests.
- Candidate authority artifact directory.

Acceptance:

- Kernel-level shallow/deep result matches the predicted mechanism.

Failure loop:

- Maximum three promoted candidate attempts before revisiting the first-principles model.
- Each failure is reverted or removed and banked before the next attempt.

### LUNA-064: full depth authority

Uses GPU: yes.

Dependencies: LUNA-063 PASS.

Actions:

- Run three same-session interleaved baseline/candidate pairs at 512/1024/2048/4096.
- Run same-session llama at 512 and 4096.
- Verify token parity, route identity, 8B non-regression, power, and fault state.
- Calculate achieved bandwidth using retained formulas.

Outputs:

- `docs/14b-decode-depth-decay-findings-20260726.md`
- Track B authority artifacts.

Acceptance:

- Context 4096 matches or beats llama.
- Context 512 remains within noise.
- Track B completion criteria pass.

### LUNA-070: regression and policy integration

Uses GPU: only already-required focused tests.

Dependencies: LUNA-055, LUNA-064.

Actions:

- Add final context-128 regression, model-aware route policy, G5 resource/mechanism, and benchmark-entry tests.
- Ensure explicit unsupported configurations fail loudly.
- Ensure no test depends on task artifact paths.

Outputs:

- Production regression tests.

### LUNA-071: manifests and current-state documentation

Uses GPU: no.

Dependencies: LUNA-070.

Actions:

- Update route/provenance manifests, current decode numbers, llama comparison, supported context range, rollback behavior, and known limitations.
- Remove stale statements superseded by Track A/B.

Outputs:

- Final production documentation and manifests.

### LUNA-072: probe and artifact cleanup

Uses GPU: no.

Dependencies: LUNA-071.

Actions:

- Delete temporary wrappers, monkeypatches, generated candidate scripts, duplicate traces, and rejected candidate code.
- Retain canonical artifacts, hashes, ledgers, and findings.
- Record recovery commits for deleted authored probes.

Outputs:

- Probe cleanup ledger.

Acceptance:

- No task-specific executable remains without owner, second use, positive control, and input contract.

### LUNA-073: promotion readiness audit

Uses GPU: no new measurement.

Dependencies: LUNA-072.

Actions:

- Audit completion criteria, commits, dirty state, test evidence, authority identities, artifact retention, and cleanup.
- Verify every promoted claim points to evidence from the final candidate commit.
- Produce an exact promotion commit list.

Outputs:

- `docs/task_workflow/output/14b-decode-ctx128-depth-decay-completion-20260726.md`

Acceptance:

- No unresolved required criterion or unrelated commit remains.

### LUNA-074: promotion and branch retirement

Uses GPU: no.

Dependencies: LUNA-073 PASS and operator approval.

Actions:

- Promote the approved commits through the repository's integration/production workflow.
- Confirm useful code, tests, findings, manifests, and authority summaries exist on the retained branch.
- Remove child worktrees and branches.
- Remove the feature worktree.
- Delete local and remote `feature/14b-decode-ctx128-and-depth-decay` after reachability is confirmed.
- Record retirement in final findings.

Acceptance:

- Production contains the accepted repair and evidence.
- Temporary branches/worktrees are gone.
- No uncommitted or unpushed work was stranded.

## 19. Orchestrator rules for low-agent continuation

The orchestrator repeats this cycle until LUNA-073 passes:

1. Read the state file.
2. Select only tasks whose dependencies have accepted verdicts.
3. Spawn CPU/static tasks in parallel only when file ownership does not overlap.
4. Spawn at most one GPU task.
5. Merge accepted child commits in dependency order.
6. Update the state file and cleanup child worktree/branch.
7. Recompute which tasks are unblocked.
8. Stop for operator input only on a listed stop condition, required external credential, unsafe GPU state, or promotion approval.

Low-agent continuation must not reinterpret `INCONCLUSIVE` as success, silently skip a failed task, or broaden a search space without updating LUNA-044 and LUNA-050.

## 20. Exhaustive completion audit

Before declaring completion, the orchestrator verifies:

- All LUNA task rows have terminal accepted verdicts.
- LUNA-021/022/023 trace llama at all required contexts.
- LUNA-031 captures the original tinygrad failure causally.
- LUNA-034 confirms or refutes implicit fallback.
- LUNA-044 provides the first-principles comparison.
- LUNA-050 constrains BoltBeam from that comparison.
- LUNA-055 proves real short prompts and context 128 work.
- LUNA-064 proves depth-decay recovery against same-session llama.
- LUNA-070 retains regressions.
- LUNA-072 deletes dead probes.
- LUNA-073 proves evidence identity on the final commits.
- LUNA-074 retires the feature branch only after promotion.

If any required row is absent, stale, collected on another commit, or missing its positive control, the task is incomplete.
