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
