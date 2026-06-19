# q8 FFN AMD scheduler/codegen work scope (2026-06-19)

This scope starts only because the primitive route is exhausted.

Known state:

- q8 decode route is valid in-model as research artifact:
  - W==D decode `1.051-1.063x`;
  - dNLL `+0.002887`;
  - default off.
- tinygrad-owned AMD DSL consumer is correct:
  - real-GGUF gate max_abs `9.54e-7`;
  - real-GGUF up max_abs `1.43e-6`.
- tinygrad-owned AMD DSL consumer is too slow:
  - fused gate/up `166.649us`;
  - target `<=60us`;
  - hipcc/LLD artifact route is the schedule oracle.

Therefore the remaining question is not primitive expressibility. It is whether tinygrad can learn enough AMD scheduling
quality to make the same primitive fast.

## Boundary

This is **compiler/scheduler work**, not q8 primitive search.

In scope:

- compare tinygrad AMD DSL ASM vs hipcc/LLD oracle at the instruction/dependency level;
- reduce address/control/SALU work in the hand-owned consumer;
- improve load/wait/dot scheduling;
- improve wave/workgroup reduction schedule;
- add small AMD DSL/assembler capabilities if a concrete q8 consumer diff proves they are missing;
- keep all changes behind probes until a local gate passes.

Out of scope:

- no model default changes;
- no producer ownership until the consumer clears its gate;
- no broad HIP importer;
- no general LLVM clone;
- no source-level COMGR reshuffling, already closed by B2a;
- no q8 quality work, already passed.

## Authority Targets

| implementation | correctness | median fused gate/up | status |
|---|---:|---:|---|
| hipcc/LLD artifact | PASS | schedule oracle / `<=60us` target | research artifact |
| COMGR fused-C | PASS | `146.88us` | closed |
| tinygrad AMD DSL/ASM | PASS | `166.649us` | closed as primitive; input to scheduler work |

B0/B1 audit instruction-class reference:

| object | dot4 | VALU | SALU | DS | branch | waitcnt |
|---|---:|---:|---:|---:|---:|---:|
| hipcc/LLD fast gate/up | 16 | 120 | 197 | 7 | 5 | 20 |
| COMGR MMVQ baseline | 16 | 167 | 299 | 2 | 18 | 9 |

The tinygrad ASM consumer still needs its own disassembly count as phase S0. Do not reason from source length alone.

## S0 — Disassembly Accounting

Goal: identify the exact schedule gap before changing code.

Tasks:

- disassemble `extra/q8_ffn_asm_gateup_full.py`'s generated code object;
- count instruction classes with the same parser as `q8_ffn_codegen_transfer_audit.py`;
- compare against hipcc/LLD oracle and COMGR fused-C;
- extract top mnemonic counts;
- record VGPR/SGPR/LDS/kernarg metadata;
- record whether local `(128,1,1)` materially changes descriptor/workgroup resource fields.

Artifact:

- `bench/q8-ffn-codegen-transfer/asm_schedule_audit.json`.

Gate:

- name the top three measured deltas to the hipcc/LLD oracle.

Kill:

- if tinygrad ASM has no large instruction-count/resource delta but is still `~3x` slower, classify as unobservable
  scheduling/latency work and do not hand-tune blindly without PMU tooling.

## S1 — Reduction Schedule Audit

Hypothesis: the correctness-first `ds_bpermute + LDS` reduction is a meaningful part of the `166.649us`.

Tasks:

- build reduction-only microbenchmarks:
  - current `ds_bpermute` wave reduce + LDS four-wave reduce;
  - one-wave-only diagnostic;
  - LDS-only four-slot reduce;
  - optional scalar-lane store variant if EXEC masking can be expressed safely;
- compare against hipcc/LLD oracle reduction shape.

Gate:

- identify a reduction replacement with predicted fused-consumer savings `>=15us`.

Kill:

- if reduction variants move less than `10us`, reduction is not the primary blocker.

## S2 — Address/Scale-Min Schedule Audit

Hypothesis: scale/min extraction and per-thread address calculation dominate SALU/VALU overhead.

Tasks:

- split diagnostics for:
  - q4/q8 address math only;
  - scale/min decode only;
  - dot loop with precomputed synthetic sc/mn;
  - dot loop with fixed sub path;
- compare instruction counts and timings.

Candidate optimizations:

- precompute invariant row base in SGPR and minimize per-lane vector math;
- replace branchless `get_scale_min` select chain with two specialized kernels or role/sub layout specialization only if
  it keeps the model route simple;
- specialize sub groups if it reduces enough control work without exploding launch count.

Gate:

- one simplification predicts `>=20us` savings and preserves the one-kernel fused gate/up lifecycle.

Kill:

- if all simplifications are correctness-equivalent but sub-`10us`, stop tuning address math.

## S3 — Load/Wait/Dot Scheduling

Hypothesis: the hand ASM serializes `load -> wait -> dot` too strictly, while hipcc/LLD interleaves independent loads,
scale/min work, q8 loads, and dot4 operations.

Tasks:

- produce an ISA dependency timeline for one thread's 8-dot loop;
- move loads for the next `k` ahead of current dot where legal;
- group q4/q8 global loads to reduce wait frequency;
- reduce `s_waitcnt` placement to the minimum correctness set;
- verify no correctness drift with the existing partial and full-row probes.

Gate:

- isolated full consumer improves by `>=25us` without correctness loss.

Kill:

- if load/wait changes cause instability or <`15us` improvement, the missing scheduler is broader than a local reorder.

## S4 — Descriptor/Local-ID Capability

Finding already banked: `local=(32,4,1)` exposed only local-x safely through the current `assemble_linear` path; using
`v1` as local-y caused an MMU fault. The consumer had to switch to `local=(128,1,1)`.

Tasks:

- audit `assemble_linear` kernel descriptor fields for enabled workitem IDs;
- determine whether enabling workitem-id Y/Z in VGPRs is supported by the runtime descriptor;
- build a safe local-y smoke if descriptor support is added;
- compare `(32,4,1)` vs `(128,1,1)` generated ISA and performance.

Gate:

- local-y support works and improves fused consumer by `>=10us`, or is needed for a larger scheduling refactor.

Kill:

- if descriptor work only changes ergonomics and not performance, do not promote it as a decode blocker.

## S5 — Minimal Scheduler/DSL Feature Decision

Only after S0-S4:

Classify the required work into one of three buckets.

| bucket | meaning | action |
|---|---|---|
| local hand schedule | a small, understandable reordering/cleanup gets close to `<=60us` | implement in probe, rerun B2b10 |
| AMD DSL feature | one missing assembler/descriptor capability blocks the fast schedule | add minimal runtime/assembler support with tests |
| project-level scheduler | fast route needs broad instruction scheduling/register allocation | stop decode route; document as compiler roadmap |

Gate to continue toward model route:

- real-GGUF fused gate/up consumer `<=75us` after local tuning, with a credible path to `<=60us`;
- final gate remains `<=60us`.

Kill:

- if best local tuning remains `>100us`, stop. Producer ownership cannot rescue the route.

## Final Success Condition

Only if scheduler work gets the consumer to `<=60us`:

1. reopen producer ownership;
2. rerun lifecycle gate `producer + gate/up <=129.2us`;
3. replace artifact route with tinygrad-owned `PROGRAM` nodes;
4. rerun final A4:
   - W==D decode `>=3%`;
   - dNLL `<=0.01`;
   - default off.

## Recommendation

Run **S0 only** first. **Executed: see `q8-ffn-amd-scheduler-s0-result-20260619.md`.**

Do not start hand-tuning until the ASM consumer's disassembly is counted against the hipcc/LLD oracle. If S0 shows a
large, obvious delta, pursue the corresponding focused phase. If S0 does not, this should be closed as project-level AMD
scheduler work rather than spending more time on local primitive probes.

S0 verdict: **S0_CLOSE_PROJECT_LEVEL_SCHEDULER**.

The tinygrad AMD DSL/ASM consumer emits the same `16` dot4 operations as the hipcc/LLD oracle and fewer static
instructions overall (`218` vs `336`), but it remains `166.649us` vs the `<=60us` target. The visible deltas are load
shape (`22` global loads vs `11`) and address/bit-manipulation VALU (`+37`), not a missing primitive or a massive
instruction-count blowup. This trips the S0 kill rule: do not continue S1-S4 as blind tuning. Native q8 decode ownership
is closed unless the project funds broader AMD scheduler/codegen work.
