# 14B decode attention ISA-cadence campaign

Date: 2026-07-27

Status: completed and promoted on 2026-07-27 (`8deca39bb`)

Production repository: `/home/ubuntu/tinygrad-arkey`

Execution copy: `/home/ubuntu/worktrees/tinygrad-g5-qgroup-parity`

Comparator: `/home/ubuntu/env/llama.cpp/build/bin/llama-bench`

Model: `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf`

Target: AMD RX 7900 XTX, gfx1100, wave32, one XCC

## 1. Objective

Fix Qwen3-14B decode depth decay by removing a proven generated-attention instruction or serialization defect.

Absolute tok/s is diagnostic only. A candidate passes when either:

1. tinygrad's ctx512 to ctx4096 decay matches same-session llama.cpp decay within measurement uncertainty; or
2. tinygrad passes llama.cpp at ctx4096 while retaining its ctx512 lead.

The final 2026-07-27 comparison is the starting point:

| Runtime | ctx512 tok/s | ctx4096 tok/s | decay | tinygrad margin |
|---|---:|---:|---:|---:|
| llama.cpp | 66.25 | 62.53 | 5.6% | reference |
| tinygrad QG2/S32/KV_BOTH | 69.13 | 61.45 | 11.1% | +4.4% / -1.7% |

The best isolated QG2 candidate improves current master at ctx4096 from 59.23 to 61.45 tok/s, but it is not approved for promotion because it fails both relative gates.

## 2. Proven attribution

Matched native tinygrad profiles show:

| Component, 40 layers | ctx512 | ctx4096 | conclusion |
|---|---:|---:|---|
| QG2 attention tile | 1.30 ms/token | 4.35 ms/token | dominant depth-sensitive cost |
| fused combine | 0.42 ms/token | 0.43 ms/token | flat |
| large quantized weight kernels | approximately flat | approximately flat | not the decay mechanism |

The llama trace reports approximately 10.9 us/layer at ctx512 and 30.0 us/layer at ctx4096 for its attention tile, plus approximately 3 us/layer for combine. Tinygrad's generated tile is therefore the only authorized optimization target in this campaign.

## 3. Closed search space

Do not repeat these tests unless a compiler/runtime change invalidates the cited mechanism:

| Axis | Result |
|---|---|
| query-group size 1 | correct, slower than QG2 |
| query-group size 3 | previously slower |
| split count 8/12/16/24/32/40 | QG2/S32 is the current-compiler optimum |
| token tile 32 | slower than token tile 16 at both depths |
| K_ONLY staging | isolated micro win, paired model loss |
| sequential K-then-V LDS reuse | correct, ctx4096 micro regression to 0.250 ms |
| dispatch growth | fixed route and programs/token; refuted |
| combine | flat across depth; not the decay mechanism |
| true occupancy as the QG2 mechanism | counters do not support it |
| L2 misses as the QG2 mechanism | misses remain nearly flat; repeated reads become L2 hits |

## 4. Repository and GPU discipline

1. Do not edit or switch branches in `/home/ubuntu/tinygrad-arkey` while a benchmark is running.
2. All compiler/kernel edits and generated binaries belong in the isolated execution copy.
3. GPU commands run serially. No profiler, benchmark, or GPU test may overlap another process.
4. Record command, tree path, model identity, binary identity, context, route, clocks, and profiler state with every measured artifact.
5. A silent or empty collector result is a failed probe until a positive control proves the collector observed a known kernel.
6. A failed PMC or rocprof session must restore profiler and power state before any timing result is admitted.
7. No candidate reaches master without explicit user promotion after all gates pass.

## 5. Evidence model

Static and dynamic evidence have different authority:

- Static disassembly proves which instructions and branches exist, resource metadata, and loop structure.
- Launch geometry proves workgroup/wave count and fixed versus depth-dependent dispatch structure.
- Hardware counters prove executed waves, instruction classes, cache behavior, and cycle changes for a concrete dispatch.
- Native kernel timelines prove the component's contribution to whole-token decay.
- End-to-end fixed-depth authority proves the actual product outcome.

Never compare unscaled static instruction counts as if they were executed counts. Loop-body counts must be multiplied by the proven trip count or joined to dynamic counters.

## 6. Required identities

Every comparison bundle must record:

- tinygrad source-tree path and source digest for the tile emitter;
- tinygrad kernel name, program binary SHA-256, kernarg layout, grid, workgroup, LDS, VGPR, SGPR, scratch, and wave size;
- llama build commit, source path, kernel name, code-object identity, grid, workgroup, LDS, VGPR, SGPR, scratch, and wave size;
- model file path, size, mtime, and identity hash;
- context, Hq=40, Hkv=8, G=5, Hd=128, split count, query-group size, staging, dtype, KV-cache dtype, and fused operations;
- exact tool versions and command lines.

If the two kernels do not implement the same semantic work, the report must state the difference and normalize comparisons per query head, KV token, output element, and layer.

## 7. Phase 0: profiler and tool preflight

### ISA-000 Tool inventory

Confirm paths and versions for:

- `llvm-objdump` or ROCm equivalent;
- `rocprofv3`;
- `rocprof-compute`, if present;
- tinygrad native `PROFILE=1`;
- the KFD PMC path and its required privilege wrapper.

### ISA-001 Positive controls

1. Disassemble a known tinygrad HSACO and prove its named kernel appears.
2. Capture one llama attention dispatch and prove `flash_attn_tile<128, 128, 2, 1, false>` appears.
3. Capture one tinygrad QG2 dispatch and prove `flash_block_tiled_xlane_score_pv_tile_whole_cache_40_128_qg2` appears.
4. For counters, prove nonzero `SQ_WAVES`, `SQ_WAVE_CYCLES`, and at least one instruction counter on the target tile.

Stop if a positive control fails. Fix the observation path before interpreting any absence.

## 8. Phase 1: comparable binary capture

### ISA-010 Tinygrad capture

Compile the exact descriptor-owned QG2/S32/KV_BOTH tile used by the model route. Persist:

- HSACO bytes and SHA-256;
- generated source or UOp program identity;
- disassembly and metadata;
- grid/workgroup geometry at ctx512 and ctx4096;
- resource row and scratch/spill status.

The capture must bind to the same program identity seen in the model timeline, not merely a shape-similar microbenchmark.

### ISA-011 Llama capture

Capture the exact llama code object and dispatch for `flash_attn_tile<128, 128, 2, 1, false>` under fixed-depth 14B decode. Persist the same binary, disassembly, launch, and resource facts as ISA-010.

### ISA-012 Semantic normalization

Map each kernel's work to:

- query heads handled per workgroup;
- KV tokens handled per split/workgroup;
- head dimension elements per lane;
- split count and combine contract;
- K and V global bytes per useful query head;
- LDS bytes and barriers per KV token tile;
- online-softmax state carried per query head.

Do not proceed to source changes until this map explains which quantities are comparable and which are not.

## 9. Phase 2: cadence analysis

### ISA-020 Instruction taxonomy

Normalize disassembly into:

- global K/V loads and widths;
- LDS loads/stores and widths;
- vector FMA/dot instructions;
- scalar and vector address arithmetic;
- cross-lane permute/shuffle/reduction instructions;
- exponent/log/reciprocal instructions;
- barriers and wait instructions;
- conditional branches and loop-control instructions;
- spills or scratch traffic.

Report static loop-body counts and depth-normalized executed estimates separately.

### ISA-021 Stage cadence

Build an ordered stage map for both kernels:

1. K load/stage;
2. QK dot;
3. score reduction;
4. scale/mask;
5. online max/sum update;
6. V load/stage;
7. PV accumulation;
8. split output store;
9. combine.

For every stage, report instructions per useful query head per 16 KV tokens and mandatory synchronization points.

### ISA-022 Dynamic join

At ctx512 and ctx4096, join the tile program identity to raw counters:

- `SQ_WAVES`;
- `SQ_WAVE_CYCLES`;
- `SQ_BUSY_CYCLES`;
- `SQ_INSTS_VALU`;
- `SQ_INSTS_SALU`;
- LDS active/conflict counters;
- L2 hit/miss counters;
- available branch, VMEM, LDS, and wait counters.

Compute deltas per layer, per useful wave, and per KV token. Do not call BoltBeam's normalized occupancy ratio physical occupancy.

## 10. Phase 3: mechanism discriminators

Only the following mechanisms are open. Each must have a discriminator that changes one property.

### ISA-030 Score reduction duplication

Claim: QK score or its lane reduction is repeated for multiple output lanes/accumulators.

Discriminator: instrument or structurally count score-producing instructions and prove whether one logical score causes more than one reduction chain per query head/token.

Pass signal: removing duplicate score computation reduces VALU/wave cycles proportional to depth without changing global bytes.

### ISA-031 Online-softmax duplication

Claim: exponent, max, normalization, or rescale work is repeated per output component instead of once per query head/token.

Discriminator: count exponent/reciprocal/reduction cadence and build a score-state sharing ablation that preserves PV arithmetic and memory traffic.

Pass signal: fewer transcendental/reduction instructions and lower deep latency with identical output.

### ISA-032 PV scalarization or lane mapping

Claim: PV accumulation is scalarized or uses a worse lane-to-output mapping than llama.

Discriminator: compare vector instruction widths and useful output elements per lane; construct one vector-width/lane-map ablation without changing split count or staging.

Pass signal: fewer VALU instructions per KV token and lower wave cycles without additional LDS conflicts or spills.

### ISA-033 Address/control overhead

Claim: generated per-token address, mask, or loop control dominates at depth.

Discriminator: hoist one invariant class or replace one dynamic index chain while preserving loads, arithmetic, and barriers.

Pass signal: SALU/branch/wave-cycle reduction proportional to trip count.

### ISA-034 Synchronization cadence

Claim: tinygrad emits redundant waits/barriers around cooperative staging or reductions.

Discriminator: prove a specific dependency makes a wait/barrier redundant; remove only that synchronization point.

Pass signal: unchanged output and memory counters with lower wave cycles. Never remove synchronization based on timing alone.

### ISA-035 Resource-induced serialization

Claim: a specific live range or lowering artifact raises VGPR/LDS/scratch enough to reduce resident workgroups or create spills.

Discriminator: shorten that live range without changing arithmetic or memory traffic and compare binary resource metadata.

Pass signal: a resource threshold crossing plus lower cycles. Resource reduction without timing change is not a win.

## 11. Phase 4: implementation gate

An implementation is allowed only when Phase 3 proves one mechanism. The patch must:

- be expressed as a reusable lowering/kernel primitive, not a model-name condition;
- be selected by descriptor facts such as G, Hd, dtype, layout, staging, or target;
- preserve existing G4 and ctx128 behavior;
- retain QG2/S32 ownership unless evidence proves a new geometry;
- add no environment-variable production switch;
- preserve fail-loud behavior for unsupported routes;
- include a stable kernel/program identity.

One mechanism per candidate. Do not combine speculative changes.

## 12. Phase 5: validation ladder

### ISA-050 Compile and resource gate

Require:

- successful gfx1100 compilation;
- no scratch/spill regression;
- resource metadata recorded;
- expected instruction-class delta visible in disassembly.

### ISA-051 Numerical gate

Compare full attention output at ctx128, 512, 1024, and 4096 with deterministic inputs. Require finite outputs and error no worse than the accepted QG2 baseline.

### ISA-052 Isolated timing gate

Use three warmups and three samples at ctx512 and ctx4096. The candidate must improve ctx4096 materially and must not create a worse slope at the isolated tile.

### ISA-053 Model correctness gate

Require deterministic generated-token evidence across repetitions and equality with the accepted route's token evidence at both flash contexts. Confirm ctx128 uses the intended fallback.

### ISA-054 Final relative performance gate

Run tinygrad and llama.cpp in the same uncontaminated GPU session, three repetitions at ctx512 and ctx4096. Pass only on decay parity or a tinygrad ctx4096 win with retained ctx512 lead.

### ISA-055 Regression gate

Only after ISA-054 passes:

- run 8B ctx512/ctx4096 decode;
- confirm G4 route and program identities are unchanged;
- confirm 14B ctx128 succeeds;
- confirm no prefill route or artifact changed;
- run targeted unit tests for descriptor identity and validation.

## 13. Phase 6: promotion and cleanup

Promotion evidence must include:

- source and binary identities;
- numerical artifact;
- static cadence comparison;
- raw counter comparison;
- isolated timing;
- same-session tinygrad/llama result;
- 8B and ctx128 regression result;
- a ledger of every refuted candidate.

Delete:

- one-off capture scripts;
- dead PMC/timing probes;
- rejected generated binaries not referenced by the ledger;
- environment switches used only for diagnosis.

Retain only reusable binary-capture/cadence tooling that has tests and a documented owner.

## 14. Stop rules

Stop and write a bounded result when any condition holds:

1. no comparable llama code object can be captured after positive-control repair;
2. static and dynamic evidence cannot distinguish the open mechanisms;
3. every proven mechanism requires a compiler primitive outside this campaign's files;
4. the best correct candidate fails the relative acceptance gate;
5. GPU health, profiler state, or source identity becomes contaminated.

A stop result must name the exact missing compiler/runtime primitive and must not recommend repeating closed geometry tests.

## 15. Task ledger

| ID | Deliverable | GPU | Completion evidence |
|---|---|---:|---|
| ISA-000 | tool/version inventory | no | paths and versions |
| ISA-001 | three positive controls | yes | named kernels and nonzero counters |
| ISA-010 | bound tinygrad HSACO bundle | compile | binary/source/resource identity |
| ISA-011 | bound llama HSACO bundle | yes | binary/source/resource identity |
| ISA-012 | semantic normalization | no | per-head/per-token work map |
| ISA-020 | instruction taxonomy | no | static counts and loop bodies |
| ISA-021 | ordered stage cadence | no | comparable stage table |
| ISA-022 | dynamic counter join | yes | raw and normalized-by-work rows |
| ISA-030..035 | one-property discriminators | mixed | accepted/refuted mechanism ledger |
| ISA-050..052 | compile/numerics/micro timing | yes | candidate artifacts |
| ISA-053..054 | model correctness and relative gate | yes | paired authority artifacts |
| ISA-055 | 8B/ctx128 regression | yes | only after acceptance |
| ISA-060 | cleanup and promotion handoff | no | no dead probes; self-contained result |

## 16. Initial execution decision

Begin with ISA-000 through ISA-012. Do not edit the kernel until the exact tinygrad and llama binaries are captured and the semantic normalization identifies a comparable loop/stage boundary.

# Execution result (2026-07-27)

## Verdict

The alternative campaign gate passed narrowly: tinygrad retained its ctx512 lead and matched/passed llama.cpp at ctx4096 within the three-run same-session uncertainty. The ctx512-to-ctx4096 slope itself did not become llama-flat, so this is a depth-parity result, not a claim that the decay mechanism was fully eliminated.

Same-session authority:

| provider | ctx512 | ctx4096 | decay |
|---|---:|---:|---:|
| llama.cpp mean | 66.266 tok/s | 62.547 tok/s | 5.61% |
| tinygrad descriptor median | 69.506 tok/s | 62.419 tok/s | 10.20% |
| tinygrad descriptor mean | 69.539 tok/s | 62.640 tok/s | 9.92% |

The mean comparison passes llama.cpp at depth by 0.15%; the median comparison trails by 0.20%. Treat the result as parity within uncertainty, not a durable large-margin win.

## Proven mechanism

The original QG2 G=5 tile scalarized cooperative K/V staging and kept many loads live:

- 64 `global_load_b32` instructions in the static code object.
- 154 VGPR, 31 SGPR, 8192 bytes LDS, no spills.

Width-4 cooperative staging changed the load cadence:

- 16 `global_load_b64` operations in the prototype code object.
- 80 VGPR, 18 SGPR, 8192 bytes LDS, no spills.
- Descriptor-owned no-env capture at its fixed compile geometry: 78 VGPR, 18 SGPR, 8192 bytes LDS, no scratch or spills.

The production implementation carries `stage_width=4` only on the G=5 route descriptor. `KernelInfo.coalesced_loads` requests the generic AMD coalesced-load pass for that kernel; codegen no longer needs a process-wide environment switch for the shipped route. G=4 explicitly remains width 1.

## Closed discriminators

- Inline score reduction: numerically equivalent but ctx4096 tile time regressed from 0.1425 ms to 0.1649 ms, about 16%.
- Split-score arm: bit-identical and timing-identical; inert for this route.
- Fast-exp2 arm: bit-identical and timing-identical; inert for this route.
- Fine split sweep after vectorization: S=32 remained the clear optimum. S=30 was 0.1755 ms and S=34 was 0.1669 ms versus S=32 at 0.1515 ms in that sweep.
- Width 8: produced 128-bit loads and lower register use but changed numerics and was slower end-to-end than width 4.

## Regression gates

- 14B ctx128: 59.89 tok/s median, SDPA route, generated tokens identical.
- 14B ctx512: 69.51 tok/s median, flash route, generated tokens identical.
- 14B ctx4096: 62.42 tok/s median, flash route, generated tokens identical.
- 8B ctx128: 94.80 tok/s median, SDPA route, generated tokens identical.
- 8B ctx512: 114.19 tok/s median, flash route, generated tokens identical.
- 8B ctx4096: 103.07 tok/s median, flash route, generated tokens identical.
- Focused CPU/contracts: 34 passed on master.

Master promotion rerun: the first three-repetition ctx512 authority produced one divergent token sequence while the
other two matched the established hash. An immediate five-repetition rerun produced the established hash in all five
repetitions at 69.72 tok/s median. This non-reproduced anomaly is retained as residual correctness evidence rather
than silently relabelled clean; ctx4096 remained identical in the promotion run.

## Counter limitation

BoltBeam's decode counter surface is still an admission schema, not an executable decode collector. Its native PMC collector invokes the prefill-only trace script, and tinygrad graph PMC documents that only SQ busy cycles are currently valid while GRBM/GL2C/SQC reads are zero. No hardware-counter causal claim was made from that path.

## Evidence

- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/isa-taxonomy.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/qg2-coalesce4-acceptance.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/llama-paired-acceptance.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/descriptor-owned-14b.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/descriptor-owned-8b-regression.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/descriptor-owned-resource.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/master-promotion-14b.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/master-promotion-14b-ctx512-repeat.json`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/reduction-ab/`
- `/home/ubuntu/boltbeam-runs/14b-decode-isa-cadence-20260727/split-fine/`

## Remaining work

Do not describe the slope as fixed. The promoted primitive closes the practical ctx4096 gap, but tinygrad still decays about twice as much as llama.cpp because it begins substantially faster at ctx512. A future campaign should first build a real decode PMC collector; static ISA and the existing env arms are exhausted as discriminators.
