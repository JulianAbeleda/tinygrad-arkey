# AMD scheduler tooling/backend project scope - 2026-06-19

Purpose: turn "fund tooling or a broader AMD scheduler/resource project" into a bounded project with gates.

This is the next layer after N1:

- q8 native N1 found no bounded `>=30us` feature;
- PMU and SQTT capture are runnable, but RDNA3 HCQ SQTT decode is not usable;
- the remaining suspected movement is scheduler/resource behavior, not a single q8 primitive patch;
- prefill/Tensile points at the same class of missing capability: software-pipelined scheduling plus spill-free
  register allocation.

## Decision

Do **Track T first** unless the project explicitly accepts a multi-week backend/compiler investment without stronger
hardware attribution.

Track T is not performance work. It is the evidence layer needed to decide whether Track B has a precise first build.

## Non-Goals

- Do not write another q8 consumer variant before attribution names a feature.
- Do not restart bounded prefill WMMA knob search; POWN/CG already closed that.
- Do not build a general profiler clone.
- Do not route external artifacts by default.
- Do not call the backend project successful because a standalone microbench improves; it must move the owning primitive
  gate.

## Track T - Attribution Tooling

Goal: make RDNA3 HCQ kernels explainable enough to assign time/stall movement to scheduler/resource features.

### T0 - Evidence Inventory

Inputs:

- `bench/q8-ffn-amd-scheduler-project/pmu_sqtt_evidence.json`
- `bench/q8-ffn-amd-scheduler-project/n1_attribution.json`
- `bench/qk-hcq-attribution/result.json`
- `bench/qk-pmu-observability/result.json`

Deliverable:

- one machine-readable inventory of what exists: PMC blobs, SQTT blobs, program metadata, code hashes, launch geometry,
  event names, and decode failures.

Gate:

- can reproduce the current SQTT decode failure from a saved blob;
- can map each captured blob to program name and code-object hash.

Stop:

- if trace blobs cannot be replayed offline, fix capture persistence before touching the decoder.

### T1 - RDNA3 HCQ SQTT Decoder

Goal: decode tinygrad HCQ instruction-trace blobs enough to map PCs to AMD instructions.

Tasks:

- isolate the packet format that currently fails with `unknown cdna format word=0xf4080100`;
- add RDNA3/gfx1100 packet handling without breaking existing decoder behavior;
- map trace addresses back to the loaded HSACO/code object and disassembly;
- emit instruction timeline rows: PC, instruction mnemonic, packet/event type, wave/SE if available.

Gate:

- at least one q8 ASM run decodes `>=90%` of instruction events to PCs or instruction rows;
- decoded rows include the q8 kernel body, not only setup packets;
- repeated capture produces stable instruction-class histograms within noise.

Stop:

- if the SQTT format is not recoverable from local traces in a bounded pass, downgrade SQTT to capture-only and proceed
  to T2/T3 with aggregate PMCs and static disassembly.

### T2 - PMC Blob Decoder / Counter Attribution

Goal: turn tinygrad's captured PMC blobs into per-kernel counter summaries.

Tasks:

- parse `ProfilePMCEvent.blob` according to its `sample_layout`;
- summarize events already captured: `SQ_BUSY_CYCLES`, `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQC_LDS_*`, `GL2C_HIT`,
  `GL2C_MISS`, `GRBM_GUI_ACTIVE`;
- normalize by dispatch time and launch geometry;
- attach counter summaries to primitive rows.

Gate:

- q8 ASM run produces non-empty per-dispatch counter summaries;
- summaries distinguish memory-bound, wait/scheduler-bound, LDS-conflict, and low-occupancy cases without relying on
  guesswork;
- HIP control and HCQ capture agree on basic counter semantics for one comparable smoke kernel.

Stop:

- if blob layout is ambiguous, keep PMCs as Level-4 only for ROCm/HIP controls and use HCQ Level-3 attribution for
  tinygrad.

### T3 - Primitive Timeline Join

Goal: connect profiler evidence to model primitives, not anonymous kernel symbols.

Tasks:

- join program name, launch dims, code hash, kernarg size, graph/eager path, role name, model block, shape, and counter
  summary;
- handle TinyJit/HCQGraph replay with rebinding;
- output one row per primitive observation.

Gate:

- q8 route can report producer, gate/up consumer, and baseline role timings with matching code hashes;
- prefill route can report WMMA/Tensile role rows when enabled;
- no row claims PMU evidence when only HCQ Level-3 evidence exists.

Stop:

- if graph replay cannot be joined reliably, keep tooling eager-only and do not use it for in-model verdicts.

### T4 - Attribution Verdict

Goal: decide whether Track B gets a concrete first feature.

Output:

- `>=30us` q8 feature attribution, or;
- `>=15 TFLOPS` prefill feature attribution, or;
- no bounded feature; backend remains roadmap-only.

Required attribution labels:

- load-shape/coalescing;
- waitcnt placement;
- `s_clause` / `s_delay_alu`;
- register/live-range pressure;
- occupancy/resource policy;
- software-pipelined global->LDS->register loop;
- LDS/barrier layout;
- graph/runtime boundary.

Gate:

- start Track B only if one label clears its movement threshold, or if the project explicitly funds the whole reusable
  backend despite missing fine-grained attribution.

## Track B - AMD Backend Scheduler/Resource Project

Goal: teach tinygrad to generate the schedule classes currently provided by hipcc/LLD or Tensile.

This is not a q8-only project. It must serve at least two authority cases:

- q8 decode consumer: tinygrad ASM `166.649us` -> hipcc/LLD oracle `93.54us`;
- prefill GEMM: tinygrad WMMA plateau around `42-50 TFLOPS` -> Tensile oracle `~66-69 TFLOPS`.

### B0 - Oracle Suite

Deliverable:

- a stable oracle suite with q8 decode, prefill ffn_gate/up, prefill ffn_down, and one small smoke kernel.

Gate:

- every oracle has correctness, timing, disassembly, resource metadata, launch contract, and a tinygrad baseline;
- clock-controlled measurement is required for any in-model claim.

### B1 - Schedule Metadata IR

Goal: add an internal representation for schedule facts the renderer currently loses.

Minimum fields:

- instruction latency class;
- memory space and vector width;
- wait dependency group;
- barrier scope;
- live-range boundary;
- preferred issue cluster/order;
- prefetch stage;
- LDS buffer stage;
- register pressure budget.

Gate:

- metadata can be attached to a q8 ASM/codegen probe and a WMMA GEMM probe without changing semantics.

Stop:

- if metadata becomes kernel-specific annotations only, do not continue; that is hand-maintained assembly, not a backend
  capability.

### B2 - Wait/Schedule Emitter

Goal: renderer can intentionally place `s_waitcnt`, `s_clause`, and `s_delay_alu` from semantic dependency groups.

Gate:

- q8-shaped probe changes ISA in the intended places;
- correctness remains stable;
- movement is `>=15us` on q8 or contributes to a larger B4/B5 gate.

Stop:

- if manual wait movement stays around the known `0.837us` result, close this as standalone and keep only as part of
  software pipelining.

### B3 - Register Allocation / Live-Range Control

Goal: prevent the renderer from collapsing or spilling schedules that need many live accumulators/prefetch registers.

Gate:

- a WMMA more-accumulator probe avoids the prior spill cliff;
- q8 register/resource metadata changes VGPR/occupancy in a controlled way;
- no broad regression in existing AMD kernels.

Stop:

- if register control requires a full allocator rewrite with no primitive movement, return to Track T/tooling.

### B4 - Software-Pipelined K-Loop

Goal: emit the Tensile-class double-buffered global->LDS->register loop.

Required behavior:

- prologue loads tile `k+1`;
- steady state overlaps global load/LDS store for the next tile with WMMA on the current tile;
- two LDS buffers avoid aliasing current reads with next writes;
- `vmcnt` waits are deferred until the data is actually consumed;
- barriers are placed for correctness, not conservatively after every global load.

Gate:

- one prefill ffn_gate/up kernel reaches `>=60 TFLOPS` without external artifacts;
- disassembly shows non-byte-identical schedule vs current single-buffer kernel;
- no correctness regression.

Stop:

- if UOps still collapse to the old ordering, the feature belongs lower in the renderer/linearizer and B4 is not done.

### B5 - q8 MMVQ Scheduler Transfer

Goal: apply B1-B3 to the q8 consumer only after the backend can preserve schedule intent.

Gate:

- native q8 gate/up consumer reaches `<=75us` to continue;
- `<=60us` to call it hipcc-quality;
- if combined with producer, lifecycle must clear `<=129.2us`;
- W==D decode must show `>=3%` sustained and dNLL must remain within the accepted q8 gate.

Stop:

- if q8 stays `>100us`, native q8 ownership remains closed even if prefill backend work succeeds.

### B6 - Machine Search Layer

Goal: search over schedule metadata, not raw accidental UOp variants.

Search knobs:

- prefetch distance;
- LDS buffer count;
- wait grouping;
- vector load width;
- wave/tile decomposition;
- accumulator grouping;
- spill budget;
- occupancy/VGPR target.

Gate:

- search rediscovers one known oracle-like schedule on a small fixed shape;
- best candidate survives clock-controlled rerun and in-model/role gate.

Stop:

- if search mostly finds non-expressible schedules or clock artifacts, keep it as diagnostic tooling only.

## Completion States

| state | meaning |
|---|---|
| `T_PASS_BOUNDED_FEATURE` | tooling identifies a concrete first backend feature; start the matching B phase |
| `T_PASS_NO_FEATURE` | tooling works but no feature clears movement gate; keep backend roadmap-only |
| `B_PREFILL_PASS` | tinygrad owns Tensile-class fp16 GEMM schedule; dependency-free prefill can approach extracted oracle |
| `B_Q8_PASS` | tinygrad owns q8 consumer schedule; producer/lifecycle route can reopen |
| `B_PROJECT_FAIL` | backend work cannot preserve schedule intent or avoids no spill; stay with artifact routes |

## Expected Payoff If Everything Passes

If Track B succeeds for both authority cases:

- decode q8 artifact-level gain becomes tinygrad-native: about `1.05-1.06x` on current measured W==D, with a cleaner
  ownership story;
- broader decode integration may reopen only if the native route also solves the in-model MMVQ occupancy/BW loss;
- prefill GEMM can recover the Tensile-class `~60-69 TFLOPS` kernel level without an external artifact;
- the real project value is not one benchmark number, but a reusable AMD renderer scheduler that can preserve
  software-pipelined, low-spill, high-MLP schedules.

## Recommendation

Fund **T0-T4 first**. It is the smallest honest next step and it protects us from starting a compiler project on a vague
label like "scheduler." If T4 cannot assign a large bucket, only start Track B as an explicit backend investment shared
by decode and prefill, not as the next decode primitive.
