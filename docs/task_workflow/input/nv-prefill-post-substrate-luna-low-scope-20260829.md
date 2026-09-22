# NVIDIA pp512 post-substrate Luna-low execution scope

Date: 2026-08-29

Status: dispatcher-ready scope. This document authorizes measurement and
isolated implementation packets only as described below. It does not authorize
model composition unless the preceding packet records PASS.

## 1. Objective

Close the settled pp512 Qwen3-8B Q4_K_M prefill gap between tinygrad and
llama.cpp by assigning bounded, decision-free packets to Luna-low agents.

The current non-profiled wall authority is:

| runtime | median | minimum |
|---|---:|---:|
| llama.cpp | 35.019399 ms | 34.680367 ms |
| tinygrad composed | 67.235719 ms | 67.153915 ms |
| gap | 32.216320 ms | 32.473548 ms |

The current tinygrad HCQ region exposure is:

| region | exclusive device time |
|---|---:|
| gate + up | 25.831808 ms |
| down | 19.076352 ms |
| Q + O | 8.972608 ms |
| residual/RoPE/KV support | 4.224608 ms |
| Flash | 3.341056 ms |
| vocabulary | 2.911360 ms |
| K | 2.205536 ms |
| norm/conversion | 1.424128 ms |
| activation/multiply | 0.745696 ms |

The region table ranks work. It is not an additive cross-runtime debt table;
the llama regional trace is from an older instrumentation arm.

## 2. Frozen facts

- Hardware authority: RTX 5090, compute capability `sm_120`, driver 595.84.
- Workload: Qwen3-8B Q4_K_M, prompt length 512, greedy token 198.
- Current tinygrad route: unroll-4 plus 18 Q4-V roles, default-off.
- Current canonical census: 198 quantized producers/mains.
- Remaining FP16 overlays: 18 Q6-V and 36 down roles.
- Current unprofiled tinygrad launch authority: 1,449 launches, unknown zero.
- Current HCQ trace: 1,467 classified intervals, unknown zero.
- Current observer overhead: 0.578% median, below the 2% ceiling.
- E1 vocabulary is numerically correct but performance STOP at 16.449309 ms
  versus 3.871647 ms control. E2 is closed.
- The isolated Flash vector kernel compiles and passes its oracle, but it has no
  installed comparator or exact 36-call performance authority.
- The HCQ submission and Buffer observer substrate is implemented and passes
  its off/on self-test.
- Default-ready Q/O placement is NO_GO. The existing safe dependency cut is
  the only admitted queue policy until P0.
- No packet may book PROFILE=1 exposure as recovered unprofiled wall.

## 3. Source authorities

- `docs/task_workflow/output/nv-prefill-exact-gap-ledger-20260829.md`
- `docs/task_workflow/output/nv-prefill-current-hcq-ledger-20260829.md`
- `docs/task_workflow/output/nv-prefill-missing-substrate-closure-20260829.md`
- `docs/task_workflow/input/nv-prefill-luna-low-execution-scope-20260829.md`
- `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/`
- `docs/task_workflow/evidence/nv-prefill-current-hcq-ledger-20260829/`

## 4. Global agent contract

Every dispatched task must contain exactly one packet ID from this document.
An agent must not continue into the next packet after producing its verdict.

The dispatcher must provide these values before a GPU packet starts:

| field | required value |
|---|---|
| starting source authority | immutable source manifest and current revision identifier |
| model | exact local Qwen3-8B Q4_K_M path |
| prompt fixture | exact pp512 prompt/token fixture path |
| GPU | exact `NV` device identifier for the RTX 5090 |
| clocks/session | fixed-clock or recorded-session protocol used by both arms |
| baseline evidence | S0 result path |
| writable files | packet-specific allowlist copied from this document |
| evidence directory | new packet-specific directory; never reuse prior raw samples |

Global prohibitions:

- Do not edit `tinygrad/llm/model.py` before a packet explicitly authorizes composition.
- Do not change queue placement, Q/O cuts, arithmetic, and ownership in the same packet.
- Do not broaden shape admission beyond exact Qwen3-8B pp512 shapes.
- Do not replace full logits with top-1 output.
- Do not use a host `sum().item()` or another host synchronization as a control.
- Do not cite a single kernel win as model-scale recovery.
- Do not average away a failed correctness sample.
- Do not continue to a second design after STOP.
- Do not modify a closed lane to improve an unrelated packet.
- Do not use stale 69.49 ms or PROFILE=1 walls as the current baseline.

## 5. Shared correctness and timing protocol

### G0 isolated numerical gate

- Use immutable real-shape inputs and canonical packed weights.
- Check complete output, not a checksum-only proxy.
- Require finite outputs, zero unwritten sentinels, and read-only input hashes.
- Use the lane's frozen tolerance and report maximum absolute, mean absolute,
  relative L2, and argmax/token where applicable.
- Any tolerance or sentinel failure is STOP before timing.

### G1 population gate

- Execute the exact live population: 18 down roles, 72 gate/up roles, or 36
  Flash calls as specified by the lane.
- State expected program, producer, overlay, copy, and workspace counts before
  the run.
- Require observed counts to equal the prediction.
- Unexpected materialization, copy, allocation, or fallback is STOP.

### G2 model gate

- Run full logits and greedy token on the exact pp512 fixture.
- Require finite logits, declared allclose tolerance, and token 198.
- Record complete graph census and wall samples.

### G3 recurrent gate

- Run the existing deep-20 recurrent protocol.
- Require every cycle finite, token/state correct, and graph census stable.
- Any recurrent drift is STOP even if one-shot wall improves.

### Matched R9 protocol

- Use three fresh-process arms in order `control_0`, `candidate_1`, `control_2`.
- Each arm performs one unrecorded warmup and nine synchronized measured rounds.
- Retain all nine samples, minimum, median, median absolute deviation, and hashes.
- The dispatcher records clocks, driver, device, environment, source manifest,
  graph digest, and program census for every arm.
- A performance PASS requires candidate minimum and median below both controls,
  unless a packet declares a stronger rule.
- Define `control_drift` as the absolute difference between control medians.
- Define `noise` as the maximum of `control_drift`, three times the larger
  control MAD, and 10 microseconds.
- A component delta does not exceed noise unless its absolute median exposure
  is strictly greater than `noise`.

### Evidence schema

Every result JSON must contain:

| key | requirement |
|---|---|
| `schema` | packet-specific versioned string |
| `packet` | exact packet ID |
| `status` | `PASS`, `STOP`, or `BLOCKED` |
| `authority` | source, model, prompt, GPU, driver, clocks, environment hashes |
| `correctness` | tolerances, error metrics, hashes, token, sentinels |
| `census` | predicted and observed program/role/copy/workspace counts |
| `samples` | raw warmup-excluded samples for every arm |
| `wall` | minimum, median, MAD, control drift, noise |
| `observer` | enabled capabilities, overhead, perturbation status |
| `decision` | one admitted mechanism or explicit reason for STOP |
| `next_packet` | exactly one packet ID or `null` |

## 6. Dependency graph

```text
S0 authority freeze
 +--> D0.1 --> D0.2 --> D0.3 --> D0.4 --> D0.5 --> D1.1 --> D1.2 --> D2
 +--> B0.1 --> B0.2 --> B0.3 --> B0.4 --> B1.1 --> B1.2
 +--> C0.1 --> C0.2 --> C0.3 --> C1.1 --> C1.2 --> C2
 +--> F1.1 --> F1.2 --> F1.3 --> F1.4 --> F2

All B/C/D/F decisions recorded
 +--> H0.1 --> H0.2

Accepted composed wins, serialized in measured-value order
 +--> P0 --> P1 --> P2
```

`D0.1`, `B0.1`, `C0.1`, and `F1.1` may run concurrently because they are
read-only or write unique harness files. Tasks within one lane are serialized.
Composition tasks are always serialized.

## 7. Packet S0: reproducible authority freeze

Objective: produce a source- and environment-pinned wall authority before new
performance claims.

Allowed writes:

- `extra/llm_research/prefill/nv_prefill_post_substrate_authority.py`
- `docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829/`
- `docs/task_workflow/output/nv-prefill-post-substrate-authority-20260829.md`

Required implementation:

- Hash every source file used by the current route, observer hooks, and runner.
- Record the exact model and prompt hashes without copying model data.
- Run fresh llama.cpp settled R9 and tinygrad composed R9 in the same session.
- Run tinygrad correctness, graph census, and deep-20 replay.
- Record observer-off wall; do not enable PROFILE for wall authority.

PASS rule:

- Both runtimes produce the expected token and finite outputs.
- tinygrad census is the declared current census with unknown zero.
- tinygrad median is within 1% of 67.235719 ms or the result explicitly becomes
  the new authority after two confirming R9 brackets.
- The source manifest is complete and immutable.

STOP rule:

- Correctness, census, clock, or session mismatch.
- More than 1% unexplained baseline drift.

Output schema: `tinygrad.nv_prefill_post_substrate_authority.v1`.

## 8. D lane: Q6-down boundary attribution and lifecycle

### D0.1 boundary ABI and fixture freeze

Objective: turn the four D0 boundaries into an executable, unambiguous ABI.

Reads:

- `extra/llm_research/prefill/nv_compiler_q6k_imma_gate.py`
- `extra/llm_research/prefill/nv_compiler_q6k_model_arm.py`
- `extra/llm_research/prefill/nv_compiler_q6k_pp512_binding.py`
- `extra/llm_research/prefill/nv_q6down_graph_profile_observer.py`
- `tinygrad/device.py`
- `tinygrad/runtime/graph/hcq.py`

Allowed writes:

- `extra/llm_research/prefill/nv_q6down_boundary_contract.py`
- `docs/task_workflow/evidence/nv-prefill-q6down-boundary-contract-20260829/`
- `docs/task_workflow/output/nv-prefill-q6down-boundary-contract-20260829.md`

Required contract:

| boundary | Q6 measured work | FP16 control work |
|---|---|---|
| `producer` | compact-Q8 production for 18 roles | matching FP16 input publication boundary |
| `main` | producer plus Q6 main, with main service separately tagged | FP16 main over the same 18 roles |
| `publication` | main plus direct output publication | FP16 main plus identical publication contract |
| `residual` | publication plus rank-preserving residual epilogue | FP16 publication plus identical residual epilogue |

The contract must freeze input/output shapes, packed weight hashes, output
buffers, residual buffers, role order, marker identities, and expected graph
counts. It must state how output correctness is checked at every cut.

PASS rule: all 18 Q6-down roles map one-to-one to all four boundaries and the
FP16 controls without an inferred or unnamed operation.

### D0.2 forced-cut runner

Dependency: D0.1 PASS.

Allowed writes:

- `extra/llm_research/prefill/nv_q6down_boundary_r9.py`
- D0.2 evidence directory only.

CLI contract:

```text
--arm fp16|q6
--boundary producer|main|publication|residual
--temperature hot|rotated-cold
--rounds 9
--profile-jsonl PATH
--buffer-events-jsonl PATH
--out PATH
```

Required behavior:

- Construct exactly the 18 real Q6-down roles in stable layer order.
- Execute only the selected cumulative boundary.
- Synchronize before and after each measured sample.
- Install `BUFFER_OBSERVER` only around selected construction/submission work.
- Set `HCQ_SUBMISSION_OBSERVER_JSON` before graph creation.
- Rotate cold buffers without changing values, shapes, or role order.
- Preserve exact paired K16 correction semantics.
- Emit raw per-role and aggregate service for every sample.

Forbidden behavior:

- No arithmetic redesign.
- No Q6-V roles.
- No model composition.
- No host checksum inside the measured interval.
- No allocation or copy count inferred from absence of profile entries.

PASS rule: every CLI arm runs, clears G0/G1, and emits all required observer
records. This packet makes no performance decision.

### D0.3 observer attribution

Dependency: D0.2 PASS.

Allowed writes:

- `extra/llm_research/prefill/nv_q6down_boundary_analyze.py`
- D0.3 evidence directory only.

Required implementation:

- Join HCQ entries, Buffer events, cut markers, roles, and samples by invocation.
- Attribute actual allocation, copyin, copyout, graph-copy, and materialization
  nodes to the selected cut.
- Compute dependency-ready time from segment-local predecessor completion.
- Preserve full successor fanout; never select a successor only by count.
- Mark a capability `UNAVAILABLE` rather than writing zero when it was not observed.
- Prove 100% record coverage and unknown zero.

PASS rule: all events have one role, boundary, arm, temperature, and sample; no
event is multiply charged.

### D0.4 paired R9 execution

Dependency: D0.3 PASS.

Allowed writes: raw evidence under
`docs/task_workflow/evidence/nv-prefill-q6down-boundary-r9-20260829/` only.

Execution matrix: 2 arms x 4 boundaries x 2 temperatures x 3 matched brackets.
Each bracket contains nine measured samples after one warmup.

Required outputs:

- Service minimum, median, MAD, and all raw samples.
- Queue-ready and dependency-wait exposure.
- Allocation, copy, graph-copy, materialization, and workspace counts/bytes.
- Per-role distribution for all 18 roles.
- Complete correctness and census results.

### D0.5 dominant-boundary verdict

Dependency: D0.4 complete.

Allowed writes:

- `docs/task_workflow/output/nv-prefill-q6down-boundary-attribution-20260829.md`
- `docs/task_workflow/evidence/nv-prefill-q6down-boundary-r9-20260829/verdict.json`

Compute these incremental exposures:

| exposure | calculation |
|---|---|
| producer | Q6 producer service minus FP16 publication-boundary service |
| main | Q6 main-only service minus FP16 main-only service |
| publication | Q6 publication increment minus FP16 publication increment |
| residual | Q6 residual increment minus FP16 residual increment |

PASS requires exactly one positive exposure greater than noise and greater than
the second-largest exposure by more than noise. The verdict must name one
removable lifecycle mechanism supported by observer evidence.

STOP if no exposure independently clears noise, two exposures are inseparable,
or the dominant exposure is arithmetic rather than a removable lifecycle
mechanism. STOP leaves all Q6-down roles FP16.

### D1.1 one lifecycle implementation

Dependency: D0.5 PASS naming exactly one mechanism.

Writable files: dispatcher supplies the minimal allowlist named by D0.5. The
allowlist must not include queue policy or unrelated kernels.

Allowed transformations by verdict:

| D0 mechanism | only authorized D1 transformation |
|---|---|
| producer publication | fold or share the producer without changing Q8 values |
| output publication | remove the observed copy/materialization while preserving output ownership |
| residual epilogue | absorb exactly the rank-preserving residual operation |
| allocation/cold setup | reuse immutable compiled assets or per-capture buffers without aliasing |

Run G0 and the complete D0 matrix. Preserve paired K16 corrections exactly.

PASS requires complete correctness and a candidate minimum and median below
both controls for hot and rotated-cold populations.

### D1.2 lifecycle population verdict

Dependency: D1.1 PASS.

Run the exact 18-role lifecycle and write
`nv-prefill-q6down-lifecycle-primitive-20260829.md`. No model integration.

PASS authorizes D2. STOP closes Q6-down.

### D2 composition

Dependency: D1.2 PASS and dispatcher serialization lock.

Integrate exactly 18 Q6-down roles. Run G1, G2, G3, and matched whole-model R9.
State the incoming census and predicted delta before execution.

## 9. B lane: gate/up issue and latency hiding

### B0.1 executed-path locator

Objective: freeze exact compiler functions, generated program identities,
register/resource baseline, and one real gate fixture before editing.

Reads:

- `extra/llm_research/prefill/nv_prefill_gateup_ncu_bridge.py`
- `extra/llm_research/prefill/nv_prefill_gateup_unroll_discriminator.py`
- `extra/llm_research/prefill/nv_compiler_q4k_gkqo_model_arm.py`
- current compiler-owned K64 route files identified by source manifest

Allowed writes:

- `docs/task_workflow/evidence/nv-prefill-gateup-schedule-locator-20260829/`
- `docs/task_workflow/output/nv-prefill-gateup-schedule-locator-20260829.md`

Required output:

- Exact source function and generated program identity for the 72 roles.
- Baseline PTX/cubin hash and launch geometry.
- Registers, spills, local memory, occupancy, tensor duty, eligible warps,
  long scoreboard, issue rate, and instruction count.
- Exact mutation specification for the three B0 variants.

No source edit is allowed.

### B0.2 three default-off schedule variants

Dependency: B0.1 PASS.

The dispatcher copies B0.1's exact compiler file into the writable allowlist.
Only that file and a new isolated runner may be changed.

Implement exactly:

| variant | transformation |
|---|---|
| `fragment_distance` | reorder existing fragment loads to increase load-to-use distance |
| `metadata_distance` | move only scale/min metadata loads relative to fragment loads |
| `double_buffer` | one register-safe alternating fragment buffer with unchanged K64 arithmetic |

Each variant must have an independent default-off flag and stable program
identity. Do not change tile, CTA ownership, IMMA count, correction arithmetic,
queue placement, cp.async, TMA, or fusion.

### B0.3 isolated correctness and counters

Dependency: B0.2 complete.

Run G0 and matched NCU collection on one real gate role for control and all
three variants.

Immediate rejection conditions:

- Any local-memory spill.
- Lower theoretical or achieved occupancy.
- Different IMMA count.
- Higher instruction count without a service win above noise.
- Incorrect packed weights, output, or sentinels.

A variant remains eligible only if tensor duty increases by at least two
percentage points or by more than counter noise, long-scoreboard exposure falls,
and service improves beyond noise.

### B0.4 exact 72-role discriminator

Dependency: B0.3 with at least one eligible variant.

Run matched R9 over the exact 72 gate/up roles for each eligible variant.

PASS names exactly one variant whose minimum and median beat both controls and
whose counters move in the llama direction. Ties within noise are STOP.

Write `nv-prefill-gateup-schedule-discriminator-20260829.md`.

### B1.1 selected schedule implementation

Dependency: B0.4 PASS.

Remove experimental branching from the selected generated route while keeping
a default-off rollback control. Run G0, G1, and exact 72-role R9.

### B1.2 model composition

Dependency: B1.1 PASS and dispatcher serialization lock.

Run G2, G3, and whole-model matched R9. PASS requires a net wall reduction,
stable census, and no queue or recurrent regression.

## 10. C lane: Q4-down numerical correction

### C0.1 immutable localization fixture

Objective: identify the first numerical divergence responsible for the existing
`max_abs=2.695646` failure.

Reads:

- existing Q4-down matched A/B evidence and primitive
- canonical Q4_K dequantization authority
- exact 18 Q4-down packed weights and real activations

Allowed writes:

- `extra/llm_research/prefill/nv_q4down_numerical_localize.py`
- `docs/task_workflow/evidence/nv-prefill-q4down-localize-20260829/`

Freeze per-role inputs, packed-weight hashes, FP32/FP16 references, residuals,
sentinels, tolerance, and expected output shapes.

### C0.2 staged independent oracle

Dependency: C0.1 PASS.

Compare these stages independently for all 18 roles:

| stage | required comparison |
|---|---|
| input | live activation versus frozen activation |
| producer | packet bytes, lane order, group scales, sums, rounding |
| weight | independent Q4_K block dequantization versus packed consumer decode |
| dot | independent dequantized dot versus consumer pre-epilogue output |
| correction | scale/min and paired correction terms separately |
| epilogue | publication and residual output versus reference |

The script must stop at the first divergent stage and emit the exact indices,
bytes, expected values, and observed values. It must not propose a performance
change.

### C0.3 one-root-cause verdict

Dependency: C0.2 complete.

PASS names exactly one root cause and one minimal correction. STOP the Q4-down
lane if divergence cannot be localized uniquely.

Write `nv-prefill-q4down-numerical-localization-20260829.md`.

### C1.1 corrected isolated primitive

Dependency: C0.3 PASS.

The dispatcher supplies an allowlist containing only the function named by
C0.3 and a new isolated harness. Implement only the numerical correction. Do
not change geometry, scheduling, queue policy, or composition.

Run G0 on all 18 roles. Require the frozen allclose class, finite/nonzero
outputs, read-only packed weights, and matching token contribution.

### C1.2 matched lifecycle verdict

Dependency: C1.1 PASS.

Run FP16 control and corrected Q4-down candidate using the matched R9 protocol.
PASS requires candidate minimum and median below both controls. Otherwise STOP.

### C2 composition

Dependency: C1.2 PASS and dispatcher serialization lock.

Integrate exactly 18 Q4-down roles. Run G1, G2, G3, and whole-model R9.

## 11. F lane: Flash installed comparator and population gate

### F1.1 live-shape comparator extraction

Objective: create an apples-to-apples comparator between the installed Flash
route and the isolated vector kernel.

Reads:

- `extra/llm_research/prefill/nv_flash_vkv_primitive.py`
- `extra/llm_research/prefill/nv_prefill_flash_program_extract.py`
- `extra/llm_research/prefill/nv_prefill_flash_oracle.py`
- installed `nv_sm120_q16_grid_hd128_loop_attention` route

Allowed writes:

- `extra/llm_research/prefill/nv_flash_f1_population_r9.py`
- `docs/task_workflow/evidence/nv-prefill-flash-f1-population-20260829/`

Capture a manifest for exactly 36 live calls containing layer, Q head, KV head,
causal length, shapes, strides, dtypes, buffer hashes, output shape, installed
program identity, and candidate identity.

The comparator must execute the same logical call in both arms. It must not
relaunch a full-T512 kernel for an installed graph-owned slice.

### F1.2 full-output and 36-call R9

Dependency: F1.1 PASS.

Run G0 for every captured call, then the matched R9 protocol for the exact
36-call population. Retain every full output and read-only Q/K/V hash outside
the measured interval.

Required census: exactly 36 installed calls versus exactly 36 candidate calls,
with no global partial buffers or support launches unless declared by F0.

### F1.3 resource-counter harness

Dependency: F1.1 PASS. May run concurrently with F1.2 using a separate evidence
directory and no source edits.

Produce a directly launchable cubin/binary and stable launch selector for NCU.
Collect registers, spills, local memory, shared memory, occupancy, DRAM bytes,
tensor/ALU utilization, and service time. If NCU reports no kernels, fix the
launch selector or isolated process; do not mark counters zero.

Resource gate:

- Zero spills.
- No undeclared local memory.
- Shared memory at or below 32768 bytes unless F0 explicitly allows more.
- Register and occupancy result must fit the exact `sm_120` launch population.

### F1.4 verdict

Dependency: F1.2 and F1.3 complete.

PASS requires full correctness, complete coverage, resource-gate PASS, and
candidate 36-call minimum and median below both installed controls. A per-call
or population tie within noise is STOP.

Write `nv-prefill-flash-vector-primitive-20260829.md` and a versioned verdict
JSON. PASS authorizes F2. STOP retains installed Flash and closes this topology.

### F2 composition

Dependency: F1.4 PASS and dispatcher serialization lock.

Integrate exactly 36 calls, regenerate the Q/O dependency cut, then run G1,
G2, G3, and whole-model R9. Any recurrent failure is STOP.

## 12. Closed E lane

No E task may be dispatched.

The corrected many-row vocabulary primitive passes full-logit correctness but
is about 4.25 times slower than control. E2 composition is unauthorized. A new
vocabulary design requires a new discriminator scope, not continuation of E1.

## 13. H lane: support semantic census

### H0.1 post-decision capture

Dependency: B0/C0/D0/F1 decisions recorded and all accepted implementations
frozen.

Allowed writes:

- `extra/llm_research/prefill/nv_prefill_support_semantic_census.py`
- `docs/task_workflow/evidence/nv-prefill-support-semantic-census-20260829/`

Map 100% of remaining support intervals to semantic family, source operation,
layer, bytes, predecessors, successors, queue, and llama counterpart.

Required families: residual, RoPE, KV transport, normalization/conversion,
activation/multiply, Q8 producer, graph copy, host transfer, materialization,
allocation/setup, vocabulary tail, and unknown.

PASS requires unknown zero and exact interval-union closure.

### H0.2 ranked support verdict

Dependency: H0.1 PASS.

Rank only independently measured unprofiled exposure above noise. Name at most
one next support mechanism. If none clears noise, STOP support optimization.

Write `nv-prefill-support-semantic-ledger-20260829.md`.

## 14. Serialized integration packets

### P0 dependency-identity queue policy

Dependency: all accepted B/C/D/F implementations composed and frozen.

Test default-ready, minimal-cut, conservative-cut, primary-only, and one-queue
controls against the regenerated graph. The fastest G3-correct arm is the only
admitted policy. Do not change kernel arithmetic.

### P1 binding and admission hardening

Dependency: P0 PASS.

Remove mutable device-global cursors, separate immutable compiled assets from
per-model/per-capture buffers, prove two models and two captures cannot alias,
and restrict packed-dtype ownership to the actual operands. Preserve exact
Qwen3-8B pp512 admission.

### P2 final parity ledger

Dependency: P1 PASS.

Run fresh llama.cpp and tinygrad R9, low-perturbation accounting, exact census,
full logits, token, deep-20 replay, memory footprint, and independent evidence
review. Report the remaining gap without extrapolation and recommend exactly
one of `PROMOTE_DEFAULT_OFF`, `CONTINUE`, or `STOP`.

## 15. File ownership and concurrency

| lane | shared files frozen during isolated work | unique writable area |
|---|---|---|
| D | `tinygrad/device.py`, `tinygrad/runtime/graph/hcq.py`, model integration | `nv_q6down_boundary_*`, D evidence |
| B | queue/runtime/model files | one locator-named compiler function, B harness/evidence |
| C | queue/runtime/model files | one locator-named Q4 function, C harness/evidence |
| F | installed Flash and model integration | F comparator/harness/evidence |
| H | all kernels and model | H census/evidence only |
| P | all isolated harnesses | one serialized integration allowlist at a time |

Agents may read shared frozen files. They may not edit them unless the packet
explicitly moves one into its writable allowlist. Two agents may never edit the
same file concurrently.

## 16. Dispatcher handoff template

```text
PACKET: <one exact ID>
OBJECTIVE: <copied verbatim from scope>
DEPENDENCY: <PASS artifact path>
STARTING AUTHORITY: <S0 manifest and revision>
MODEL: <exact path and hash>
PROMPT: <exact path and hash>
GPU/CLOCK SESSION: <exact values>
READS: <closed list>
WRITES: <closed list>
COMMANDS: <exact CLI invocations>
EXPECTED CENSUS: <numbers before execution>
CORRECTNESS GATE: <exact tolerance and sentinels>
TIMING GATE: <matched R9 rule>
OUTPUTS: <JSON and verdict paths>
FORBIDDEN: <copied packet prohibitions>
STOP CONDITION: <copied verbatim>
```

An assignment missing any template field is not ready for a Luna-low agent.

## 17. Completion definition

This campaign is complete when one of these conditions holds:

- Accepted mechanisms pass serialized composition, P0, P1, and P2, producing a
  new exact llama/tinygrad ledger.
- Every B/C/D/F/H lane records STOP with executable evidence, leaving the
  current 67.235719 ms default-off route as the retained result.

Neither condition permits extrapolated recovery or an unmeasured composition.
