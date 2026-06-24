# q8 FFN dynamic scheduler observability result (2026-06-19)

Executed DSO-0 through DSO-5 from `q8-ffn-dynamic-scheduler-observability-scope-20260619.md`.

Verdict: **wait_scheduler_bound / do not reopen q8 native ownership**.

This does not change the q8 research finding: A4 still proves the q8 route works behind a research flag. It does close
the remaining "maybe the visible gap is a bounded load-shape tweak" question. The dynamic evidence says the local AMD
DSL variants are body-insensitive at this granularity, so this is broader AMD scheduling/work-decomposition/codegen
behavior, not a single primitive edit.

## Artifacts

Probe:

- `extra/q8_ffn_dynamic_scheduler_observability.py`

Artifacts:

- `bench/q8-ffn-dynamic-scheduler-observability/preflight.json`
- `bench/q8-ffn-dynamic-scheduler-observability/hcq_rows.json`
- `bench/q8-ffn-dynamic-scheduler-observability/resource_audit.json`
- `bench/q8-ffn-dynamic-scheduler-observability/variant_ladder.json`
- `bench/q8-ffn-dynamic-scheduler-observability/pmc_attempt.json`
- `bench/q8-ffn-dynamic-scheduler-observability/pmc_q8_gateup_full.json`
- `bench/q8-ffn-dynamic-scheduler-observability/result.json`

## DSO-0 / S0 refresh

S0 was rerun and preserved the same static conclusion:

| object | static instructions | dot4 | global loads | DS | waitcnt |
|---|---:|---:|---:|---:|---:|
| tinygrad AMD DSL/ASM | `218` | `16` | `22` | `10` | `17` |
| hipcc/LLD oracle | `336` | `16` | `11` | `7` | `20` |
| COMGR fused-C | `482` | `16` | `12` | `2` | `9` |

Static deltas still show load shape/address work (`+11` global loads, `+37` VALU), but not a missing dot4 or massive
instruction-count blowup.

## DSO-1 / DSO-2 HCQ and resources

The HCQ attribution rows captured real program metadata and warm device times for the two externally compiled controls:

| program | warm device ms | kernarg | LDS/group | private | wave32 |
|---|---:|---:|---:|---:|---:|
| hipcc/LLD `q8_mmvq_gateup` | `0.087` | `40` | `16` | `0` | yes |
| COMGR `q8_mmvq_gateup` | `0.122-0.123` | `40` | `16` | `0` | yes |

This confirms the known hierarchy under direct HCQ timing:

`hipcc/LLD` faster than `COMGR`, both faster than tinygrad AMD DSL/ASM.

Resource metadata does not explain the whole gap. The controls have the same kernarg size, same tiny LDS requirement,
no private segment, and wave32. The remaining difference is codegen/scheduling/resource encoding quality, not a large
resource allocation mismatch.

## DSO-3 variant ladder

The dynamic ladder is the decisive new evidence.

| variant | median ms | vs full ASM | static instructions | dot4 | global loads | waitcnt |
|---|---:|---:|---:|---:|---:|---:|
| full real-GGUF ASM authority | `0.166649` | `1.00x` | `218` | `16` | `22` | `17` |
| reduction-only | `0.153344` | `0.92x` | `51` | `0` | `0` | `8` |
| synthetic-dot | `0.150879` | `0.91x` | `69` | `16` | `0` | `8` |
| scalar load/wait only | `0.152562` | `0.92x` | `116` | `0` | `16` | `16` |
| grouped-wait load only | `0.151725` | `0.91x` | `96` | `0` | `16` | `9` |

Interpretation:

- Removing real Q4_K scale/min math does not move enough.
- Removing all global loads does not move enough.
- Removing all dot4 does not move enough.
- Reducing waits from `16` to `9` in the load-only variant does not move enough.
- A tiny reduction-only body over the same row/workgroup shape is still ~92% of full ASM time.

That is body-insensitive behavior. At this granularity, the cost follows the AMD DSL work decomposition / dispatch /
wave-reduction schedule much more than the Q4/q8 inner-loop body.

So the S0 visible `22` vs `11` global-load delta is real, but it is not sufficient to explain or recover the
`166us -> <=60us` target. A load-shape-only feature would not be a credible reopen gate.

## DSO-4 PMC attempt

The actual q8 ASM consumer ran under `PROFILE=1 PMC=1 SQTT=0`:

- subprocess return code: `0`;
- correctness still passed;
- median under profiling: `5.37ms`.

This proves the q8 path is profile-runnable with tinygrad's built-in AMD hooks, but the profiling overhead is too high
for the timing verdict. This run was used only as a capability check, not as a performance number.

SQTT was not required for the final classifier because DSO-3 already produced a decisive Level-3 result. If the project
wants instruction-issue proof later, SQTT/PMC can be scoped as a separate AMD scheduler project.

## Classifier

Final label: **`wait_scheduler_bound`**.

Reason:

- body-insensitive variant ladder: true;
- reduction-only remains `0.153ms` vs full `0.166ms`;
- grouped wait placement does not materially reduce the load-only variant;
- resource metadata does not show a simple occupancy/LDS/private-memory explanation;
- static instruction counts are not the limiting story.

Action: **do not reopen q8 native ownership**.

The next legitimate work is a project-level AMD scheduler/codegen effort:

- better work decomposition for one-row-per-workgroup MMVQ-style kernels;
- latency-aware load/wait/dot scheduling;
- vector/coalesced load selection as part of that scheduler, not as a standalone q8 fix;
- renderer/assembler support that can emit hipcc-quality schedules or import mature schedules.

## Consequence

For decode:

- A4 q8 remains a valid research artifact: W==D `1.051-1.063x`, dNLL `+0.002887`;
- native tinygrad q8 ownership stays closed;
- producer ownership should not be funded before the consumer scheduler wall is solved;
- the performance-primitive source of truth should classify this as project-level AMD scheduling/codegen, not open
  primitive search.
