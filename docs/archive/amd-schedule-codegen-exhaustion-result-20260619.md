# AMD schedule/codegen exhaustion result (2026-06-19)

Executed SCE-0/SCE-1 from `amd-schedule-codegen-exhaustion-scope-20260619.md`.

Verdict: **PASS_MATRIX_BUILT**.

The AMD schedule/codegen question is now exhausted at the primitive-map level. No bounded native codegen feature is
identified. The remaining native work is a project-level AMD renderer/scheduler/register-allocation effort; the bounded
near-term path is artifact/policy/graph routing.

## Artifacts

Probe:

- `extra/amd_schedule_codegen_exhaustion.py`

Output:

- `bench/amd-schedule-codegen-exhaustion/oracle_matrix.json`

Inputs:

- q8 Route A contract/capability/PMU artifacts under `bench/q8-ffn-amd-scheduler-project/`
- Tensile TPE artifacts under `bench/qk-tensile-extraction/`

## Matrix result

| classification | count |
|---|---:|
| `expressible_now` | 1 |
| `project_level` | 7 |
| `not_worth_owning` | 1 |
| `artifact_only` | 1 |
| `bounded_extension` | 1 |
| `tooling_blocked` | 1 |

Feature verdicts:

| feature | classification | next action |
|---|---|---|
| special instruction selection | `expressible_now` | do not build as standalone |
| vector/global load shape | `project_level` | own only as part of scheduler/register layout |
| waitcnt placement | `project_level` | needs latency-aware scheduler |
| `s_clause` / `s_delay_alu` | `project_level` | no manual emission without semantic insertion rules |
| register allocation / live ranges | `project_level` | renderer/register allocator project |
| occupancy / VGPR / SGPR policy | `project_level` | bounded knob search closed |
| software pipelining | `project_level` | AMD renderer scheduler work |
| LDS staging / layout | `project_level` | needs staged-kernel/scheduler capability |
| reduction topology | `not_worth_owning` | do not reopen standalone |
| launch/kernarg contract | `artifact_only` | policy decision if artifacts are accepted |
| graph/rebind boundary | `bounded_extension` | finish only if accepting artifact route |
| attribution tooling | `tooling_blocked` | SQTT decode/counter attribution before stall-level claims |

## What this means

For q8 decode:

- the mature artifact route works in-model;
- native tinygrad emits the important math instruction (`v_dot4_i32_iu8`);
- the visible gap is not one missing primitive;
- PMU/SQTT capture works, but decoded attribution does not;
- no bounded `>=30us` A2 feature exists.

For prefill:

- extracted Tensile kernels are the concrete oracle;
- tinygrad already matches the macro tile and WMMA instruction class;
- the gap is the software-pipelined K-loop plus spill-free large-accumulator scheduling;
- POWN-1 already killed bounded knobs like more waves, bigger tiles, BK variants, and noLDS;
- in-model artifact route has already passed pp512 as policy-gated research evidence.

## Native codegen decision

Do **not** start a q8-specific native A2 or another bounded pure-tinygrad prefill sweep.

The native compiler path is valid only as a broader AMD backend project:

- latency-aware instruction scheduling;
- register allocation / live-range control;
- software-pipelined global->LDS->register K-loop scheduling;
- semantic waitcnt / `s_clause` / `s_delay_alu` placement;
- staged reductions and post-barrier multi-output stores;
- robust SQTT/PMU attribution if we want hardware-feedback-guided mutation.

That project may be worth doing, but it should be funded as a reusable backend capability, not as a single q8 or prefill
primitive edit.

## Bounded next path

The only bounded near-term path left by this matrix is not native codegen. It is:

1. artifact policy decision;
2. graph/rebind/fallback hardening for the extracted routes;
3. in-model measurement by phase and shape;
4. keep default off unless policy and portability are accepted.

If the artifact route is rejected, the honest resting point is:

- shipped decode / q8 artifact banked but off;
- PREFILL_V2 / pure tinygrad prefill as the no-dependency baseline;
- AMD schedule/codegen recorded as project-level future work.
