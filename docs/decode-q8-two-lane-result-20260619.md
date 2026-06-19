# Decode q8 two-lane result - 2026-06-19

Purpose: execute `decode-q8-two-lane-scope-20260619.md`.

Artifacts:

- `extra/qk_decode_q8_two_lane_closeout.py`
- `bench/qk-decode-mmvq-large-project/q8_two_lane_closeout.json`

## Verdict

**BOTH_LANES_SCOPED_ARTIFACT_READY_NATIVE_PROJECT_LEVEL**.

## Lane 1 - Research Flag

Status: **PASS_RESEARCH_FLAG_READY**.

Route:

- flag: `Q8_FFN_HANDWRITTEN=1`;
- default: off;
- scope: Qwen3-8B Q4_K_M-style dense FFN, `dim=4096`, `hidden=12288`, Q4_K gate/up, gfx1100;
- runtime: tinygrad AMD HCQ / `AMDProgram`;
- no in-process HIP runtime.

Measured evidence:

| ctx | baseline tok/s | q8 tok/s | speedup |
|---:|---:|---:|---:|
| 128 | `79.5` | `84.5` | `1.063x` |
| 512 | `73.0` | `77.4` | `1.060x` |
| 1024 | `71.3` | `75.4` | `1.058x` |
| 4096 | `65.1` | `68.4` | `1.051x` |

Quality:

- baseline NLL: `2.855476`;
- q8 route NLL: `2.858363`;
- dNLL: `+0.002887`;
- gate: `<=0.01`.

Artifact boundary:

- producer hash: `dd119afa0ef41c8dbf5de6ec365f8c04fd3b7018553dee3cf179bdde99ae8682`;
- gate/up hash: `9d00b0723a6aa92d54f18e152678352d6b19d04ace9cbf605637c6abcf0287a5`;
- gate/up consumer: `93.54us`;
- producer + gate/up lifecycle: `115.24us`.

Decision: keep as a default-off research flag. Do not promote beyond research unless the external hipcc/LLD HSACO
dependency is explicitly accepted.

## Lane 2 - Native Transfer

Status: **PROJECT_LEVEL_NOT_BOUNDED_PATCH**.

Oracle:

- artifact lifecycle: `115.24us`;
- local speedup vs current gate/up: `1.46x`;
- graph route: PASS.

Current native failures:

- COMGR fused lifecycle: `177.72us`;
- AMD DSL consumer-only: `166.65us`;
- DSL capability verdict: `FAIL_A1_NO_BOUNDED_FEATURE`;
- no A2 candidate clears the `>=30us` start gate.

Required project capabilities:

- latency-aware AMD instruction scheduling;
- register allocation and live-range control for low-VGPR high-occupancy kernels;
- semantic waitcnt / `s_clause` / `s_delay_alu` placement;
- global load grouping as part of scheduling, not a standalone knob;
- staged reductions and post-barrier multi-output stores;
- SQTT/PMU attribution good enough to assign `>=30us` movement to a bounded feature.

Decision: do not start native transfer as a q8-specific patch. Start only if the project funds the broader AMD backend
scheduler effort, or if new attribution identifies one bounded `>=30us` feature.
