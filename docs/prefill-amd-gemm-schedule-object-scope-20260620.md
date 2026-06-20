# Prefill AMD GEMM Schedule Object — Scope

Date: 2026-06-20

## Verdict

`SCOPED_AMD_GEMM_SCHEDULE_OBJECT_STRUCTURAL_SCAFFOLD_LANDED`

The audit is sufficient for a **non-bitexact native schedule-object design**, so this pass turns the selected
rocBLAS ffn_gate/up Tensile contract into a **first-class AMD GEMM schedule object** plus a structural probe.
It does **not** build a kernel, does **not** change any default behavior or production routing, and makes
**no performance claim**. It is the "the schedule object exists and its structural contract holds" gate the
transfer table named as the prerequisite before any timing or BEAM/search.

The missing surface was never the WMMA atom and never "add LDS." It is a Tensile-class GEMM schedule object:
`global_load → LDS store → wait/barrier → LDS read → WMMA consume → buffer swap across K`, with shape,
resource, and dependency policy as first-class fields. This scope makes that object representable and
verifiable; the still-heuristic K-loop schedule and the lowering-to-ISA remain explicitly open.

## Deliverables in this pass

| artifact | role |
|---|---|
| `tinygrad/renderer/amd/schedule.py` (appended, **unwired**) | the first-class `AMDGemmScheduleObject` and its component dataclasses + structural gate |
| `extra/qk_amd_gemm_schedule_object_probe.py` | instantiates the selected ffn_gate/up object from extracted JSON and verifies the structural contract only |
| `bench/amd-broad-backend-roadmap/amd_gemm_schedule_object_structural_result.json` | emitted structural result (28/28 checks pass; `bench/**` gitignored, reproducible) |

Run:

```bash
PYTHONPATH=. python3 extra/qk_amd_gemm_schedule_object_probe.py
```

The probe reads three already-extracted artifacts and reconstructs nothing it cannot source:
`bench/qk-tensile-extraction/ffn_gate_up_contract.json` (resources/shape/launch),
`bench/qk-tensile-extraction/ffn_gate_up_schedule_template.json` (decoded `.dat` solution row),
`bench/amd-broad-backend-roadmap/bb5a10_tensile_layout_audit_result.json` (disasm instruction counts +
handoff inference).

## Design row 1 — Selected shape and layout contract

`AMDGemmShapeContract`. Source: `ffn_gate_up_contract.json` + decoded `.dat` solution row `52/290`.

| field | value |
|---|---|
| role | `ffn_gate/up` |
| M × N × K | `512 × 12288 × 4096` (flops `51,539,607,552`) |
| dtype | fp16 in, fp32 accumulate (`HHS_BH`) |
| macroTile (M, N, unroll) | `128 × 128 × 16` |
| threadTile | `4 × 64` |
| workGroup | `32 × 4 × 1` = 128 threads |
| depthU | `16` |
| workGroupMapping | `8` (WGM8) |
| launch grid / workgroup | `[512, 96, 1]` / `[128, 1, 1]` |

Structural checks: shape positive, macroTile 3-D, workgroup threads match launch (128), `K % depthU == 0`,
grid×tile covers M and N.

## Design row 2 — LDS layout object (A0/B0/A1/B1 bases/spans/padding)

`AMDGemmLDSLayout` of `AMDGemmLDSRegion`. Non-bitexact, reconciled from Tensile `getLdsNumElements` and the
selected offsets (`prefill-tensile-lds-tile-map-sketch-20260620.md`). fp16 = 2 bytes; B carries
`LdsPadB = 8` elems per `128`-byte block (256 pad elems = 512 pad bytes); PGR1 second buffer base is the
power-of-two ceil of `(A+B)` elems (`8192` elems = `16384` bytes).

| region | operand | slot | byte base | byte span | pad bytes | role |
|---|---|---:|---:|---:|---:|---|
| A0 | A | 0 | `0` | `4096` | `0` | current/prefetch A tile (lower buffer) |
| B0 | B | 0 | `4096` | `4608` | `512` | current/prefetch B tile (lower buffer, padded) |
| — gap | — | — | `8704` | `7680` | — | PGR1 power-of-two second-buffer alignment gap |
| A1 | A | 1 | `16384` | `4096` | `0` | alternate A tile (PGR1 second buffer) |
| B1 | B | 1 | `20480` | `4608` | `512` | alternate B tile (PGR1 second buffer, padded) |

`sum(spans) + gap = 17408 + 7680 = 25088 = group_segment_fixed_size`. Structural checks: nonzero LDS,
double-buffer present (`{A,B}×{0,1}`), alias-safe (no region overlap), all regions within total, spans+gap
== total, gap non-negative.

## Design row 3 — Pipeline stages

`AMDGemmPipelineStage[]`, ordered, matching the named `GEMM_PIPELINE_STAGES` contract. One prologue plus a
steady DepthU=16 K iteration; `isa_evidence` ties each stage to the audited opcode that realizes it.

| order | stage | phase | op_class | operand | slot | produces_for | wait | ISA evidence |
|---:|---|---|---|---|---:|---|---|---|
| 0 | global_load_A | prologue | global_load | A | 0 | lds_store_A | vmcnt | `buffer_load_b64` |
| 1 | global_load_B | prologue | global_load | B | 0 | lds_store_B | vmcnt | `buffer_load_b64` |
| 2 | wait_global_before_lds | prologue | wait | — | — | lds_store_A | vmcnt | `s_waitcnt` |
| 3 | lds_store_A | prologue | lds_store | A | 0 | barrier | lgkmcnt | `ds_store_b64` |
| 4 | lds_store_B | prologue | lds_store | B | 0 | barrier | lgkmcnt | `ds_store_b64` |
| 5 | barrier_after_lds_store | prologue | barrier | — | — | lds_read_A | barrier | `s_barrier` |
| 6 | lds_read_A | steady | lds_load | A | 0 | wmma_consume | lgkmcnt | `ds_load_b128` |
| 7 | lds_read_B | steady | lds_load | B | 0 | wmma_consume | lgkmcnt | `ds_load_b128` |
| 8 | wait_lds_before_wmma | steady | wait | — | — | wmma_consume | lgkmcnt | `s_waitcnt` |
| 9 | wmma_consume | steady | wmma | — | — | buffer_swap | wmma_dependency | `v_wmma` |
| 10 | store_output | epilogue | global_store | — | — | — | vscnt | — |
| 11 | buffer_swap | steady | swap | — | — | global_load_A | — | — |

Structural checks: all named stages present, order monotonic, prologue+steady phases present, a global_load
precedes the first wmma, a buffer_swap exists.

## Design row 4 — Resource gate

`AMDGemmResourceGate`. Source: `ffn_gate_up_contract.json`.

| field | target | actual (selected) | gate |
|---|---:|---:|---|
| LDS bytes | `25088` | `25088` | `0 < actual ≤ target` |
| private/scratch | `0` required | `0` | `actual == required == 0` |
| VGPR budget | — | `256` | `0 < budget ≤ 256` |
| SGPR budget | — | `58` | `0 < budget ≤ 128` |

The intent (carried from the implementation spec): **reject a candidate before timing** if scratch/private
appears or the VGPR/SGPR/LDS envelope cannot preserve the authority occupancy.

## Design row 5 — Structural gates before any timing

Evaluated by `AMDGemmScheduleObject.structural_gate()` (28 checks; all pass for the selected contract). The
ISA subset is sourced from `bb5a10_tensile_layout_audit_result.json` (`buffer_load_b64=24`, `ds_store_b64=32`,
`ds_load_b128=40`, `v_wmma=80`, `s_waitcnt=545`, `s_barrier=6`; handoff `32/32` and `80/80`):

- nonzero LDS metadata (`lds.nonzero_lds`, `resource.lds_within_target`)
- visible global load (`isa.visible_global_load`)
- visible `ds_store` (`isa.visible_ds_store`)
- visible `ds_load_b128` (`isa.visible_ds_load_b128`)
- visible `v_wmma` (`isa.visible_v_wmma`)
- waits and barriers present (`isa.waits_present`, `isa.barriers_present`)
- WMMA operands fed from LDS-loaded VGPRs (`isa.wmma_fed_from_lds`, 80/80) and global→LDS handoff
  (`isa.global_to_lds_handoff`, 32/32)

These gates exist so that no candidate is ever timed until it is structurally a Tensile-class LDS-staged
pipeline — the failure mode of the prior slow native LDS macro family was to time first and discover the
wrong family later.

## Explicit blocked / unknown rows

Carried verbatim into the object's `blocked_unknown` and the probe result:

- **non-bitexact LDS layout** — per-element A/B coordinate map not reconstructed (region sketch only); the
  disasm immediates are not full logical addresses (Tensile carries A/B base in address VGPRs).
- **exact K-loop / buffer-swap schedule still heuristic** — segmentation is first-WMMA/last-WMMA, not a
  symbolic loop; the loop-carried buffer swap points are inferred, not labeled.
- **source-level per-element tile map not fully reconstructed** — address-VGPR base carry not replayed.
- **bank/padding rationale beyond `LdsPadB=8`/128 B not replayed** from the Tensile generator.
- **no lowering to ISA and no performance claim** — this is the structural contract only; the `≥60 TFLOPS`
  authority gate is a separate, later step.

## What this is NOT

- Not a kernel build, not another toy macro, not a bit-identical Tensile clone.
- Not wired into the live compile path; `AMDGemmScheduleObject` is structural metadata with a gate, exactly
  like the surrounding `schedule.py` scaffolding (`AMDPipelineStageMeta`, `AMDLDSStagePlan`), which is also
  unwired analysis.
- Not a performance result; `performance_claim` is `False` and the structural gate refuses to assert timing.

## Next (still gated, not authorized here)

1. Reconstruct the steady-state K-loop from heuristic first/last-WMMA into an exact symbolic schedule (close
   the first `blocked_unknown` row that matters for lowering).
2. Resource-gated lowering of the schedule object to ISA (the schedule object feeds the existing
   `AMDLDSStagePlan` → `define_local` lowering path; the K-loop scheduler is the net-new capability and
   remains the codegen wall named in `prefill-sw-pipeline-codegen-charter-20260620.md`).
3. Only then time against the `≥60 TFLOPS` pure-tinygrad authority gate, under the PTM-1 interleaved,
   one-clock harness. **No BEAM/search until the schedule object lowers to ISA** — until then search would
   tune the wrong space.
