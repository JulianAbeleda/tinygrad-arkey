# Decode MMVQ large project P5/P6 result - 2026-06-19

Purpose: continue the funded source/object-import path from
`decode-mmvq-large-project-scope-20260619.md` after P2-P4 proved the imported llama Q4_K consumer.

Artifacts:

- `extra/qk_decode_mmvq_p5_lifecycle_probe.py`
- `extra/qk_decode_mmvq_p6_q4_shape_matrix.py`
- `bench/qk-decode-mmvq-large-project/p5_lifecycle_probe.json`
- `bench/qk-decode-mmvq-large-project/p6_q4_shape_matrix.json`

## P5 - one-role lifecycle probe

Role: `blk.0.attn_output.weight`.

Baseline: current tinygrad Q4_K `attn_output` decode path.

Candidate: real model attention activation -> explicit `block_q8_1` producer -> imported llama Q4_K MMVQ consumer.

Result:

| item | result |
|---|---:|
| q8 producer | byte-exact vs CPU `q8_blocks` |
| imported consumer max_abs vs q8 reference | `1.19e-7` |
| q8 producer device time | `0.00630ms` |
| imported consumer device time | `0.01304ms` |
| lifecycle device sum | `0.01934ms` |
| lifecycle bandwidth | `488.0 GB/s` (`50.8%` of 960 GB/s) |
| current `attn_q/o` frontier | about `29%` HBM |

Verdict: **PASS_DEVICE_LIFECYCLE**.

Important measurement note: eager wall timing for the current tinygrad role is not a valid device-kernel baseline here;
it includes Python graph construction. P5 therefore gates on lifecycle device bandwidth versus the banked current
`attn_q/o` in-model frontier. The wall number is retained in the artifact only as a diagnostic.

## P6 - Q4_K shape matrix

The same imported no-fusion Q4_K template was tested on all Q4 shapes that matter for the first decode route:

| tensor | rows x K | device ms | Q4 GB/s | correctness |
|---|---:|---:|---:|---:|
| `blk.0.attn_output.weight` | `4096 x 4096` | `0.01056` | `893.6` | PASS |
| `blk.0.ffn_gate.weight` | `12288 x 4096` | `0.02514` | `1126.2` | PASS |
| `blk.0.ffn_up.weight` | `12288 x 4096` | `0.02539` | `1115.2` | PASS |

Verdict: **PASS_Q4_MATRIX**.

The >100% "Q4 GB/s" values are an effective packed-weight-read metric from queue timestamps. They should not be read as
physical HBM saturation; the actionable result is that the imported Q4 consumer is not the blocker and easily clears the
current tinygrad role frontiers.

## Consequence

P5/P6 move the project boundary:

- solved: source/object import, named descriptor load, raw kernarg rebinding, Q4 correctness, Q4 standalone speed, Q4
  activation-producer lifecycle for a real role, Q4 shape generalization to `ffn_gate/up`;
- still open: graph-safe route, Q6 imported-kernel correctness/perf, dNLL gate for the lossy q8 activation lifecycle,
  and final W==D ctx sweep.

The next buildable phase is **P7a graph-safe Q4 route**, not more Q4 kernel work. Q6 should be a parallel coverage
track, but Q4 already covers the largest traffic bucket (`ffn_gate/up`) plus `attn_q/o`.
