# AMD Decode Flywheel Kernel Triage Dataset v1 Featured

This Phase 3E artifact preserves the Phase 3D rows and split while adding
real source/compile evidence where committed artifacts expose it. It does
not add synthetic outcomes or move holdout rows into train.

- rows: `83`
- train rows: `45`
- holdout rows: `38`
- feature schema: `candidate_outcome_v1_featured`
- real UOp/source rows: `13`
- real UOp/source train rows: `7`
- real UOp/source holdout rows: `6`

## Real Feature Coverage By Mechanism

| mechanism | rows |
|---|---:|
| `packed_word_lane_unroll` | 2 |
| `qk_block_dot` | 2 |
| `tile_custom` | 7 |
| `vector_load` | 2 |

## Top Feature Sources

| source | rows |
|---|---:|
| `bench/qk-packed-tile-lowering-analysis-20260613/source/load-width-report.json` | 5 |
| `bench/qk-ansor-transition-20260612/semantic-codegen-v3/load-width/report.json` | 2 |
| `bench/qk-ansor-transition-20260612/semantic-codegen-v4/load-width/report.json` | 2 |
| `bench/qk-packed-tile-lowering-20260613/load-width/report.json` | 2 |
| `bench/qk-block-dot-compile-gate-20260613/compile-gate.json` | 2 |
