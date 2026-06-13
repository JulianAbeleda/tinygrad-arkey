# QK Semantic Codegen v3

Family C v0: packed-load Q4_K `ffn_gate` probe.

This surface is a memory-access probe, not another schedule-only sweep. It
rewrites the Q4_K partial GEMV reduce axis from per-position qword indexing to
explicit packed-word lanes that unroll four nibbles from each loaded `uint32`.

## Regenerate

```sh
base=bench/qk-ansor-transition-20260612/semantic-codegen-v3
for model in 8b 14b; do
  PYTHONPATH=. .venv/bin/python extra/qk_semantic_codegen_v3.py \
    --descriptor bench/qk-ansor-transition-20260612/descriptors/$model.json \
    --json $base/$model/candidates.json \
    --md $base/$model/candidates.md \
    --gate-json $base/$model/static-gate.json \
    --gate-md $base/$model/static-gate.md

  PYTHONPATH=. DEV=AMD .venv/bin/python extra/qk_semantic_schedule_bench.py \
    --model $model \
    --candidates $base/$model/candidates.json \
    --static-gate $base/$model/static-gate.json \
    --out $base/$model/microbench-runs \
    --json $base/$model/microbench.json \
    --md $base/$model/microbench.md \
    --device AMD \
    --iters 3 \
    --min-gain 0.10 \
    --tie-band 0.03
done

PYTHONPATH=. .venv/bin/python extra/qk_semantic_codegen_v3_verdict.py \
  --base $base \
  --json $base/verdict.json \
  --md $base/verdict.md
```

Load-width evidence was captured with `DEBUG=4` and summarized by:

```sh
PYTHONPATH=. .venv/bin/python extra/qk_load_width_report.py \
  $base/load-width/8b-ffn-gate-current-debug4.log \
  $base/load-width/8b-ffn-gate-packed-load-debug4.log \
  --json $base/load-width/report.json \
  --md $base/load-width/report.md
```

## Verdict

`semantic_codegen_v3_rejected`.

| model | status | current GB/s | candidate GB/s | gain |
|---|---|---:|---:|---:|
| 8B | tie | `206.42` | `205.07` | `-0.65%` |
| 14B | tie | `367.98` | `366.84` | `-0.31%` |

The source parser confirms the candidate produced a distinct
`q4k_gemv_packed_load_partial_*` kernel, but still inferred scalar `u32` loads
and no vector-load evidence. This v0 rewrite changed loop expression shape, not
the underlying memory transaction width/coalescing enough to matter.
