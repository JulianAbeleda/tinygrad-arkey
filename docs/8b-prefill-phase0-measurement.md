# 8B Prefill Phase 0 Measurement Notes

Date: 2026-07-08.

## Scope

Phase 0 should reuse the existing whole-prefill authority harness:

```text
extra/qk/prefill_whole_synced.py
```

Do not add another throughput harness for the 8B oracle/candidate comparison. This script already measures the
synced `model.__call__` prefill-v2 warmstart path, records route attribution, and writes JSON artifacts.

The current 5k oracle artifact is:

```text
bench/prefill-whole-synced/graph-gemm-8b-refresh-20260708.json
```

It reports `PREFILL_GRAPH_GEMM=1`, route attribution
`prefill_pipe_role_selective_generated`, `prefill_route_pure=false`, and whole-prefill tok/s:

| length | tok/s |
|---:|---:|
| 512 | 5110.58 |
| 1024 | 4909.52 |
| 2048 | 4427.54 |
| 4096 | 3676.86 |

## Repeatable Side-By-Side Commands

Run from the repo root:

```sh
cd /home/ubuntu/tinygrad-arkey
```

Oracle / escape hatch:

```sh
PYTHONPATH=. PREFILL_V2=1 PREFILL_GRAPH_GEMM=1 \
  python3 extra/qk/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --mode authority \
  --artifact bench/prefill-whole-synced/graph-gemm-8b-authority.json \
  --json
```

Generated scheduler candidate:

```sh
PYTHONPATH=. PREFILL_V2=1 PREFILL_GRAPH_GEMM=0 \
  python3 extra/qk/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --mode authority \
  --artifact bench/prefill-whole-synced/generated-8b-authority.json \
  --json
```

For faster smoke checks, use the same commands with `--mode smoke`. For timing-sensitive authority runs, add
`--pin-clock` on machines where the clock-pin helper has the needed permissions.

## Result Locations

The two commands above write:

```text
bench/prefill-whole-synced/graph-gemm-8b-authority.json
bench/prefill-whole-synced/generated-8b-authority.json
```

If `--artifact` is omitted, the harness writes `bench/prefill-whole-synced/latest.json`, which is convenient for local
iteration but not acceptable for a side-by-side record because the second run overwrites the first.

Each JSON report includes:

- `whole_tok_s`: throughput at the requested whole-prefill lengths.
- `chunk_ms` and `chunk_samples_ms`: measured 512-token chunk timings.
- `graph_gemm`: whether `PREFILL_GRAPH_GEMM` was active in the imported runtime.
- `route_attribution`: generated-vs-oracle route family, purity, rollback, and provenance fields.
- `timing_authority`: the measurement method string.

## Structural Inventory Command

For Phase 0 lifecycle counters, use the existing census script rather than adding a parallel tracer:

```sh
PYTHONPATH=. python3 extra/qk/prefill/prefill_route_census.py \
  --routes generated-direct,generated-kmajor,hand-lds2 \
  --shapes '2,2' \
  --structural-only \
  --json
```

On the current tree this command returns a valid hand `hand-lds2` row, but both generated rows fail structurally with:

```text
AttributeError: 'int' object has no attribute '_fields'
```

That is missing Phase 0 glue for generated lifecycle/instruction-count parity. Whole-prefill throughput comparison is
not blocked by this; generated structural counters are.

## Missing Glue

- There is no single A/B wrapper that runs oracle and generated authority commands together. This is not required for
  Phase 0 because `prefill_whole_synced.py --artifact ...` gives repeatable side-by-side artifacts.
- `prefill_route_census.py` does not currently write an artifact. Capture stdout or extend it only if a persistent
  lifecycle-census artifact becomes required.
- Generated ISA lifecycle census currently errors before producing generated instruction counters. Fix that before
  claiming per-role generated instruction/wait/WMMA parity against the hand oracle.
