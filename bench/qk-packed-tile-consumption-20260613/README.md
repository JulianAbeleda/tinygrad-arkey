# QK Packed Tile Consumption Probe

Date: 2026-06-13

Purpose: construction gate for consuming a `PackedQKTile` Q4_K
`u32x4_aligned` load inside tinygrad.

Result: `semantic_custom_op_required`.

Summary:

- normal UOp scalar lane extraction from `uint32.vec(4)` fails verifier;
- normal UOp vector integer arithmetic fails shape validation;
- a custom semantic kernel can load `tg_uint4`, index lanes, unpack Q4 low/high
  nibbles, and accumulate an exact dot;
- DEBUG=4 source parsing confirms `vector_u32x4` evidence for that custom
  semantic probe;
- no microbench or full-decode gate should run until a first-class packed QK
  load/decode/dot op or renderer PatternMatcher lowering exists.

Artifacts:

- `probe.json` / `probe.md`: construction verdict.
- `load-width/probe-debug4.log`: DEBUG=4 source log.
- `load-width/report.json` / `load-width/report.md`: generated-source
  load-width parser output.

Commands:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_packed_tile_consumption_probe.py \
  --device AMD --iters 3 \
  --json bench/qk-packed-tile-consumption-20260613/probe.json \
  --md bench/qk-packed-tile-consumption-20260613/probe.md

DEV=AMD DEBUG=4 PYTHONPATH=. .venv/bin/python extra/qk_packed_tile_consumption_probe.py \
  --device AMD --iters 1 \
  --json bench/qk-packed-tile-consumption-20260613/probe.json \
  --md bench/qk-packed-tile-consumption-20260613/probe.md \
  > bench/qk-packed-tile-consumption-20260613/load-width/probe-debug4.log 2>&1

PYTHONPATH=. .venv/bin/python extra/qk_load_width_report.py \
  bench/qk-packed-tile-consumption-20260613/load-width/probe-debug4.log \
  --json bench/qk-packed-tile-consumption-20260613/load-width/report.json \
  --md bench/qk-packed-tile-consumption-20260613/load-width/report.md \
  --repo .
```
