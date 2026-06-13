# AMD Decode Memory Access Audit

Date: 2026-06-13

Status: source gate passed for Family C v1.

Canonical artifact:

- `bench/qk-memory-access-20260613/audit.md`
- `bench/qk-memory-access-20260613/audit.json`
- `bench/qk-memory-access-20260613/vector-probe.md`
- `bench/qk-memory-access-20260613/load-width/report.md`

Regenerate:

```sh
DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_integer_vector_load_probe.py \
  --device AMD --n-words 4096 --iters 5 \
  --json bench/qk-memory-access-20260613/vector-probe.json \
  --md bench/qk-memory-access-20260613/vector-probe.md

PYTHONPATH=. .venv/bin/python extra/qk_memory_access_audit.py \
  --json bench/qk-memory-access-20260613/audit.json \
  --md bench/qk-memory-access-20260613/audit.md
```

Load-width source logs are captured separately with `DEBUG=4` for the
`uop_vec_request` and `custom_uint4` probe modes, then parsed with
`extra/qk_load_width_report.py`.

## Verdict

The core source gate for Family C v1 now passes.

The normal tinygrad codegen path preserves a requested aligned `uint32.vec(4)`
global load/store on AMD. The probe copies all lanes exactly, and DEBUG=4 source
shows an `unsigned_int4` load/store through vector pointer casts. Raw custom C
still works, but is no longer the only way to reach this memory shape.

The next implementation surface is therefore Family C v1:

- add a generated Q4_K memory-access candidate that requests aligned `uint32x4`
  packed-weight loads directly;
- keep 8B/14B as the gate models;
- require reference unpack, AMD GEMV correctness, generated source load-width
  evidence, and a strong microbench result before full-decode promotion.

## Why This Matters

The model-scope roofline says the remaining gap to llama.cpp is memory-load
efficiency, not another compute schedule. Family C v0 already tried the cheap
packed-word lane rewrite and tied on 8B/14B while DEBUG=4 still showed scalar
`u32` loads. The new lowering capability gives Family C v1 a materially
different source shape to test.

## Stop Rule

Do not broaden packed-load expression rewrites unless generated source shows a
real vector/coalesced integer load shape. 32B remains skipped unless 8B/14B
show promise with Family C v1.
