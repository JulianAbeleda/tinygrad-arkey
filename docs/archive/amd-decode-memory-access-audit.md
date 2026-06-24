# AMD Decode Memory Access Audit

Date: 2026-06-13

Status: source gate passed; Family C v1 GEMV construction gate rejected.

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

The core source gate for Family C v1 passes.

The normal tinygrad codegen path preserves a requested aligned `uint32.vec(4)`
global load/store on AMD. The probe copies all lanes exactly, and DEBUG=4 source
shows an `unsigned_int4` load/store through vector pointer casts. Raw custom C
still works, but is no longer the only way to reach this memory shape.

Family C v1 then tried to consume that memory shape inside the real Q4_K
`ffn_gate` GEMV. That follow-up is recorded in:

- `bench/qk-ansor-transition-20260612/semantic-codegen-v4/verdict.md`
- `bench/qk-ansor-transition-20260612/semantic-codegen-v4/load-width/report.md`

The result is a construction reject on both gate models. The candidate reaches
the benchmark harness, but tinygrad's current vector UOp shape rules cannot yet
use the loaded `uint32x4` value in the unpack/dot expression: scalar lane
extraction fails the verifier, and vector-lane partial arithmetic fails later
shape checks before AMD source is emitted.

The next implementation surface is therefore no longer "another packed-load
candidate"; it is the missing core representation support:

- represent packed-load lane extraction / vector-lane arithmetic as valid UOps,
  or add a first-class packed QK load/decode operation that lowers to this
  source shape;
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

Do not broaden packed-load expression rewrites until the vector-load consumption
blocker is fixed. 32B remains skipped unless 8B/14B show promise after that
core representation issue is resolved.
