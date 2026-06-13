# AMD Decode Memory Access Audit

Date: 2026-06-13

Status: current go/no-go for Family C v1.

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

Do not build Family C v1 as another semantic descriptor candidate yet.

The normal tinygrad codegen path does not currently preserve aligned integer
vector global loads for Q4_K `uint32` storage. Raw custom C can force a vector
load by defining a `uint4`-style type inside an `Ops.CUSTOM` source block, but
that is not a search-visible capability.

The next implementation surface is therefore a core lowering capability:

- teach `correct_load_store.split_load_store` when it is legal to keep aligned
  integer global vector loads/stores, initially `uint32x4`;
- add renderer/codegen tests proving the generated source contains the vector
  load and the copy is exact;
- then rerun Family C v1 as a generated memory-access candidate.

## Why This Matters

The model-scope roofline says the remaining gap to llama.cpp is memory-load
efficiency, not another compute schedule. Family C v0 already tried the cheap
packed-word lane rewrite and tied on 8B/14B while DEBUG=4 still showed scalar
`u32` loads. A Family C v1 that cannot emit wider/coalesced integer loads would
repeat the same failure with a larger search surface.

## Stop Rule

Do not broaden packed-load expression rewrites unless generated source shows a
real vector/coalesced integer load shape. 32B remains skipped unless 8B/14B
show promise after the core lowering capability exists.
