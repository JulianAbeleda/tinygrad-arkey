# AMD dynamic tile owner validation — 2026-07-15

Validation owner: `extra/qk/amd_dynamic_tile_owner_validation.py`.  The probe
uses one symbolic output tile and four-element dynamic indexed writeback.  It
does not modify production emitter/compiler code.

Run:

```sh
PYTHONPATH=. python3 extra/qk/amd_dynamic_tile_owner_validation.py \
  --out bench/amd-dynamic-tile-owner/latest.json
```

The focused CPU graph tests pass. The earlier apparent AMD pass was a no-op:
the owner returned a void `SINK` and `Tensor(sink).realize()` did not schedule
it. The corrected probe attaches the effect to a real output and rejects
launch-only evidence. Current gfx1100 status is fail-closed:

| tile count | UOps | graph | compile | runtime | classification |
|---:|---:|---|---|---|---|
| 1 | 159 | passed | not attempted | not attempted | `INDEX_or_dynamic_store_unsupported` |

Current exact failure: `END src[0] should be KERNEL, not Ops.STORE`. The
symbolic indexed writeback is present in the UOp graph, but current callify /
rangeify does not turn that owned `END(STORE, range)` into a schedulable kernel.
The route remains disabled; no runtime or tok/s claim is made from this owner.
