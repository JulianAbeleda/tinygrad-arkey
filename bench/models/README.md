# Model Benchmarks

End-to-end, per-model performance numbers (decode tok/s, prefill, VRAM), organized by **model family** and then
by **GPU backend**. This is the central place to answer "how fast does model X run on GPU Y here."

```
bench/models/
  README.md                       # this index
  <family>/                       # e.g. qwen/
    README.md                     # family index across backends
    <backend>.md                  # one file per GPU backend (rendered table)
    data/<backend>/<model>.json   # raw per-model artifacts (source of truth)
```

## How numbers are produced

- Measurement harness: `extra/model_e2e_bench.py` — clean whole-decode `model.generate` (W==D), `PROFILE=0`,
  auto clock, warmed JIT, steady-state median + spread band. Authority + rules: `bench/qk-decode-eval/HARNESS_GUIDE.md`.
- The `.md` tables are **rendered views**; the `data/<backend>/*.json` artifacts are the source of truth.
  Regenerate a table with `extra/gen_model_bench_doc.py`.

## Why quantization is a column

Decode is HBM-bandwidth bound: each token re-reads the weights. The **quant** sets bytes-per-weight, which is the
dominant decode cost — so a Q8_0 model moves ~2x the bytes of a Q4_K_M model of the same parameter count. Always
read tok/s next to its quant, not parameter count alone.

## Families

- [qwen/](qwen/README.md)
