# Prefill Graph GEMM Corpus Quality Scope - 2026-06-20

Verdict: `SCOPE_PREFILL_GRAPH_GEMM_CORPUS_QUALITY_READY`

The graph GEMM route now has three banked facts:

- full in-model performance passed: `4895.9 tok/s`, `104.6ms / 512`,
- one-role numeric correctness passed: rel RMSE `0.0002077`,
- VRAM-safe sampled quality passed across two real `T=512` windows: `max_abs_dNLL = 0.0`.

The route should still remain default-off until broader quality coverage exists. The next action is a corpus-style
quality gate that keeps the memory behavior of the sampled probe but scores enough positions to make promotion
defensible.

## Why This Gate Exists

The old full-window NLL harness materializes `(512, vocab)` logits and OOMs near the current VRAM ceiling. The
sampled gate fixed the memory shape by slicing to one vocab vector before realization, but it only scored two
positions. The missing tool is a middle path:

- exercise the real `T=512` graph route,
- score many teacher-forced positions,
- avoid retaining all prompt logits,
- compare baseline `PREFILL_V2` and `PREFILL_GRAPH_GEMM=1` in isolated subprocesses.

## Proposed Tool

Add `extra/qk_prefill_graph_gemm_corpus_quality.py`.

### Inputs

| input | default | reason |
|---|---:|---|
| `model` | required or `QK_MODEL` | explicit artifact |
| `--windows` | `8` | enough to cover more contexts without long runs |
| `--stride` | `64` | overlapping windows give different scored positions while keeping `T=512` |
| `--score-offsets` | `128,256,384,510` | multiple positions per window, including the current final-token smoke |
| `--eps-mean` | `0.002` | mean dNLL should be effectively zero |
| `--eps-max` | `0.01` | keeps compatibility with the decode/prefill convention |
| `--argmax-mismatch-max` | `0` | promotion smoke should not change greedy picks on sampled points |

### Execution Model

Run two subprocesses:

| child | env |
|---|---|
| baseline | `DEV=AMD PREFILL_V2=1` |
| graph | `DEV=AMD PREFILL_V2=1 PREFILL_GRAPH_GEMM=1` |

Each child loads the model once. For each window and each score offset:

```python
for q4k_linear in model._q4k_linears.linears:
  q4k_linear.decode_enabled = False
for block in model.blk:
  block._prefill_v2 = True
  block._use_flash = False
pr._WARMSTART_OPTS = model._pf16_warmstart
logits = model.logits(tokens_512, 0)[:, offset, :].realize()[0].numpy()
target = tokens_512[offset + 1]
nll = logsumexp(logits) - logits[target]
argmax = logits.argmax()
```

This realizes one vocab vector at a time, not `(512, vocab)`. It intentionally recomputes the `T=512` forward per
offset unless/until tinygrad can safely materialize a small gather of vocab rows without keeping the full logits
matrix. Correctness and memory safety matter more than speed for this gate.

Important: calling `model.logits` directly does not pass through `Transformer.__call__`, so the tool must install
the prefill-v2 block state and warmstart table itself. Otherwise it can silently measure the fallback logits path
instead of the graph GEMM route.

## Acceptance Gates

| gate | threshold |
|---|---:|
| baseline finite | true |
| graph finite | true |
| scored positions | `windows * len(score_offsets)` |
| mean graph-baseline dNLL | `<= 0.002` |
| max positive dNLL | `<= 0.01` |
| max abs dNLL | `<= 0.01` |
| argmax mismatches vs baseline | `0` |
| child OOM/retry failures | `0` |

If the graph route is mathematically equivalent at this layer of precision, the expected result is still exactly
zero or near-zero dNLL, as in the two-window sampled gate.

## Promotion Ladder

| level | status | requirements |
|---|---|---|
| explicit developer flag | passed | current `PREFILL_GRAPH_GEMM=1` route |
| experimental user flag | next | corpus quality pass plus existing perf pass |
| default for gfx1100/Qwen3-8B-like shapes | later | corpus quality, repeated perf, OOM behavior, and shape fallback audit |
| generic default | out of scope | requires broader model/device/shape coverage |

## Out Of Scope

- default-on behavior,
- BEAM/search integration,
- decode-route promotion,
- replacing the existing full-window NLL harness,
- claiming full corpus perplexity parity beyond the scored sample set.

## Next Command

After implementing the corpus-quality probe:

```bash
DEV=AMD PREFILL_V2=1 PYTHONPATH=. python3 extra/qk_prefill_graph_gemm_corpus_quality.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --windows 8
```

Expected pass name: `PASS_PREFILL_GRAPH_GEMM_CORPUS_QUALITY`.
