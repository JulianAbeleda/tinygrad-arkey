# Prefill authority versus shared-attention benchmark

## Scope and conclusion

This is a static comparison only.  No GPU command was run for this report.

`extra/qk/prefill/prefill_whole_synced.py --mode authority` and
`extra/qk/benchmark_shared_attention.py` do not exercise the same executable
path.  The latter's valid 8B/14B rows are **attention-only TinyJit replay
numbers**, not whole-model prefill numbers.  They therefore cannot establish
that the authority call should complete, nor can an authority flag make the
benchmark wrapper reproduce the model path.

The smallest authority experiment below retains the known production path and
isolates its first capture/replay boundary.  A stall there is not explained by
the attention-only benchmark succeeding.

## Static comparison

| Area | Whole-prefill authority | Shared-attention benchmark / replay |
| --- | --- | --- |
| Model load | Calls `load_model_and_tokenizer(GGUF, max_context, seed)` and retains all blocks, quantized linears, cache/state and model config. | Does not load a GGUF or `Model`; constructs synthetic Q/K/V tensors from the selected 8B/14B head geometry. |
| Prompt/construction | Deterministic `int32[1,512]` token chunk; every position calls `model(chunk, concrete_start_pos, temp, use_flash=True)`. Whole throughput sums interpolated per-chunk timings. | Fixed `Q=512`, synthetic fp16 Q/K/V, causal mask, and a single `shared_prefill_attention` candidate or reference-SDPA closure at one `kv`. |
| Routes and environment | Applies model-profile env defaults with `setdefault`; sets `_use_flash=True` and `_prefill_v2=True` per block; serializes the model's graph-GEMM candidate registry into `BOLTBEAM_FULL_KERNEL_CANDIDATE_SET_JSON`; route binding derives from that derived env. `--require-route` can hard fail after execution. | No model prefill route, graph-GEMM registry, Q4K/GEMM path, or `--require-route`. Candidate execution needs an admitted closure programmatically and the public CLI always exits for `--mode candidate`; its normal CLI route is baseline SDPA. |
| Capture/JIT | For each concrete `start_pos`, executes `warmups` full `model.__call__` forwards, realizes them, then measures further full calls. This enters the model's per-`start_pos` prefill-v2 JIT and warmstart schedule path. Each position is a separate capture/burst. | Performs candidate/reference residency and (for candidate) numeric materialization before timing; then captures one attention closure in `TinyJit` and times replay only. |
| Synchronization | Synchronizes after warmup, before every timed round, and after each `K`-call timed burst. Reports the **minimum** of `rounds`, then constructs whole-prefill rates from those per-position minima. | `synced_time` synchronizes around each replay sample; summary uses replay samples (the recorded replay matrix reports medians). |
| Output/side effects | Optional profile-event collection, route census, artifact JSON, human output, and optional JSON stdout occur outside the timed section. `--logits-only` changes the forward to `model.logits` and skips sampling/argmax. Normal path retains sampling/argmax. | Writes a compact attention artifact with timing/numeric metadata. Q/K/V graph construction and candidate numeric gate are deliberately before timing; no sampling, LM head, GGUF dequantization, KV/cache update or model artifact generation is timed. |

## What can explain authority stalling while replay succeeds

1. **The authority first touches almost the entire model graph.** It loads the
   GGUF and captures Q4K projections, FFN, cache/state, rotary, attention,
   LM-head and, unless `--logits-only`, sampling/argmax. Replay captures only
   an attention closure with already-resident synthetic inputs.
2. **Authority has five independently captured concrete positions by default**
   (`0,512,1024,2048,3584`), and each has four warmups, three timed bursts and
   eight calls per burst. A single replay benchmark has one `kv` shape and one
   closure. A delay can therefore be initial compilation/capture, a later
   position's distinct graph, or a non-attention model component.
3. **Authority mutates model route state after loading** (`_use_flash`,
   `_prefill_v2`, model warmstart state and derived candidate-set env). The
   standalone benchmark does none of this. In particular, an apparent replay
   candidate result cannot prove the model selected that route.
4. **The public benchmark's candidate mode is not the source of the saved
   candidate replay matrix.** Its CLI refuses candidate mode; saved valid
   candidate rows necessarily used an admitted programmatic closure/replay
   path. Invoking the public baseline wrapper is useful only as an
   attention-only control.
5. **The authority fault investigation identifies a separate full-path
   suspect:** the non-`--logits-only` path includes the sampling/argmax
   expression. The documented diagnostic predicts `--logits-only` can avoid
   that faulting subgraph. This is absent from the attention benchmark.
6. **Output is not an in-burst cause.** Artifact/JSON serialization is after
   measurement, but it can make a completed run appear to pause after the
   final burst. `PREFILL_POST_GRAPH_MARKERS=1` distinguishes this post-graph
   phase from capture or device synchronization.

The attention-only source is therefore a valid component performance control,
not a known-working wrapper for whole-prefill authority. The older
`shared-attention-benchmark-20260723` artifact explicitly says its
graph-rebuilding callbacks were invalid replay performance evidence; use the
`shared-attention-benchmark-replay-20260723` protocol/control instead.

## Exact discriminating commands (GPU; not run for this report)

Run these serially on an otherwise idle GPU. Both use the same Python
environment; neither is intended to create a promotion artifact.

First, the minimal whole-model authority call. It keeps the production
`model.__call__` prefill-v2/warmstart route but reduces it to one position, one
warmup/capture and one timed replay. `--logits-only` excludes the documented
sampling/argmax suspect. It still reports `authority_incomplete` because no
promotion metadata was supplied; that stamp change occurs after timing and
does not change the route.

```bash
DEV=AMD JIT=1 PROFILE=1 PYTHONPATH=. PREFILL_POST_GRAPH_MARKERS=1 \
  .venv/bin/python extra/qk/prefill/prefill_whole_synced.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --model-profile 8b \
  --mode authority --start-positions 0 --whole-lengths 512 --max-context 512 \
  -K 1 --warmups 1 --rounds 1 --logits-only --no-artifact
```

Then, the matching attention-only baseline control. It confirms that a single
8B geometry/`kv=512` SDPA replay can capture and synchronize, but it cannot
reuse or validate the authority's model route.

```bash
DEV=AMD JIT=1 PYTHONPATH=. \
  .venv/bin/python extra/qk/benchmark_shared_attention.py \
  --profile qwen3_8b_q4k_m_gfx1100 --kv 512 --mode baseline \
  --warmup 1 --samples 10 --output /tmp/qwen3-8b-attention-baseline-512.json
```

Interpretation:

- If the control succeeds and the minimal authority command stalls before its
  first post-graph marker, the differentiator is model loading/full capture or
  a model-only route, not replay synchronization.
- If it reaches a marker after the timed burst and stalls later, inspect
  artifact/stdout handling separately; rerun once with `--no-artifact` (as
  above) and without `--json`.
- If logits-only completes, repeat the exact authority command without
  `--logits-only`; the only intentional functional addition is sampling/argmax
  and its dependent full-forward behavior.

## Reuse verdict

No minimal `prefill_whole_synced.py --mode authority` command can reuse the
known-working attention benchmark path without ceasing to be authority. There
is no flag that substitutes `shared_prefill_attention` replay for
`model.__call__`, and the benchmark wrapper has no flag that loads the GGUF or
installs the model's prefill warmstart/graph-GEMM state. The first command
above is the smallest valid authority path; the second is its deliberately
non-equivalent control.
