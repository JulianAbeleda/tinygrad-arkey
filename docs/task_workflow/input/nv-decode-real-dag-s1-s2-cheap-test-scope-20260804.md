# NV decode real-DAG S=1/S=2 cheap wall test scope

Date: 2026-08-04

Status: executable diagnostic scope. Branch: `nvidia-bringup-20260731`.
Drafting boundary: `64507f994`. This scope authorizes the smallest lock-held
GPU A/B needed to decide whether explicit two-stream scheduling converts any
of the approximately 22% no-contention scheduling potential in the real d512
CUDA decode DAG.

This is not a parity, promotion, or default-flip scope. It changes no production
default and does not authorize S=2 outside the measurement process.

## 1. Question

With the model, token workload, physical memory plan, decode route, and kernel
set held fixed, does `CUDA_GRAPH_STREAMS=2` improve real d512 decode wall over
the default `CUDA_GRAPH_STREAMS=1`?

The test distinguishes these outcomes:

```text
S2 gain >= 5%:
  explicit scheduling is a real wall lever; authorize a profiled follow-up

S2 within +/- 5%:
  theoretical DAG slack does not convert cheaply; classify wall-neutral

S2 regression >= 5%:
  capture/event overhead or concurrent resource contention dominates

construction/correctness failure:
  lowerer/ABI blocker; no performance interpretation
```

## 2. Fixed arms

| arm | environment | graph construction |
| --- | --- | --- |
| A | `DEV=CUDA CUDA_GRAPH_STREAMS=1` | production programmatic graph |
| B | `DEV=CUDA CUDA_GRAPH_STREAMS=2` | gated capture-based two-stream graph |

Both arms use:

- commit `64507f994` and its unchanged working tree code;
- `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`;
- fixed depth 512, max context 4608, chunk size 32;
- canonical `extra/llm_research/decode/decode_runtime_overhead.py`;
- production generate path (`W`) only;
- 8 measured tokens per repetition, 3 retained repetitions;
- 3 decode warmups before retained measurement;
- independent processes because `CUDA_GRAPH_STREAMS` is process configuration;
- GPU lock `/tmp/gpu-bench.lock`.

The dispatch-only diagnostic and CUPTI are omitted from this cheap pass.

## 3. CPU preflight

Before reserving the GPU:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  test/unit/test_cuda_graph_multi_stream_schedule.py \
  test/unit/test_decode_runtime_overhead_evidence.py -q
```

PASS requires all tests green. Any failure stops the live test.

## 4. Live commands

Run sequentially under one outer lock acquisition:

```bash
DEV=CUDA CUDA_GRAPH_STREAMS=1 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 8 --reps 3 \
  --warmup-decode 3 --chunk-size 32 --skip-dispatch-diagnostic \
  --out /tmp/nv_decode_real_dag_s1_20260804.json

DEV=CUDA CUDA_GRAPH_STREAMS=2 PYTHONPATH=. .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 8 --reps 3 \
  --warmup-decode 3 --chunk-size 32 --skip-dispatch-diagnostic \
  --out /tmp/nv_decode_real_dag_s2_20260804.json
```

## 5. Fail-closed identity and correctness gates

No timing comparison is interpreted unless:

1. both processes exit zero;
2. both report CUDA, d512, the same model identity, and the same decode route
   (the live authority reports `sdpa` at this threshold);
3. route sequences are identical;
4. captured programs per token are identical; if the canonical introspector
   reports that field unavailable, a same-session DEBUG census must instead
   prove identical kernel count, class count, and graph-group count;
5. prelude token hashes are identical within and across arms;
6. generated token hashes are identical within each arm and across arms;
7. every repetition contains exactly 8 generated tokens;
8. neither arm reports a graph construction, update, launch, or synchronization
   error.

Failure verdict: `CONSTRUCTION_OR_CORRECTNESS_BLOCKED`. Do not report a speed
ratio from a failing arm.

## 6. Wall arithmetic

Use median production tok/s from each artifact:

```text
s1_ms_per_token = 1000 / median_s1_tok_s
s2_ms_per_token = 1000 / median_s2_tok_s
gain_pct         = 100 * (median_s2_tok_s / median_s1_tok_s - 1)
saved_us         = 1000 * (s1_ms_per_token - s2_ms_per_token)
```

Record all repetition rows. Do not discard a slow retained repetition unless a
named correctness or process failure invalidates the entire arm.

## 7. Decision and follow-up

- `SCHEDULER_WALL_GO`: correctness PASS and S2 gain `>=5%`. Next run the same
  A/B with CUPTI to measure realized overlap and kernel-duration inflation.
- `SCHEDULER_WALL_NEUTRAL`: correctness PASS and absolute gain `<5%`. Do not
  spend on CUPTI; return to kernel/resource analysis.
- `SCHEDULER_WALL_REGRESSION`: correctness PASS and S2 regression `>=5%`.
  Profile only if needed to distinguish capture/event tax from contention.
- `CONSTRUCTION_OR_CORRECTNESS_BLOCKED`: fix the gated lowerer only; no wall
  claim.

No outcome authorizes a default change. A GO requires a separate full protocol
with interleaved same-session arms, at least 20 tokens x 3 reps, correctness,
CUPTI mechanism evidence, and d2048/d4096 qualification.
