# NV decode real-DAG S=1/S=2 cheap wall measurement record

Date: 2026-08-04

Authority: `nv-decode-real-dag-s1-s2-cheap-test-scope-20260804.md`.
Branch/commit: `nvidia-bringup-20260731` / `64507f994`.
Status: complete. Verdict: **SCHEDULER_WALL_NEUTRAL**.

## 1. Finding

Explicit two-stream capture does not convert the real d512 CUDA DAG's
approximately 22% no-contention schedule potential into token wall:

| arm | graph construction | median tok/s | median ms/token | ratio vs S=1 |
| --- | --- | ---: | ---: | ---: |
| S=1 | production programmatic | 178.159 | 5.61295 | 1.00000x |
| S=2 | capture-based two stream | 178.265 | 5.60961 | 1.00060x |

Derived S=2 change: **+0.0595%**, **3.34 us/token saved**. This is far below
the declared 5% belief-flip threshold and below ordinary repetition spread.
The result is wall-neutral, not a speedup.

Correctness and structure passed:

- all three repetitions within each arm produced identical token sequences;
- S=1 and S=2 produced the same eight generated token IDs and hashes;
- both structural arms emitted 1021 kernels/token in six graph groups;
- kernel-class counts were identical, including 216 Q4_K, 36 Q6_K, and 72
  flash-attention kernels;
- both census arms produced the same token hash and first token;
- S=2 constructed, updated, launched, and replayed the full real graph without
  error.

No CUPTI follow-up was run because the cheap wall gate did not pass.

## 2. Protocol

GPU: NVIDIA GeForce RTX 5090, driver 595.84, compute capability 12.0. Both arms
ran sequentially under one `/tmp/gpu-bench.lock` acquisition in independent
processes.

Canonical authority command shape:

```bash
DEV=CUDA CUDA_GRAPH_STREAMS={1|2} PYTHONPATH=. .venv/bin/python \
  extra/llm_research/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 8 --reps 3 \
  --warmup-decode 3 --chunk-size 32 --skip-dispatch-diagnostic \
  --out /tmp/nv_decode_real_dag_s{1|2}_20260804.json
```

The production generate `W` interval is authoritative. Dispatch-only timing
and profiling were intentionally omitted.

CPU preflight:

```text
test_cuda_graph_multi_stream_schedule.py
test_decode_runtime_overhead_evidence.py
16 passed in 0.30s
```

## 3. Raw repetition rows

| arm | rep 0 tok/s | rep 1 tok/s | rep 2 tok/s | median |
| --- | ---: | ---: | ---: | ---: |
| S=1 | 177.691 | 178.159 | 178.271 | 178.159 |
| S=2 | 177.432 | 178.265 | 178.309 | 178.265 |

No retained repetition was removed. Both arms show a slightly slower first
retained repetition and nearly identical later repetitions.

Canonical token evidence, identical in every repetition and across arms:

```text
prelude sha256:
  51a6b5bb6e00b5ccc0cbc75622616a20ca9e806403d9c290ab44d664d6bb5ac1

generated token ids:
  38835,34208,13,279,3974,13876,38835,34208

generated sha256:
  9e6664fd1d67a6124e786daaa1d895bdb64b972c3991c54dd5fcc6cea16f6881
```

## 4. Structural identity fallback

The canonical authority's `programs_per_token_by_route` introspector returned
`None` for both arms. Per the amended scope, a minimal DEBUG census was run to
replace that unavailable field:

| quantity | S=1 | S=2 |
| --- | ---: | ---: |
| kernels/token prime | 1021 | 1021 |
| kernel launch lines | 1021 | 1021 |
| graph groups/replay | 6 | 6 |
| Q4_K kernels | 216 | 216 |
| Q6_K kernels | 36 | 36 |
| flash-attention kernels | 72 | 72 |
| elementwise/fusion kernels | 690 | 690 |

The census used `[1] * 512`, three measured tokens, and one repetition. Its
purpose was topology and cross-arm token identity, not wall authority. Both
arms produced first token `271` and token SHA-256
`349e2acf8aa49a50616754b9caf53bc8093afc4646ed3e8d9f65f403e488d94b`.
The local census tool was untracked at measurement time; its exact source
SHA-256 was
`35c9c4e7febf113cb582fa48e6c5e93e031b19bb7af5ff9e535104984c11a521`.
It is supporting structural evidence only; the canonical committed authority
owns the wall and token-correctness verdict.

The live canonical route label was `sdpa` in both arms. The scope's initial
assumption that it would report `flash` was corrected before classification;
the 1021-kernel/six-group topology matches the anchored real CUDA route.

## 5. Artifact hashes

The compact anchored payload is
`docs/task_workflow/output/nv-decode-real-dag-s1-s2-cheap-20260804.json`.

Ephemeral raw artifacts:

| artifact | SHA-256 |
| --- | --- |
| canonical S=1 | `e7adeb51e5d5055c6acb7bdcf2d88b16be21b8a40bb383387c14c65507351a1d` |
| canonical S=2 | `70aca9dc9de8c7caa07d581ed8add59d5072ba028356e0fcb122775edbb9e594` |
| census S=1 | `dcbc5a1793823be2f4c7a8fa2076c5ba5fa7322b0586810d340edd3266ec7d06` |
| census S=2 | `faeefe68508987925de940f92e048fa9fa2c79e78cc3bc5b3f196eee8cefb100` |

The compact payload carries the repetition rows, identities, correctness
hashes, structural counts, arithmetic, and evidence classifications needed to
reproduce the verdict without the `/tmp` files.

## 6. Interpretation

This result closes a narrower claim than “overlap is impossible”:

> The existing explicit S=2 capture lowerer is not a useful d512 wall lever on
> the real physical decode DAG.

It does not reveal whether S=2 realizes concurrency that is canceled by kernel
stretch, or whether the driver still schedules the graph serially. That
mechanism distinction would require CUPTI. The scope deliberately declines
that expense because either mechanism has the same immediate decision: do not
advance this S=2 route toward a default or parity claim.

Combined with B3.2:

```text
physical DAG no-contention potential: about 22%
planner-attributed loss:              about 35 us/token
explicit S=2 measured recovery:       about 3 us/token, wall-neutral
```

Therefore neither planner unaliasing nor this explicit two-stream construction
explains or closes the llama parity gap. The highest-value remaining work is:

1. Q4_K/Q6_K MMV instruction mapping and achieved bandwidth;
2. llama-kernel constant-DAG oracle to isolate kernel quality;
3. six graph-launch/inter-group gaps and removable boundaries;
4. profiling S=2 only if a materially different scheduler or kernel mix first
   creates a reason to revisit it.

## 7. Verdict

`SCHEDULER_WALL_NEUTRAL`:

- correctness: PASS;
- structural identity: PASS;
- wall gain: FAIL (`+0.0595% < 5%`);
- CUPTI follow-up: not authorized by the cheap-test scope;
- default change: not authorized;
- parity claim: none.
