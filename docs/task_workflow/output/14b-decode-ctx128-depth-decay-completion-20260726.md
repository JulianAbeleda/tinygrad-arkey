# 14B Decode Context-128 and Depth-Decay Campaign Closeout

Verdict: `BLOCKED`; this is a no-production-code-change closeout.

## Task verdicts

| Task | Verdict | Closeout basis |
|---|---|---|
| LUNA-070 | `NOT_RUN` | Requires LUNA-055 and LUNA-064; neither has a passing candidate. |
| LUNA-071 | `NOT_RUN` | No current numbers, supported-range change, or production manifest can be truthfully updated. |
| LUNA-072 | `PASS` | Deleted only the unconsumed token-fixture generator and its generated fixture. |
| LUNA-073 | `FAIL` | Promotion audit cannot pass: Track A and Track B authority, final candidate identity, and regressions are absent. |
| LUNA-074 | `NOT_RUN` | Pending a passing LUNA-073 and explicit operator approval. No promotion or retirement action occurred. |

## What is proven

- The context-128 failure occurs before timed decode, but no same-regime STORE ancestry, route selection log, or working 8B/512 capture identifies a safe repair layer.
- G5 has a focused deep-context cliff, but no final-commit family attribution, resource delta, or Amdahl proof names a full-model owner.
- No context-128 repair candidate passed LUNA-052. No depth candidate was admitted by LUNA-062 or implemented by LUNA-063.
- `TINYGRAD_PREFILL_PACKED_WMMA=0` remains the only source-proven unsafe configuration guard. Implicit fallback at context 128 is unproven.

## Exact block and reopening order

The host has `rocm-smi`, `rocminfo`, ROCtx headers, and `libroctx64.so.4`, but lacks `rocprofv3`, `rocprof`, `rocprof-compute`, and `omniperf`. Provide a supported gfx1100 ROCm trace collector plus resource/disassembly extraction. Install or replace no system ROCm package within this campaign without operator approval.

After that prerequisite and a bounded collector positive control, reopen in dependency order: LUNA-030, LUNA-031, LUNA-034, LUNA-051, LUNA-052, LUNA-053 through LUNA-055; then recollect final-Track-A 512/4096 evidence for LUNA-060 through LUNA-064. Do not promote a speculative candidate before those gates.

## Deferred regression and authority commands

Run these only after a reviewed candidate exists and its compile-only, route, and numerical gates have passed. Every GPU command remains serialized by `/tmp/gpu-bench.lock`.

```bash
flock /tmp/gpu-bench.lock env -u TINYGRAD_PREFILL_PACKED_WMMA PYTHONPATH=. DEV=AMD \
  python3 extra/qk/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --ckpts 128 --max-context 4608 --nmeas 1 --reps 1 --warmup-decode 2 --chunk-size 32 \
  --out bench/14b-decode-ctx128-depth-decay-20260726/tinygrad-ctx128-trace.json

flock /tmp/gpu-bench.lock env -u TINYGRAD_PREFILL_PACKED_WMMA PYTHONPATH=. DEV=AMD \
  python3 extra/qk/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 40 --reps 5 --warmup-decode 3 --chunk-size 32 \
  --out bench/14b-decode-ctx128-depth-decay-20260726/tinygrad-ctx512-authority.json

flock /tmp/gpu-bench.lock /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json \
  > bench/14b-decode-ctx128-depth-decay-20260726/llama-ctx128-trace.json
```

For authority, use one process per checkpoint, replace `-d 128` with the checkpoint, use `-r 5`, and retain the explicit `-p 0 -n 128 -b 512 -ub 512 -fa 1 -ngl 99` settings. The final gates also require the LUNA-052 compile-only shape controls, deterministic token/KV-bounds controls, the LUNA-054 short-prompt matrix, and LUNA-064 three interleaved candidate/baseline pairs at `512,1024,2048,4096` with same-session llama at `512,4096`.

## Cleanup and retirement

Deleted campaign-only dead probes: `extra/qk/decode/token_fixture.py` and `bench/14b-decode-ctx128-depth-decay-20260726/token-fixture.json`. Retained canonical ledgers, manifests, failure evidence, and the reusable manifest collector with its unit test. No unrelated file, worktree, branch, stash, or system package was touched.

Do not retire `feature/14b-decode-ctx128-and-depth-decay`. LUNA-074 remains `NOT_RUN` pending operator approval after LUNA-073 passes.
