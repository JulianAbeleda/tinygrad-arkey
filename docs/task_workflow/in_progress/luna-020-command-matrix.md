# LUNA-020: Comparable Workload and Command Matrix

Status: fixture produced CPU-only; cross-runtime token-ID consumption is blocked by the current entry points.

## Fixture contract

`bench/14b-decode-ctx128-depth-decay-20260726/token-fixture.json` is generated from the Qwen3 GGUF header only. It records BOS/EOS/EOT policy, exact IDs for 128, 512, and 4096, and hashes. It uses the same corpus construction as `decode_runtime_overhead.py`, but the canonical authority has not yet been extended to accept the fixture as input.

Temperature is explicitly zero. Tinygrad uses the temperature-zero argmax expression in `Transformer.forward`; token identity is an acceptance check to perform after a bounded input adapter exists, not a claim made by this fixture.

## Commands

All GPU commands below require `flock /tmp/gpu-bench.lock`; they are retained commands, not run by LUNA-020.

Tinygrad trace-smoke, one process and one checkpoint:

```bash
flock /tmp/gpu-bench.lock env -u TINYGRAD_PREFILL_PACKED_WMMA PYTHONPATH=. DEV=AMD \
  python3 extra/qk/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --ckpts 128 --max-context 4608 --nmeas 1 --reps 1 --warmup-decode 2 --chunk-size 32 \
  --out bench/14b-decode-ctx128-depth-decay-20260726/tinygrad-ctx128-trace.json
```

Tinygrad authority, one process per checkpoint:

```bash
flock /tmp/gpu-bench.lock env -u TINYGRAD_PREFILL_PACKED_WMMA PYTHONPATH=. DEV=AMD \
  python3 extra/qk/decode/decode_runtime_overhead.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --ckpts 512 --max-context 4608 --nmeas 40 --reps 5 --warmup-decode 3 --chunk-size 32 \
  --out bench/14b-decode-ctx128-depth-decay-20260726/tinygrad-ctx512-authority.json
```

llama.cpp length/depth comparator, explicitly setting prompt length, generation length, decode depth, batch, ubatch, flash attention, offload, repetitions, and JSON capture:

```bash
flock /tmp/gpu-bench.lock /home/ubuntu/env/llama.cpp/build/bin/llama-bench \
  -m /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf -ngl 99 \
  -p 0 -n 128 -d 128 -b 512 -ub 512 -fa 1 -r 1 -o json \
  > bench/14b-decode-ctx128-depth-decay-20260726/llama-ctx128-trace.json
```

For authority, replace `-d 128` with one checkpoint per process and `-r 5`; keep `-p 0`, `-n 128`, `-b 512`, `-ub 512`, `-fa 1`, `-ngl 99`, and the unique output path explicit. `llama-bench` has no token-ID argument in the retained wrapper (`extra/llm/llama_bench.py`), so this command is a comparable geometry/depth benchmark, not a shared-prompt-token benchmark.

## Blocking contract

Neither canonical entry point consumes `token-fixture.json`: tinygrad internally tokenizes and llama-bench accepts benchmark lengths. LUNA-020 therefore cannot claim exact cross-runtime prompt equivalence. A bounded adapter must accept `token_ids`, preserve BOS policy, prefill exactly the requested IDs, generate a fixed number of temperature-zero tokens, and retain IDs/hashes in its artifact. Until then the fixture is an exact tinygrad-input reference and a required contract for the adapter.
