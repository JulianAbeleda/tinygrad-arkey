# 14B Direct-Packed Prefill Authority Baseline - 2026-07-10

Command:

```bash
DEV=AMD DEBUG=0 PROFILE=0 python3 extra/qk/bench.py \
  --model /home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \
  --model-profile 14b \
  --prefill --prefill-mode authority
```

Artifact:

```text
bench/prefill-whole-synced/qwen3-14b-direct-packed-authority-baseline-20260710.json
```

Result:

```text
profile: qwen3_14b_q4k_m_gfx1100
route env: PREFILL_ROUTE=direct_packed, PREFILL_PACKED_STREAM=1, PREFILL_V2=1, ALLOW_DEVICE_USAGE=1
K/warmups/rounds: 8 / 4 / 3
chunk@0:    1404.6601 ms, 364.5 tok/s
chunk@512:  1424.9633 ms, 359.3 tok/s
chunk@1024: 1457.8411 ms, 351.2 tok/s
chunk@2048: 1498.8213 ms, 341.6 tok/s
chunk@3584: 1530.3323 ms, 334.6 tok/s

WHOLE-PREFILL@512:  364.50 tok/s
WHOLE-PREFILL@1024: 361.89 tok/s
WHOLE-PREFILL@2048: 355.20 tok/s
WHOLE-PREFILL@4096: 346.41 tok/s
```

Reproducibility band:

```text
worst_cv: 0.00424
worst_spread: 0.01032
single_sample: false
```

Notes:

- The harness completed successfully and wrote the shared-harness artifact.
- The artifact stamps `mode: authority_incomplete`, because comparator id, candidate id, primitive class, threshold, ledger, and quality gate were not supplied.
- Route binding passed for the default effective route set.
- Git state was dirty at measurement time: `git_short=2cf0c794f`, `git_dirty=true`.
