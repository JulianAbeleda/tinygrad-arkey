# Q4-down numerical localization - C0 (2026-08-29)

## Verdict: BLOCKED

The authoritative failure reproduces, but the named harness set does not retain
the intermediate tensors required by C0. No correction mechanism is claimed and
C1 is not authorized.

## Reproduction

Command:

```sh
PYTHONPATH=. python3 extra/llm_research/prefill/nv_q4down_matched_ab.py \
  --z docs/task_workflow/evidence/nv-q4down-capture-20260829/z.npy \
  --out docs/task_workflow/evidence/nv-q4down-localization-20260829/matched-ab.json \
  --rounds 1
```

The exact saved-Z input is float16 `(512,12288)`, SHA-256
`c1a31d73c201d2682bce8b0798054c9bf2a3bfcabd4efc1f71fd6f48772c61d1`.
The run used all 18 real GGML type-12 down roles, 18 candidate mains, 18
control mains, and 18 producer records. Candidate and control were finite;
`max_abs=2.695646286010742`, `mean_abs=0.005390220787376165`, and the declared
`rtol=0.02, atol=0.5` comparison failed.

## Required localization status

| Stage | Status | Retained evidence |
|---|---|---|
| compact-Q8 record | producer-only v4 gate passes q/scale/sum exact | `nv-q4down-capture-20260829/saved-z-k12288-gate-v4.json` |
| decoded Q4 group metadata | not exposed by binding | missing |
| per-K32 corrected subtotal | not exposed by kernel ABI | missing |
| pre-epilogue FP32 output | not exposed by binding | missing |
| post-epilogue output | mismatch reproduced | `nv-q4down-localization-20260829/matched-ab.json` |

The first divergent stage, first output coordinate, expected/actual value, and
owning source expression therefore cannot be proven from the retained tensors.
The current source contains multiple possible ownership points in Q4 nibble
decode, Q4 metadata addressing, and the corrected subtotal expression; selecting
one would violate the C0 stop rule.

## Narrow unblock

Add a diagnostic-only, default-unreachable single-role harness that uses the
same saved-Z input and retains: decoded Q4 metadata, each K32 integer subtotal,
the corrected subtotal before epilogue, and final output for `blk.4.ffn_down`.
Compare each against an independent CPU/static oracle, then rerun this packet.
This is measurement infrastructure only; do not edit the production binding or
implement C1 until one stage and one owning expression are proven.

No timing claim, model edit, geometry sweep, or implementation change was made.
