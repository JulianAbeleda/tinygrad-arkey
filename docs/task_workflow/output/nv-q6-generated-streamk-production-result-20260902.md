# NVIDIA generated Q6 Stream-K FFN-down result

## Decision

Promote the exact generated Q6_K FFN-down route inside the qualified Qwen3-8B
pp512 compiler gate/up+K stack.  Keep attention-V research-only.  The rollback
is `NV_COMPILER_Q6_IMMA_PP512=0`.

## Primitive

- Input activation: FP16 `[512, 12288]`
- Weight: canonical packed Q6_K `[4096, 12288]`; no expansion or hot-path copy
- Producer: graph-owned Q8_1 Stream-K records
- Main: generated tinygrad UOps lowered through the CUDA renderer
- Geometry: 170 CTAs, 256 threads
- Reduction: deterministic destination-major generated fixup
- Output: FP32 `[512, 4096]`
- Per-call workspace: 22,282,240 bytes
- Model census: 18 FFN-down calls and 401,080,320 workspace bytes

## Same-session oracle gate

The historical pinned llama total of 209.856 us is retained as context, not as
the promotion authority, because GPU state moves both implementations.  The
authority is an interleaved same-session comparison over 31 samples.

| route | main us | fixup us | total us |
|---|---:|---:|---:|
| generated tile8+phase | 217.152 | 7.904 | 224.992 |
| live llama | 217.408 | 8.832 | 226.208 |

Generated-minus-llama total was -1.440 us median with 0.704 us MAD and 27/31
wins.  Output was bit-exact.

## Full pp512 model gate

The candidate and control used the same compiler gate/up+K stack.  The only
difference was generated Q6 FFN-down.

| arm | minimum ms | throughput tok/s | status |
|---|---:|---:|---|
| generated Q6 down | 65.107056 | 7863.971 | PASS |
| rollback control | 70.356494 | 7277.224 | PASS |

The promoted generated route recovered 5.249438 ms per 512-token chunk, an 8.1% model
throughput increase.  Replay was deterministic, both arms selected token 198,
and logits passed the recorded tolerance (`max_abs=0.12507677`,
`mean_abs=0.02871314`).

## Scope boundary

This proves and promotes FFN-down only.  It does not prove attention-V, does
not make an architecture-wide claim, and does not claim a universal win over
llama across uncontrolled GPU states.
