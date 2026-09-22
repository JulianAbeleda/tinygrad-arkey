# Q4-down saved-z producer gate (2026-08-29)

Target was the first real Q4-down tensor, `blk.4.ffn_down.weight`, GGML type
12, with shape M=512, K=12288. The producer used the specialized 24-segment
K12288 geometry (grid 512x24, block 128).

The gate is **FAIL / STOP**. The producer was finite and its FP32 scales were
bit-exact after matching the CUDA reciprocal literal; FP16-rounded raw sums
were also bit-exact. Quantized values were not bit-exact against the current
host reference: 158 of 6,291,456 values differed. Timing was 7.488 us minimum
(9 samples; median 7.552 us), with 36 registers, 1024 shared bytes, and 576
local bytes.

Because the value ABI is not exact, the compiler composition and model
integration gates were not attempted. This lane remains substrate-blocked;
the next required test is an independent CUDA reference for the producer's
round-to-int8 instruction sequence, then rerun this gate before any timing
claim.

Evidence: `docs/task_workflow/evidence/nv-q4down-capture-20260829/saved-z-k12288-gate.json`.
