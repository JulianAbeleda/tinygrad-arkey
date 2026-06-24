# Q4_K MMVQ llama-scheduler probe — baseline/design (2026-06-18)
Probe of llama's 128-thread/row MMVQ decomposition. Baselines: base fp ~40%, fp coop ~48%, _sdot4 8-thread ~49%,
opaque asm ~52%, llama/READRAW ~70%. Llama scheduler (audited): 128 threads/row, 16 K-blocks parallel across
threads, warp-shuffle + shared reduce, one write (vs tinygrad's 8 threads/row + serial K + global partials +
stage-2). Build results + verdict: see qk-mmvq-llama-scheduler-probe-verdict-20260618.md.
