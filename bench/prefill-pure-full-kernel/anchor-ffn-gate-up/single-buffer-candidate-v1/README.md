# Exact pure single-buffer candidate authority

This bundle records the first end-to-end passing generated candidate for the
Qwen3 8B `ffn_gate_up` anchor (`M=512, N=12288, K=4096`) on AMD gfx1100.

- Candidate: `81c27275d1aad1bb8147c5c5cdaa8000e9375e81f3d085b49d62064a731313d6`
- Source commit: `c52e562358c31506fd1d55cdd012e66b48cf1ec1`
- Binary: `e5b988f008de36242bff886b46daae1dc82816547832c0c63da72ef7c84b6c1c`
- Generated topology: tile `128x128x32`, waves `4x2`, local size `(32,4,2)`
- Generated LDS: one 20,480-byte allocation, A/B stride 80, zero spills/scratch
- Correctness: full 6,291,456-element constant and row/column-varying cases, zero error
- Kernel timing: 0.953 ms median / 54.10 TFLOPS; 0.778 ms best / 66.25 TFLOPS
- Evaluator: all five authority stages pass with candidate/binary/commit joins

This proves one exact candidate can own generated TC topology and LDS address
semantics through Tinygrad. It does not claim a generalized machine-search
space. Tile, wave, stride, vector, pipeline, dependency, and epilogue fields
remain capability-limited as documented in the pure-path scope.
