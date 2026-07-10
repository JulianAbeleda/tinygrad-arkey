# llama.cpp MMQ Research Source

This directory is a research-only snapshot of llama.cpp's MMQ source used to
reduce the Qwen3 14B Q4_K prefill kernel into tinygrad-owned atoms.

- Source: `/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh`
- Source commit: `ac4cddeb0dbd778f650bf568f6f08344a06abe3a`
- License: MIT, preserved in `LICENSE.llama.cpp`
- SHA256: `6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f`

This file is not imported by production dispatch and is not a selectable
backend. Converted pieces must still move through bounded correctness,
resource, route, and promotion gates before becoming tinygrad runtime code.
