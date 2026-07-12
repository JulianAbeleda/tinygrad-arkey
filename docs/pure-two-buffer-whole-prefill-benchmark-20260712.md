# Pure two-buffer whole-prefill benchmark

The generated two-buffer candidate was benchmarked through the canonical
whole-prefill harness against the ordinary pure PREFILL_V2 scheduler. Both
runs used the same Qwen3-8B Q4_K_M GGUF, 512-token chunks, `K=8`, four
warmups, three rounds, device synchronization, and requested clock pinning.

The candidate was scoped to `ffn_gate_up`; all other roles stayed on the
ordinary generated scheduler. Oracle rollback was disabled.

| Context | Pure scheduler | Two-buffer pure | Speedup |
|---:|---:|---:|---:|
| 512 | 1,511 tok/s | 2,431 tok/s | 1.61x |
| 1,024 | 1,473 tok/s | 2,384 tok/s | 1.62x |
| 2,048 | 1,410 tok/s | 2,241 tok/s | 1.59x |
| 4,096 | 1,324 tok/s | 2,019 tok/s | 1.52x |

The initial one-warmup smoke values were discarded because the timed call still
included TinyJit capture. The table uses the established authority protocol.

Kernel-level authority for the same candidate reports full-output zero-error
correctness, 40,960-byte LDS, zero spills/scratch, and 76.54 TFLOPS median on
the joined binary. Whole-model speedup is smaller because only the gate/up role
uses this candidate and attention/other linear roles remain unchanged.
