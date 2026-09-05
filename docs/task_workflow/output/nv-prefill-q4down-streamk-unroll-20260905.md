# Q4-down Stream-K unroll comparison

Fresh interleaved live llama runs use the same 18 packed down tensors, graph-owned producer/main/fixup lifecycle, two activation correctness replay, and 31 alternating calls. Baseline Stream-K median was 6.152 ms; unroll4 was 5.636 ms; unroll8 is 5.481 ms. The unroll8 result is finite and tolerance-correct (`max_abs=0.0121557`) but remains slower than live llama at 3.710 ms. All variants remain research-only.
