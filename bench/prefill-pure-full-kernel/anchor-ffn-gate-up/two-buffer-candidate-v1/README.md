# Generated two-buffer candidate authority

This bundle records the first passing generated two-buffer/stage-1 candidate
for the Qwen3 8B `ffn_gate_up` anchor (`M=512, N=12288, K=4096`) on gfx1100.

- Candidate: `7e37ad6e13dee573758b3264aa9ec71b0cd91f1ca08e012d279d67300360c2d9`
- Source commit: `774ab015b85c3663064bb1e89ce2f3aa597761cc`
- Binary: `800ffd5209078db2da6a3de2beb1f70cc6b1926610a7be400030b0338df955e5`
- Topology: tile `128x128x32`, waves `4x2`, local size `(32,4,2)`
- Pipeline: two 20,480-byte slots, prologue/body/drain, K16 -> K16 recurrence
- Resources: 40,960-byte LDS, 188 VGPR, 18 SGPR, zero spills/scratch
- Correctness: full 6,291,456-element row/column-varying output, zero error
- Kernel timing: 0.6734 ms median / 76.54 TFLOPS; 0.5821 ms best / 88.54 TFLOPS
- Evaluator: all five authority stages pass with candidate/binary/commit joins

The generated median exceeds the prior approximately 75-TFLOPS S9 oracle
reference. This is a fixed, proven two-buffer schedule; deeper stages, wait
policies, K unroll choices, and broader shape search remain future dimensions.
