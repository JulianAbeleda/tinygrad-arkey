# Attention compact VGPR lease experiment: negative result

## Question

Can single-wave Hd=128 attention reduce its current physical VGPR ceiling by moving its serialized A/B fragments from the generic `v200..v237` fragment window into a private partition?

The tested partition was:

- persistent PV C: `v8..v71`
- persistent m/l: `v72..v87`
- QK/alpha state: `v88..v103`
- serialized attention A/B: `v104..v119`
- allocator temporaries: `v120..v191`
- generic `FRAG_BASE=200..238`: unchanged for non-attention WMMA

The implementation was explicitly disabled for `AMDMultiWaveAttentionGridSpec`, so it did not change the concurrent multi-wave state ABI.

## Gate result

The candidate compiled and ran without a spill/private-memory failure. Numeric output remained correct, but corrected TinyJit replay did not improve:

| Profile | KV | Existing median | Compact median | Change | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 8B Q4_K_M | 512 | 0.5445 ms | 0.5523 ms | +1.43% | 6.10e-05 |
| Qwen3 8B Q4_K_M | 4096 | 3.4832 ms | 3.5689 ms | +2.46% | 1.91e-06 |

Commands used the corrected replay harness with `DEV=AMD`, 1 warmup, and 10 samples. The KV4096 run used `DEBUG=4` and completed normally.

## Decision

Do not retain the compact partition. It adds an attention-specific allocator ABI without a measured performance benefit and slightly regresses both tested endpoints. The implementation was reverted; this report is the retained artifact.

The primitive conclusion is that merely lowering physical register numbers is not the next useful lever. Further register work should first prove an occupancy change from compiler resource metadata, then test a lifetime reduction that lowers the allocated register count rather than remapping the same live state.
