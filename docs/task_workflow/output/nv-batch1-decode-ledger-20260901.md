# NVIDIA batch-1 decode ledger

This is the retained strict Qwen3-8B Q4_K_M batch-1 comparison on NVIDIA. It is
a protocol-specific result, not a universal claim across GPU temperature and
clock regimes.

## Endpoint

| Runtime | Latency | Throughput |
|---|---:|---:|
| tinygrad promoted | 4.035563 ms/token | approximately 247.80 tok/s |
| llama retained strict | 4.058359 ms/token | 246.405 tok/s |
| tinygrad advantage | 22.796 us/token | approximately 1.40 tok/s |

Under this retained strict protocol tinygrad is approximately 0.56% faster.
Historical llama measurements range from approximately 240.6 to 250.7 tok/s,
so this does not establish a universal thermal-regime win.

## Active-body regional ledger

| Region | tinygrad advantage | Winner |
|---|---:|---|
| gate/up projections | 26.220 us/token | tinygrad |
| FFN down | 41.164 us/token | tinygrad |
| QKV/provider | 16.385 us/token | tinygrad |
| norms | 92.898 us/token | tinygrad |
| Flash score | 4.608 us/token | tinygrad |
| O projection | -3.007 us/token | llama |
| vocabulary lifecycle | -8.372 us/token | llama |
| **net active-body advantage** | **169.896 us/token** | **tinygrad** |

Negative values mean tinygrad loses. The active-body ledger is larger than the
endpoint margin because runtime scheduling, graph boundaries, token feedback,
and sampling are outside or differently charged by the regional kernel ledger.

## Vocabulary and roofline

| Stage | tinygrad | llama | tinygrad delta |
|---|---:|---:|---:|
| activation preparation | no separate Q8 node | 0.672 us | +0.672 us |
| projection body | 309.974 us | 300.930 us | -9.044 us |
| complete projection lifecycle | 309.974 us | 301.602 us | -8.372 us |

The practical streaming floor is 300.064 us. Llama is 0.866 us above that
floor; the promoted tinygrad body is 9.910 us above it. The experimental
packed-word Q8 route removed a millisecond-scale bad consumer expression, but
its complete provider-inclusive lifecycle remained slower than the promoted
FP16 route and was not promoted.

## Current claim

Tinygrad wins this retained strict endpoint because its norm, FFN, and QKV
advantages outweigh losses in O projection and vocabulary. The next comparison
must be a fixed-depth curve; a depth-512 endpoint alone cannot establish how the
lead behaves as KV context grows.
