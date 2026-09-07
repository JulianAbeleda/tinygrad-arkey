# Decode ctx1024 cold and independent-request qualification

Three separate ordinary generated processes return identical tokens, select Flash, and capture the same 418-program decode graph with the same 29 unique renderer sources. Runs 1 and 2 use the normal persistent compiler cache. The third run uses a unique isolated empty `CACHEDB` namespace; it creates 80 NV compile-cache rows in a 1,818,624-byte SQLite database without clearing or modifying the normal cache.

| run | cache condition | construct | prefill + prelude | first decode | steady median | steady min | sampled memory after prefill |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | fresh process, warm disk cache | 18.211 s | 246.203 s | 4.520 ms | 4.236 ms | 4.217 ms | 30,538 MiB |
| 2 | fresh process, warm disk cache | 18.125 s | 245.765 s | 4.546 ms | 4.235 ms | 4.219 ms | 30,538 MiB |
| isolated | fresh process, empty isolated cache | 18.355 s | 246.436 s | 4.540 ms | 4.248 ms | 4.227 ms | 30,538 MiB |

Each run starts at 117 MiB GPU memory, reaches 18,472 MiB after model construction, and reports 19,259,343,980 tinygrad live bytes and 30,538 MiB NVIDIA memory after prefill. Memory values are phase samples rather than a continuous peak. Every process exits cleanly and returns the GPU to 117 MiB.

Token evidence is identical across all three processes: prelude `34208`, first decode `13`, then steady tokens `[279, 3974, 13876, 38835, 34208, 13]`. The first decode overhead relative to its run's steady median is 0.284 ms, 0.311 ms, and 0.292 ms. Compilation is therefore accounted primarily in model construction and the 246-second prompt prefill/capture phase, while first decode after the prelude has a small bounded transition cost.

This qualifies fresh-process and isolated-empty-compiler-cache behavior for the generated route. It does not compare cold compile latency against llama because the llama measurements do not use an equivalent empty build/cache condition.
