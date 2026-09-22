# NV flash-score floor test at HEAD (2026-08-16)

Date: 2026-08-16
Branch: `nvidia-bringup-20260731`
HEAD: `86d653651`
Status: **floor re-pinned at HEAD, closed NO-GO with fresh measurements.**

## 1. What this test establishes

Test-first floor re-measurement for the biggest open flash row
(`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`) at current HEAD,
run before any implementation is touched. It answers three questions with fresh
measurements, not inherited prose:

1. Have the installed flash rows drifted since the last census (`ef24c46ae`)?
2. Is the isolated production-tile body still 4.19 us (the 08-13 CUPTI audit),
   or has it changed at HEAD?
3. Where does the "~90 us structural floor" figure come from, and is it a
   reachable installed-graph number or a microbench artifact?

## 2. Evidence (all re-measured this session)

### 2a. NV census at HEAD (no drift)

`route_kernel_census.py --depth 512` on DEV=NV at `86d653651`:

| row | ef24c46ae (old pin) | 86d653651 (HEAD) | delta |
| --- | ---: | ---: | ---: |
| score med us x 36 | 6.52 x 36 = 239.87 | 6.56 x 36 = 241.39 | +1.5 us |
| combine med us x 36 | 3.39 x 36 = 123.02 | 3.39 x 36 = 122.80 | -0.2 us |
| tok/s | 207.427 | 205.415 | -2.0 |
| token sha | `227ad3ce...` | `227ad3ce...` | identical 3/3 |
| first token | 271 3/3 | 271 3/3 | identical |

Flash source (`tinygrad/llm/flash_decode_attention.py`) is unchanged since the
08-13 audit commit (`8b1acc998`); the census confirms no runtime drift either.

### 2b. Isolated production-tile body at HEAD (CUPTI)

`nv_flash_body_device_timing.py --only tile_s48_prod --replays 400` under
`nsys --trace=cuda`, DEV=CUDA (same metric as the llama node ledger), grid
48 x 8 x 32 x 4 = the production S=48 config:

| measure | HEAD (this session) | 08-13 audit | llama in-situ |
| --- | ---: | ---: | ---: |
| tile_s48_prod median us | **4.192** | 4.192 | 3.16 |
| tile_s48_prod avg us | 4.181 | 4.187 | - |

Body is unchanged at HEAD: still 4.19 us isolated, 1.33x llama. The isolated
body gap is +37 us/36 nodes; the honest in-situ gap (same metric as wall) is
the 08-13 +68 us.

### 2c. Production-shape microbench floor (the "90 us" source)

`flash_score_tile_peak_cuda.cu` at the production d512 shape (LPER=16,
nvalid=11, LANES=32, WARPS=4, TK=16, SW=1, MODE=2 installed structure),
RTX 5090 sm_120:

| run | us/kernel | note |
| --- | ---: | --- |
| warm, NACC=8, qhoist=1 (3000 iters) | **1.946** | steady-state peak |
| warm, NACC=8, qhoist=0 (3000 iters) | 1.986 | steady-state peak |
| warm, NACC=1, qhoist=1 (08-03 record) | 2.334 | installed render serial chain |
| cold, single launch (iters=1) | **7.616** | cold L2 per launch |

The "~90 us structural floor" (= 36 x 2.5 us) comes from the warm 08-03 record
peak (2.33 us). That peak is a microbench artifact: back-to-back passes inside
one launch, full cache L2-resident, operand setup hoisted. It is not reachable
in the installed graph, where each launch is cold-L2 with surrounding kernel
traffic: the cold single-launch reading is 7.6 us and the installed NV row is
6.56 us. The only measured floor that reflects the installed graph is the
isolated CUPTI body: 4.19 us.

## 3. Verdict

**Closed NO-GO at HEAD, no implementation change.** The flash score body is
unchanged (4.19 us isolated vs llama 3.16 us), the installed rows have not
drifted, and the "~90 us floor" is a warm microbench peak, not an
installed-graph floor. No new mechanism appeared on this test:

- tile geometry sweep (08-03): NO-GO
- 512-thread single-stage combine (08-05): NO-GO
- llama-vec single-pass as-is (08-13): 10.2 us NCHUNK=2, slower than tile
- multi-stream overlap (08-15): FLAT

The honest recoverable flash score mass remains the 08-13 figure: ~+68 us
in-situ (~+2.7 tok/s ceiling), not the +128 us the old ledger attributed, and
there is no ready kernel to capture it. Any future attempt must first show a
device-side body at production config faster than the 4.19 us tile, otherwise
it is not a win.
