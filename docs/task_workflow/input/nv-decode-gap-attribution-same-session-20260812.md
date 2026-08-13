# NV decode gap attribution - same-session llama vs tinygrad pair (2026-08-12)

Date: 2026-08-12
Branch: `nvidia-bringup-20260731` (HEAD `a8b560457`, fp32 q/k promoted)
Status: **attribution record with evidence.** Fresh unprofiled llama-bench +
same-session nsys node ledger (llama) and DEBUG=2 prime-token kernel trace
(tinygrad) at d512 on Qwen3-8B-Q4_K_M / RTX 5090. Answers "where exactly is
the lag" in us/token per class, and maps each recoverable row onto the
existing ledger with a tok/s ceiling. No implementation authorized here.
Process: trace llama -> arithmetic-validate -> e2e dependency map -> implement
(standing process, `0515f2539`).

## 1. Headline (d512, same session)

| side | tok/s | ms/token | source |
| --- | ---: | ---: | --- |
| llama (`ac4cddeb0`, fresh `llama-bench -p 512 -n 10 -d 512 -r 5`) | **245.45 +/- 12.52** | **4.0741** | pair run; nsys ledger `/tmp/llama_tg10_node_20260812.sqlite`; second fresh rep confirms 247.98 +/- 8.53 = 4.0366 ms (raw JSON below) |
| tinygrad (production HEAD, all promoted policies) | **192.19** | **5.2031** | M2c/M2d/fp32-q/k promotion brackets, bit-exact token hash `9e6664fd` |
| **gap** | | **+1.129 ms/token (1.278x)** | |

Fresh llama sweep (same build, unprofiled, raw JSON in evidence): d512
247.98 +/- 8.53, d2048 234.76 +/- 7.14, d4096 224.02 +/- 10.30, pp512
14,302.6 +/- 494.3 tok/s (the paired run measured d512 245.45 +/- 12.52;
rep-to-rep spread ~1%).

## 2. Wall decomposition (where the 1.129 ms lives)

llama per replay (nsys node ledger, graphId 5, 762 nodes, 16 replays, 1 stream):
node-sum **4774.4 us**, kernel union **3826.9 us**, span **3835.2 us**, median
inter-replay host gap **212.5 us**. Check: 3835 + 212 = 4047 us ~= the 4074 us
bench wall (94% GPU busy). The 4774 us node-sum is an upper bound: ~946 us of
per-node launch accounting sits inside it (overlap_mass), so llama's true GPU
busy is the 3835 us span.

tinygrad (DEBUG=2 prime token, kernels 1545-2126, 582 rows): kernel-sum
**5424.9 us**, prime wall span **5430 us**. Production wall is 5.2031 ms, so the
campaign replay factor (0.883, graphs amortize launch gaps) gives a
replay-equivalent busy of **~4790 us** and a host/gap share of ~413 us (92%
busy).

| component | llama | tinygrad | delta |
| --- | ---: | ---: | ---: |
| GPU busy (kernels) | ~3835 us | ~4790 us (est) | **+955 us (85% of gap)** |
| host launch / inter-token gap | ~239 us | ~413 us | +174 us (15%) |
| wall | 4074 us | 5203 us | +1129 us |

The gap is GPU/kernel-bound, not host-bound: llama's graph replay hides launch
cost better, but the dominant delta is more kernel work per token.

## 3. Per-class same-metric comparison (the 652 us of kernel accounting)

Both columns are the same kind of measurement: sum of per-kernel durations over
one decode token in the same session (llama nsys node ledger; tinygrad
DEBUG=2 prime window). Class deltas sum exactly to +652.4 us.

| class | llama us/token | tinygrad us/token | delta us | note |
| --- | ---: | ---: | ---: | --- |
| matmul+quant+output-reduce (non-vocab) | 3721.2 (mmq 3239.0 + quantize_q8_1 482.2) | 4116.3 (q4k 2812.5 + q6k 911.8 + reduce_output 392.0) | +395.1 | GEMV bodies alone are at parity (3724.3 vs 3721.2, +3.1); the +392 is our fp32 q/k reduce (240.5) + FFN-down output reduce (151.5), which llama absorbs in-kernel |
| vocab head | 303.6 (mmq node) | 380.8 (151936 GEMV 323.5 + E_/r_ aux 57.3) | +77.2 | GEMV at 1.07x; the aux chain is extra plumbing |
| flash score | 113.9 (36 x 3.16) | 280.4 (36 x 7.79) | +166.5 | 2.5x/node; combine is exact parity (3.35 both) |
| flash combine | 120.5 | 120.5 | 0.0 | parity |
| norms | 307.6 (145 x 2.12) | 49.6 (17 x 2.92) | -258.0 | llama runs standalone rms_norm; we fuse the norm epilogue into GEMVs |
| rope | 126.8 (72 x 1.76) | 0 | -126.8 | fused into q/k GEMV epilogue |
| kv cache store | 74.3 (36 x 2.06) | 0 | -74.3 | fused into flash/plumbing |
| small elt/reduce plumbing | 4.8 | 477.1 (e_ 233.6 + r_ 243.5) | +472.3 | the E_/r_ set: M1 norm chains (229.5), attention K/V extras, fp32 q/k epilogue |
| **total** | **4774.4** | **5424.9** | **+652.4** | |

Two striking results: (1) **matmul+quant is at parity once GEMV bodies are
compared** - our in-kernel q8/quant folding cancels llama's separate
quantize_q8_1 (482 us); the +392 us is the reduce-output epilogue we have not
absorbed. (2) **the norm and rope rows are already better than llama** (fused);
the entire recoverable mass is reduce-output epilogue, flash score, vocab aux,
and the E_/r_ plumbing set.

## 4. Per-shape quant cross-check (same session, fresh llama floors)

| shape | llama us/node | tinygrad us/node | ratio | tinygrad total us | ledger status |
| --- | ---: | ---: | ---: | ---: | --- |
| gate/up 12288x4096 (w1w3fused16) | 37.86 | 39.12 | 1.03x | 1408.3 | landed (MC3+M2a), parity |
| attention O 4096x4096 (epi_resadd) | 11.78 | 9.97 | 0.85x | 359.0 | landed (M4), better than llama |
| attention Q 4096x4096 | 9.54 | 9.51 | 1.00x | ~171 | parity |
| attention K/V Q4 1024x4096 | 3.33 | 4.83 | 1.45x | 130.4 | shared-Q8 on 26/54 |
| FFN-down Q4 4096x12288 (epi_ffnresadd) | 11.78 | 26.94 | **2.29x** | 484.9 | **never swept (audit rank 1)** |
| FFN-down Q6 4096x12288 | 28.75 | 35.00 | 1.22x | 630.0 | MC2 coop NO-GO |
| attention V Q6 partial 1024x4096 | 4.90 | 17.89 | 3.65x | 178.9 | MC2 partial NO-GO (local optimum) |
| vocab 151936x4096 | 303.6 | 323.5 | 1.07x | 323.5 | landed, near parity |

Confirms the quant audit (`nv-quant-gemv-llama-audit-20260812.md`): the Q4
FFN-down shape is the one open quant row. At llama's floor it is worth
18 x 15.2 = **~273 us/token**; the other shapes are closed NO-GO or landed.

## 5. Ranked recoverable rows with tok/s ceilings

Baseline 5.2031 ms/token = 192.19 tok/s. Ceiling = full class delta recovery at
1:1 census-to-wall (body-adding changes historically map ~0.6, pure removal
~1.0+).

| rank | row | us/token | ceiling tok/s | ledger mapping |
| --- | --- | ---: | ---: | --- |
| 1 | reduce-output epilogue (fp32 q/k 240.5 + FFN-down 151.5) | 392.0 | **207.9** (+15.7) | fp32 q/k route booked +83.5 us wall; the reduces are the leftover epilogue mass (M1/M2-style fold sites) |
| 2 | Q4 FFN-down 4096x12288 GEMV body | ~273 | **202.8** (+10.6) | quant audit rank 1, never swept; in-loop offset 1.26-1.7x keeps partial wins positive |
| 3 | flash score | 166.5 (90 at structural floor) | **198.5** (+6.4; +3.4 at floor) | 08-03 tile sweep NO-GO: zero-load floor 5.3 us is still 1.7x llama; structural, not geometry |
| 4 | M1 norm chains (r_16_256 + E_f14a5cc0, 36 chains) | 229.5 census | ~197-193 line | ledger line to 193; prior fold attempt NO-GO +81.9 us (cost gate contradicted) |
| 5 | vocab aux chain (E_1187, r_32_4_1187, r_128_16_8_1187, r_16_8) | 57.3 | **194.4** (+2.2) | L4 substrate landed; aux is the remaining tail |
| 6 | attention K/V extras + Q6 partials | ~330 | closed | NO-GO per MC2/shared-Q8 sweeps; no new mechanism |

Closing all open rows (1-5, at floor) lands at ~4.55 ms/token ~= 220 tok/s on
kernel accounting alone; full wall parity (busy+host) is 4.074 ms = 245 tok/s.
The last ~0.30 ms of that is llama's superior launch hiding (graph replay), not
kernel work.

## 6. Verdict

- **192 -> 193 needs ~-22 us wall.** The ledger's line is the M1 norm row; this
  trace prices it at 229.5 us census (36 chains), i.e. ~18-23 us wall only if
  the fold is body-free (pure removal of r_16_256 + E_f14a5cc0). The prior M1
  attempt added body work and lost 81.9 us, so a fresh M1 needs a body-free
  site or stays NO-GO.
- **193 -> ~200 needs the quant row**: Q4 FFN-down sweep (rank 2, ~273 us
  ceiling) plus the reduce-output folds (rank 1, 392 us ceiling).
- **200 -> parity (~245) needs flash-score structure (rank 3) and launch
  hiding** - llama's 94% vs our 92% busy, plus the ~0.30 ms replay overhead
  delta.

## Evidence

- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json` (nsys
  node ledger, graphId 5)
- `docs/task_workflow/evidence/nv-tinygrad-prime-gap-table-20260812.json`
  (parsed prime-token table, 582 kernels)
- `docs/task_workflow/evidence/nv-llama-bench-fresh-20260812.json` (raw
  llama-bench JSON, d512/d2048/d4096/pp512)
- raw traces: `/tmp/llama_tg10_node_20260812.sqlite`,
  `/tmp/tg_debug_probe_20260812.log`
