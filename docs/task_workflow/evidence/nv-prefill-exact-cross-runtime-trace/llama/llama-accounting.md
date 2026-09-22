# llama.cpp pp512 exact lifecycle accounting

## Result

The settled profiled repetition contains **1,186 classified launches and zero unknowns**. Its measured wall is **35.430083 ms**. The device interval union is **32.683341 ms**, internal device idle is **0.201442 ms**, and the measured boundary residual is **2.545300 ms**. Those quantities close the profiled wall exactly; no profile share is scaled onto the unprofiled wall.

## Workload and settled convention

The frozen workload is Qwen3-8B-Q4_K_M, CUDA, `-ngl 99 -fa 1 -p 512 -n 0`, with llama.cpp build `ac4cddeb0` (build 9592). Exact binary, model, shared-library, source-state, command, machine, and trace hashes are in `llama-provenance.json`.

The fresh unprofiled R9 samples, in benchmark order, are:

`42.288655, 34.680367, 34.714590, 34.881845, 35.046325, 34.992473, 35.118109, 35.197799, 35.194633 ms`

Sample 0 is retained but excluded from the settled statistic. The marker R2 trace shows why: the first timed repetition performs first-use CUDA graph setup, while the second is a replay of the completed graph. The settled convention is therefore samples 1-8, whose median is **35.019399 ms** (min 34.680367, max 35.197799). This is a lifecycle-state rule, not post-hoc statistical outlier removal.

The profiled R2 samples are 50.281306, 35.430083 ms. Accounting selects repetition 1, matching the settled state. Profiling adds 0.410684 ms (1.173%) versus the unprofiled settled median. The first timed interval contains stream capture, graph instantiation, graph-exec update, and launch; the settled interval contains only graph launch among those APIs.

## Exact profiled-wall reconciliation

A preload-only shim brackets the unchanged binary's `system_clock::now()` reads with NVTX marks and records the returned clock values. It does not replace the clock result or modify the benchmark binary.

| Quantity | Exact value |
|---|---:|
| Returned start epoch | 1787975917445559422 ns |
| Returned end epoch | 1787975917480989505 ns |
| Start-read trace bracket | 2630425746-2630426788 ns (1042 ns wide) |
| End-read trace bracket | 2665850208-2665857192 ns (6984 ns wide) |
| Profiled benchmark wall | 35.430083 ms |
| Device span | 32.884783 ms |
| Device interval union | 32.683341 ms |
| Device idle inside span | 0.201442 ms across 1,071 gaps |
| Boundary residual | 2.545300 ms |
| Closure error | 0 ns |

The pre-device residual is bounded to 2.491062-2.492104 ms and the post-device residual to 0.047575-0.054559 ms by the marker brackets. The exact identity is:

`35.430083 ms wall = 32.683341 ms device union + 0.201442 ms internal idle + 2.545300 ms boundary residual`

Individual kernel durations sum to 32.759118 ms, exceeding the interval union by 0.075777 ms because CUDA graph programmatic dependencies permit small overlaps.

## Semantic-region accounting

`Active` is the sum of individual launch durations in a primary region. `Exclusive union` is wall time during which only that primary region is active. Cross-region overlap is kept separate rather than assigned twice.

| Primary region | Launch-active ms | Exclusive-union ms |
|---|---:|---:|
| input/embed and graph setup | 0.000000 | 0.000000 |
| RMSNorm and activation conversion | 2.145511 | 2.109191 |
| Q | 2.514450 | 2.514450 |
| K | 1.251141 | 1.251141 |
| V | 1.053720 | 1.053720 |
| Flash score/reduction | 1.657447 | 1.639559 |
| O | 2.397213 | 2.397213 |
| gate | 6.086529 | 6.082529 |
| up | 6.111329 | 6.111329 |
| activation/multiply | 0.903843 | 0.903843 |
| down | 6.940482 | 6.940162 |
| residual, RoPE, KV write, and other support | 1.379275 | 1.320458 |
| final-row gather/prune | 0.005024 | 0.001312 |
| vocabulary | 0.313154 | 0.310530 |
| output/token transfer | 0.000000 | 0.000000 |
| unknown | 0.000000 | 0.000000 |

Cross-region overlap union:

- Flash score/reduction + residual, RoPE, KV write, and other support: 0.017888 ms
- RMSNorm and activation conversion + down: 0.000320 ms
- RMSNorm and activation conversion + final-row gather/prune: 0.000896 ms
- RMSNorm and activation conversion + final-row gather/prune + gate: 0.000672 ms
- RMSNorm and activation conversion + gate: 0.003328 ms
- RMSNorm and activation conversion + residual, RoPE, KV write, and other support: 0.022176 ms
- RMSNorm and activation conversion + vocabulary: 0.002624 ms

The complete launch-by-launch role map, all per-role totals, all 36 per-layer totals, and the interval timeline are machine-readable in `llama-role-map.json`, `llama-accounting.json`, and `llama-intervals.json`. The role assignment uses source-defined graph order plus exact template, grid, block, and graph-node sequence; a topology mismatch aborts the parser.

## Per-layer projection view

All values below are launch-active milliseconds. `gate+up` is combined because the final M=1 gate/up launch is physically fused and cannot be honestly split.

| Layer | Total | Q | K | V | Flash | O | gate+up | down |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.952547 | 0.072127 | 0.040673 | 0.029024 | 0.046208 | 0.066367 | 0.349217 | 0.224193 |
| 1 | 0.946179 | 0.072159 | 0.039649 | 0.028896 | 0.045664 | 0.065984 | 0.346112 | 0.223137 |
| 2 | 0.946466 | 0.072416 | 0.039553 | 0.028959 | 0.045728 | 0.066528 | 0.347298 | 0.222689 |
| 3 | 0.949186 | 0.071967 | 0.039552 | 0.029024 | 0.046816 | 0.066688 | 0.346626 | 0.224257 |
| 4 | 0.885443 | 0.072095 | 0.029088 | 0.029504 | 0.046624 | 0.066304 | 0.346210 | 0.170913 |
| 5 | 0.882371 | 0.067456 | 0.029056 | 0.029921 | 0.045888 | 0.066751 | 0.347266 | 0.172001 |
| 6 | 0.945796 | 0.068032 | 0.041248 | 0.028448 | 0.046241 | 0.066304 | 0.346402 | 0.224737 |
| 7 | 0.887108 | 0.072159 | 0.029120 | 0.029696 | 0.045569 | 0.066720 | 0.347138 | 0.172897 |
| 8 | 0.880291 | 0.067041 | 0.029056 | 0.029855 | 0.045856 | 0.066400 | 0.346978 | 0.170977 |
| 9 | 0.946338 | 0.067295 | 0.041408 | 0.028511 | 0.046368 | 0.066561 | 0.346914 | 0.225473 |
| 10 | 0.888452 | 0.072160 | 0.029344 | 0.029888 | 0.046400 | 0.067040 | 0.347330 | 0.172513 |
| 11 | 0.880864 | 0.067167 | 0.029120 | 0.029759 | 0.045728 | 0.066527 | 0.347458 | 0.172097 |
| 12 | 0.942883 | 0.067264 | 0.041536 | 0.028703 | 0.045185 | 0.065888 | 0.346946 | 0.224257 |
| 13 | 0.886565 | 0.072064 | 0.028800 | 0.029504 | 0.046721 | 0.066688 | 0.346818 | 0.172193 |
| 14 | 0.882051 | 0.067520 | 0.029248 | 0.028992 | 0.046560 | 0.066368 | 0.347586 | 0.171041 |
| 15 | 0.946596 | 0.067103 | 0.040832 | 0.029472 | 0.045792 | 0.066305 | 0.347458 | 0.225441 |
| 16 | 0.887459 | 0.072480 | 0.029184 | 0.029728 | 0.046848 | 0.066400 | 0.347330 | 0.171072 |
| 17 | 0.882081 | 0.067232 | 0.029088 | 0.029856 | 0.046240 | 0.066559 | 0.346658 | 0.172256 |
| 18 | 0.944357 | 0.067936 | 0.040800 | 0.028736 | 0.044960 | 0.066624 | 0.347362 | 0.223745 |
| 19 | 0.886789 | 0.071999 | 0.029152 | 0.029792 | 0.046112 | 0.066368 | 0.347362 | 0.171617 |
| 20 | 0.881956 | 0.067297 | 0.029088 | 0.029823 | 0.045984 | 0.066689 | 0.346626 | 0.172353 |
| 21 | 0.943268 | 0.067360 | 0.040768 | 0.028640 | 0.045856 | 0.066240 | 0.346658 | 0.223489 |
| 22 | 0.885540 | 0.071904 | 0.029152 | 0.029088 | 0.046208 | 0.066816 | 0.346882 | 0.171649 |
| 23 | 0.881346 | 0.066752 | 0.029280 | 0.029632 | 0.045376 | 0.066335 | 0.347266 | 0.172705 |
| 24 | 0.944449 | 0.067232 | 0.041216 | 0.028863 | 0.046081 | 0.066079 | 0.346498 | 0.224288 |
| 25 | 0.884738 | 0.071583 | 0.028864 | 0.029952 | 0.046145 | 0.066559 | 0.346690 | 0.171041 |
| 26 | 0.883300 | 0.067809 | 0.029024 | 0.029695 | 0.046496 | 0.066657 | 0.347586 | 0.172129 |
| 27 | 0.943907 | 0.067743 | 0.040096 | 0.028479 | 0.046240 | 0.066177 | 0.347106 | 0.223393 |
| 28 | 0.886212 | 0.072384 | 0.028736 | 0.029664 | 0.045824 | 0.066656 | 0.347170 | 0.170945 |
| 29 | 0.886498 | 0.067967 | 0.029568 | 0.028896 | 0.045920 | 0.066912 | 0.349410 | 0.172513 |
| 30 | 0.947621 | 0.067552 | 0.040768 | 0.028863 | 0.045697 | 0.066688 | 0.348066 | 0.224993 |
| 31 | 0.952034 | 0.072511 | 0.040288 | 0.028992 | 0.045568 | 0.067584 | 0.348066 | 0.223586 |
| 32 | 0.952227 | 0.072863 | 0.039264 | 0.029313 | 0.046496 | 0.066720 | 0.347266 | 0.224290 |
| 33 | 0.954308 | 0.072703 | 0.040193 | 0.029280 | 0.045696 | 0.067200 | 0.349442 | 0.224993 |
| 34 | 0.955491 | 0.072607 | 0.039681 | 0.029152 | 0.046272 | 0.067263 | 0.349697 | 0.226017 |
| 35 | 0.726401 | 0.072511 | 0.039648 | 0.029120 | 0.046080 | 0.067264 | 0.040960 | 0.034592 |

## Final-row pruning proof

Executed launches prove that layers 0-34 contain 35 full M=512 FFNs: 35 gate main, 35 up main, and 35 down main launches. After layer 35 O, exactly 3 gather/add launches reduce to one row. Layer 35 then uses an M=1 FFN norm (grid [1, 1, 1]), one fused gate+up MMVQ (grid [12288, 1, 1]), and one down MMVQ (grid [4096, 1, 1]). The selected graph therefore does not execute a 36th full M=512 FFN.

## Tail-work equivalence caveat

llama-bench executes the final Q6_K vocabulary MMVQ (0.313154 ms active), then synchronizes. Its `test_prompt` does not read logits or invoke GPU argmax/sampling. The matched tinygrad safe cut executes its vocabulary kernels and also a GPU finite-fp32 argmax plus token copy (0.009248 ms active). Vocabulary projection is comparable; post-vocabulary token selection is not one-to-one because llama-bench does not perform that work. No absent llama work is assigned a synthetic charge.

## Artifacts

- Raw: `llama-trace.nsys-rep`, `llama-trace.sqlite`, and benchmark JSON outputs.
- Exports: `llama-kernels.csv`, `llama-cuda-api.csv`, and `llama-graph.csv`.
- Accounting: `llama-intervals.csv`, `llama-intervals.json`, `llama-role-map.json`, and `llama-accounting.json`.
- Provenance and source mapping: `llama-provenance.json` and `llama-source-map.json`.
