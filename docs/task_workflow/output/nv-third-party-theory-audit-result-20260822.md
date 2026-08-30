# Independent NV theory audit — findings first

Claim labels in this report are literal: **MEASURED** is reconstructed from a retained artifact or a fresh run, **INFERRED** is a conclusion from measured observables, and **UNMEASURED** identifies a quantity that the available experiment did not measure. A projected device-time ceiling is never called a wall result.

## Executive verdict

1. **MEASURED — CRITICAL:** The claimed `reduce + residual = +452 us` hotspot is not a real semantic row. The classifier maps every generated `r_*` program to “reduce” and every `E_*` program to “residual” at `extra/llm_research/decode/nv_inter_anchor_analysis.py:104-116`; those buckets include unrelated generated work. Current model code already absorbs the attention residual in the O epilogue at `tinygrad/llm/model.py:654-670` and the FFN residual in the down epilogue at `tinygrad/llm/model.py:675-689`.

2. **MEASURED — CRITICAL:** The closest executable version of the proposed boundary fusion lost. Across 36 layers, the old `r_16_256_*` plus `E_32_32_4_f14*` norm pair used **185.152 us node_sum**; the one-pass boundary row `reduce_output_rmsnorm_1_4096` used **254.592 us node_sum**, a **+69.440 us** regression. Its independent control/candidate/control wall bracket was **+59.602 us/token**, with bitwise-equal logits and equal token SHA. This is the specific row where the fusion approach failed.

3. **MEASURED — CRITICAL:** The handoff identity `union == useful_body - useful_overlap` at `docs/task_workflow/input/nv-third-party-theory-audit-handoff-20260822.md:50` is false whenever the GPU has resident kernels but none has passed its wait. The missing term is `spin_only_union`. On the retained full H1 replay set, the claimed identity misses by **63.806 us** using the earliest wait exit and **2427.944 us** using the latest; the corrected identity closes at **0.000 us** in both cases.

4. **MEASURED — HIGH:** The anchor deficit is real dot work, not the proposed `+13..46 us` label artifact. On fresh profiles, the comparable Q+reduction/O/gate-up/down bodies are **2968.608 us tinygrad** versus **2633.088 us llama PDL-off**, a **+335.520 us device-body deficit**. Tinygrad rope, quant-provider, and Q/K reductions are separate nodes, refuting the “folded into this row” premise at handoff lines 133-140.

5. **INFERRED — HIGH:** View A is a correct residence-time arithmetic description but an incorrect causal account of the speed gap. View B has the correct mechanism—most PDL-on overlap is wait residence—but its `+760..800 us useful_body` ledger is not valid. A fresh identical-topology PDL on/off comparison removed **1122.680 us** of resident overlap while changing interval union by only **+7.705 us**; aggregate DRAM traffic changed by only **4352 bytes out of 4.7275 GB**. The prior useful-body calculation omits spin-only time and uses span as union.

6. **MEASURED — HIGH:** The current flash-score gap is **64.540 us/device token**, not the handed-off `109..112 us`. Isolated current bodies were tinygrad **4.160 us**, llama **3.744 us**, tinygrad **4.160 us** per call; across 36 calls that assigns **14.976 us** to body and leaves **49.564 us** as launch/L2/timeline residual. No wall A/B isolated that residual.

7. **UNMEASURED — MEDIUM:** Production `DEV=NV` full-token DRAM bytes and effective bandwidth remain unmeasured. Nsight Compute reported no kernels for the NV driver path. Oracle PDL-off measured **4,727,499,264 bytes**. A tinygrad `DEV=CUDA` semantic-route fallback measured **2,494,133,248 bytes**, but it is not the production NV backend and cannot replace the missing result.

8. **MEASURED — MEDIUM:** Submit-ahead did not recover the locked `+100.648 us` host-gap delta. Its fresh reverse bracket was **-3.690 us/token**, with the candidate only **-0.011 us** versus the second control; verdict `WALL_NEUTRAL`.

9. **MEASURED — PROVENANCE:** The locked wall artifact is for tinygrad commit `07e9b2abe`, while the fresh probes ran at HEAD `6570abc0` with a pre-existing dirty production worktree. The tracked diff SHA before and after this audit is `017935b7de60675857c624e71ae0e84b3839cbccb7c982706d194c865302c7a5`; this audit did not modify production, renderer, scheduler, or model code. Cross-epoch numbers are kept in separate ledgers below.

## 1. Wall identities

**MEASURED:** The locked artifact `docs/task_workflow/evidence/nv-240-exact-wall-account-20260817.json` supports the first two identities by definition:

```text
resident_union   = measure(union of resident kernel intervals)
resident_overlap = node_sum - resident_union
host_gap         = wall - resident_union

wall             = resident_union + host_gap
resident_union   = node_sum - resident_overlap
```

**MEASURED:** The displayed three-decimal fields close to rounding precision, not literal mathematical zero. Tinygrad gives `4519.316 + 268.971 - 4788.2868 = +0.0002 us`; llama gives `3890.534 + 168.322 - 4058.8565205 = -0.0005205 us`. The artifact's underlying equation records `residual_us: 0.0`; the sub-nanosecond residual in the printed fields is rounding.

### Corrected locked wall ledger

| Label | System | node_sum us | resident overlap us | resident union us | host_gap us | wall us | useful_body us | useful_overlap us |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEASURED | tinygrad, locked `07e9b2abe` | 4519.316 | 0.000 | 4519.316 | 268.971 | 4788.2868 | UNMEASURED | UNMEASURED |
| MEASURED | llama locked binary | 5015.666 | 1125.132 | 3890.534 | 168.322 | 4058.8565 | UNMEASURED | UNMEASURED |
| MEASURED | tinygrad − llama | -496.350 | -1125.132 | +628.782 | +100.648 | +729.430 | UNMEASURED | UNMEASURED |

**MEASURED:** The locked wall identity therefore attributes the **+729.430 us** wall gap to **+628.782 us resident union** and **+100.648 us host gap**. It does not measure useful body or useful overlap.

### Useful-body identity

**MEASURED:** Let `R(t)` be the number of resident kernels and `U(t)` the number that have passed their dependency wait. Direct interval integration gives:

```text
node_sum          = integral R(t) dt
resident_union    = integral 1[R(t) > 0] dt
resident_overlap  = integral max(R(t)-1, 0) dt
spin_sum          = integral (R(t)-U(t)) dt
useful_body       = integral U(t) dt
useful_union      = integral 1[U(t) > 0] dt
useful_overlap    = integral max(U(t)-1, 0) dt
spin_only_union   = integral 1[R(t) > 0 and U(t) = 0] dt
```

**INFERRED:** Algebra then requires:

```text
resident_union = useful_union + spin_only_union
               = useful_body - useful_overlap + spin_only_union
```

**MEASURED:** Re-integration of all seven steady full-H1 replays produced:

| Label | Wait-exit bound | node_sum | resident union | resident overlap | spin_sum | useful_body | useful_overlap | useful union | spin-only union | old residual | corrected residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEASURED | earliest | 5432.090 | 4048.086 | 1384.004 | 1336.245 | 4095.844 | 111.564 | 3984.280 | 63.806 | 63.806 | 0.000 |
| MEASURED | latest | 5432.090 | 4048.086 | 1384.004 | 3739.030 | 1693.060 | 72.918 | 1620.142 | 2427.944 | 2427.944 | 0.000 |

**MEASURED:** The corresponding final-H1 capture also closes only with the new term: old residual **64.140/2435.885 us**, corrected residual **0.000/0.000 us** for earliest/latest bounds. Raw replay-level values and SHA-256s are in [identity-rederive.json](../evidence/nv-third-party-theory-audit-20260822/identity-rederive.json).

**MEASURED:** Two implementation defects explain the earlier zero-residual claim. `extra/llm_research/decode/nv_llama_useful_body_h1.py:209-210` calls `max(end)-min(start)` “union,” and `extra/llm_research/decode/nv_useful_body_reconciliation.py:173-178` repeats that substitution. On the fresh PDL-off trace, `node_sum = true union = 3878.254 us` while span is `4018.961 us`; the old formula would report impossible negative overlap. At `nv_llama_useful_body_h1.py:218-224`, the sweep accounts for only excess residents and leaves one all-spin critical resident outside both useful excess and shadow.

**INFERRED:** The prior `llama useful body = node_sum - shadow` construction at `nv_useful_body_reconciliation.py:13-14,285-288` is consequently invalid. The claimed `+760..800 us` tinygrad useful-body deficit is **UNMEASURED**, not a corrected wall ledger.

## 2. View A versus View B

**MEASURED:** Fresh PDL-on A and B traces and a PDL-off trace each contain the same 762-node topology. True interval integration gives:

| Label | Arm | node_sum us | resident union us | resident overlap us | span us |
| --- | --- | ---: | ---: | ---: | ---: |
| MEASURED | PDL on, A/B median | 5008.638 | 3885.959 | 1122.680 | — |
| MEASURED | PDL off | 3878.254 | 3878.254 | 0.000 | 4018.961 |
| MEASURED | on − off | +1130.384 | +7.705 | +1122.680 | — |

**MEASURED:** Full-token NCU counter replay reports **4,727,503,616 bytes PDL-on** and **4,727,499,264 bytes PDL-off**, a difference of **4352 bytes (0.0000921%)** over the same 762 kernels.

**INFERRED — adjudication:** View A is correct only as the identity `node_sum - overlap = resident_union`. It is refuted as the statement “1125 us of overlap is 1125 us of useful wall leverage”: turning PDL off removes essentially all residence overlap without increasing useful work or reducing traffic, and changes interval union by only 7.705 us. View B is the better causal model because wait residence explains the overlap, but View B's numeric `+760..800 us useful_body` conclusion is refuted by the missing spin-only term. Neither view, as written, is a complete quantitative wall explanation.

**UNMEASURED:** Aggregate NCU counter replay cannot prove that exactly 91.9-95.4% of every overlapped interval is shadow; replay serialization destroys the original overlap timing, and bytes are not timestamped relative to each kernel's wait exit.

**INFERRED — minimum decisive tests:** A promotable shadow claim needs all four observables in one matched experiment: (1) identical PDL-on/off topology and true interval union; (2) cross-clock-calibrated per-kernel wait-exit timestamps; (3) time-resolved DRAM sectors issued before and after each wait exit, not aggregate replay totals; and (4) an unprofiled control/on/off/control wall bracket with one token SHA. This audit measured (1), aggregate rather than time-resolved (3), and did not obtain (2) and (4) together; the precise percentage remains **UNMEASURED**.

## 3. Six ordered probes

### Probe 1 — reduce/norm boundary one-pass fold

**MEASURED — `NO_GO_WALL`, exact delta `+59.602 us/token`:** Fresh control/candidate/control medians were **4722.508 / 4780.669 / 4719.626 us/token**; the control bracket median was **4721.067 us**. All timing arms had token-stream SHA `f25083e5d0a754131283b40c03f52e688fee9f175bea7ae106805e7d628d7905`.

**MEASURED:** Qualification was bitwise exact. Both arms had logit SHA `c36d7268975c5b10a8619db9753ebf008c57ab5356eae112ccb128098b81d523`; topology changed only `36 r_16_256 + 36 E_32_32_4_f14` to `36 reduce_output_rmsnorm_1_4096`, with no weight materialization.

**MEASURED:** The exact boundary row explains the loss: old pair **185.152 us node_sum**, fused row **254.592 us node_sum**, delta **+69.440 us node_sum**. `resident_union`, `resident_overlap`, `useful_body`, and `useful_overlap` were **UNMEASURED** in the unprofiled wall bracket.

**INFERRED:** There was no additional residual kernel to fold at this boundary. The premise that a separate residual contributed +218 us is the classifier error described in Finding 1.

### Probe 2 — anchor mapping

**MEASURED — `UNMEASURED` wall delta; device-body delta `+335.520 us`:** Fresh rows were:

| Label | Comparable dot body | tinygrad us | llama PDL-off us | tinygrad − llama us |
| --- | --- | ---: | ---: | ---: |
| MEASURED | Q projection plus its `r_32_32_4_4` completion | 333.504 | 249.092 | +84.412 |
| MEASURED | `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | 335.040 | 259.809 | +75.231 |
| MEASURED | `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | 1369.696 | 1268.370 | +101.326 |
| MEASURED | q4/q6 down `*_4096_12288_epi_ffnresadd` | 930.368 | 855.817 | +74.551 |
| MEASURED | **total** | **2968.608** | **2633.088** | **+335.520** |

**MEASURED:** Tinygrad's fresh full-token profile was `node_sum 4677.920 us`, `resident union 4671.500 us`, and `resident overlap 5.500 us`. Llama PDL-off was `node_sum = resident union 3878.254 us`, overlap `0`. Useful quantities were **UNMEASURED**.

**INFERRED:** The `+13..46 us` alternative double-counts llama rope/quant/KV against tinygrad anchor rows. Tinygrad has separate Q/K reduce-output, `E_*` rope/store, and `rmsnorm_q8_1_llama_provider_4096` nodes. The adjudication is therefore the real-dot alternative, at the high end of the handed-off `~320..345 us` range.

### Probe 3 — flash-score launch versus body

**MEASURED — `UNMEASURED` wall delta:** Current installed rows total **227.488 us tinygrad** versus **162.948 us llama PDL-off**, delta **+64.540 us node_sum**.

**MEASURED:** An isolated A/B/A CUPTI run of 400 calls gave medians **4.160 / 3.744 / 4.160 us**. Multiplying the per-call **+0.416 us** by 36 yields **+14.976 us** body. Subtraction leaves **+49.564 us** launch/L2/timeline residual.

**INFERRED:** Tile-body work is not the main remaining flash-score lever. The residual has not been isolated by a wall A/B, so it is not a measured wall recovery. The old `109..112 us` installed-row claim was not reproduced on the current mapping.

### Probe 4 — useful overlap versus shadow with DRAM counters

**UNMEASURED — verdict `UNMEASURED`:** Exact useful-versus-shadow time with DRAM counters was not obtained. Production `DEV=NV` returned `cuProfilerStart/Stop rc=3` and Nsight Compute reported “No kernels were profiled.” NCU counter replay on llama changes execution timing and supplies aggregate bytes rather than original-timeline traffic.

**MEASURED:** PDL-on/off full-token aggregate traffic differs by only **4352 bytes**, and fresh PDL on/off interval accounting differs by **1122.680 us resident overlap** but **7.705 us resident union**.

**INFERRED:** These observables support dependency-wait/launch shadow qualitatively. They do not validate the exact 91.9-95.4% percentage on a traffic-bearing timeline. `node_sum`, union, overlap, and useful-body values from separate timeline and NCU replay domains are not combined into a synthetic result.

### Probe 5 — real-byte effective bandwidth

**MEASURED — oracle only:** Llama PDL-off issued **4,727,499,264 DRAM bytes** across 762 kernels. NCU replay kernel time was **5112.960 us**, giving **924.611 GB/s** in the NCU profile domain.

**UNMEASURED — verdict `UNMEASURED`:** Production tinygrad `DEV=NV` bytes, production-NV effective bandwidth, and a backend-matched bandwidth delta remain unmeasured.

**MEASURED — labeled fallback:** The same semantic model route through `DEV=CUDA` issued **2,494,133,248 bytes**, with NCU replay kernel time **4191.008 us** and **595.115 GB/s**. The real-byte difference from oracle is **-2,233,366,016 bytes**, but it is cross-backend.

**INFERRED:** The handoff's `5.04 GB` tinygrad value is an accounting estimate, not a measured hardware-byte result. The CUDA fallback contradicts that estimate for its own backend but does not refute or replace the missing production-NV measurement.

### Probe 6 — submit-ahead / multi-replay handoff

**MEASURED — `WALL_NEUTRAL`, exact bracket delta `-3.690 us/token`:** Fresh control/candidate/control medians were **4647.860 / 4640.491 / 4640.503 us/token**; the bracket median was **4644.181 us**. All 15 windows shared SHA `5ede6924aaaa9acc69f9cb48a3f3cdeb7a0386be7689be2b69f0ff368110af46`; the candidate was eligible and the ping-pong pair was captured before route selection.

**INFERRED:** The `-7.369 us` versus control A and `-0.011 us` versus control B spread is consistent with noise/drift, not a recovered 100.648-us host term. `node_sum`, resident union/overlap, and useful-body fields were **UNMEASURED** in this wall-only bracket.

### Probe verdict matrix

| Label | # | Verdict | Exact reported delta | node_sum | resident union | resident overlap | useful_body / useful_overlap |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| MEASURED | 1 | `NO_GO_WALL` | **+59.602 us wall**; +69.440 us boundary-row mass | row measured; bracket total unmeasured | UNMEASURED | UNMEASURED | UNMEASURED |
| MEASURED | 2 | `UNMEASURED` | wall UNMEASURED; **+335.520 us dot body** | tg 4677.920; ll 3878.254 | tg 4671.500; ll 3878.254 | tg 5.500; ll 0 | UNMEASURED |
| MEASURED | 3 | `UNMEASURED` | wall UNMEASURED; **+64.540 us installed row** | row measured | UNMEASURED | UNMEASURED | UNMEASURED |
| UNMEASURED | 4 | `UNMEASURED` | exact shadow share and wall delta UNMEASURED; traffic **+4352 B** | timeline and counter domains separate | same | same | UNMEASURED |
| UNMEASURED | 5 | `UNMEASURED` | production NV delta UNMEASURED; oracle **4,727,499,264 B** | NCU replay only | UNMEASURED | UNMEASURED | UNMEASURED |
| MEASURED | 6 | `WALL_NEUTRAL` | **-3.690 us wall** | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED |

## 4. Where the fusion approach failed

**MEASURED:** It failed in `reduce_output_rmsnorm_1_4096`, the FFN norm/output boundary row immediately after each `q4k_g3_lanemap_gemv_epi_resadd_4096_4096`. Replacing the two old norm passes reduced launch count by 36 but increased their aggregate device residence from **185.152 to 254.592 us**. The wall moved in the same direction by **+59.602 us/token**.

**INFERRED:** The likely row-level reason is that the fused body does more costly work per invocation than the pair it replaces; this audit measured the mass increase but did not collect instruction, occupancy, or per-row DRAM counters on `DEV=NV`, so the microarchitectural cause is **UNMEASURED**.

**MEASURED:** The fusion theory did not fail because a +218-us residual launch remained exposed. That row is absent; residual absorption is already present in the O/down program names and model control flow.

## 5. Ranked “where we should have worked” ledger

**INFERRED:** The following values are fresh, non-additive **device-excess ceilings**, not measured wall recoveries. A successful replacement cannot recover more wall than the excess device mass assigned to its comparable oracle row; only a token-SHA reverse wall bracket can promote a ceiling to a result.

| Rank | Label | Concrete tinygrad row | Fresh device excess ceiling us/token | Expected wall recovery status |
| ---: | --- | --- | ---: | --- |
| 1 | MEASURED input / INFERRED ceiling | `q4k_g3_lanemap_gemv_w1w3fused16_12288_4096` | **101.326** | UNMEASURED; highest clean single-row ceiling |
| 2 | MEASURED input / INFERRED ceiling | Q `q4k_*_4096_4096` plus `r_32_32_4_4_*` completion | **84.412** | UNMEASURED |
| 3 | MEASURED input / INFERRED ceiling | `q4k_g3_lanemap_gemv_epi_resadd_4096_4096` | **75.231** | UNMEASURED |
| 4 | MEASURED input / INFERRED ceiling | q4/q6 `*_4096_12288_epi_ffnresadd` | **74.551** | UNMEASURED |
| 5 | MEASURED input / INFERRED ceiling | vocab `q6k_gen_coop_151936_4096_inkernel` plus reduction tail | **67.612** | UNMEASURED |
| 6 | MEASURED input / INFERRED ceiling | `flash_fused_gmax_combine_f16_32_128` | **66.943** | UNMEASURED |
| 7 | MEASURED input / INFERRED ceiling | `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` | **64.540** | UNMEASURED; only 14.976 us assigned to isolated body |
| 8 | MEASURED input / INFERRED ceiling | K projection plus `r_8_32_4_4_*` completion | **58.368** | UNMEASURED |
| 9 | MEASURED input / INFERRED ceiling | V projection plus `r_8_32_4_4_*` completion | **53.855** | UNMEASURED |
| — | MEASURED wall result | `reduce_output_rmsnorm_1_4096` proposed fusion | **-59.602 recovery** | `NO_GO_WALL`; it regressed wall by 59.602 us |
| — | MEASURED wall result | submit-ahead handoff | **+3.690 recovery** | `WALL_NEUTRAL`; not distinguishable from bracket drift |

**INFERRED:** If work must be prioritized from this audit alone, gate/up is first because it is the largest clean, apples-to-apples row; Q/O/down follow as the rest of the measured **+335.520 us** anchor-body deficit. Flash-combine, vocab, and K/V remain substantial but need wall A/Bs. The previously primary reduce/residual target should be closed, not optimized further in its tested form.

## 6. Prior evidence that is wrong, overstated, or miscounted

| Label | Prior claim | Audit disposition |
| --- | --- | --- |
| MEASURED | `union == useful_body - useful_overlap` at handoff line 50 | Wrong; misses `spin_only_union`, with 63.806-2427.944 us residual on full H1 bounds. |
| MEASURED | `objective_real = useful_body - useful_overlap + host_gap` at line 61 | Wrong for resident-wall accounting; also needs `spin_only_union`. |
| MEASURED | H1/reconciliation “union” | Wrong implementation: it is span at `nv_llama_useful_body_h1.py:209` and `nv_useful_body_reconciliation.py:174`. Fresh PDL-off demonstrates 140.707 us of gaps. |
| INFERRED | `llama useful body = node_sum - shadow` and `+760..800 us` gap | Unsupported because a critical all-spin resident and spin-only union are omitted; useful-body gap is UNMEASURED. |
| MEASURED / INFERRED | `52..91 us useful overlap` | Not a same-run measurement; it scales a locked 1125.132-us overlap by a percentage from another H1 run. Direct same-run H1 re-integration is 72.918-111.564 us for the full capture and 65.557-106.170 us for the final capture. The locked-epoch 52-91 us value is a projection. |
| MEASURED | `reduce + residual = +234.3 +218.4 us` | Misclassified buckets, not semantic rows. Every `r_*`/`E_*` is assigned wholesale at `nv_inter_anchor_analysis.py:104-116`. |
| MEASURED | “llama folds residual; tinygrad exposes it” | Wrong at current HEAD: O and down residuals are already epilogue-absorbed at `model.py:654-689`. |
| MEASURED | primary `+452 us` fusion lever | Refuted by exact A/B: proposed row regressed wall +59.602 us. |
| MEASURED | tinygrad anchor includes rope/quant/KV, producing only +13..46 us | Wrong mapping. Those are separate nodes; comparable dot body is +335.520 us. |
| MEASURED | flash score +109..112 us and isolated parity | Stale/currently unreproduced. Fresh installed delta is +64.540 us; fresh isolated body delta is +14.976 us/token. |
| UNMEASURED | exact 91.9-95.4% shadow verified by DRAM counters | Overstated. Aggregate on/off bytes are equal within 4352 B, but counters are not time-resolved on the overlap timeline. |
| UNMEASURED | tinygrad production real bytes ~5.04 GB | Accounting estimate only. Production NV counters failed; CUDA fallback is 2.494 GB but is not a substitute. |
| MEASURED | +100.648 us host gap is recoverable with submit-ahead | The locked accounting term is real; the tested handoff recovery is not. Fresh wall delta is -3.690 us and neutral. |
| MEASURED | llama identity is commit `ac4cddeb0` | Incomplete provenance: the llama tree is dirty. The measured binary is fixed by SHA `947eb29052871f151719762c2fc265024e14f833b98df7801af9eb09da1625a8`. |
| MEASURED | locked and fresh figures describe one candidate | Wrong if mixed. Locked wall uses tinygrad `07e9b2abe`; probes use current dirty `6570abc0`. This report keeps them separate. |

## 7. Measurement provenance and retained evidence

**MEASURED:** Fresh timing runs used fresh child processes, `DEV=NV`, `/tmp/gpu-bench.lock`, depth 512, settled continuous windows, and control/candidate/control reverse brackets. The model SHA is `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`; Python SHA is `1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118`; GPU UUID is `GPU-c800ade9-21ea-2e55-f75c-6d7a458fb186`; driver is 595.84.

**MEASURED:** The analysis added or adjusted only analysis helpers: `nv_useful_identity_audit.py`, `nv_ffn_reduce_output_ab.py`, `nv_submit_ahead_wall_audit.py`, and an analysis-route selector in the untracked `nv_rmsnorm_current_head_topology.py`. No tracked production diff changed during the audit.

**MEASURED:** The machine's consumer application-clock interface reports “deprecated”; serialization used the repository's GPU bench lock rather than a fixed-frequency application clock. Reverse brackets are therefore the timing control for drift.

**MEASURED:** Principal raw artifacts are retained in [nv-third-party-theory-audit-20260822](../evidence/nv-third-party-theory-audit-20260822/):

- **MEASURED:** `probe1-qualify.json`, `probe1-timing.json`, `probe1-fold-candidate-profile.jsonl`, and `probe1-fold-candidate-capture.json`.
- **MEASURED:** `probe2-tinygrad-profile.jsonl`, `probe2-tinygrad-capture.json`, and all three llama `.nsys-rep`, `.sqlite`, graph, and DAG exports.
- **MEASURED:** Probe 3 A/B/A `.nsys-rep` and `.sqlite` captures plus the exact isolated llama executable.
- **MEASURED:** Probe 4/5 PDL-on, PDL-off, and CUDA-fallback `.ncu-rep` counter reports.
- **MEASURED:** `probe6-submit-ahead-bracket.json` and its three child records.
- **MEASURED:** [audit-summary.json](../evidence/nv-third-party-theory-audit-20260822/audit-summary.json) contains the machine-readable corrected ledgers and verdicts; `sha256.txt` fixes every retained artifact after report generation.

**UNMEASURED:** No claimed projected ceiling in Section 5 is a measured production speedup. No production-NV DRAM result was synthesized from the CUDA fallback, and no useful-body value was inserted into the locked wall ledger.
