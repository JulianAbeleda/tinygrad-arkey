# NVIDIA pp512 Prefill Exact Gap Ledger — 2026-08-29

Scope: exact, settled, non-profiled cross-runtime prefill comparison (pp512, Qwen3-8B-Q4_K_M, settled R9 replay, warmup excluded). Repetition logic: llama = exclude sample 0 (warmup), median of samples 1–8 of `.[0].samples_ns`; tinygrad = median of 9 samples.

Session mode: build-only. No GPU runs, no new R9 arms, no `PROFILE=1` walls, no code edits. Every figure below was read from the cited artifact in this session (spot-verified 2026-08-29 against `git rev-parse --short HEAD = 131b22a8b`).

## 1. Baselines (cited)

| Side | Settled median | Settled min | Launches | Unknown |
|---|---|---|---|---|
| llama | **35.019399 ms** | **34.680367 ms** | 1186 | 0 |
| tinygrad composed (current best) | **67.235719 ms** | **67.153915 ms** | 1449 | 0 |

- llama: `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/cross-runtime-accounting.json → .unprofiled.llama_settled_median_ns / .llama_settled_min_ns` (35019399 / 34680367 ns). Raw: `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/llama/llama-unprofiled-r9.json` — **array of length 1**; `.[0]` is the full bench object, `.[0].samples_ns` is the sample list (sample 0 = warmup, excluded). Do not jq as a flat list.
- tinygrad: `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/candidate.json → .wall` (9 samples: 67.153915 … 67.267569; median 67.235719, min 67.153915, tok_s 7624.28), `→ .route.default_enabled:false`, `→ .census` (198 canonical weight args, `all_weights_canonical:true`, 0 weight-copy kernels, **54 remaining FP16 overlays** = 18 type-14 Q6-V + 36 down), `→ .deep_replay.all_cycles_exact:true`.
- Correctness: `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/compare.json → {status:PASS, performance_pass:true, correctness:{same_token:true, finite:true, max_abs:0.10494756698608398, mean_abs:0.013181783258914948, allclose_rtol_0p02_atol_0p5:true, candidate_token:198, control_token:198}}`.

**GAP (recomputed; do NOT cite the stale debt fields in the cross-runtime JSON):**
- median **32.216320 ms** = 67.235719 − 35.019399
- min **32.473548 ms** = 67.153915 − 34.680367
- (`cross-runtime-accounting.json → .unprofiled.median_wall_debt_ns:34472779` / `minimum_wall_debt_ns:34697787` are 69.49-precursor debt — superseded; see §9.)

**Instrumentation asymmetry (profiled, mapping-only):** `cross-runtime-accounting.json → .profiled` = `instrumentation_delta_llama_ns:749716` (+0.75 ms), `instrumentation_delta_tiny_ns:5307577` (+5.31 ms). Caveat: the profiled tiny side (wall 74.685731 ms, `tiny_wall_ns:74685731`) is the **69.49-precursor arm, not the current 67.24 composed arm**. Deltas are instrumentation-perturbation evidence only; they are not used in any wall claim.

**Provenance / tree state (BLOCKED #1 basis, verified this session):** `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/tinygrad/tinygrad-provenance.json → .git_head = 4d117c8e018573986e6b61600dbdff7959bb0139` ≠ current HEAD `131b22a8b`; working tree has **611** modified/untracked entries; `.source_state_note` confirms the compiler substrate is commit-pinned while the tree carries pre-existing uncommitted HCQ queue/timestamp work; `.capture.selected_group:7` = first recorded profiled wall 74.685731 ms. ⇒ **No new performance claim is admissible until a clean/pinned re-run** (forbidden this session).

## 2. Role-by-role ledger

### 2a. Cross-runtime region table — **FLAGGED STALE** (69.49-precursor arm, pre-Q4-V shift; do not treat as current)

Order: **llama / tinygrad** (ms):

| region | llama | tiny (stale arm) |
|---|---|---|
| norm | 2.145511 | 1.400224 |
| Q | 2.514450 | 4.492896 |
| K | 1.251141 | 2.210560 |
| V | 1.053720 | 6.381184 |
| Flash | 1.657447 | 3.328096 |
| O | 2.397213 | 4.477664 |
| gate | 6.086529 | 12.761504 |
| up | 6.111329 | 13.006080 |
| act | 0.903843 | 0.758880 |
| down | 6.940482 | 19.007040 |
| residual | 1.379275 | 3.118624 |
| vocab | 0.313154 | 2.921568 |

Source: `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/cross-runtime-accounting.json → .regions`. Stale because (a) the tiny side is the 69.49-precursor arm and (b) Q4-V shifts the V role (−1.886203 ms standalone / −2.011928 ms composed) after this capture. **Gate 0 must regenerate this table** before any role-level subtraction is admissible.

### 2b. Post-unroll PROFILE=1 exact trace — **mapping-only** (not wall authority; different arm/instrumentation from 2a)

- 1,449 launches / 0 unknown; device span 541,417.728 us.
- Role debts vs llama: gate+up **+8,750.686 us**, Q+O +4,070.161, vocab **+2,606.654**, Flash +1,673.529, K +970.171, io +15.424.
- Source: `docs/task_workflow/output/nv-prefill-post-unroll-exact-trace-20260829.md`.
- Support split: down **36 launches / 461,006.816 us**; residual/RoPE/KV transport 1,233 / 79,869.184 us; q8 producer 180 / 525.088 us; V 0.
- Source: `docs/task_workflow/output/nv-prefill-post-unroll-support-attribution-20260829.md` + `docs/task_workflow/evidence/nv-prefill-post-unroll-trace-20260829/support-attribution.json`.
- ⚠️ Never mix the two "support" region tables (2a cross-runtime regions vs 2b post-unroll trace regions): different arms, different instrumentation. Cite each with its own path.

### 2c. Accounting completeness

**No composed-graph accounting exists for the current 67.235719 arm.** Role-level subtraction against the 32.47 ms gap is therefore **not yet admissible** — Gate 0 (HCQ-native low-perturbation census of the exact composed graph) is the prerequisite that unblocks it.

## 3. Proven (path-cited)

1. **Composed graph PASS, default-off.** Census 198 canonical / 54 overlays, `all_weights_canonical:true`, deep-replay all-cycles-exact, compare `same_token:true` + token 198, cut-policy v2, `HCQ_NV_READY_PLACEMENT:0`. → `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/{candidate,compare}.json`.
2. **Q4-V (18 type-12):** −2.011928 ms composed (`nv-prefill-composed-unroll4-q4v-20260829/compare.json`) / −1.886203 ms standalone (`docs/task_workflow/evidence/nv-prefill-q4v-pass-final-20260829.json` vs `docs/task_workflow/evidence/nv-prefill-noq4v-control-20260829.json`).
3. **Q/O safe-cut PASS** (isolated + combined correctness gates clean).
4. **K integration recovery 4.004704 ms** (70.390585 vs 74.395289), 71.690 us/call proxy, residual 0.959419 ms → CLOSE. → `docs/task_workflow/output/nv-prefill-k-residual-service-audit-20260829.md`.
5. **Support mapping closure:** 1,449 launches / 0 unknown. → §2b paths.
6. **Flash installed oracle PASS:** max_abs 0.00930290096, route `nv_sm120_q16_grid_hd128_loop_attention`, 36 calls.
7. **gate/up NCU bridge (identical instruction mix):** 6,291,456 IMMA both; tensor duty **14.65% vs 31.71%**; issue 37.44% vs 52.38%; eligible warps 0.554 vs 0.807; long scoreboard 0.535 vs 0.324; instructions 194,359,296 vs 156,188,112 (+24.4%) ⇒ 1.87× slower, loss = **issue / latency hiding**, not arithmetic. → `docs/task_workflow/output/nv-prefill-gate-up-service-audit-20260829.md`.
8. **final-row prune: isolated-correct but graph-regressive** (isolated max_abs 0.0059030056; matched graph **+0.277480 ms**) → STOP, do not re-invest. → `docs/task_workflow/output/nv-final-row-prune-result-20260829.md`.

## 4. Uncertain

1. **gate/up occupancy gap + tile-k=64 primitive is negative:** +4.22% (483.920 vs 464.352 us); tile-k=32 **fails compile** ("current atomic staging requires at least two tensor-core K steps"). Mechanism (issue/latency hiding per §3.7) is proven; the fix is not.
2. **54 FP16 overlays remain** (18 type-14 Q6-V + 36 down): `nv-prefill-composed-unroll4-q4v-20260829/candidate.json → .census.remaining_v_down_fp16_overlays`.
3. **Flash S6 cooperative closed 4,431.904 us** vs installed ~6.7 us/call — whether a genuinely new vectorized topology beats installed S6-class service is unproven. → `docs/task_workflow/output/nv-prefill-flash-s6-cooperative-installed-ab-20260829.md`.
4. **vocab Q8:** `docs/task_workflow/evidence/nv-vocab-q8-fullshape-20260828/gate-r9.json → verdict: PERFORMANCE_PASS_NEEDS_RECURRENT_QUALITY`; `gate-corrected-v2-r9.json → verdict: NO_GO_FULLSHAPE`; nacc4 327.8 us at 86.9% DRAM. No full-logit many-row Q6 exists.
5. **default-ready placement NO_GO** (drift 0.096721); only cut-policy is safe today.

## 5. Ship next — ranked top-5 highest-confidence gaps (evidence-backed, not guesses)

1. **down FP16 (36 overlays)** — largest overlay owner; PROFILE=1 support share 36 launches / 461,006.816 us (§2b, `support-attribution.json`); every prior down attempt failed on substrate, not on the idea (`nv-q4down-matched-ab-20260829/result.json` max_abs 2.695646; `nv-compiler-q6k-model-20260828/final-compare.json` +1.61→+1.95 ms regression). → **Gate 4**.
2. **gate/up issue/latency hiding** — 1.87× slower at identical IMMA count, tensor duty 14.65% vs 31.71% (§3.7, `nv-prefill-gate-up-service-audit-20260829.md`); largest single role debt in §2b (+8,750.686 us). → **Gate 1**.
3. **vocab many-row Q6** — active debt +2,606.654 us (§2b); llama 0.313154 vs tiny 2.921568 ms (stale table §2a, direction still valid); fullshape Q8 gate NO_GO (§4.4) means a new many-row full-logit Q6 kernel is required. → **Gate 2**.
4. **Flash vectorized topology** — active debt +1,673.529 us (§2b); llama 1.657447 vs tiny 3.328096 ms; installed route proven correct (§3.6), service gap remains. → **Gate 3**.
5. **18 type-14 Q6-V + support/transport families** — 1,233 transport launches / 79,869.184 us + 180 q8 producer / 525.088 us (§2b); V role already shifted by Q4-V, so residual must be re-measured, not assumed. → **Gate 5**.

(**Gate 0** is the prerequisite that unblocks all role-level subtraction against the 32.47 ms gap.)

## 6. Substrate required before further claims

(a) **HCQ-native low-perturbation timestamp census (Gate 0)** — regenerate the stale §2a role table without PROFILE=1 (whose tiny-side perturbation is +5.31 ms vs llama's +0.75 ms, §1) and split the 461 ms-class down share at unprofiled scale.
(b) **Pinned clean-tree re-run** before any new performance claim — HEAD `131b22a8b` ≠ trace pin `4d117c8e0…`, tree dirty (611 entries) (`tinygrad-provenance.json → .git_head / .source_state_note`).
(c) **down:** role-matched Q4-down compiler geometry/provider with **≥2 tensor-core K steps** (tile-k=32 fails: "current atomic staging requires at least two tensor-core K steps"), or a separate static full-output oracle for the matched A/B.
(d) **vocab:** new many-row full-logit Q6 kernel (current Q8 fullshape is NO_GO).
(e) **Producer gates are kernel-only scope** — they must not be cited as model-level evidence.

## 7. Test ladder — next 2–4 weeks (Gates 0–5)

**Gate 0 — HCQ-native low-perturbation timestamp census of the exact composed graph (prerequisite).**
- Purpose: rank the 32.47 ms gap by subtraction without PROFILE=1; regenerate the stale §2a table; split the 461 ms-class down share at unprofiled scale.
- Fail-closed correctness: 100% of device span accounted, **zero unknown**, on the pinned clean tree; census reproduces 67.235719 median ± drift band and 1,449-launch closure.
- Measured gain/loss: census overhead per sample (must stay ≪1% of wall); wall delta vs 67.235719 baseline.
- Decision rule: role-level subtraction is admissible only after this passes; otherwise all role-debt claims stay BLOCKED and only whole-wall comparisons are citable.

**Gate 1 — gate/up register-safe K-step / fragment load-to-use scheduling.**
- Purpose: close the issue/latency-hiding gap (tensor duty 14.65% → toward 31.71%).
- Fail-closed correctness: full-output oracle (max_abs ≤ 2.136e-4 class), canonical/read-only weights, **no local spill** (register pressure verified), full-vocab `same_token` + deep-20 exact.
- Measured gain/loss: NCU counters (tensor duty, long scoreboard, issue, instruction count) + exact 72-role native population wall + whole-model wall bracket vs 67.235719.
- Decision rule: adopt only if counters move in the llama direction (higher tensor duty, lower long scoreboard, fewer instructions) AND the candidate wins the exact 72-role population AND the whole-model wall bracket shrinks vs 67.235719 without drift.

**Gate 2 — new many-row full-logit Q6 vocabulary kernel (512 rows, full logits).**
- Purpose: attack the +2,606.654 us vocab debt with a kernel the Q8 fullshape gate rejected.
- Fail-closed correctness: full-vocab logits exact vs Q6_K FP16 control (allclose class) + `same_token`.
- Measured gain/loss: matched R9 wall delta (must be < 0) vs the 1,368.1 us corrected-v2 service baseline (`nv-vocab-q8-fullshape-20260828/gate-corrected-v2-r9.json`).
- Decision rule: ship only if matched R9 wall delta < 0 AND it beats the 1,368.1 us corrected-v2 service AND the +2,606.654 us debt shrinks in a Gate 0 census.

**Gate 3 — Flash new vectorized topology.**
- Purpose: close the +1,673.529 us Flash debt (llama 1.657447 vs tiny 3.328096 ms).
- Fail-closed correctness: capture/oracle (installed-class max_abs ≤ 0.00930290096) + same-population service comparison; respect the graph-ownership boundary — **NO full-T512 relaunch of the installed capture**.
- Measured gain/loss: per-call service vs installed S6-class (~6.72 us/call; S6 cooperative closed 4,431.904 us, `nv-prefill-flash-s6-cooperative-installed-ab-20260829.md`) + 36-call total.
- Decision rule: ship only if it beats installed S6-class service at 36 calls; otherwise keep the installed `nv_sm160…` — i.e. `nv_sm120_q16_grid_hd128_loop_attention` — and re-rank Flash. (Reopen only with a genuinely new topology.)

**Gate 4 — Q4/Q6 down new lifecycle (largest overlay owner).**
- Purpose: replace the 36 FP16 down overlays; every prior attempt failed on substrate (max_abs 2.695646 matched A/B; +1.61→+1.95 ms regression; population gate failed pre-timing).
- Order: oracle → readonly/sentinel/finite → matched FP16 comparator → population → model.
- Fail-closed correctness: matched 18-role A/B allclose + `same_token` (must clear the 2.695646 failure), no host-sync control artifacts (no `sum().item()` controls).
- Measured gain/loss: net whole-model wall delta vs 67.235719 (prior down work was a +1.61→+1.95 ms **regression** — the bar is net reduction, not isolated kernel wins).
- Decision rule: ship only if net wall reduction with full correctness; else down stays BLOCKED #3 and remains FP16.

**Gate 5 — support/transport families (refined split: 1,233 transport / 180 producer / 36 down).**
- Purpose: close the residual transport/producer debt after Gates 1–4.
- Fail-closed correctness: per-family oracle + replay-exact on the composed graph.
- Measured gain/loss: Gate 0 census delta per family.
- Decision rule: run only after Gates 1–4; rank families from the Gate 0 census, not from the stale §2a table.

## 8. BLOCKED (7, file pointers)

1. **Tree ≠ trace state.** HEAD `131b22a8b` ≠ pin `4d117c8e018573986e6b61600dbdff7959bb0139`; dirty tree (611 entries). New perf claims require a clean/pinned re-run (forbidden this session). → `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/tinygrad/tinygrad-provenance.json → .git_head / .file_sha256`.
2. **cross-runtime-accounting.json STALE.** Tiny side = 69.49-precursor arm; `median_wall_debt_ns:34472779` / `minimum_wall_debt_ns:34697787` are 69.49-debt, not current; role table must shift with Q4-V; no composed-graph accounting exists. → `docs/task_workflow/evidence/nv-prefill-exact-cross-runtime-trace/cross-runtime-accounting.json → .unprofiled / .regions`. Unblocked by Gate 0.
3. **down lanes STOP.** Matched A/B FAIL max_abs 2.695646 + slower (`docs/task_workflow/evidence/nv-q4down-matched-ab-20260829/result.json`); Q6 +1.61→+1.95 ms (`docs/task_workflow/evidence/nv-compiler-q6k-model-20260828/final-compare.json`); population gate failed pre-timing (`docs/task_workflow/output/nv-q4down-population-gate-result-20260829.md`); missing substrate = role-matched down binding/shape contract + ≥2 tensor-core K steps.
4. **54 FP16 overlays** (18 type-14 Q6-V + 36 down) → `docs/task_workflow/evidence/nv-prefill-composed-unroll4-q4v-20260829/candidate.json → .census.remaining_v_down_fp16_overlays`.
5. **vocab Q8** `PERFORMANCE_PASS_NEEDS_RECURRENT_QUALITY` / `NO_GO_FULLSHAPE` → `docs/task_workflow/evidence/nv-vocab-q8-fullshape-20260828/gate-r9.json`, `gate-corrected-v2-r9.json`.
6. **default-ready placement NO_GO** (drift 0.096721); cut-policy only.
7. **All PROFILE=1 walls non-authoritative.** 87 ms Q6 preliminary rejected; q4down 356–365 ms control = host-sum harness artifact, not full-model.

## 9. Stale / superseded references

- **69.49 / 69.38 / 74.92 / 70.35 ms** = precursors, not current best (current = 67.235719).
- **68.992404 / 69.269884** = final-row-prune bracket (STOP; `docs/task_workflow/output/nv-final-row-prune-result-20260829.md`).
- **87 ms Q6** rejected; **q4down 356–365 ms** = harness artifact (`docs/task_workflow/output/nv-q4down-candidate-control-decision-20260829.md` is SUPERSEDED/INVALID — host `sum().item()` control).
- **cross-runtime tiny side** (69.492178 / 69.378154 ms in `.unprofiled.tiny_median_ns / .tiny_min_ns`) = stale arm; use `nv-prefill-composed-unroll4-q4v-20260829/candidate.json` instead.
- **`q8-k12288-finish.json`** superseded — PASS evidence is `saved-z-k12288-gate-v3.json` & `saved-z-k12288-gate-v4.json` (q_mismatch 0).
- **`population.json`** (zero checksums, FAIL) superseded by **`population-v4.json`** (PASS) — cite v4 only.
- **`docs/task_workflow/output/nv-prefill-complete-lifecycle-ledger.{md,json}`** superseded by this ledger.
- **`nv-compiler-q4k-gkqo-combined-result.md`** does **not** exist — do not cite.
- Decode-era documents (`nv-genuine-llma-gap-audit-result.md`, `nv-dense-next-lever-ledger.md`) = **decode**, context only; `nv-s4-g32/g64` NCU JSONs = residual-O, not gate/up.

## 10. One-line summary

- **Proven:** composed 67.235719 ms PASS default-off with all-cycles-exact replay (§3); gap to llama 32.216320 ms recomputed on settled non-profiled R9 (§1); gate/up loss mechanism = issue/latency hiding (§3.7).
- **Uncertain:** gate/up fix, 54 overlays, Flash topology, vocab Q6, default-ready placement (§4).
- **Ship next:** Gate 0 (census, unblocks subtraction) → Gate 4 (down) / Gate 1 (gate/up) / Gate 2 (vocab) / Gate 3 (Flash) / Gate 5 (support), in §5 rank order (§7).
- **Substrate before further claims:** (a) Gate 0 census, (b) pinned clean-tree re-run, (c) down ≥2 tensor-core K steps, (d) many-row Q6 vocab kernel, (e) kernel-scope producer gates only (§6).
