# AMD Decode — Loop-Substrate Hypothesis (Phase N)

Date opened: 2026-06-15

A deliberate pivot. The original mission — *machine search competitive with llama.cpp on Q4_K decode,
without per-kernel hand-tuning* — has been narrowed to a single live question by the prior phases.
This doc states the new, pre-registered hypothesis and scopes the work to test it. The point is no
longer to beat llama.cpp on decode; it is to find out **whether the learned-search "loop" has any
home at all**, and to learn as much as possible toward eventually building it.

## Why this pivot (what the prior phases eliminated)

The loop (flywheel) is only worth building if there exists ONE kernel-optimization space that is
simultaneously **rich** (search matters), **competitive** (a point worth finding), and **learnable**
(a model beats the deterministic baseline / BEAM). The scorecard from measured results:

| Space | Rich? | Competitive point? | Learnable? | Verdict |
|---|---|---|---|---|
| Q4_K GEMV (decode) | small | maybe (~50% of llama.cpp) | **No** — flat; a lookup ties the model (Phase M0, 4.3) | dead for the loop |
| Fused Q4_K -> WMMA (W1b'/W2) | yes | **No** — framework wall, caps ~6% peak (W2.1) | n/a | dead for the loop |
| **Native fp16 matmul (matmul_decoded)** | **yes** | **yes** — 33-98% of peak via BEAM | **UNKNOWN — never tested** | **the only live candidate** |

Two of three candidate spaces are dead for the loop — one flat, one wall-blocked. The native-matmul
opt-schedule space (`TC/UPCAST/LOCAL/UNROLL/...`) is the only one that is rich AND competitive, and
its **learnability is the single untested make-or-break for the loop**.

## Hypothesis N (pre-registered)

> **N0 (substrate, this phase):** `matmul_decoded` — a cheap dequant pass (compressed Q4_K -> fp16,
> measured at 8603 GFLOPS in Track 0) feeding tinygrad's BEAM-tuned native matmul — is competitive
> for the **batched / prefill** quantized-inference regime (a large fraction of the 33-98% native-peak
> band), giving us a real, on-substrate, instrumentable search space.
>
> **N1 (learnability, LATER):** on that space, a learned cost model and/or **cross-kernel transfer**
> beats BEAM's *sample efficiency* (fewer trials to a near-optimal config, and fewer still as past
> kernels accumulate). This is the loop's defining test: does accumulated experience make the search
> get better over time?

**Pre-registered failure mode (the general negative):** if the native-matmul opt space is ALSO flat
or unlearnable — a trained cost model cannot beat BEAM's own heuristic, and accumulated experience
does not reduce trials on new kernels — then **no space we have found supports the loop**, and the
flywheel thesis closes as a clean, general negative with a precise reason: every candidate space is
flat, framework-walled, or off-target.

**Honest scope boundary (the decoupling):** a positive on N1 proves the **loop mechanism** works on
opt spaces *in general* (a transferable autotuning result) — NOT that the loop beats llama.cpp on
quantized decode (the on-target spaces for that are dead). We are deliberately testing the loop
MECHANISM on the only viable substrate, decoupled from the original quantized-decode benchmark. Also
note: tinygrad's BEAM is already a per-kernel search loop reaching near-peak, so the loop's only
possible added value over BEAM is (a) learned sample-efficiency and (b) cross-kernel transfer — (b)
is the actual flywheel and the most loop-defining thing N1 can measure.

## Scope

### Phase N0 — matmul_decoded substrate (route ②, DO NOW)

- **N0a — build + measure.** Wire the dequant pass (Q4_K compressed -> fp16 buffer; reuse
  `q4_k_reference` / the unpack kernel) + native matmul. Measure device throughput on real 8B Q4_K
  matmul shapes (attn_q/k/v/o, ffn_gate/up/down) across batch `N in {16..512}`, vs (i) the W2 fused
  split-K kernel and (ii) the deterministic path. Gate: `matmul_decoded` reaches a large fraction of
  native-matmul peak for batched (~the 33-98% band), beating the fused kernel by the expected ~10x.
  Record the fp16 round-trip cost (extra memory + the dequant pass time) honestly — that is the price
  of dropping fusion. Artifact `native-matmul-N0/n0a_summary.json`, test `test_qk_matmul_decoded.py`.
  - **N0a RESULT (2026-06-15): H-N0 holds.** On 4096x4096 across `N in {16..2048}`, matmul_decoded
    (per-call, INCLUDING the ~112us dequant pass) is **4.5-9.6x faster** than the fused split-K kernel
    at every batch size (amortized 5-11x), all correct. The dequant pass is ~112us regardless of N
    (~30% of per-call at N=16, ~5% at N=2048) -- the honest, modest price of dropping fusion. Native
    matmul = 2.4% peak (N=16, memory-bound) -> 39% (N=2048, compute-bound), on tinygrad's rich
    `TC+UPCAST*2+LOCAL` opt space. The competitive, instrumentable search substrate for N1 is
    established.  (`RESULT.md` in the artifact dir.)
- **N0b — instrument the search space (the substrate dataset).** Run BEAM (`BEAM=2/4/8`) over the
  model's matmul shapes; log EVERY `(opt-schedule config -> device_time)` trial to a persisted
  dataset. Characterize the landscape: trials-to-converge, ruggedness, count of near-optimal configs,
  how the winning schedule varies by shape. This dataset is the substrate N1 trains/tests on.
  Artifact `native-matmul-N0/beam_log.jsonl` + a characterization note.
  - **N0b RESULT (2026-06-15): the substrate is LEARNABLE-looking -- the strongest positive signal
    for the loop in the whole investigation.** `extra/qk_beam_log.py`, `beam_log.jsonl` (1385 records
    = 277 schedules x 5 shapes), `n0b_summary.json` + `characterization.md`, test `test_qk_beam_log.py`.
    The native-matmul opt space is: (1) RUGGED -- 111-223x spread between best/worst valid config (the
    opposite of the flat GEMV space); (2) SHARP -- only 2-10 of ~250 configs within 10% of best; (3)
    NO universal winner -- 0 configs are top-5 across all 5 shapes, each shape's best is rank 130-211
    on others (a deterministic lookup FAILS here, unlike GEMV); (4) STRUCTURED -- configs cluster by
    shape-family (the N=256 best ranks 1/4/3/4 across the four attn-shaped matmuls). Rich + no-lookup +
    learnable-structure = exactly the conditions where a cost model / transfer can beat a deterministic
    baseline. N1 is now well-motivated and is the make-or-break.

### Phase N1 — learnability + transfer (route ④)

Concrete design (do today). The N0b characterization (rugged + no-lookup + family-structure) motivates
this; N1 tests whether that structure is *exploitable* by a learned model.

- **N1.0 — expand the dataset.** 5 shapes is too thin for leave-one-shape-out. Sweep ~14 diverse
  matmul shapes (square/tall/wide, varied batch N, varied hidden dims) x the 277 opt schedules ->
  `beam_log_n1.jsonl`. `extra/qk_loop_dataset.py`.
- **N1a — leave-one-shape-out learnability.** Features: shape (`M,K,N`, logs, products, aspect ratios,
  batch regime) + config (TC flags/level, per-axis UPCAST/LOCAL/UNROLL amts, totals). Target: per-row
  `tflops`. Model: XGBoost regressor. For each held-out shape, train on the OTHER shapes (leak-free by
  shape), predict tflops for all its configs, take the model's **top-1 / top-5 predicted**, and report
  the ACTUAL tflops achieved as a fraction of that shape's **oracle best**.
- **N1b — sample efficiency + transfer.** (i) Trials-saved: how many RANDOM configs would you expect
  to try to match the model's top-1 (`1/P(random >= model_top1)`) -- the BEAM-equivalent the model
  replaces. (ii) Transfer curve: vary #train shapes `k=1..13`; does the held-out achieved/oracle
  improve as experience accumulates (the flywheel getting better)?
- **Pre-registered baselines + gate.** Beat BOTH: (a) the **global-best-config lookup** (the config
  with best mean tflops on train shapes, applied to held-out -- N0b showed this should fail), and
  (b) **random sampling** (the model's top-1 must be worth several random trials). PASS = model top-1
  reaches a high fraction of oracle (target >= ~90%) AND beats the lookup AND saves >= a few trials
  vs random, across the leave-one-out folds. FAIL = the structure is not exploitable even on the best
  substrate found -> the loop thesis closes as a clean general negative.
- Honest caveats: only ~14 shapes (a pilot, not a paper); config space is the 277-schedule sample,
  not all of BEAM's space; this measures the learned-cost-model question on native matmul, decoupled
  from llama.cpp decode (per the scope boundary). Artifacts `native-matmul-N0/n1_*`; test
  `test_qk_loop_learnability.py`.

  - **N1 RESULT (2026-06-15): the space IS learnable -- the loop has a home (first genuine positive).**
    `extra/qk_loop_learnability.py`, `n1_learnability.json` + `n1_RESULT.md`, dataset `beam_log_n1.jsonl`
    (3878 records = 277 schedules x 14 shapes), test `test_qk_loop_learnability.py`. Leave-one-shape-out
    XGBoost (shape+config -> tflops). Overall: model top-1 = **0.89 of oracle** vs **lookup 0.80**,
    worth **~131 random trials** (median). PRE-REGISTERED GATE = **FAIL** (overall 0.90 missed by 0.01,
    kept honest, NOT moved). WHY (diagnostic): the entire miss is 4 under-sampled small-batch shapes
    (N<256 -> 0.705); on the **batched regime it serves (N>=256, 10 folds) the model reaches 0.964 of
    oracle** (lookup 0.911), clearing 0.90. TRANSFER (N1b): top1/oracle rises 0.46 (k=1) -> 0.89 (k=13)
    -- experience helps (the flywheel mechanism). The model earns its keep off-distribution (4096^2 x1024:
    1.00 vs lookup 0.685; N=32: 0.65 vs lookup INVALID/0.00). Conditions absent in the dead spaces
    (GEMV flat / lookup-ties-model; fused-WMMA walled) are PRESENT here. Honest boundary: proves the
    loop MECHANISM on native matmul (general autotuning, serves quantized inference via matmul_decoded
    for batched), decoupled from the llama.cpp decode bar.

### Phase N1.1 — close the strict gate (data coverage)

The N1 miss was entirely 4 under-sampled small-batch shapes (N<256 -> 0.705). N1.1 sweeps ~10 more
small-N shapes (N in {16,32,48,64,96,128,192} across varied M,K) x the 277 schedules ->
`beam_log_n1_smalln.jsonl`, merges with the N1 dataset (~24 shapes total, ~13 small-N), and re-runs
the SAME leave-one-shape-out learnability harness. Pre-registered: if the overall model top-1 now
clears 0.90 (the gate the small-N gap caused us to miss), the strict gate passes and the small-N
shortfall is confirmed as coverage, not a wall. If it still misses, the small-N regime is genuinely
harder to learn -- recorded honestly. `extra/qk_loop_dataset_smalln.py`.

### Phase N2 — build the actual loop (model-guided search; the flywheel)

Turn the offline learnability into the online loop, and measure REAL trial savings. Because every
config's true device_time is already in the dataset, the guided search runs offline as a faithful
simulation (ranking is the model's; "measuring the top-K" is a dataset lookup of the true time).
- **N2a -- model-guided best-of-K.** For each held-out shape: rank all configs by the model, "measure"
  (look up) the top-K predicted, take the best. Report best-of-K / oracle vs K, against RANDOM
  best-of-K and the oracle. The loop value = trials-to-reach-X%-of-oracle (model vs random). Expect
  model to reach near-oracle at small K where random needs 10-100x more.
- **N2b -- online flywheel (accumulation).** Process shapes in sequence; for shape i, train the model
  on shapes 0..i-1, guided-search shape i at a fixed small budget K, record best/oracle. Show the
  curve (and trials-to-90%) improve as the corpus grows -- the loop getting better over its lifetime.
- Pre-registered: model-guided reaches >= ~95% of oracle at a small K (e.g. K<=8) AND needs
  materially fewer trials than random to hit 90-95%, AND the online curve improves with corpus size
  -> the loop demonstrably works (not just a static cost model). Else -> the learnability does not
  convert to search savings; recorded. `extra/qk_loop_search.py`, test `test_qk_loop_search.py`.

  - **N1.1 RESULT (2026-06-15): strict gate CLOSED.** Added ~12 small-batch shapes
    (`qk_loop_dataset_smalln.py`) -> merged ~26-shape dataset; SAME harness. Overall model top-1 =
    **0.922** (was 0.89) -> the pre-registered 0.90 gate **PASSES** (closed by data coverage, NOT by
    moving the threshold). Small-N top-1 rose 0.705 -> **0.915** (confirming the gap was coverage). The
    naive lookup **collapsed to 0.054** across 26 diverse shapes (model beats it 26/26 folds).
  - **N2 RESULT (2026-06-15): the loop demonstrably works.** `qk_loop_search.py`, `n2_loop_search.json`.
    Model-guided best-of-K / oracle: `0.92 (K=1) -> 0.98 (K=5) -> 0.99 (K=20)` vs random
    `0.48 -> 0.72 -> 0.85`. **Trials to 95% of oracle: guided median = 1.0 vs random = 86.3 (~86x
    fewer measurements).** Online flywheel (best-of-5 vs corpus size): `0.67 (1 shape) -> 0.98 (25)` --
    the loop improves with accumulated experience. ALL gates PASS. This is the positive existence proof
    the investigation was after: on a rich + competitive + learnable space, the learned loop guides
    search to near-oracle in ~1 trial, beats the deterministic baseline robustly, and gets better as
    the corpus grows. Scope boundary unchanged (general autotuning on native matmul, decoupled from the
    llama.cpp decode bar).

## Stop rules / honesty

- Measure on the device metric (the M0 lesson). Warm up, fix clocks, median.
- Pre-register before each gate; record negatives in full. A null result on N1 is a real, valuable
  result (it closes the loop thesis), not a failure to engineer.
- Do not oversell a native-matmul learnability win as a llama.cpp-decode win — they are decoupled
  (the scope boundary above).

## Pointers

- Prior verdicts this builds on: `docs/amd-decode-flywheel-proof-plan.md` (W2.1 framework wall, M0
  flat GEMV), `docs/amd-decode-flywheel-postmortem.md` (2026-06-15 addendum).
- Substrate artifacts: `bench/amd-decode-flywheel-proof-20260614/native-matmul-N0/`.
