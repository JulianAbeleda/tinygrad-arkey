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
- **N0b — instrument the search space (the substrate dataset).** Run BEAM (`BEAM=2/4/8`) over the
  model's matmul shapes; log EVERY `(opt-schedule config -> device_time)` trial to a persisted
  dataset. Characterize the landscape: trials-to-converge, ruggedness, count of near-optimal configs,
  how the winning schedule varies by shape. This dataset is the substrate N1 trains/tests on.
  Artifact `native-matmul-N0/beam_log.jsonl` + a characterization note.

### Phase N1 — learnability + transfer (route ④, LATER, per user)

- **N1a — single-kernel learnability.** Train a cost model on BEAM logs for a set of shapes; predict
  good configs on HELD-OUT shapes; does it cut BEAM's trials at equal final quality? Leak-free,
  family-split holdout (reuse the Phase 3 cost-model discipline).
- **N1b — cross-kernel transfer (the flywheel core).** Warm-start BEAM on a NEW kernel from past
  kernels' data; does accumulated experience monotonically reduce trials over a sequence of kernels
  (the loop getting better)? This is THE flywheel question.
- Pre-registered: beats BEAM sample-efficiency on N1a AND shows transfer on N1b -> the loop mechanism
  has a home (general autotuning result). Ties/loses BEAM on either -> general negative, thesis closed.

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
