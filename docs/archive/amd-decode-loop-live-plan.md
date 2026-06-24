# Phase L — Make the loop LIVE (turn N2's offline simulation into a real autotuner)

Date: 2026-06-15

## Why
The loop mechanism is PROVEN but OFFLINE. N1/N2 (final report) showed a learned cost model guides
search to 95% of oracle in a median of 1 measured config (~86x fewer than random) and beats the
deterministic lookup 26/26 folds — but N2's "guided search" **looks up already-measured device times**
from the dataset (`qk_loop_search._train_predict` → argsort → `test[i]["tflops"]`). It has never timed a
candidate live. Final-report follow-up #1: "wire the cost model into a BEAM warm-start and measure real
wall-clock autotuning speedup on FRESH shapes — turn the offline simulation into a tool." This is that.

## What's reused (no fork — both grounded by code maps)
- **Model**: `extra/qk_loop_learnability._train_predict(train, test)` — XGBoost regressor, 23 features
  (13 shape: M/K/N + logs + products + flops + aspect + N_small/N_big; 10 opt-aggregate: n_opts, has_tc,
  tc_level, up0/up1, loc0/loc1, unr0, tot_up, tot_loc), target = tflops, retrained in-process (no
  persisted artifact; fast). Predicts tflops per (shape, complete-config).
- **Live timing path**: `extra/qk_beam_log.py` → `tinygrad.codegen.opt.search._time_program(prg, {},
  bufs, cnt=3)` via `to_program(replace_opts(ast, opts), renderer)`. THIS IS EXACTLY how the dataset was
  built — so timing a fresh shape's configs live reuses the audited path, no new timing code.
- **Candidate set**: the same 277 opt-schedules (`qk_beam_log.gen_candidates`: 1 baseline + 3 TC +
  2-opt/3-opt combos over UPCAST/LOCAL/UNROLL). Encoding `[{op,axis,arg}]` == tinygrad `Opt(op,axis,arg)`
  (verified: direct serialization, no mismatch).
- **Corpus**: `bench/amd-decode-flywheel-proof-20260614/native-matmul-N0/beam_log_n1*.jsonl` (26 shapes
  × 277 configs, 5456 deduped records).

## The structural fact that sets the sequencing
The N1/N2 model ranks COMPLETE schedules over the fixed 277-config space. tinygrad's native
`beam_search` (search.py:114) instead builds schedules INCREMENTALLY from a larger action pool, timing
PARTIAL schedules with beam-width pruning. So:
- **L0/L1 (faithful, low-risk)**: live-time the model's ranked candidates over the **277-config substrate
  the model knows**, on FRESH shapes. Directly converts N2's lookup into real device timing. This is the
  honest "make it a tool."
- **L2 (deeper, stretch)**: inject a cost-model reorder hook into native `beam_search` (search.py:143).
  This is the final report's literal "BEAM warm-start," but the model would predict on PARTIAL schedules
  from a bigger action space than its 277-config training distribution (OOD) — a real research risk,
  gated behind L0/L1 and with a correctness-safe fallback (reorder-only: still time every candidate, so
  worst case = no speedup, never wrong).

## L0 — make-or-break (cheapest, FIRST): live guided search on ONE fresh shape
1. Pick 1 real Qwen3 GEMM shape NOT in the 26-corpus (verify absence by (M,K,N) tuple).
2. Train the model on the full 26-shape corpus; predict-rank the 277 configs for the fresh shape.
3. **LIVE**: `_time_program` the model's top-K (K=1,2,4,8) on device; record real wall-clock + best tflops.
4. Also sweep all 277 live = the **oracle** for this shape (gives % of oracle + the exhaustive wall-clock).
5. Random-K baseline: Monte-Carlo over the same 277 live times.
- **Gate**: guided top-8 reaches >=95% of the live oracle, in real wall-clock << exhaustive, AND beats
  random-8. If live device noise breaks the offline result (top-1 unreliable) -> that is a REAL finding;
  diagnose (cnt, clear_l2, warmup), do not paper over.

## L1 — generalize: H fresh held-out shapes
4–6 fresh shapes; per shape repeat L0 (leave-it-out training). Report median trials-to-95%, guided vs
random vs exhaustive WALL-CLOCK, and the % -oracle distribution — now on LIVE device times on UNSEEN
shapes. Pre-registered gate mirrors N2 (guided hits 95% by K<=8; far fewer trials than random), but
honest: if live degrades from the offline 0.92 / 86x, report the real numbers.

## L2 — native BEAM warm-start (stretch, gated on L1)
Inject an optional scorer at `tinygrad/codegen/opt/search.py:143` (after `candidates = flatten([...])`):
reorder candidates by predicted time so the predicted-best are timed first; early-stop (`_time_program`
early_stop, search.py:157) then prunes faster. Handle train/serve shift: extract the 10 opt-aggregate
features from each candidate's `applied_opts` + M/K/N from `full_shape`; the model is OOD on partial
schedules and unseen opts. Fallback = reorder-only (correctness identical — we still time every kept
candidate; we only change ORDER + early-stop budget). Measure end-to-end BEAM wall-clock with/without
the warm-start on fresh shapes. Optional retrain on partial-schedule timings harvested from real BEAM
logs if reorder-only doesn't pay.

## Touch points
- New: `extra/qk_loop_live.py` (train → rank → live-time top-K + full-sweep oracle + random baseline +
  wall-clock accounting). Reuses `qk_loop_learnability._train_predict`, `qk_beam_log` timing/candidate
  gen, the N0 dataset.
- Results: `bench/amd-decode-flywheel-proof-20260614/loop-live-L0/` (+ L1). Test:
  `test/external/test_qk_loop_live.py` (fresh-shape absence; live-time path; guided-beats-random on the
  live numbers; deterministic ranking under fixed seed).
- L2 (if reached): guarded optional hook in `tinygrad/codegen/opt/search.py` +
  `extra/qk_loop_beam_warmstart.py`.

## Pre-registered honesty
- Live device times are noisier than the dataset's min-of-3; the offline 0.92 / 86x may not fully
  reproduce — report the real degradation, do not re-tune the gate to pass.
- The "speedup" is autotuning SEARCH wall-clock to reach a quality bar on FRESH shapes — NOT a kernel
  that beats llama.cpp (those on-target spaces stay dead; unchanged).
- Integrity: fresh shapes must be genuinely absent from the corpus (assert by tuple). Fixed seed →
  deterministic ranking (re-run twice, identical).

## Out of scope
New ops (conv/attention — that's the "scale the substrate" follow-up), 14B/32B cross-model transfer,
expanding the 277-config space, and any claim about single-stream decode parity.
