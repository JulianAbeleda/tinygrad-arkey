# NV campaign - forward path review

Status: review scope, response recorded. This document is the request-of-record;
`nv-campaign-forward-review-amendment-20260803.md` is the response-of-record and
supersedes sections 2-6 below wherever they differ (section 7 here is the reconciliation).
Records where the NV performance campaign stands after the decode-norm tail closed.
Branch boundary: tinygrad `nvidia-bringup-20260731` @ `72dca8010`, pushed and in sync.
Does not authorize promotion to `dev`/`exp`/`master`.

## 1. Recorded, measured state (all committed and pushed)

Prefill: P1 gate MET (warm pp512 11.2-11.6k tok/s from ~110, GPU busy 961->98.8ms, wall
4.3s->44-46ms). P2 resolved by P1's mechanism (all 252 linears on the fp16 overlay path,
no build). P5 full sweep: pp1024/2048/4096 at 0.97x/1.05x/0.99x of llama (pp2048 measured
above llama). Remaining prefill lever: host overhead B3 (1.9x wall/busy vs llama's
1.15-1.35x envelope; wait() CPU 23.7-23.8ms MEASURED on the tuned schedule). The
per-poll mechanism (fresh memoryview + cast per poll, `hcq.py:262`) is code-verified, but
its tuned-schedule cost is INFERRED pending same-run instrumentation (B3 scope section
1.1); the cause is NOT named. Requires an AMD control before landing.

Decode: final same-session parity rows (2026-08-03, `nv-decode-parity-final-20260802.md`):
d512 172.8 vs 248.2 (1.44x), d2048 161.5 vs 235.1 (1.46x), d4096 149.0 vs 226.0 (1.52x).
Decode is GPU/kernel-bound (95% busy) at 46% of the measured 1792 GB/s ceiling vs llama's
64%. Correctness pins hold at every depth (sha `9d6b3787...`, first token `151936`, 3/3).

Lever classes are diagnosed, not assumed (decode-gap-per-target-lever-scope-20260802.md):
L3 flash score = SUBSTRATE (36-row sweep, no values win), L5 lanemap = SUBSTRATE (lane
hypothesis falsified), L2 partial = SUBSTRATE (single-pass variant is a separate scope),
L4 vocab = values row landed (row_tile=2, d512 163.5->172.6 tok/s). M2 in-kernel Q6K down
coop merge landed promoted; M3/M4/M5/Path3 measured non-landing behind the custom-kernel
boundary copy tax and are correctly closed. The copy tax itself is scoped as Path 1
(copy-free opaque boundary) in `decode-norm-fusion-paths-forward-20260802.md`.

## 2. The decision point

The largest single decode lever is L1 plumbing fusion (~0.9-1.0ms of the ~1.98-2.18ms
realistic total, ~45-50%; corrected budget 4.37-4.93 ms/token end-state vs llama's 4.07ms
= ~1.07-1.21x). Its design is written and deliberately stopped for review:
`l1-decode-plumbing-fusion-design-20260802.md` ends at a HARD STOP with nine open
questions. Nothing is blocked on missing measurement; the campaign is at a review gate on
one design doc.

## 3. What we know vs do not know

Known and measured: prefill parity at pp512+ is reached with the fp16 mechanism; llama's
int8 IMMA mechanism is 25-40% faster at raw GEMM rate, so full prefill parity is not
reachable with fp16 mma (P0, section 8.2 of the campaign scope); decode parity at
1.07-1.21x is in reach only if L1 delivers both halves (norm pair-fusion ~0.58ms AND
GEMV/flash epilogue absorption ~0.4ms); B3 is the last prefill lever.

Open, honestly: whether the L1 fused norm survives the "no new kernel" review question
(Q2); whether L1 shape (a) epilogue absorption re-hits the boundary copy tax that sank
M3/M4/M5 (the design absorbs inputs under a gate, but the transport contiguous()s
non-identity inputs - the overlap with Path 1 needs a decision, not an assumption);
whether the partial-merge 466.6us anomaly resolves under the substrate scope (recorded,
not chased); and whether B3's fix removes the pp512 gap without moving AMD (needs the AMD
control run).

## 4. Non-competing paths

The remaining paths do not compete; they are ordered dependencies:

1. L1 plumbing fusion (decode, SUBSTRATE, largest recovery) - blocked only on review of
   the nine open questions.
2. Path 1 copy-free opaque boundary (transport substrate) - the M3 reopen condition and
   the L1 epilogue-absorption risk; determines whether L1 shape (a) lands copy-free or
   must be sequenced after the transport fix.
3. B3 host overhead (prefill, pp512) - runtime build, AMD control required, independent
   of decode work.

The short-prompt cliff is recorded-not-prioritised by its own scope and should not enter
sequencing. L4's vocab substrate variant and L2's single-pass partial are parkable
follow-ons behind L1; each is small and measured.

## 5. Proposed forward sequence

1. Review and resolve the L1 design's nine open questions (section 6 below).
2. Decide the L1-vs-Path-1 ordering: if L1 shape (a) requires the copy-free boundary to
   avoid the M3-class tax, Path 1 lands first as the transport fix; otherwise L1 lands
   directly. This decision is the single new question this scope adds.
3. Implement L1 capability-gated and additive, with pg3 HIP/CUDA render equality and the
   existing NV pins (first-token digits, decode sha, census row).
4. Re-measure decode at d512/d2048/d4096 against the same-session llama rows pinned in
   `nv-decode-parity-final-20260802.md`.
5. Only then size and build B3 (prefill host overhead) with the AMD control it requires.

## 6. Questions for review

1. Is the L1 design's nine-question set complete, or does the L1/Path-1 ordering question
   (section 5 item 2) need to be resolved inside L1's review rather than after it?
2. Is the "no new kernel" discipline in the campaign scope binding for a fused decode
   RMSNorm emitter (L1 Q2), or is it scoped to the prefill GEMM verdicts as the design
   argues?
3. Does the boundary copy tax (Path 1) gate L1 shape (a) epilogue absorption, or is the
   design's per-emitter opt-in sufficient to land L1 first?
4. Is B3 (prefill host overhead) correctly sequenced after decode L1, or should the
   prefill work proceed in parallel given the AMD control requirement is independent?
5. Are the end-state expectations (decode ~1.07-1.21x, prefill parity at pp512+, pp2048
   above llama) stated with the right evidence-class discipline - node-sum upper bounds
   until re-measured, never wall forecasts?

HARD STOP after this section. No implementation beyond the L1 design's own review until
this scope is reviewed.

---

## 7. Response to the reviewer amendment (2026-08-03)

The amendment (`nv-campaign-forward-review-amendment-20260803.md`) is accepted; this
section is the reconciliation. The five review questions are answered in the amendment's
sections 2-3; the corrected forward sequence is its section 4. Key acceptances, mapped to
the superseded text:

1. **Lifecycle (supersedes section 2 and section 6 Q1)**: the nine L1 design questions are
   no longer the blockers. M1-M5 and Path 3 are implemented and measured; the next
   decision is which existing CLOSED variant earns a narrowly scoped reopen, not whether
   to implement the original L1 design.
2. **Boundary verdict (supersedes section 4 item 2 and section 6 Q3)**: per-emitter
   opt-in does NOT solve transport. The forced-open measurements answer Q3 with
   "no" (amendment section 2.2). The boundary work is reframed as a variant-reopen
   boundary P0, NOT a general Path-1 transport campaign (the non-norm census NO-GO
   stands; the new claim is narrower: copies introduced by a specific forced-open
   variant).
3. **M4 decomposition (supersedes section 3's open-items framing)**: M4's +1264us is not
   explained by 126 copies (~189us). The `ffn_down_fused` prelude recomputes
   `_silu_uop(gate)*up` inside the per-row reduction, once per 4096 output row - the
   ~1075us arithmetic residual. This is now MEASURED, not estimated:
   `m4-decomposition-measurement-record-20260803.md` shows the fused kernel at 98.16us
   (d512) / 98.56us (d4096) vs legacy 26.23 / 26.34us (3.74x), with +1295us recompute
   mass. M4's combined record stays closed; residual-add, k/v fp16-output, and FFN-down
   are decomposed and measured independently, and the current FFN-down prelude shape is
   rejected until a compute-once design exists (amendment section 2.3).
4. **Forecast (supersedes section 3's "known" list and section 6 Q5)**: `0.9-1.0ms` and
   `1.07-1.21x` are withdrawn. The norm hypothesis stands at approximately -144 launches /
   -0.16ms node-sum (paths-forward section 9); no composed decode endpoint is stated
   until isolated same-session wall measurements exist. The wall authority is
   `nv-decode-parity-final-20260802.md` (1.44x/1.46x/1.52x), with the next evidenced wall
   lever in decode GEMV efficiency.
5. **B3 (supersedes section 4 item 3 and section 6 Q4)**: B3 is independent of decode and
   may proceed under its own scope with the AMD control; nothing in the decode boundary
   decision serializes it (amendment section 3 Q4).
6. **Parity record (supersedes section 1's implicit baseline)**: the reproducible state is
   M2 OPEN for NV:sm_120, M3/M4/M5/Path 3 CLOSED; the parity record is amended
   accordingly (amendment section 2.5).

Corrected forward sequence: amendment section 4 (decode: correct the state-of-record,
decompose M4, one minimal variant-reopen boundary P0 starting with M5, re-measure each
route independently, prioritize the current wall authority; prefill: B3 independently).
No implementation is authorized by this scope as amended; the next step is a
variant-specific reopen scope with its settling command, legacy hash controls,
correctness pins, and fixed-depth wall gate.
