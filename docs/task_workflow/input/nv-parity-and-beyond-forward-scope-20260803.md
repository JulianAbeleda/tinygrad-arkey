# NV parity and beyond - canonical forward scope

Date: 2026-08-03
Status: canonical forward authority for parity and beyond-parity sequencing. Docs only;
authorizes no implementation, no route-record change, no promotion to
`dev`/`exp`/`master`, and no composed performance endpoint. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `3f52f00fd`. This document is the response to
`nv-beyond-parity-forward-scope-review-amendment-20260803.md` and implements its section
6 requirements.

## 1. Authority and scope

This document is the SOLE forward authority for parity and beyond-parity sequencing.
It supersedes, for forward purposes:

- `decode-parity-endgame-design-20260803.md` sections 5-8 and its role as the
  beyond-parity umbrella (the design record stays as history, section 8);
- the endpoint and L1-first ordering claims in
  `decode-gap-per-target-lever-scope-20260802.md` (sections 6 and 8.2) and
  `decode-norm-fusion-paths-forward-20260802.md`;
- the sequencing language in `nv-campaign-forward-review-20260803.md` sections 2-3
  (already superseded by its amendment; this document is now the forward authority);
- `nv-campaign-forward-review-comments-addressed-20260803.md` as a sequencing record
  (it remains the disposition record for the earlier amendment).

The existing variant-specific scopes continue UNCHANGED under their own HARD STOP and
measurement gates: the decode GEMV-efficiency scope
(`decode-gemv-efficiency-forward-scope-20260803.md`), the M5 boundary P0 scope
(`m5-variant-reopen-boundary-p0-scope-20260803.md`), the B3 prefill scope
(`b3-prefill-host-overhead-scope-20260803.md`), and the short-prompt prefill scope. This
umbrella adds nothing to them and removes nothing from them.

## 2. Measured baselines (exact, with provenance)

### 2.1 Prefill (same-session P5 warm rows, `nv-performance-campaign-scope-20260801.md` section 13.1)

| target | tinygrad tok/s | llama tok/s | ratio | state |
| --- | ---: | ---: | --- | --- |
| pp512 | 11,158 | 14,468.4 | 0.77x | BELOW PARITY |
| pp1024 | 14,003 | 14,450.3 | 0.97x | NEAR PARITY, NOT ABOVE |
| pp2048 | 14,947 | 14,231.6 | 1.05x | ABOVE IN THIS MEASURED SESSION |
| pp4096 | 13,657 | 13,793.7 | 0.99x | NEAR PARITY, NOT ABOVE |

pp128/pp256 remain the separately scoped short-prompt cliff and are not covered by the
promoted pp512+ path. Provenance: 2026-08-02, RTX 5090 / sm_120, Qwen3-8B-Q4_K_M, warm
steady-state passes, llama-bench same-session rows.

B3 authority on the tuned pp512 schedule is limited to: 44-46 ms warm wall, 24.1 ms GPU
busy, 23.7-23.8 ms elapsed inside `wait()`, and a ~20-22 ms wall-minus-busy residual.
The residual's submit/poll/overlap split is UNRESOLVED pending the same-run
instrumentation in `b3-prefill-host-overhead-scope-20260803.md` section 1.1.

### 2.2 Decode (same-session M2-open rows, `nv-decode-parity-final-20260802.md`)

| target | tinygrad tok/s | llama tok/s | tinygrad/llama | state |
| --- | ---: | ---: | ---: | --- |
| d512 | 172.80 | 248.20 | 0.696x | BELOW PARITY |
| d2048 | 161.50 | 235.14 | 0.687x | BELOW PARITY |
| d4096 | 149.00 | 225.95 | 0.659x | BELOW PARITY |

Reproducible promotion state: M2 `decode_epilogue_fusion` OPEN for `NV:sm_120`; M3 norm,
M4 Q4K epilogue, M5 combine, and Path 3 semantic RMSNorm CLOSED. Correctness pins at
every depth: token sha `9d6b3787...` 3/3, first token `151936` 3/3; decode sha
`0721c16f...`; census row `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d...`.

## 3. Lifecycle states (adopted, used for every claim in this document)

- OBSERVED: directly measured accounting or behavior, with session provenance.
- INFERRED: hypothesis supported by observations but not directly isolated.
- SCOPED: mechanism and settling gate written; no implementation claim.
- IMPLEMENTED-CLOSED: mechanism exists behind a closed default; no performance promotion.
- MEASURED: isolated same-session wall and correctness record exists.
- LANDED: the measured route/default change is committed under its promotion controls.
- PARITY-QUALIFIED: a named target point has a same-session tinygrad/llama row meeting
  the declared criterion (section 5).
- BEYOND-PARITY-QUALIFIED: a named target point has a same-session result above the
  declared criterion; it does not generalize to other depths/prompts.
- CLOSED: measured non-landing or rejected mechanism.
- UNRESOLVED: required evidence or external control does not exist.

## 4. Current work (SCOPED; may proceed now, each under its own gate)

| item | state | gate |
| --- | --- | --- |
| Decode GEMV characterization (L2 partial, L4 vocab substrate, flash structure) | SCOPED; A-C order is a provisional node-sum upper-bound order; wall ranking PENDING | isolated same-session d512 measurement per item before any ranking claim |
| M5 typed-boundary P0 | SCOPED; infrastructure may land closed | route opens only with isolated measured wall benefit, fixed-depth correctness, legacy hash controls |
| B3 characterization | SCOPED; cause split UNRESOLVED | tuned-schedule same-run poll count, exclusive polling cost, submission latency, residual overlap; AMD runtime leg before any shared change lands |
| Short-prompt prefill (pp128/256) | SCOPED, independent | its own scope; inherits no pp512+ qualification |

No item is sequenced by another. L1-first ordering is withdrawn (section 8).

## 5. Parity phase (per target point, per depth)

Parity criterion (declared): a target point is PARITY-QUALIFIED when the same-session
harness median satisfies `tinygrad >= llama` at that point (ratio >= 1.00), with the
matching llama row from the same session family, the repetition protocol of the
referencing harness (decode fixed-depth: nmeas=20, reps=3, median; prefill: warm
steady-state passes median), and all correctness pins (section 2.2). The exact ratio is
reported at that target point only.

- Prefill: pp2048 is ABOVE-PARITY-QUALIFIED for the P5 session only (1.05x row). pp512,
  pp1024, and pp4096 are not qualified; pp128/256 are out of scope here. B3 or any other
  change must be measured independently at pp512/1024/2048/4096 before any broader
  prefill qualification is claimed.
- Decode: no depth is parity-qualified today. A win at d512 qualifies d512 only; d2048
  and d4096 each need their own rows. Campaign-wide "decode parity" phrasing is allowed
  only after every depth in the claimed matrix (minimum d512/d2048/d4096) is individually
  PARITY-QUALIFIED.

Each work item is implemented and measured independently. Re-rank only from isolated
same-session wall results. Compose only LANDED pieces, then re-measure the complete
target matrix. Node-sum remains diagnostic and may not be converted into a wall endpoint
by arithmetic or a haircut.

## 6. Beyond-parity phase (per target point, not a pre-authorized list)

Beyond-parity work activates per target point only after that point is
PARITY-QUALIFIED. It is not a pre-authorized list of L1/L2/L4/flash/B3 changes. Candidate
work is selected from newly measured residuals after parity, receives its own scope, and
must demonstrate additional isolated wall benefit before composition.

No beyond endpoint is published from hardware busy ceilings, class deltas, node-sum
recovery, or unlanded candidates. In particular: the `18-21k tok/s` figure is a
busy-ceiling bound derived from 512/24.1 ms, not an outcome; "landing B3 pushes prefill
past llama" is not licensed; and "decode beyond parity" is not an active state while all
measured decode depths remain below parity.

## 7. UNRESOLVED (explicitly, per amendment section 2 and 6.5)

- Decode wall ranking: PENDING isolated same-session d512 measurements for L2/L4/flash.
- B3 cause decomposition: the wall-minus-busy residual's submit/poll/overlap split is
  UNRESOLVED pending the section 1.1 instruments.
- AMD leg: no AMD GPU exists on this machine; the runtime control required before any
  shared runtime change lands has not run.
- Route openings: `decode_flash_combine_fusion`, `decode_q4k_epilogue_fusion`, the norm
  family, and Path 3 all stay CLOSED until their own measured gates pass.
- Any composed endpoint: none exists; none is published from this document.

## 8. Supersession map and status updates applied

| document | remains authoritative for | no longer |
| --- | --- | --- |
| `decode-parity-endgame-design-20260803.md` | design history (sections 1-4) | a live forward umbrella; sections 5-8 are historical (header updated) |
| `decode-gap-per-target-lever-scope-20260802.md` | lever evidence and verdict taxonomy | endpoint expectations and ordering (sections 6, 8.2 marked withdrawn) |
| `decode-norm-fusion-paths-forward-20260802.md` | M3 path analysis | the 195-210 tok/s target phrasing (marked withdrawn) |
| `nv-campaign-forward-review-20260803.md` (+ amendment) | review history | forward sequencing authority (header names this document) |
| `nv-campaign-forward-review-comments-addressed-20260803.md` | disposition record for the earlier amendment | sequencing authority |
| variant-specific scopes (GEMV, M5, B3, short-prompt) | unchanged | nothing |

Status headers of the above documents were updated in the same commit as this document
where their live-forward framing could mislead a reader.

## 9. Adversarial consistency report (search of every withdrawn claim)

Searches run 2026-08-03 over `docs/task_workflow/input/*.md` for each claim in amendment
section 5. Classification: REMOVED = rewritten out of the source; HISTORICAL = retained
inside a document whose header names the superseding authority; SUPERSESSION = the hit is
itself a withdrawal statement; REQUIREMENT = the reviewer amendment or a quote of it;
EVIDENCE = per-target matrix, not a campaign-wide claim.

| withdrawn claim | hits (file:line after rewrite) | classification |
| --- | --- | --- |
| `195-210 tok/s` as target/outcome | `decode-gap-per-target-lever-scope-20260802.md:353`; `decode-norm-fusion-paths-forward-20260802.md:182,185,315,413`; `decode-parity-endgame-design-20260803.md:9,169`; amendment | SUPERSESSION (decode-gap, paths-forward); HISTORICAL (endgame, header-labeled); REQUIREMENT (amendment, this doc) |
| decode `1.07-1.21x` | `nv-campaign-forward-review-20260803.md:42,52,102,138`; `decode-gap-per-target-lever-scope-20260802.md:420`; `decode-gemv-efficiency-forward-scope-20260803.md:48,310`; amendment | HISTORICAL (forward review, header-labeled); SUPERSESSION (decode-gap, GEMV scope, amendment) |
| L1 `0.9-1.0ms` as available lever | `nv-campaign-forward-review-20260803.md:40,137`; `decode-gap-per-target-lever-scope-20260802.md:172,411,423`; `decode-norm-fusion-paths-forward-20260802.md:109`; `decode-gemv-efficiency-forward-scope-20260803.md:48,310`; amendment | HISTORICAL (forward review, decode-gap taxonomy, endgame); SUPERSESSION (GEMV scope, amendment, paths-forward, decode-gap) |
| `18-21k tok/s` as B3 outcome | `decode-parity-endgame-design-20260803.md:9,170`; amendment | HISTORICAL (endgame, header-labeled); SUPERSESSION (amendment, this doc section 6) |
| "Landing B3 pushes prefill past llama" | `decode-parity-endgame-design-20260803.md:9,156`; amendment | HISTORICAL (endgame, header-labeled); SUPERSESSION (amendment, this doc section 6) |
| "decode beyond parity" | `decode-parity-endgame-design-20260803.md:10,158`; amendment | HISTORICAL (endgame, header-labeled); SUPERSESSION (amendment, this doc section 6) |
| L2/L4/flash after L1 (L1-first) | `decode-parity-endgame-design-20260803.md:160`; `decode-gap-per-target-lever-scope-20260802.md:423`; `nv-campaign-forward-review-20260803.md:95` | HISTORICAL (endgame, forward review); SUPERSESSION (decode-gap ordering sentence) |
| broad "prefill pp512+ parity" | `nv-campaign-forward-review-20260803.md:45,98`; `nv-performance-campaign-scope-20260801.md:663`; amendment; comments-addressed | HISTORICAL (forward review, header-labeled); EVIDENCE (campaign line 663 states the 0.77x-1.05x matrix); REQUIREMENT (amendment 4.2.1) |

No remaining hit is presented as current authority outside this document or the labeled
historical records named in section 8.

## 10. Answers to the review questions

1. **Parity criterion**: ratio >= 1.00 at a named target point, same-session harness
   median vs the matching llama row, referencing harness's repetition protocol (decode
   nmeas=20 reps=3 median; prefill warm steady-state median), all correctness pins; the
   exact ratio is reported at that point only. Beyond-parity adds: the point is
   PARITY-QUALIFIED first, and the candidate change shows an additional isolated
   same-session wall gain.
2. **First work item after isolated wall measurements**: no a-priori order. The first
   item run is the one whose isolated d512 measurement shows the largest same-session
   wall recovery within its own scope and controls; the measured result replaces the
   provisional node-sum order.
3. **B3 one scope or split**: B3 remains one scope until the section 1.1 instruments
   produce the poll/submit/overlap split. If the split shows independent mechanisms, the
   evidence then splits into polling, submission, and whole-schedule sub-scopes, each
   with its own gate. The decision is deferred to characterization; this document does
   not pre-split it.
4. **Decode matrix before campaign-wide phrasing**: every depth in the claimed matrix,
   minimum d512/d2048/d4096, must be individually PARITY-QUALIFIED before any
   campaign-wide "decode parity" phrase is allowed; a single-depth win qualifies that
   depth only.
5. **Historical documents and the one canonical authority**: the historical set is
   `decode-parity-endgame-design-20260803.md` (sections 5-8), the endpoint/ordering
   claims in `decode-gap-per-target-lever-scope-20260802.md` and
   `decode-norm-fusion-paths-forward-20260802.md`, and the review chain
   (`nv-campaign-forward-review-20260803.md`, its amendment, and the comments-addressed
   record). The one canonical forward authority is this document.

HARD STOP. This document authorizes no implementation, no route-record change, no
promotion, and no composed endpoint. Each work item proceeds only under its own
variant-specific scope and measurement gate.
