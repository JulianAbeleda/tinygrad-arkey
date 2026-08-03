# NV beyond-parity forward scope - reviewer amendment

Date: 2026-08-03

Status: reviewer amendment, docs only. This amendment corrects the lifecycle,
evidence, and sequencing of the proposed "parity and beyond" work. It does not
authorize implementation, route-record changes, promotion to `dev`/`exp`/`master`,
or a composed performance forecast. Branch boundary: tinygrad
`nvidia-bringup-20260731` at `3f52f00fd`.

This amendment applies to `decode-parity-endgame-design-20260803.md`, especially
sections 5-8, and to any older endpoint or ordering claim in
`decode-gap-per-target-lever-scope-20260802.md` and
`l1-decode-plumbing-fusion-design-20260802.md`. The existing variant-specific
scopes may continue under their own HARD STOP and measurement gates; this amendment
stops only the obsolete umbrella from authorizing or sequencing implementation.

---

## 1. Verdict

**REVISION REQUIRED before the beyond-parity umbrella may guide work.**

The direction remains valid, but the current umbrella still contains claims that
were superseded by the forward-review amendment and the global consistency pass:

1. Decode is not at parity today, so "decode beyond parity" is not an active
   lifecycle state.
2. The old `195-210 tok/s`, `0.9-1.0ms`, and `1.07-1.21x` decode endpoint stack is
   withdrawn. It cannot be carried into a new scope as a target or expectation.
3. B3 has a measured host-side residual, but its tuned-schedule cause split and
   recoverable wall mass are not measured. The `18-21k tok/s` busy-ceiling estimate
   is not an achievable endpoint, and "landing B3 pushes prefill past llama" is not
   licensed.
4. L1 is no longer the sequencing authority. L2, L4, flash, M5, B3, and any later
   route are ordered only by isolated same-session wall evidence and their own
   controls.
5. "Parity" and "beyond parity" must be target-specific measured states, not one
   campaign-wide label.

HARD STOP on any beyond-parity implementation or composed endpoint until the
canonical replacement requested in section 6 is reviewed. This does not stop the
already approved characterization, microbench, and variant-specific scope work.

---

## 2. Current measured authority

The replacement scope must begin with these exact lifecycle baselines and must not
collapse them into a single parity statement.

### 2.1 Prefill

Same-session P5 warm rows from
`nv-performance-campaign-scope-20260801.md` section 13.1:

| target | tinygrad tok/s | llama tok/s | ratio | state |
| --- | ---: | ---: | ---: | --- |
| pp512 | 11,158 | 14,468.4 | 0.77x | BELOW PARITY |
| pp1024 | 14,003 | 14,450.3 | 0.97x | NEAR PARITY, NOT ABOVE |
| pp2048 | 14,947 | 14,231.6 | 1.05x | ABOVE IN THIS MEASURED SESSION |
| pp4096 | 13,657 | 13,793.7 | 0.99x | NEAR PARITY, NOT ABOVE |

pp128/pp256 remain the separately scoped short-prompt cliff and are not covered by
the promoted pp512+ path.

B3 authority on the tuned pp512 schedule is limited to: 44-46ms wall, 24.1ms GPU
busy, 23.7-23.8ms elapsed inside `wait()`, and a ~20-22ms wall-minus-busy residual.
The residual's submit/poll/overlap split is UNRESOLVED pending the same-run
instrumentation in `b3-prefill-host-overhead-scope-20260803.md` section 1.1.

### 2.2 Decode

Same-session M2-open rows from `nv-decode-parity-final-20260802.md`:

| target | tinygrad tok/s | llama tok/s | tinygrad/llama | state |
| --- | ---: | ---: | ---: | --- |
| d512 | 172.80 | 248.20 | 0.696x | BELOW PARITY |
| d2048 | 161.50 | 235.14 | 0.687x | BELOW PARITY |
| d4096 | 149.00 | 225.95 | 0.659x | BELOW PARITY |

The current reproducible state is M2 OPEN for `NV:sm_120`; M3, M4, M5, and Path 3
are CLOSED. The next evidenced work is isolated decode GEMV efficiency and the
already scoped M5 boundary P0, not a composed L1 delivery sequence.

---

## 3. Required lifecycle model

The replacement must use explicit states and may not use "parity" as a narrative
goal that silently licenses later phases.

### 3.1 Allowed states

- **OBSERVED**: directly measured accounting or behavior, with session provenance.
- **INFERRED**: hypothesis supported by observations but not directly isolated.
- **SCOPED**: mechanism and settling gate written; no implementation claim.
- **IMPLEMENTED-CLOSED**: mechanism exists behind a closed default; no performance
  promotion claim.
- **MEASURED**: isolated same-session wall and correctness record exists.
- **LANDED**: the measured route/default change is committed under its promotion
  controls.
- **PARITY-QUALIFIED**: a named target point has a same-session tinygrad/llama row
  meeting the scope's declared parity criterion.
- **BEYOND-PARITY-QUALIFIED**: a named target point has a same-session result above
  the declared parity criterion. It does not generalize to other depths/prompts.
- **CLOSED**: measured non-landing or rejected mechanism.
- **UNRESOLVED**: required evidence or external control does not exist.

### 3.2 Claim boundary

Prefill pp2048 may be called "above llama in the measured P5 session." It does not
license "prefill beyond parity" at pp512, pp1024, pp4096, or pp128/256. Decode may
not be called parity-qualified at any recorded depth today.

The replacement must state its numerical parity criterion and repetition protocol.
At minimum, every qualification claim uses the existing same-session harness,
repeated medians, the matching llama row, and all correctness pins. A result at one
target point is reported at that target point only.

---

## 4. Corrected forward structure

### 4.1 Work that may proceed now

1. **Decode GEMV characterization:** L2 partial single-pass, L4 vocab substrate,
   and flash structure remain separate. Their A-C order is a provisional node-sum
   upper-bound order; wall ranking is PENDING until isolated d512 measurements exist.
2. **M5 typed-boundary P0:** infrastructure may land closed. The
   `decode_flash_combine_fusion` route opens only with isolated measured wall benefit,
   fixed-depth correctness, and legacy hash controls.
3. **B3 characterization:** run the tuned-schedule same-run poll-count, exclusive
   polling-cost, submission-latency, and residual-overlap instruments before choosing
   a fix. Any shared runtime change requires the AMD runtime leg before landing.
4. **Short-prompt prefill:** remains independent and does not inherit pp512+
   qualification.

### 4.2 Parity phase

Each work item is implemented and measured independently. Re-rank only from isolated
same-session wall results. Compose only landed pieces, then remeasure the complete
target matrix. Node-sum remains diagnostic and may not be converted into a wall
endpoint by arithmetic or a haircut.

For decode, a parity phase remains active until the declared criterion is met at each
claimed depth. A win at d512 does not qualify d2048 or d4096.

For prefill, the current pp2048 win stands only as its measured row. B3 or another
change must be measured independently at pp512/1024/2048/4096 before any broader
prefill qualification.

### 4.3 Beyond-parity phase

Beyond-parity work activates per target point only after that point is
PARITY-QUALIFIED. It is not a pre-authorized list of L1/L2/L4/flash/B3 changes.
Candidate work is selected from newly measured residuals after parity, receives its
own scope, and must demonstrate additional isolated wall benefit before composition.

No "beyond" endpoint is published from hardware busy ceilings, class deltas,
node-sum recovery, or unlanded candidates.

---

## 5. Claims to withdraw or rewrite globally

DeepSeek must search canonical sources and either remove these claims or retain them
only inside an explicitly labeled historical/supersession record:

- `195-210 tok/s` as a current decode target or expected outcome;
- decode `1.07-1.21x` as a forward expectation;
- L1 `0.9-1.0ms` as an available composed lever;
- `18-21k tok/s` as a B3 outcome rather than a busy-ceiling bound;
- "Landing B3 pushes prefill past llama";
- "decode beyond parity" while all measured decode depths remain below parity;
- "L2/L4/flash after L1" or any equivalent L1-first sequencing;
- broad "prefill pp512+ parity" wording that hides the measured
  `0.77x/0.97x/1.05x/0.99x` per-target matrix.

Historical documents may preserve the original text only when their status names the
superseding authority at the top and the canonical forward document does not require a
reader to reconcile conflicting sections manually.

---

## 6. Required DeepSeek response

Produce one new canonical document:

`docs/task_workflow/input/nv-parity-and-beyond-forward-scope-20260803.md`

It must:

1. Declare itself the sole forward authority for parity and beyond-parity sequencing.
2. Reproduce the measured baseline tables in section 2 with provenance.
3. Use the lifecycle states in section 3.
4. Separate current work, parity qualification, and beyond-parity qualification.
5. Treat decode wall ranking, B3 cause decomposition, the AMD leg, route openings,
   and any composed endpoint as UNRESOLVED where evidence is missing.
6. State that existing variant-specific scopes continue under their own gates and
   that this umbrella authorizes no implementation.
7. Name which older sections/documents it supersedes, then update their status headers
   or canonical claims so the repository does not tell two different stories.
8. Include an adversarial consistency report: search results for every withdrawn claim
   in section 5, classifying each remaining hit as current authority, historical quote,
   or supersession statement.

The response should also amend `decode-parity-endgame-design-20260803.md` so sections
5-8 are not presented as a live forward scope. Prefer a clear historical/superseded
status over another reconciliation paragraph.

---

## 7. Review questions DeepSeek must answer

1. What exact numerical and repetition criterion qualifies a single target point as
   parity and beyond parity?
2. Which current work item runs first after isolated wall measurements replace the
   provisional node-sum order?
3. Does B3 remain one scope after characterization, or does the evidence split it into
   polling, submission, and whole-schedule scopes?
4. Which decode target matrix must pass before any campaign-wide decode-parity phrase
   is allowed?
5. Which older documents become historical only, and what is the one canonical forward
   authority after the rewrite?

HARD STOP. Do not authorize a beyond-parity implementation, route promotion, or
composed endpoint until the canonical replacement and global consistency response are
reviewed.
