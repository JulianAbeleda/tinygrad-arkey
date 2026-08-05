# NV Q4 cooperative tail-subset extension record — 2026-08-05

## Verdict

Extending the qualified 17-block lease (`1–12,14–18`) into blocks `19–35`
did not produce another wall-bookable promotion.  Thirteen of the seventeen
singleton additions pass the full-logit semantic contract, but the best
qualified singleton regresses settled wall by `12.917031 us/token`.  The two
highest-cardinality combinations selected by the exhaustive additive rank both
fail fresh real-model validation.  Nothing is promoted or enabled by default.

This is a bounded **NO-GO for incremental credit**, not proof that every one of
the `131072` tail subsets fails real-model semantics.  The exhaustive CPU rank
is a direction oracle only: transformer propagation proved materially
non-additive when the selected combinations were run on the GPU.

## Authority and protocol

- Original unleased oracle: `/tmp/nv-q4-subset-search/g0.{json,npz}`.
- Qualified reference lease: blocks `1–12,14–18` (17 blocks).
- Tail search space: every block `19–35` as a fresh singleton, followed by all
  `2^17` additive subsets ranked from signed full-logit deltas.
- Every semantic child used d512, eight autoregressive logits, exact requested
  topology, the primitive binding check, and two-capture census.
- Semantic authority was relative L2 `<= 1e-3` plus finite logits, exact token
  stream, exact argmax, and exact top-10 set and order.
- Fresh real-model validation was limited to the nearest four-tail boundary and
  the best predicted three-tail candidate.  Wall authority was one settled
  continuous base17 / candidate / base17 reverse bracket.

## Singleton census

All 17 singleton arms had exact tokens, argmax and top-10.  Only the relative-L2
gate distinguishes PASS from FAIL.

| added block | relative L2 | verdict |
|---:|---:|:---:|
| 19 | `0.000968088` | PASS |
| 20 | `0.001027628` | FAIL |
| 21 | `0.000874874` | PASS |
| 22 | `0.000912821` | PASS |
| 23 | `0.000908661` | PASS |
| 24 | `0.000876298` | PASS |
| 25 | `0.000874804` | **PASS; best singleton** |
| 26 | `0.000986495` | PASS |
| 27 | `0.000899477` | PASS |
| 28 | `0.000930604` | PASS |
| 29 | `0.001198538` | FAIL |
| 30 | `0.000969715` | PASS |
| 31 | `0.000882946` | PASS |
| 32 | `0.000941336` | PASS |
| 33 | `0.000883147` | PASS |
| 34 | `0.001205941` | FAIL |
| 35 | `0.001171132` | FAIL |

## Additive rank and fresh validation

The signed-delta optimizer found no four-tail subset predicted below the
`1e-3` gate.  The nearest predicted boundary was `{23,24,27,33}` at
`0.001008359`; its fresh real-model result was worse at `0.001053048` and
failed.  The best predicted three-tail candidate `{23,24,33}` was expected at
`0.000917685`, but fresh real-model validation measured `0.001064545` and also
failed.  Both retained exact tokens, argmax and top-10.

The inversion between predicted and measured ordering is itself decisive:
adding independently observed final-logit deltas does not model how multiple
Q8 leases change later transformer state.  Further multi-block qualification
must either run real subsets or build an interaction-aware model; it must not
promote from additive prediction.

## Settled incremental wall

Block 25 was selected for the sole wall bracket because it is the best
already-real-qualified singleton.  Each fresh arm used six untimed decode
calls followed by five uninterrupted 32-token windows.  No high-side sample
was rejected and all three arms had the same exact 160-token stream hash.

| arm | median ms/token |
|---|---:|
| base17 A | `5.374582906` |
| base17 + block 25 | `5.395949656` |
| base17 C | `5.391482344` |
| base17 bracket midpoint | `5.383032625` |

The candidate is `0.012917031 ms/token` slower, a **`12.917031 us/token`
regression** (`-0.239384%` speedup).  There is no causal-ledger credit.

## Disposition

- Keep the 17-block lease as the last qualified, wall-positive subset.
- Do not add block 25 and do not promote any tail subset.
- Book zero incremental recovery from blocks `19–35`.
- Treat additive subset ranking as prioritization only, never semantic proof.
- Reopen this tail family only with either a cheaper real-subset oracle or a
  concrete interaction-aware hypothesis; the current best qualified extension
  is already wall-negative.

Compact evidence is in
`docs/task_workflow/output/nv-q4-cooperative-tail-subset-extension-20260805.json`.

