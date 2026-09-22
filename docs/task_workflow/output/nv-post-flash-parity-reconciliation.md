# Post-Flash parity reconciliation

## Verdict

The installed dense endpoint has not yet reached llama wall parity.  The
authority remains 4.060523 ms/token (246.274 tok/s) versus llama at 4.021721
ms/token (248.711 tok/s), a 38.802-us/token gap.

Flash conversion is now in llama's percentage band, but absolute Flash service
is not at parity.  The current production score population is 193.296--194.080
us/token versus llama PDL-off at 162.948 us/token, leaving about 30--31 us/token
of score debt.  Current combine is about 48.3 us/token versus llama's 37.057-us
production row, but matched native replay makes tinygrad's combine faster; that
unmatched row difference is not an admissible recovery pool.

## Fresh ledger correction

The first post-promotion ledger was invalid because its replay parser selected
the older 32/64/128 symbolic/prefill graph family.  That produced a physically
impossible 362-ms node sum and negative host residual.  The installed decode
signature is 33/66/132/185.  The parser now requires that exact signature and
rejects any replay containing a `_toks` kernel.

The corrected profile contains 182 complete decode replays and 179 steady
replays.  Its median device accounting is:

| quantity | result |
|---|---:|
| node sum | 3925.824 us/token |
| device union | 3925.500 us/token |
| overlap | 0.324 us/token |
| retained llama PDL-off node sum | 3878.254 us/token |
| tinygrad device-service debt | 47.570 us/token |

The profiled 5040.163-us wall includes 1114.663 us of profiling/instrumentation
tax and is not an endpoint authority.  The unprofiled endpoint above remains
the wall authority.

The per-name table now records `replay_samples` and
`steady_replay_fraction`.  This prevents the three S8 transition replays from
being added to the 176-replay S6 steady population as though both paths ran in
every token.

## V-tail translation test

The V-tail construction is bit-exact and improves the isolated repeated score
primitive from 4.080960 to 3.935406 us.  Its canonical unprofiled 24-token x
9-repetition A/C/A bracket is also positive:

| arm | wall |
|---|---:|
| control A | 4069.504 us/token |
| installed V-tail candidate | 4064.019 us/token |
| control C | 4070.290 us/token |
| candidate versus control midpoint | **-5.878 us/token, +0.355 tok/s** |

All token hashes match.  V-tail is already the installed default, so this is a
confirmation of the existing endpoint construction, not additional recovery
that can be subtracted from 4.060523 ms/token.

The result also constrains the remaining Flash work: only a small fraction of
the hot primitive advantage converts at token wall.  Another scheduling
spelling must not claim the isolated delta without a production bracket.

## Current decision

- Dense wall parity: **not yet reached**.
- Flash hot-to-production percentage parity: **approximately reached**.
- Absolute production Flash parity: **not reached**; score retains about
  30--31 us/token of matched-row debt.
- Existing V-tail spelling: **booked and reconfirmed, now exhausted as a new
  recovery source**.
- Remaining full-device debt: about 47.6 us/token, consistent with the
  38.8-us unprofiled endpoint gap once small lifecycle/host differences are
  charged.

## Evidence

- `docs/task_workflow/evidence/nv-post-flash-parity-ledger-20260827/ledger-corrected.json`
- `docs/task_workflow/evidence/nv-post-flash-parity-ledger-20260827/ledger/production.profile.jsonl`
- `docs/task_workflow/evidence/nv-flash-score-parity-reconcile-20260827/installed-vtail1-r9.json`
- `docs/task_workflow/evidence/nv-flash-score-parity-reconcile-20260827/vtail1-wall-canonical-r9.json`

