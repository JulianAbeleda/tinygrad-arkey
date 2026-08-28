# Flash-to-O overlap reopen

## Result

Flash-to-O overlap is physically real, but the tested split-grid spelling is
not profitable.  The remaining construction is narrower than “more overlap”:
retain the full production Flash grid and publish per-head readiness from
inside it, then let an exact O partial consumer start without splitting Flash.

No recovery is booked and no production route changed.

## Exact decomposition

The corrected complete-span gate compares four bit-exact arms with legal,
finite outputs:

| arm | cold median | delta from control |
| --- | ---: | ---: |
| unsplit Flash + ordinary O | 25.264 us | — |
| split Flash + ordinary O | 29.448 us | +4.184 us |
| unsplit Flash + split exact O | 25.176 us | -0.088 us |
| split Flash/O with overlap | 27.096 us | +1.832 us |

The overlapped arm exposes about 3.84 us of timeline overlap.  Relative to the
two isolated split costs it recovers about 2.26 us, so the overlap is useful.
The dominant tax is instead splitting one 192-CTA Flash score grid and one
combine into two underfilled producer waves.  Exact O partial materialization
is cold-neutral in this gate.

This rejects further stream scheduling of the same split producer.  A useful
next precursor must preserve the full score geometry.  The proposed mechanism
is:

1. Every production score CTA writes its ordinary split partial.
2. A release counter is incremented for that query head.
3. The last split CTA for a head performs the existing combine association and
   publishes a head-ready epoch.
4. Reserved O worker CTAs consume ready 16-head groups and compute the existing
   exact lane-half partials.
5. The second half retains the installed low-plus-high operand order and final
   XOR ladder.

The first gate is score-plus-last-CTA-combine versus the current score and
combine launches.  It must be bit-exact, retain the 192-CTA score population,
avoid spills, and not regress the cold score/combine span.  Only a passing
precursor justifies adding resident O workers.

## Full-grid readiness precursor result

The precursor passes.  Each score CTA writes the unchanged partial ABI and
increments a per-head release counter.  The sixth CTA for that head performs
the existing six-split combine in the existing operand order, writes FP16, and
resets the epoch.  The score launch remains 192 CTAs.

The independent-ring, reversed-order R9 result is bit-exact.  The score body
remains at 56 registers with zero spills; shared memory rises by only four
bytes.  Hot score+combine falls from 4.862 to 4.452 us.  Independently rotated
cold service is neutral within dispersion, 7.716 versus 7.668 us.  The earlier
same-allocation cold result is rejected because it let the candidate inherit
the control's cache state.

This books no token recovery by itself.  It establishes the missing substrate:
head readiness can be exposed without splitting or slowing the Flash producer.
The next complete-span gate attaches a bounded persistent O consumer to these
head-ready epochs.

## Other reopened Q/O/K/V tests

The repaired static Q/K/V stripe uses legal finite Q4_K blocks.  All arms are
bit-exact and finite; interleaving recovers only about 0.05 us cold against the
Q-first full-grid control, below the 0.15 us advance threshold.

The transposed-qdata near-use `uint2` O spelling is bit-exact on three legal
fixtures.  It saves about 0.21 us hot but only 0.026 us cold.  Like the prior
`uint4` spelling, it reduces cache-hot instruction cost without shortening the
streamed service episode.

Primary evidence:

- `docs/task_workflow/evidence/nv-qokv-stripe-legal-20260828/gate-r9.json`
- `docs/task_workflow/evidence/nv-q4k-qdata-transpose-u2-20260828/gate-r9.json`
- `docs/task_workflow/evidence/nv-flash-o-tax-decomposition-20260828/gate-r9.json`
- `docs/task_workflow/evidence/nv-flash-last-cta-combine-20260828/gate-independent-r9.json`
