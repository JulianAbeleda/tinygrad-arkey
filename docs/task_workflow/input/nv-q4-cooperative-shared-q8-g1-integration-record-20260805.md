# NV Q4 cooperative shared-Q8 g1 integration record (2026-08-05)

## Verdict

`g18 = SEMANTIC_STOP`; g12 is the last settled wall-qualified depth.

The closed integration is numerically and structurally valid through g12.  A
first reverse A/B/A was invalid because its timer included ping-pong eager /
capture and generation-reset lifecycle.  DEBUG=2 exposed that contradiction;
a corrected settled continuous protocol then measured small, repeatable wins
at g1, g4, and g12.  No g35 expansion, default, or promotion change is
authorized because cumulative llama-Q8 numerical error crosses the contract
at g18.

The compact machine-readable record is
`docs/task_workflow/output/nv-q4-cooperative-shared-q8-g1-gate-20260805.json`.

## Construction repaired before measurement

The isolated microgate used a bound scalar UOp to retain a real four-iteration
loop.  The first model integration tried two invalid spellings:

1. A resident scalar Tensor extent executed correctly but left a LOAD-derived
   UOp in `Estimates`; replay `sym_infer` failed with `NameError: UOp`.
2. Using the bound `start_pos` directly retained an `Ops.BIND` inside the
   kernel AST and failed spec verification.

The final construction strictly accepts either `DEFINE_VAR` or
`BIND(DEFINE_VAR, CONST)`, strips only the latter BIND, requires a nonnegative
lower bound, and uses:

```python
blocks_per_warp = start_pos_var // (start_pos_var + 1) + 4
```

For every admitted decode position this is exactly four.  It uses the graph's
existing variable binding, adds no scalar provider, and retains one runtime
loop.  The hermetic bound-input test proves the emitted AST contains one
`DEFINE_VAR(start_pos)` and no `BIND`.  Fourteen focused tests pass.

## g1 semantic and topology gate

The composed baseline was redirect=1 plus direct greedy plus feedback
ping-pong.  At d512 for eight exact autoregressive tokens:

- tokens, argmax, top-10 sets, and top-10 order are equal;
- relative L2 is `4.884647408926116e-4` (gate `<=1e-3`);
- max absolute error is `0.008861303329467773`;
- perturbation/top-1-margin is `0.01452855` (gate `<1`);
- two captures contain exactly two fused RMSNorm/Q8 providers, four
  cooperative Q4 consumers, zero legacy shared-Q4 consumers, and no duplicate
  provider.

The candidate has four more programs than control (`1756` vs `1752`): Q and K
each need one external four-part sum in each ping-pong capture.  The governing
topology gate is exact substitution/provider reuse, not raw program-count
reduction.

## Invalid first reverse wall gate

All token hashes were identical.  Ignoring each fresh process's compilation
outlier, control A is `5.410648 ms/token`, control C is `5.421305`, and their
midpoint is `5.415976`.  Candidate median is `10.856981`, a regression of
`+5.441005 ms/token` (`-50.12%` reported speedup).  Candidate samples
are also anomalously unstable (`7.350380`, `10.856981`) while controls remain
tight.  This initially looked like a hard wall failure, but it is not authority
evidence: the timer mixed eager/capture/reset lifecycle into each short rep.

## Device attribution

The first DEBUG=2 trace measured capture execution, not settled graph replay.
Its complete token kernel sums were `5.82826 ms` for g0 and `5.83443 ms` for
g1, only `+6.17 us`.  The affected g1 rows were:

- fused RMSNorm/Q8 provider: `2.62 us`;
- cooperative Q: `9.73 us`;
- cooperative K: `3.94 us`;
- Q sum: `1.54 us`;
- K sum: `1.50 us`;
- shared Q6 consumer: `14.94 us`.

A composed six-token trace then exercised both ping-pong captures and one true
replay of each slot.  The final five device batches sum to:

- g0: `5.23249`, `5.20901 ms` (median `5.22075`);
- g1: `5.21972`, `5.19243 ms` (median `5.206075`).

Thus g1's settled device graph is `14.675 us` faster.  The logs contain no
recapture/recompile at those settled iterations.  Compared with the wall gate,
the apparent excess was outside device graph execution and therefore required
a corrected wall protocol rather than a kernel change.

This evidence rejected three tempting but false conclusions: the cooperative
body is not 5.4 ms slow, graph replay is not 5.4 ms slow, and replay is not
recompiling at the settled iterations.  It does not yet distinguish
timing-child lifecycle behavior from another host-side synchronization or
enqueue boundary.  The corrected wall test below settles it as timing-protocol
contamination; changing the kernel construction was not supported.

## Corrected settled wall and progression

Each fresh arm executed prelude plus six untimed decode calls, putting both
ping-pong JITs at replay (`cnt>=2`).  It then measured five uninterrupted
32-token windows without reset or capture inside the timer.  A fixed high-side
contention rule was declared before measurement; all 15 samples were accepted.

At g1, exact 160-token stream hashes matched across A/B/A.  Control A was
`5.3491478`, candidate `5.3376988`, and control C `5.3460380 ms/token`.
Against the `5.3475929` control midpoint, g1 saves `9.894 us/token` (`+0.185%`).
This agrees in sign and scale with the `14.675 us` device replay result.

g2 and g4 then passed the established exact-token/argmax/top-10 contract.  g2
relative L2 was `4.3991e-4`, with exactly four providers, eight cooperative Q4
consumers, and zero legacy consumers.  g4 relative L2 was `5.6029e-4`, with
exactly eight providers, 18 cooperative Q4 consumers (one Q4 V adds the extra
pair across captures), and zero legacy consumers.

At g4 the same settled A/B/A measured control A `5.3524958`, candidate
`5.3410370`, and control C `5.3542248 ms/token`.  Against the `5.3533603`
midpoint, g4 saves `12.323 us/token` (`+0.231%`), with exact 160-token streams
and no rejected samples.

The g4 aggregate is real, but it is only `2.429 us/token` better than g1.
That rejects linear extrapolation from the isolated per-group microgate.

The coarse g4--g18 gap was then closed without changing the contract.  g8
passes at relative L2 `6.98738e-4`, with 16 exact providers and 40 cooperative
Q4 consumers.  g12 passes at `8.36963e-4`, with 24 exact providers and 60
cooperative consumers.  Both retain exact tokens/argmax/top-10 and zero legacy
shared-Q4 consumers.

One settled g12 A/B/A measured control A `5.3420231`, candidate `5.3226408`,
and control C `5.3526114 ms/token`.  Against the `5.3473173` midpoint, g12
saves `24.676 us/token` (`+0.464%`).  All 15 windows were accepted and all arms
produced the exact same 160-token stream.  Scaling therefore partially
recovers after g4, but g18's numerical stop remains authoritative.

## g18 semantic stop

The cooperative qualification tuple was explicitly expanded to g18/g35 for a
bounded full-depth decision.  g18 preserves exact tokens, argmax, top-10 sets,
and top-10 order.  Its topology is also exact: 36 providers for 18 blocks and
two captures, 92 cooperative Q4 consumers, zero legacy shared-Q4 consumers,
and no duplicate providers.

The cumulative numerical contract fails, however.  Relative L2 is
`0.0012714405`, above the established `0.001` limit; max absolute difference
is `0.0231905`.  The perturbation/top-1-margin ratio remains small (`0.0380`),
but that does not override the declared relative-L2 authority gate.  Therefore
g18 is `semantic_pass=false`; g35 semantics and wall are skipped.  g12 is the
last qualified depth, not a full-depth promotion candidate.

## Ledger-ready summary

| depth | semantic | relative L2 | settled wall delta | topology |
|---:|:---:|---:|---:|:---|
| g1 | PASS | `4.88465e-4` | `-9.894 us/token` | providers `2/2`, coop `4`, legacy `0` |
| g4 | PASS | `5.60287e-4` | `-12.323 us/token` | providers `8/8`, coop `18`, legacy `0` |
| g8 | PASS | `6.98738e-4` | not run | providers `16/16`, coop `40`, legacy `0` |
| g12 | PASS | `8.36963e-4` | `-24.676 us/token` | providers `24/24`, coop `60`, legacy `0` |
| g18 | **FAIL** | `1.27144e-3 > 1e-3` | prohibited | providers `36/36`, coop `92`, legacy `0` |

Ledger classification: real bounded decode win, last qualified depth g12,
g18 cumulative-numerics stop, no full-depth or parity credit, no default flip.

## Stop conditions

- g2/g4/g8/g12 pass; g18 fails semantics; do not run g35 from this evidence.
- Do not change defaults or route policy.
- The settled g12 incremental wall win is valid bounded causal-ledger credit;
  it is not full-depth parity qualification or a default-promotion result.
- Do not reject the cooperative mapping as a kernel-body failure.
- Reopen expansion only after localizing cumulative llama-Q8 numerical error
  between g12 and g18 by block.
