# Gate Stream-K warp-topology discriminator

The retained tile-major generated gate/up body uses a `2 x 4` warp grid.  The
existing compiler schedule plumbing was exposed through
`NV_COMPILER_Q4_GATE_WARP` so alternate topologies can be tested without
changing the graph, main ABI, canonical packed weights, or Q8 handoff.

## Result

`4 x 2` is incompatible with the retained body.  The exact current252 smoke
faulted on every SM during capture (`warp_pc=0x2265a024c0`) and ended with an
HCQ timeout waiting for timeline 2957 at 2954.  It produced no timing result,
so no performance claim is booked.

The default `2 x 4` route subsequently passed the same smoke contract:

- token: 198
- compiler mains / canonical weight arguments: 252 / 252
- admitted overlays: 0
- replay cycles: 2, exact logits and token
- median: 53.353781 ms (warmup-1, three-round recovery smoke; not booked)

This closes the warp-grid transpose as a gate convergence branch.  Keep `2 x
4` as the production topology and target a structural difference within that
known-good geometry.

## Commands

Candidate: set `NV_COMPILER_Q4_GATE_WARP=4,2` on the ordinary current252 arm.
Control: omit the variable (equivalent to `NV_COMPILER_Q4_GATE_WARP=2,4`).

Both commands used `--gate-streamk --gate-q8-reuse --warmups 1 --rounds 3
--replay-cycles 2` with the current Q4/Q6 generated role set.
