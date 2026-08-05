# NV decode P4 dependency-coherent cut reopen

Date: 2026-08-05
Status: **CPU candidate; GPU not run**

## Question and answer

The previous exact-name allowlist moved 51 scattered calls and regressed native
wall time by 24--55 us.  Its static accounting charged 761 raw cross-queue DAG
edges.  That is not the hardware command count: `HCQGraph._resolve_deps` caches
the largest signal value already observed from the other queue, so several raw
edges can collapse into one wait.

A strict one-entrance/one-exit attention cut does **not** exist in the captured
graph.  The query-side rope branch is consumed once by the shared merge and
again by its reduction tail before flash joins it.  Across 36 blocks, the
smallest coherent repeated branch therefore needs 105 effective wait events,
or 2.917 per block, rather than two per block.  This falsifies the literal
one-entrance/one-exit hypothesis while still finding a materially coarser cut
than the name-scattered policy.

## CPU ledger

Input: `/tmp/full_token_dag_e2_reconstructed.json`, 948 duration-bearing nodes
and 4,222 captured dependencies.  The analyzer mirrors program order on each
queue and HCQ's monotonically cached cross-queue signal values.  The wait cost
is the prior conservative calibration, 0.363 us per effective wait.

| occurrence-pinned cut | aux nodes | effective waits | raw span saving | costed span saving |
| --- | ---: | ---: | ---: | ---: |
| attention K rope chain | 108 | 105 | 175.008 us | 153.567 us |
| attention Q rope/reduction chain | 144 | 105 | 194.912 us | **169.694 us** |

This is a falsifiable scheduling forecast, not booked recovery.  The duration
model cannot represent SM/DRAM contention, signal encoding overhead beyond the
calibrated wait, or drift between the old 948-node capture and the current
graph.  It does establish a >=50-us CPU gate and materially fewer effective
boundaries than the earlier raw-edge accounting suggested.

## Closed-default substrate

`HCQ_NV_MULTI_QUEUE_INDICES` accepts exact call indices and ranges.  Empty is
the shipped default and changes no placement.  It complements, but does not
broaden, the old program-name experiment: only explicitly listed occurrences
move to queue 1.  The construction census now records `graph_idx` with program
name and assigned queue.  Malformed or descending ranges fail closed.

`extra/llm_research/decode/nv_dependency_closed_cut.py` recognizes both
attention branches by exact local generated-program structure, emits the exact
index selector, and simulates effective waits.  Hermetic tests cover parsing,
one-entrance/one-exit signal caching, and redundant-edge collapse.

## Required gate before any GPU timing

1. Regenerate a duration-bearing DAG from the current closed model graph; do
   not reuse the 948-index selector after topology changes.
2. Require the analyzer's structural name checks to pass and costed saving to
   remain >=50 us.
3. Run a construction census and require every queue-1 `(graph_idx, name)` to
   equal the analyzer output; otherwise stop.
4. Run full float32 logits, returned argmax, and generated-token correctness.
5. Only then run reverse A/B wall timing.  Promotion requires >=50 us/token
   versus both bracketing one-queue controls.  A wall regression closes this
   exact occurrence cut without weakening correctness or changing defaults.

No GPU evidence or token-time credit is claimed here.

## Control recapture (not selector authority)

A later live capture with the precompiled-output redirect explicitly disabled
produced a 946-node control DAG (4,360 edges, 124 cross-group edges; ordered-name
digest `4001027a599ece9d53bd3ade0e7a49aafb7a63c3d8282cd1dd0e9fcb888ca700`).
The same fail-closed analyzer retained the CPU gate: Q chain 185.440 us raw,
159.667 us costed with 107 effective waits; K chain 158.880 us raw, 144.592 us
costed.  This proves the result is not an artifact of the older 948-node capture,
but it is deliberately quarantined: redirect-off is no longer the candidate
topology.  Its indices must not be applied to the current redirect-on graph.

## Current graph qualification and final verdict

The redirect-on authority capture contained 875 nodes, 4,080 edges, and 147
cross-group edges (ordered-name digest
`49838b8ab2e7118d0c384fb93d2b4c3085b3732f1fe8d5abc69d51d232a6b413`).
It strengthened the Q-cut forecast to 184.992 us raw / 171.486 us costed with
73 effective waits, approximately 2.03 per attention block.

The first live selector attempt exposed and stopped on a construction mistake:
HCQ `graph_idx` is local to each split graph, while the initial selector used
full-token global indices.  Its exact logits were not sufficient evidence;
only 66 of 144 intended indices could be represented.  No wall timing was run.

The corrected policy is keyed by each graph's normalized ordered-program prefix
through the first post-cut consumer and verifies every selected `(local index,
structural program identity)`.  This tolerates only diagnostic-only suffixes;
generated E/r hashes are normalized while shapes and custom-program names remain
load-bearing.  The corrected census selected zero setup calls and exactly
`4 + 12 + 20 + 44 + 64 = 144` decode calls.  Full float32 logits were bitwise
identical to control (`71c0a2b092cbc2e40c22b42cd4f6f3c84fe56fd40f2bfd008efc5b76be0ae0f0`),
with exact returned argmax and token sequence.

The included-cost reverse bracket is a wall **NO-GO**:

| arm | settled samples, ms/token | median, ms/token |
| --- | --- | ---: |
| A1, redirect-on / one queue | 5.49779135, 5.47594915, 5.47844975 | 5.47844975 |
| B, redirect-on / exact Q cut | 5.49688340, 5.45380080, 5.48892370 | 5.48892370 |
| A2, redirect-on / one queue | 5.49308565, 5.47690510, 5.47972435 | 5.47972435 |

Candidate B is 10.474 us slower than A1 and 9.199 us slower than A2.  Disjoint
recovery is zero.  The selector and correctness mechanisms worked; the static
duration scheduler failed to represent execution contention and/or remaining
multi-queue command overhead.  This exact Q cut is closed.  K, combined cuts,
default enablement, and promotion were not tested.
