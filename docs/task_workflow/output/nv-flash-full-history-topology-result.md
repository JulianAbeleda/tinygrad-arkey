# Flash full-history and topology test

## Decision

The missing tinygrad Flash conversion state is now accounted for. It is not a
mysterious graph-launch tax.

The earlier replay retained only selected named producers and serialized their
execution. Restoring every kernel between consecutive score launches explains
part of the gap. Restoring the production Q versus K/V fork/join explains the
rest: the target rises to the same latency band as the retained production
layer. Reheating the target working set after that fork returns the score to
the hot latency band, so the mechanism is topology-conditioned cache residency.

## Native result

All rows use the installed dense Q4 endpoint at depth 512. Only the final score
kernel is timestamped.

| target state | score median | from hot | interpretation |
|---|---:|---:|---|
| target immediately repeated | **4.576 us** | reference | L2-hot body |
| exact one-score interval, serialized | **5.712 us** | +1.136 us | complete predecessor contents, wrong topology |
| exact two-score history, serialized | 5.568--5.600 us | +0.992--1.024 us | wider history does not deepen the penalty |
| exact one-score interval, production Q parallel K/V fork/join | **6.240 us** | **+1.664 us** | production topology restored |
| fork/join, then target-working-set reheat | **4.608 us** | +0.032 us | cold penalty removed |
| retained production layer 35 | **6.144 us median**, 6.169 us mean | +1.568 us versus hot | 68 graph observations |

The fork/join replay is within 0.096 us/layer of the production median and
0.071 us/layer of the production mean. That is tighter than the original
roughly 0.45-us/layer unexplained residual.

The exact serial result replicated independently at 5.696 and 5.712 us. Its
reheated result replicated at 4.608 us. The two-history arm remained no slower
than one-history, rejecting insufficient history depth as the remaining cause.

## What was omitted

The current production interval contains eleven launches between consecutive
scores. In addition to the old selected producer names, it includes the prior
Flash combine, O projection, attention norm, and the installed ordinary Q, K,
and V producer topology. More importantly, production does not serialize Q and
K/V: the branches execute independently and join before score.

Serializing the exact same interval measured 5.712 us. Reinstating that fork
and join measured 6.240 us, a +0.528-us/layer change. A target reheat then
recovered 1.632 of the 1.664-us/layer fork-conditioned penalty. The evidence
therefore assigns the conversion to working-set residency shaped by concurrent
producer service, not to additional old layers or intrinsic graph dispatch.

## Token translation

The installed endpoint remains 4.094502 ms/token, or 244.230 tok/s. Nothing is
booked by this diagnostic.

Mechanically removing the measured +0.528-us/layer topology-conditioned
increment would expose 19.008 us/token and a 245.369-tok/s ceiling, or about
+1.14 tok/s. That is not an admissible optimization claim: serializing Q and
K/V would sacrifice producer overlap, and the replay does not include that
added token-wall cost.

Removing the complete production-median cold penalty is a 56.448-us/token,
247.644-tok/s counterfactual ceiling. A literal reheat is not useful because it
executes the target a second time. This number bounds the exposure; it is not a
construction.

## Consequence for the next lever

The accounting wall is closed. More predecessor-history capture is no longer
the priority. The useful target is the cold Flash body itself: reduce the bytes
it must fetch, improve its service rate in the production residency state, or
change producer/cache topology without giving back Q/K/V overlap. Every such
candidate still needs a full-token bracket before promotion.

## Evidence

- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/full-history-probe.json`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/full-history-probe-confirm.json`
- `docs/task_workflow/evidence/nv-flash-kernel-to-production-conversion/full-history-fork-join.json`
- `docs/task_workflow/evidence/nv-flash-causal-reopen/post-wide-installed-ledger/production.profile.jsonl`
- `extra/llm_research/decode/nv_flash_full_history_probe.py`
