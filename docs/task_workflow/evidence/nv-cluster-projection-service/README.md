# Exact cluster projection-service gates

This directory records the feasibility campaign for using CUDA thread-block
clusters to improve the short dense Q/O/K/V service episodes.

## Verdict

The hardware and compiler substrate works, but the exact current-Q4_K O
ownership does not pay. Production routing was not changed and no token
recovery is booked.

## Authority

- `gate-r9.json`: generic cluster/DSM handoff at cluster sizes 2, 4, and 8.
- `o-exactness-r9.json`: ClusterFusion-style FP16 atomic O join versus an
  exact deterministic FP32 scratch join.
- `o-cluster-r9.json`: installed 4096-by-4096 Q4_K residual-add O control
  versus exact cluster-2 and cluster-4 ownership.
- `o-cluster-artifacts/`: generated CUDA source and executable for the final
  Q4_K gate.

The final Q4_K gate uses legal finite fixtures, checks all output words on
three rotated weights, preserves the installed lane reduction order, and
reports zero spills. Its rotated-cold medians are 9.430 us for control,
11.228 us for cluster-2, and 11.642 us for cluster-4.

The result closes this topology only. A first-class persistent O emitter is a
different mechanism and remains open.
