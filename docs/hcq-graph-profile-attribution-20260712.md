# HCQGraph ctx512 attribution

One ctx512 four-role run was captured with `PROFILE=1` and
`HCQ_GRAPH_PROFILE_JSON=/tmp/prefill-graph.json`. The graph emitted 801 dispatch
rows and 143,797.08 device timestamp ticks in total. This is a diagnostic run
(`K=1`, one warmup/round), not the pinned throughput authority; timestamps are
GPU-clock units and are only compared within this capture.

The exact generated-name join proves 180 dispatches: `attn_kv` (72),
`ffn_gate_up` (36), and the candidate dense name shared by `attn_qo`/`ffn_down`
(72). The remaining 621 dispatches are conservatively `unknown`, including the
`E_4_96_32...` dense name likely representing another large projection. The
route census proves candidate identity and call counts, but does not attach
semantic role metadata to generated graph names, so splitting that ambiguous
name would be speculation.

Use `extra/qk/graph_profile_attribution.py` to summarize subsequent exports.
No route or synchronization changes are involved.
