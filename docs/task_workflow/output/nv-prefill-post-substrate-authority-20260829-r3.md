# NVIDIA pp512 post-substrate S0 authority freeze, r3

Status: **PASS (new authority established)**

Both fresh candidate R9 brackets passed bitwise deep-20 replay, finite logits,
and greedy token `198`. Bracket 0: minimum `72.127022 ms`, median
`72.142030 ms`. Bracket 1: minimum `72.109648 ms`, median `72.150726 ms`.
They agree within `0.008696 ms`, establishing the new settled authority:
minimum `72.109648 ms`, median `72.146378 ms`. The prior `67.235719 ms`
figure is superseded and unreproducible because its source and clock
provenance are absent.

The fresh llama R9 reference is retained at
`docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829-r3/llama_r9.json`.
Both tinygrad arms used one GPU flock session, three warmups, R9, and
`PROFILE=0`. Recorded GPU session: RTX 5090, compute capability `12.0`,
driver `595.84`, current clocks graphics/SM/memory `435/435/7001 MHz`, P3.

Census per arm: 198 canonical weights, 198 mains, 198 Q8 producers, 72
gate/up, 36 K, 72 Q/O, 18 V, zero copies, zero partial workspace, 54 FP16
overlays, unknown zero. Safe-cut environment was pinned to two compute
queues, the normalized `combined-flash-direct-deps-cut-v2.json` policy path,
and `HCQ_NV_READY_PLACEMENT=0`.

Evidence: `docs/task_workflow/evidence/nv-prefill-post-substrate-authority-20260829-r3/`

Fresh llama settled median is `35.3648185 ms`; the new median gap is
`36.7815595 ms`. No non-derivable S0 field remains missing.
