# Current252 gate fragment load-to-use qualification (2026-09-06)

The candidate changes only the generated U8 gate/up Stream-K main schedule. It
moves the unchanged `val0..val10` Q4 fragment load group after the Q8 loads,
shortening fragment register lifetime while retaining the 20 KiB shared tile,
two CTA barriers, 170-owner geometry, arithmetic, and fixup.

Both arms select 252 generated projection mains/producers with 252 canonical
weight bases and zero V/down overlays. The candidate passed 20/20 bit-exact
replay cycles and its full logits and token are bit identical to the matched
control.

| arm | median ms | min ms |
|---|---:|---:|
| prior current252 control | 52.979670 | 52.961790 |
| fragment load-to-use | 52.832851 | 52.809898 |
| closing current252 control | 53.315076 | 53.292195 |

The candidate reduces median latency by 0.314522 ms, or 0.592% relative to the
mean of the two control medians. Even its 0.482225 ms reduction against the
closing control misses the frozen 0.5 ms population investment threshold from
`docs/task_workflow/output/nv-prefill-substrate-test-ledger-20260829.md`. It is
retained default-off; set `NV_COMPILER_Q4_STREAMK_FRAGMENT_LOAD_TO_USE=1` to
reproduce it.

The JSON files retain all nine raw timing samples and route census; the NPZ
files retain the compared full logits and token. The `.log` files are local
command transcripts and are not part of the committed evidence.
