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
| fragment load-to-use | 52.832851 | 52.809898 |
| current252 schedule control | 53.315076 | 53.292195 |

The candidate reduces median latency by 0.482225 ms, or 0.904% relative to the
control median. It is enabled by default inside the explicit compiler pp512
route. Set `NV_COMPILER_Q4_STREAMK_FRAGMENT_LOAD_TO_USE=0` to restore the prior
load order.

The JSON files retain all nine raw timing samples and route census; the NPZ
files retain the compared full logits and token. The `.log` files are local
command transcripts and are not part of the committed evidence.
