# NVIDIA pp512 clean-room Flash result

The clean-room Flash candidate produced fp16 HCQ-exact output, but the
one-query CTA architecture is rejected at the production seam.

| result | value |
|---|---:|
| HCQ numerical result | exact fp16 |
| measured device time | 1,337.344 us/layer |
| whole-model wall | 84.924811 ms |
| production seam | removed |

The candidate does not establish a recoverable whole-model win. The one-query
CTA design is therefore rejected and no model integration is retained.

After this rejection, the remaining measured support/O levers have no
independent evidence-backed candidate at or above the 0.5 ms whole-model
threshold. The Flash category remains a measurement observation, not an
admitted implementation target; any future work requires a new ABI and a
matched full-model correctness/performance gate.

Status: **REJECTED_CLEAN_ROOM_FLASH**
