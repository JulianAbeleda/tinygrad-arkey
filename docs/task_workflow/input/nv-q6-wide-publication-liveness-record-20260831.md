# NV Q6 wide publication liveness record

## Verdict

**NO-GO.** Dead scale-padding elimination is exact and reduces registers, but
does not reduce shared-store issue count or full-route latency.

## Address proof

For every `(threadIdx.x, threadIdx.y, threadIdx.z)` in the promoted
`(32,2,4)` CTA, the 80 source stores are a bijection over the 20,480-byte
shared allocation. Exhaustive evaluation of the 288 downstream `buf1` load
expressions finds 17,920 unique consumer-visible bytes and 2,560 bytes of dead
scale padding. All four 16-byte payload planes are fully live.

The only legal reduction masks odd `alu3` producer lanes in the four scale
planes and narrows the first two scale stores from four bytes to two. It leaves
the payload stores and both barriers unchanged.

## Full-route gate

The candidate uses the same canonical Q6/Q8 inputs, 170 Stream-K owners,
unroll two, partial workspace, and fixup as the control.

| Measurement | Control | Live-publication candidate |
|---|---:|---:|
| main median, R9 | 334.912 us | 335.072 us |
| fixup median, R9 | 11.648 us | 11.648 us |
| main registers | 250 | 232 |
| stack/local | 0/0 B | 0/0 B |
| static STS | 24 | 24 |
| static BAR | 6 | 6 |
| max absolute output difference vs direct | 0.0006866455 | 0.0006866455 |

The R21 candidate median is 335.616 us. The candidate therefore fails the
performance gate despite exact full-route consumer output and lower register
allocation. NVRTC already lowers each body to four scale stores plus four
`STS.128` payload stores; eliminating unused scale bytes changes store width
and lane predicates, not the number of issued store instructions.

Evidence is under
`docs/task_workflow/evidence/nv-q6-wide-live-publication-20260831/`.
