# Packed Q4 gate publication qualification (2026-09-07)

The exact weight-A gate/up source pattern publishes 32 scalar Q4 nibbles to LDS. The typed, fail-closed transform replaces those stores with eight aligned `uint32` stores while preserving the same lane addresses and nibble shifts. It is admitted only for the qualified swapped Q4 gate context.

## Gates

- Production-shaped R15: 347.714 us control versus 336.282 us candidate, an 11.432 us/call win (about 0.823 ms across 72 gate/up calls), with identical output and read-only inputs.
- SASS: IMMA 256 unchanged, PRMT 368 -> 176, LOP 3343 -> 3147, LDG 152 and LDS 448 unchanged, no local spill. Registers rise 207 -> 237 without a measured cliff.
- Deep model smoke: PASS, token 198, 252 generated mains, 234 Q8 producers, 126 active fixups, canonical packed weights, no copies/overlays, and exact logits/stages across three replay cycles.
- Stable warmup-9 A/B/C R9: control midpoint 45.3411465 ms, candidate 44.817677 ms, median win 0.5234695 ms. Minimum midpoint 45.2001665 ms, candidate minimum 44.406544 ms, win 0.7936225 ms. Control endpoint median drift is 0.002215 ms.

Ordinary qualified gate/up selection enables this transform. Set `NV_COMPILER_Q4_GATE_Q4_PACKED_PUBLICATION=0` to roll it back.
