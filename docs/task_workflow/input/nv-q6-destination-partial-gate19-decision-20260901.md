# NV Q6 destination-major partial Gate19 decision (2026-09-01)

## Decision

`PROMOTE_DESTINATION_MAJOR_PARTIAL_LAYOUT`

The Q6 Stream-K main previously wrote each 128x128 partial in tile-row-major order while the final model output is destination-major. The fixup therefore had to choose between coalesced partial reads and coalesced output stores. Gate19 changes only the partial workspace permutation to `slot*16384 + col*128 + row` and uses a 512-block, 128-thread fixup whose lanes are contiguous for every partial read and destination store.

## Correctness

- Active candidate partials transpose back to the admitted partials bit-for-bit.
- Unused candidate slots retain their NaN sentinels.
- Candidate and admitted final outputs are bit-exact and finite.
- Ownership, segment descriptors, contributor order, and three-step `__fadd_rn` fold are unchanged.
- No atomics, memory barriers, counters, resets, spins, or inter-CTA dependencies are introduced.

## Locked route-level R31

| Component | Admitted | Destination-major | Paired candidate-admitted |
|---|---:|---:|---:|
| Main | 231.072 us | 228.160 us | -2.816 us, 28/31 wins |
| Fixup | 25.376 us | 7.648 us | -18.112 us, 31/31 wins |
| Total | 256.320 us | 235.808 us | -20.640 us, 31/31 wins |

The total improvement clears the `3 us`, `24/31` promotion bar by a wide margin. Against the pinned llama total of `209.856 us`, the isolated Q6 route gap falls from approximately `46.4 us` to approximately `25.95 us`.

## Binary identities

- Frozen anchor main: `6eb663b3a3fd628e3394a0ce8f8780e108e47f40b887b0a75a0756dcf33e9137`
- Destination-major main: `fc93c201002b45e1b2d4ae45db8fc43fa32f277af788474bfd92377cd0733685`
- Destination-major fixup: `f22d72755e1b3836f2f6ca0b41da13ead66889f4643dfa8b99b4f93b1ae1a081`

Evidence: `docs/task_workflow/evidence/nv-q6-destination-partial-full-gate19-20260901/result.json`
