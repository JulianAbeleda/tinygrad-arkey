# Q4 weight-A x4 model integration closure (2026-09-06)

The first composed attempts faulted before an artifact because the swapped program ABI is `(out, weight, record)` while the conventional binding supplied `(out, record, weight)`. The binding now derives that ordering from the typed Q context and asserts canonical buffer capacities. With the corrected ABI, explicit transpose materialization passes token 198, exact replay2, canonical weights, and no copies, but costs 52.784 ms and adds 36 services.

Fusing `weight @ activation.T` followed by logical transpose into one generated program removes all 36 services and remains exact in the composed graph. The one-round composed readings were 52.888 and 53.609 ms; their formal FAIL is a now-repaired identity census omission (Q and conventional O have separate typed identities), not correctness.

The isolated conventional/fused-x4/conventional R9 medians are 305.745/310.926/305.555 us. Fused x4 regresses 5.276 us against the control midpoint. SASS explains the conversion loss: x4 improves LDS 80->56 and registers 222->218 with zero spill, but physical transpose writeback doubles STG 32->64. This route remains default-off under `--q-x4`. Promotion requires a typed coalesced C-coordinate store remap that retains 32 STG; rollback omits `--q-x4`.
