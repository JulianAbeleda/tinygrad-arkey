# MMQ host structure audit

Artifact: `mmq-host-structure-audit-20260711.json`

This pure-host 2x2 generated noncandidate audit holds UOp count, opcode histogram, edge count, shared-edge count, and Python UOp builder events fixed while varying dependency depth 32/64 and unique GROUP fanout 256/512.

## Exact admission

Every cell has:

- 1,246 UOps: `ADD1243 / CONST1 / GROUP1 / SINK1`
- 3,120 source edges, inside the required 3,050-3,200 range
- 1,875 shared extra edges, inside the required 1,800-1,950 range
- Builder events `raw_add1243 / group1 / sink1`
- GROUP total source edges fixed at 633

Depth and GROUP unique-source fanout match each requested cell exactly. Full structural identities and representation hashes are recorded.

## Failure audit

The first timing implementation selected unique GROUP sources with a fanout-dependent linear membership loop. It produced an artificial 3.49 ms versus 10.74 ms fanout difference even though admitted UOp events were fixed. That run was rejected because the Python bookkeeping was outside the declared builder-event contract.

The admitted implementation scans the same 1,243-node candidate list in every cell using constant-time set membership, then slices to the requested unique fanout. Exact graph metrics were unchanged. The artifact contains only the corrected run.

## Result

| Depth | Fanout | Construction median |
| ---: | ---: | ---: |
| 32 | 256 | 1.1392 ms |
| 32 | 512 | 1.1351 ms |
| 64 | 256 | 1.1323 ms |
| 64 | 512 | 1.1393 ms |

The saturated contrast coefficients are `depth -562.3 ns`, `fanout -59.3 ns`, and `depth*fanout +1.353 ns`; with zero residual degrees of freedom they are descriptive only. Depth changes sign across fanout levels, fanout changes sign across depth levels, and the full cell spread is below 0.7%. This is a bounded null result: after fixing node, opcode, edge, sharing, and builder-event volume, these depth/fanout changes do not show a stable Python construction-cost relationship.

No schedule, device, candidate binary, or candidate timing was collected.
