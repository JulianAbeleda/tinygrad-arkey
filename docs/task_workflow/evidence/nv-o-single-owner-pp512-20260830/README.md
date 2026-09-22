# NV pp512 O single-owner promotion

Decision: **PROMOTE** under `NV_LLAMA_FULL_PACKED_PP512`, with explicit
rollback `NV_LLAMA_O_SINGLE_OWNER_PP512=0`.

The O native program declares output and workspace writes. Retaining separate
lazy `AFTER(main)` owners for both caused HCQ to materialize the same q4 main
twice at every O site. The promoted route retains one output owner and passes
the raw, program-owned workspace allocation to fixup.

| arm | median wall |
|---|---:|
| control A | 37.245973 ms |
| candidate | 34.963556 ms |
| control C | 37.035831 ms |

Mean control is 37.140902 ms. Recovery is 2.177346 ms and median throughput
increases 6.23% to 14,644 tok/s. Full logits are bit-exact (`max_abs=0`), all
arms select token 198, and rollback is bit-exact.

The device interval census proves the mechanism:

| physical quantity | control | candidate | delta |
|---|---:|---:|---:|
| all launches | 1649 | 1613 | -36 |
| q4 main | 252 | 216 | -36 |
| q6 main | 36 | 36 | 0 |
| O producer | 36 | 36 | 0 |
| Flash | 36 | 36 | 0 |
| summed active | 36.583040 ms | 34.745088 ms | -1.837952 ms |

Exactly one redundant q4 main per layer is removed. No other named primitive
count changes. `promotion-result.json` is the machine-readable authority.
