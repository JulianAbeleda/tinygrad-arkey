# Flash S6 cooperative vs installed body — closure

The standalone cooperative candidate is **closed and not integrated**. It preserves
the oracle-qualified ABI, but its full 32×512×6 service row is not competitive with
the installed Flash population.

| body | measured service |
|---|---:|
| cooperative S6 score | 4,422.016 µs |
| cooperative S6 combine | 9.888 µs |
| cooperative total | 4,431.904 µs |
| installed Flash score population (existing live captured rows) | 6.720–6.752 µs per captured Flash score call |
| installed Flash combine population | 1.344–1.408 µs |

The installed rows are from the live graph-bound Flash population and its exact
compiled body (`flash_vec_llama_score_pv_32_128_8_widekv16`); the cooperative row is
the full 32×512×6 standalone fixture. Because the installed graph capture does not
export a directly re-launchable buffer-bound full-T512 Flash body, an apples-to-apples
full-T512 relaunch cannot be performed without violating the graph ownership boundary.
The available same-population evidence is nevertheless decisive: the candidate is
orders of magnitude slower at the service granularity and cannot be promoted.

The cooperative candidate passed exact oracle/read-only checks at 32×1×6, 32×8×6,
and 32×512×6, but is retained only as research evidence. No model integration.
