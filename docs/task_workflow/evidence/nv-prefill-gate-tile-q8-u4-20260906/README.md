# Tile-major gate/up outer-K unroll 4

The tile-major Q8 conversion materially changed the generated gate/up body, so the previously layout-bound outer-K unroll discriminator was rechecked once on the new body. Unroll 4 preserves finite outputs, token 198, five exact replay cycles, and the complete generated/canonical census.

It regresses current252 to 48.908469 ms median and 48.652056 ms minimum versus the promoted unroll-8 tile-major baseline at 47.642309 / 47.870729 ms medians. Unroll 4 is rejected; the qualified unroll 8 remains selected.
