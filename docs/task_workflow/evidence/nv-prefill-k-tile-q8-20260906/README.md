# Generated K tile-major Q8 integration

The generated K binding now consumes the same retained 144-byte tile-major Q8 record contract promoted for gate/up. Its tensor expression remains compiler-owned; the record carrier and FP16 DS4 producer replace the disconnected flat-record conversion.

Two complete current252 runs measure 47.642309 and 47.870729 ms medians, compared with 48.619011 and 48.577254 ms for the immediately preceding gate tile-major route. Both runs are finite, choose token 198, pass five exact recurrent replay cycles, and retain 252 generated mains, canonical packed weights, zero copies and zero overlays.

Tile-major K is selected by default within the explicit compiler K lease. `NV_COMPILER_Q4_K_TILE_Q8=0` restores the flat record.
