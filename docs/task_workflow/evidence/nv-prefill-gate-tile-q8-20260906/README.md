# Generated gate/up tile-major Q8 integration

The retained `TileMajorQ8ActivationRecordTransform` was disconnected from the selected gate/up path. Production generated a flat compact Q8 record, while the faster reference topology consumes 144-byte tile-major DS4 records. This change pairs the existing generated tile-major fragment provider with the existing FP16 DS4 producer and the same generated 170-owner Stream-K main/fixup.

The Stream-K converter also assumed the flat renderer's terminal index name `alu242`; tile-major lowering names the identical output expression `alu246`. It now derives the terminal output index from the compiler source, preserving both renderer spellings without changing arithmetic.

A same-process isolated A/B produces bit-identical 6,291,456-element FP32 outputs. Median synchronized main+fixup time improves from 521.882 to 509.088 us in that host-heavy diagnostic. More decisively, two complete current252 runs measure 48.619011 and 48.577254 ms medians, versus the immediately preceding pair-reuse candidate/control/control medians of 49.368896 / 49.598182 / 49.690918 ms. Both tile-major runs are finite, select token 198, pass five exact replay cycles, retain 252 generated mains and canonical packed weights, and use no weight copies or FP16 overlays.

Tile-major Q8 is now the ordinary generated gate/up Stream-K layout. `NV_COMPILER_Q4_GATE_TILE_Q8=0` is the explicit flat-record rollback.
