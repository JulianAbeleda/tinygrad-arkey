# Dual flat/tile Q8 producer closure

The bounded producer emits the established flat Q/Q4-V record and tile-major K
record from one fp16 activation read.  The isolated canonical fixture is byte
exact for both complete payloads and metadata, preserves its input, and checks
the exact capacities (655360 and 594432 uint32 words).  Fifteen alternating
host-heavy rounds measured 1751.379 us dual versus 1735.308 us for the two
reference producers (+16.071 us median), so the primitive itself has no timing
win.

The model integration was rejected and removed.  Its only bounded full smoke
kept token 198, 252 canonical generated weight arguments, 252 generated mains,
36 dual calls, 90 active fixups, and zero overlays/copies, but the expected Q8
count did not fall: compact producers dropped from 72 to 36 while the 36 dual
calls replaced them one-for-one, leaving 198 total.  This disproves the assumed
36-service removal in the current graph.  Replay diverged on cycle zero
(max_abs 0.127616; first K record/output mismatch at layer 8) and propagated to
later stages.  The 51.940 ms warmup-1 wall is non-authoritative.

Artifacts contain the isolated correctness/timing JSON and exact failed model
smoke.  No selection or model wiring remains.  The standalone provider is kept
as corpus for a future graph whose two layouts truly replace two visible
producer services.
