# Generated O-only Stream-K qualification

The generated Stream-K Q4 body was isolated on the 36 attention-output
projections while Q stayed on the selected wide body and Q/Q4-V retained its
shared flat Q8 record.  The identical-buffer R15 lifecycle comparison includes
the required Q8 producer, projection main/fixup, and logical residual output.

`isolated_r15.json` is exact/allclose (`max_abs=1.33514e-5`,
`mean_abs=2.804e-7`) with read-only activation, canonical weight, and residual.
Wide measured 233.549 us and Stream-K 208.873 us, a 24.676 us/call advantage
or 0.888336 ms over 36 layers.  Producer count is unchanged; Stream-K adds 36
active fixups and replaces 36 wide O mains.

Both deep-smoke artifacts executed the intended graph and have token 198,
252 generated mains, 252 canonical weight arguments/bases, 234 necessary Q8
producers, 126 active fixups, zero copies/overlays, and exact logits plus every
tracked stage for three replay cycles.  Their formal FAIL is confined to the
diagnostic enumerating the same 36 O output/record allocations twice through
two captured call entries.  The harness now deduplicates O stage ownership by
exact output-buffer identity and requires 36 records/outputs.  No kernel or
graph path changed in that correction.

The route remains research-only pending the stable full-model timing gate.

The stable warmup-9 control/candidate/control R9 model bracket rejects the
route.  Control medians were 47.130337 and 47.527812 ms (midpoint
47.3290745 ms); O Stream-K measured 49.570100 ms, a 2.2410255 ms regression.
Control minima were 47.097054 and 46.927554 ms (midpoint 47.012304 ms) versus
candidate 49.107910 ms, a 2.095606 ms regression.  The isolated body/lifecycle
win therefore does not compose with the production graph.  Wide O remains
selected.

A composition audit found the original generated Stream-K capture retained AFTER
owners for output, partial workspace, and IDs.  This could schedule its opaque
multi-output main more than once.  Mirroring the native single-owner contract
keeps only output's AFTER owner and passes the raw partial/ID allocations to
fixup.  A one-projection TinyJit then contains exactly one producer, one main,
one fixup, and one residual service while preserving exact/allclose output.

The full deep graph after this repair has exactly 36 O mains and 36 fixups, 36
O record/output stage owners without identity deduplication, total mains 252,
Q8 producers 234, active fixups 126, canonical weights, no copies/overlays, and
exact three-cycle replay.  Its warmup-1 wall was 50.890940 ms.  The JSON formal
FAIL is only the then-stale wide-Q/O stage expectation; the harness now expects
wide Q 36 plus Stream-K O 36.

Stable warmup-9 control/candidate/control R9 measured control medians 47.399535
and 47.358540 ms (midpoint 47.3790375) against candidate 46.964699 ms: a
0.4143385 ms win.  Control minimum midpoint was 46.989235 ms against candidate
45.938449 ms: a 1.050786 ms win.  The median fails the frozen 0.5 ms promotion
threshold, so single-owner O Stream-K remains default-off despite removing the
composition regression.

Composing the qualified Q4-A x4 fragment, WMMA-update interleave, and logical
XOR4 stores with the single-owner O Stream-K route clears promotion.  Isolated
R15 measures 206.697 us versus wide 235.080 us (28.383 us/call); output is
allclose with max_abs 9.537e-6 and inputs are read-only.  Source/SASS census is
recorded in `x4_resource_census.json`; it has 32 x4 calls, 64 LDSM, 256 IMMA,
128 SHFL, 64 vector stores, and zero LDL/STL/local/stack spill.

Deep model smoke formally passes with exact three-cycle logits/stages, token198,
O main/fixup/owners 36 each, generated mains252, Q8 producers234, fixups126,
canonical weights, and no copies/overlays. Stable control/candidate/control R9
has control median midpoint47.170311 ms versus candidate46.179898 ms, a
0.990413 ms win. Minimum midpoint46.9749945 ms versus45.815785 ms is a
1.1592095 ms win. Both frozen0.5ms gates clear. The exact qualified compiler
pp512 route now selects this O body ordinarily; rollback is
`NV_COMPILER_Q4_O_STREAMK_X4=0`.
