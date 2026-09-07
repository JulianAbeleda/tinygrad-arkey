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
