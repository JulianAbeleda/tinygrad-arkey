# NV buildable lever rows: exhaustive scope for L1-L4 (2026-08-17)

Date: 2026-08-17. Branch `nvidia-bringup-20260731`, HEAD `b1ba6c556`.
Numbers are from `nv-llama-full-trace-lever-ledger-20260817.md` (same session,
d512, Qwen3-8B-Q4_K_M, RTX 5090): current wall 4788.3 us / 208.84 tok/s vs
llama 4058.9 us / 246.37 tok/s; node_sum -496.3 us AHEAD; overlap 0 vs 1125.1.

Status: **scope record for the four buildable kernel-work rows.** Rows L1-L4
are the only places we are BEHIND llama on kernel mass; L5-L7 (overlap, host
gap, PDL) are construction/wall rows and are covered by
`nv-substrate-exhaustive-scope-20260817.md` and
`nv-pdl-substrate-verdict-20260817.md`. Each row below follows the standing
pipeline: audit llama -> arithmetic (mass + tok/s ceiling) -> implement ->
gate. Tok/s ceilings are exact at 1:1 wall recovery, sublinear after that.

**Test status at HEAD (honest): this session tested ONLY the PDL substrate
(`nv-pdl-substrate-verdict-20260817.md`). The four rows below are SCOPED, not
re-gated at HEAD.** Their numbers come from the 08-17 same-session ledger and
the cited prior records. L1's audit census harness (`nv_reduce_output_rmsnorm_census.py`)
was attempted at HEAD and FAILS to run: flash admission is capability-gated
now, so the CPU-only capture no longer binds the live-split route ("is not
served by the generated live-split route"). The L1 audit tool must be updated
to the capability-gated route before its gate can run.

## Summary table (from the ledger)

| row | node mass | wall ceiling | tok/s ceiling | tested at HEAD? | substrate / prior evidence | status |
| --- | ---: | ---: | ---: | --- | --- | --- |
| L1 reduce_output | 312.1 us | +312.1 | 223.4 | NO (census harness stale) | geometry split 08-13: P1 promoted (+55-67 us), P2 NO-GO reverted | OPEN |
| L2 vocab_aux | 59.5 us | +59.5 | 211.5 | NO | 08-03/08-12/08-14 fusion records; F5 keys.clone landed | OPEN |
| L3 flash_score parity | 39.4 us | +39.4 | 210.6 | PARTIAL (floor re-pinned 08-16, closed NO-GO) | body 4.19 vs 3.16 us; all shape attempts NO-GO | OPEN |
| L4 other (residual launches) | 47.2 us | +47.2 | 210.9 | NO | none - pure census needed first | OPEN |

L1+L2 at 1:1 = +17.6 tok/s exact (per ledger 4.1). L1-L4 all at 1:1 = 230.9
tok/s; 240 still requires L5+L6+L7 (substrate docs above).

## L1 - reduce_output elimination/absorption (+312.1 us)

What: 36 q-norm + 36 k-norm reduce_output launches (ours 312.1 us, llama 0 -
its mmvq absorbs the output reduce in-kernel).

Audit (llama): llama pays 0 for reduce_output; `rms_norm_f32` writes logits
and the GEMV reduce happens inside the next kernel. Our emitter
(`tinygrad/codegen/late/reduce_output.py`) emits one block per row today.

Prior work (committed): stage-3 geometry scope/outcome
(`nv-reduce-output-stage3-geometry-scope-20260813.md`,
`nv-reduce-output-stage3-p3-outcome-record-20260813.md`). P1 per-row grid for
the multi-row q/k bodies BOOKED (median 3.245 / 3.07 us, +55-67 us bracket,
promoted at `dc58ae57f`). P2 lean single-row 1_4096 launch NO-GO (45.63
us/launch, reverted). Remaining gap: ours ~3.2 us/launch vs llama 1.30 for
q/k; the 4096 side is at parity and closed unless a bitwise-preserving wider
geometry is found.

Arithmetic: 36 x (3.245 - 1.30) ~= 70 us is the bounded remainder; the full
312.1 us row requires absorbing the reduce into the GEMV epilogue (the llama
shape), which is a fusion/elimination change, not a primitive.

Substrate: present. Emitters, census, and bitwise-exact harness all exist
(594-program smoke, exact full-logit sha, 91-body census).

Implement order:
1. Re-open the P1 geometry against HEAD census; confirm the 3.245/3.07 pins.
2. Absorption prototype: fold the q/k output reduce into the rmsnorm epilogue
   as a vectorized tail (llama's grid-per-row shape), keep serial association
   bitwise-exact.
3. If bitwise-exact absorption is impossible, land the bounded P1 remainder
   (~70 us) and re-measure.

Gates: smoke survive; exact full-logit sha identical; census 19/36/36 with
zero weight materializations; reverse wall bracket; +50 us bar vs both
controls; tok/s translation recorded.

## L2 - vocab_aux elimination (+59.5 us)

What: the 4-kernel aux scatter chain after the vocab GEMV
(`E_1187_32_4` + `r_32_4_1187` + `r_128_16_8_1187` + `r_16_8` = 57.3 us
measured 08-12; ledger now 59.5 us). Llama: one mmvq node, top-1 on host.

Audit: `nv-vocab-aux-chain-fusion-scope-20260812.md` priced the aux chain at
57.3 us and authorized fusing into the landed `q6k_gen_coop_151936_4096_inkernel`
epilogue (or a packed-argmax reduce). `F5 keys.clone` landed (ledger notes),
row still open.

Substrate: present - `in_kernel` coop route landed (L4 substrate fusion,
~-85 us booked 08-03), epilogue fusion pattern proven.

Implement order:
1. Fresh census of the 4 aux nodes at HEAD (drift check vs 08-12 pins).
2. Packed-argmax reduce prototype in the vocab epilogue (top-1 bit-exact to
   `9e6664fd...`).
3. Wall A/B; promote only past the +50 us bar.

Gates: bit-exact top-1 token; aux node count 4 -> 0 in census; reverse wall
bracket; +50 us bar; tok/s recorded.

## L3 - flash_score parity (+39.4 us)

What: score 213.1 us (ours) vs 173.6 us (llama). Body is at structural
parity; the gap is shape/vectorization (our tile
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`, S=48; llama
`flash_attn_ext_vec` 8-lane reduce, 128 columns parallel, 2 KV splits).

Audit: `nv-flash-score-floor-test-head-20260816.md` re-pinned the body at HEAD
(no drift; 4.19 us isolated vs llama 3.16, ~+37 us/36 nodes; in-situ ~+68 us);
`nv-flash-score-llama-trace-20260813.md` traced llama's shape. Every prior
attempt at the shape change is NO-GO (tile geometry sweep 08-03, single-stage
combine 08-05, llama-vec single-pass as-is 08-13, multi-stream overlap 08-15).
The 08-16 verdict: no ready kernel to capture the mass; any future attempt
must first show a device-side body at production config faster than the 4.19
us tile.
The flash-attention kernel search generalization task (BubbleBeam vocab axes,
FutureSight legality, Coder emit) is scoped separately; the cheap first step
is a compile-only capability probe: can our lowering emit llama's shape
(8-lane / 3-stage / 128-parallel / 2-split)? Case A = expose levers and
search; Case B = build the missing reduce/output primitive first.

Substrate: present for the current tile; llama's exact shape is unproven
(probe decides Case A vs Case B).

Implement order:
1. Compile-only capability probe (no GPU) for llama's flash shape.
2. Case A: expose shape levers to the search, run, measure.
3. Case B: build the missing primitive (reduce-output-style), then 2.

Gates: exact logits sha; census score med; wall bracket; +39.4 us row target
in chunks.

## L4 - other residual launches (+47.2 us)

What: 49.1 us of small elementwise/get_rows/bcast nodes vs llama 1.9.

Audit: llama's residual is one `k_bin_bcast` class node (1.9 us). Ours are
scattered small launches (elementwise, get_rows).

Substrate: present (fusion/elimination only).

Implement order:
1. Census the residual nodes at HEAD (name, count, us each).
2. Classify: absorbable into an existing epilogue (fusion) vs must-stay.
3. Absorb the absorbable set; re-measure.

Gates: census row -> 0 for absorbed nodes; bitwise token; wall bracket;
tok/s recorded.

## Working rule (standing)

Every row ships only after its gate passes on the native NV production route
at HEAD, with the exact-token and reverse-wall-bracket harness. No row is
promoted on microbench numbers alone. After L1-L4 land, re-run the exact wall
account to refresh the L5-L7 residual before touching the overlap rows.
