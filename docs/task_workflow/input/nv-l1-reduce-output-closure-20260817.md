# NV L1 reduce_output: closure after fresh test (2026-08-17)

Date: 2026-08-17. Branch `nvidia-bringup-20260731`, HEAD `5c30155dd`.
Evidence: `/tmp/census_nv_head_20260817.json` (fresh at-HEAD census),
`nv-reduce-output-stage3-p3-outcome-record-20260813.md` (P1/P2/P3 geometry
verdicts), `nv-reduce-output-stage3-geometry-scope-20260813.md` (corrected
premise).

Status: **L1 closes at its buildable ceiling. The ledger's +312.1 us row was
double-counted; the honest compare is llama 303.5 us norm family vs ours
441.6 (old) / 384.1 (fresh reduce_output only). P1 (q/k per-row grid) is
already promoted at HEAD and captured +55-67 us wall. The remaining q/k gap
to llama's 1.30 us/launch is blocked by the bitwise-equality constraint
(tree block_reduce would flip the logits sha), and the 4096 side is at
parity or ahead. No bitwise-safe implementation work remains.**

## 1. Fresh census at HEAD (this session)

| body | count | med us | total us |
| --- | ---: | ---: | ---: |
| `reduce_output_rmsnorm_32_128` | 36 | 3.07 | 116.4 |
| `reduce_output_rmsnorm_8_128` | 36 | 3.16 | 115.4 |
| `reduce_output_rmsnorm_1_4096` | 19 | 7.94 | 152.3 |
| total | 91 | - | 384.1 |

The q/k medians (3.07 / 3.16) are the P1-promoted per-row grid geometry
(P3: 3.245 / 3.07 at promotion; within cross-session noise). P1 is in the
tree at HEAD (emitter docstring: multi-row = one 32-lane block per row,
grid = rows, serial chain association preserved).

## 2. The corrected premise (from the 08-13 scope, still valid)

Llama does NOT pay 0 for reduce_output. Its norm family is standalone
`rms_norm_f32` kernels: q/k grid-per-row at 1.30 us/launch, 4096 at 2.88
us/launch, 303.5 us total. The honest gap was llama 303.5 vs tinygrad
441.6 = +138.1 us (norm family), of which the q/k share was +147.2 us and
the 4096 side already over-earned (-9.1 us).

P1 (per-row grid) landed the bitwise-safe part of the q/k gap: body medians
3.70 -> 3.245 (32_128), 3.17 -> 3.07 (8_128), bracket +55.31 / +66.55 us,
promoted at `dc58ae57f`. P2 (lean 1_4096) NO-GO and reverted; phase6
single-fused-program NO-GO (18.5 us slower); M1 norm-into-GEMV fold NO-GO
(+81.92 us).

## 3. Why the remaining q/k gap is closed (not blocked by effort)

The remaining gap is ours ~3.1 us/launch vs llama 1.30 us/launch (~130 us
at 1:1). Llama reaches 1.30 with 256-thread blocks and a tree
`block_reduce`; our logits are pinned bitwise to the ordinary serial-chain
association (per-row 8-lane x 16-serial / 16-lane x 8-serial), and a tree
reduce changes the fp32 summation order - flipping the full-logit SHA
(`9e6664fd...`). The 08-13 scope and P3 outcome both record tree reduce as
out of scope for that reason. No bitwise-preserving geometry between the
current per-row block and llama's tree shape was found (P2 was the attempt;
it regressed 7.97 -> 45.63 us/launch).

## 4. Ledger correction

| ledger row | old value | corrected | basis |
| --- | ---: | ---: | --- |
| L1 reduce_output node | +312.1 us | +384.1 us fresh, of which P1 is landed | fresh census |
| L1 wall ceiling | +312.1 (223.4 tok/s) | buildable part already captured (+55-67 us, promoted); remaining gap bitwise-blocked | P3 outcome |
| L1+L2 combined claim | +17.6 tok/s at 1:1 | L2 NO-GO (see L2 verdict); L1's open remainder ~0 | L2 A/B + this record |

## 5. Conclusion

1. L1's buildable geometry was landed before this session (P1, promoted).
2. The ledger's L1 row should be re-read as "P1 landed; remainder
   bitwise-blocked at ~130 us (tree reduce)" - not an open +312.1 us.
3. Together with the L2 NO-GO verdict, the kernel-work rows L1-L4 resolve
   to: L1 mostly landed, L2 wall-neutral, L3 at parity, L4 ~13 us. The path
   to 240 remains overlap + host gap + PDL (substrate docs), not these rows.
