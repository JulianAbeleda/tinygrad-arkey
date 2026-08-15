# NV substrate definition - what "the substrate" means, and the 220 substrate specifically

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `b9dd814c8`)
Status: **definition record.** No code, no measurement, no GPU session. Pins the
word "substrate" to one meaning so the next scope builds against that meaning
instead of the loose usage accumulated across the ledger.

## 1. One-line definition

**Substrate = a capability in the compile/lower/emit/runtime stack that makes a
target construction expressible as one valid, replayable program.** It answers
"can we emit this at all?" and is the gate that must clear before any wall can
be measured.

The repo already draws this boundary in
`nv-substrate-capability-vs-ledger-scope-20260807.md`, and this definition just
names it:

- **capability-blocked**: the construction is not expressible; no schedule
  search reaches it. This is a substrate problem.
- **wall-blocked**: the construction renders but loses on wall. This is a
  values problem, not a substrate problem.

Operational test: a fold that cannot render (`CONSTRUCTION_GAP`) is missing
substrate. A fold that renders but is slower (`NO_GO_WALL`) is a values result.
The two must never be conflated, because they route to different work.

## 2. The three layers

Every "substrate" claim in the ledger is one of three layers:

| layer | what it is | status |
| --- | --- | --- |
| primitive | generic lowerer/emitter primitives: cooperative reduce-to-output, typed boundary ABI, epilogue authoring into opaque kernels, rangeify SPECIAL reads, late-store selector (C1-C8) | mostly built + promoted |
| topology | the composed shape of one decode token: a long fused-quant GEMV anchor with support folded into its epilogue and a shadow (`overlap_mass > 0`), instead of 253 serial ~16us GEMVs | **missing** |
| hardware | compute queues can physically co-schedule | built + proven; current DAG has nothing to hide |

Most "do we have the substrate" confusion is between layer 1 (yes) and layer 2
(no). The primitive layer is the bricks; the topology layer is the wall.

## 3. The 220 substrate, precisely

The `219.8 tok/s` figure in `nv-decode-gap-attribution-same-session-20260812.md`
section 7 step 7 is the arithmetic ceiling of composing the already-built
primitives. Its substrate is **layer 2**, and it has three requirements that
must hold simultaneously in one replayable token:

1. in-kernel quantized matvec (dequant/quant stays inside the GEMV) - already
   present and promoted;
2. body-free epilogue absorption: reduce/norm/residual/cast/vocab-aux run inside
   the consuming GEMV with no added body work;
3. anchor shadow: one long GEMV provides a shadow so support-kernel durations
   stop landing on the critical path (`overlap_mass` goes from `0.0` to `> 0`).

The three must compose **at once**, because 219.8 is "all class deltas closed at
1:1," not one fold at a time. Piecewise folds that each add body work do not sum
to this number.

## 4. Reconciling "the anchor does not transfer"

Two records look contradictory until the word "anchor" is disambiguated:

- `nv-anchor-verdict-20260814.md` says llama's anchor does **not** transfer.
  That is true of llama's *mechanism*: split the `quantize_q8_1` pass out of
  `mul_mat_vec_q` and pipeline it. We already fused that quantize away, so the
  mechanism has no corresponding node on our side.
- `nv-overlap-layer1-substrate-test-20260814.md` says the missing substrate is
  llama's *anchor+fusion topology*.

The reconciliation is structural, not mechanical. We do not want llama's
quantize pass; we want its **outcome** - one long fused-quant GEMV anchor per
token with support work hidden behind it. Our GEMVs are 253 chained ~16us
kernels with `overlap_mass = 0.0`, so there is no shadow to hide support in.
The substrate to build is the topology that produces that outcome, not a copy
of llama's kernel decomposition.

## 5. What the 220 substrate is NOT

- not per-shape GEMV tuning (Q4 FFN-down, Q6 attention-V) - that is values; the
  sweeps closed those rows NO-GO and they stay closed;
- not the multi-stream knob - the CUDA multi-stream lowerer exists and buys ~0
  on the current chain-like DAG;
- not llama's `946 us` overlap mass transferred 1:1 - most of it is
  quantize/norm/rope work we already fused, so the transferable overlap mass is
  much smaller.

## 6. Acceptance test

We "have the 220 substrate" when one decode token clears all three:

1. CPU construction gate PASS: the folded graph renders as one replayable
   ordinary program per fused site, no custom-program boundary, no CONTIGUOUS,
   token-exact;
2. program count `1021 -> ~762` (llama's count) with `overlap_mass > 0`;
3. wall reaches `4.545 ms` (220 tok/s) with the composition body-free.

## 7. Current position against that test

| requirement | present | evidence |
| --- | --- | --- |
| in-kernel quantized matvec | yes | `8440c51b5`, `d457a3bf0`, `87fa01a97` |
| reduce-output primitive + absorption machinery | yes, piecewise | `0358a07fe`, `a8b560457`, `882ce66a5` |
| body-free absorption across all open classes | no - several folds add body and lose wall | `77f8f4ebd`, `6843e83db`, `b9dd814c8` |
| one composed full-token anchor+fusion graph | no - decode DAG is a chain, `overlap_mass = 0.0` | `nv-tinygrad-node-ledger-gap-record-20260813.md` |

Net: primitive substrate present and promoted; the composed topology that turns
it into 220 is the missing piece. The next work is to define that composition,
not to build another primitive.
