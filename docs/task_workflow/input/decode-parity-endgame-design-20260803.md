# Decode parity endgame - view-preserving boundary first, then L1

Status: design, not implemented. The forward-path review
(`nv-campaign-forward-review-20260803.md`) identified one open decision - whether the
copy-free opaque boundary (the M3 reopen condition) must land before L1 epilogue
absorption. This design answers that question with the measured evidence and specifies
the build order to decode parity and beyond. Branch boundary:
tinygrad `nvidia-bringup-20260731` @ `1d668e3bb`. Does not authorize promotion to
`dev`/`exp`/`master`.

Bans for this scope: no `prefill_routes.py`, no dtype/precision cleanup, no
`if backend == "NV"` branches in lowering (data lookups only), no commits to
`master`/`dev`/`exp`, and never commit the untracked scratchpads
(`scratchpad/t6_metal_admission_probe.py`, `extra/llm_research/microbench/dp4a_peak_cuda*`).
The short-prompt cliff is recorded-not-prioritised by its own scope and stays out.

## 1. Why the boundary comes first - four non-landings, one root cause

Four measured, committed non-landings share the same mechanism: the opaque custom-kernel
transport materializes per-call input copies for lazy producers, and the copy tax exceeds
the fusion win.

| piece | measured verdict | copy tax |
| --- | --- | --- |
| M3 fused decode RMSNorm | -3% wall (173.45 -> 168.42) | 144 input copies + 72 output materializations |
| M4 q4k GEMV epilogue absorption | -18.8% wall (172.69 -> 140.20) | one scheduler copy per extra epilogue input |
| M5 flash combine fp16 | net zero kernels/us | new fp16->fp16 copy class, 36x |
| Path 3 semantic RMSNorm | -0.9% to -1.3% wall, +110 kernels | +72 input copies, +36 per-layer boundary, +1 output class |

The shared root cause is one line of shared code:
`UOp.custom_kernel` (`uop/ops.py:1264-1273`) preserves `AFTER` and
`MEMORY_SEMANTIC(has_buffer_identity)` sources but calls `.contiguous()` on every other
input. Decode norm inputs are lazy producer chains (residual adds, attention outputs, the
layer-0 embedding gather) with no buffer identity at lowering time, so every fused call
materializes a copy. The non-norm census (`non-norm-copy-inventory-census-20260802.md`)
measured category A (flash q/kv views) at 0us - those consumers already bind identity
views. The tax is specific to lazy producers, and the norms are the biggest class of them.

The no-copy alternative already exists as a code path and was tried: binding the producer's
`(1,1,4096)` MEMORY_SEMANTIC view straight into a manual `kernel.call` (rank-3 param, no
reshape/contiguous). It crashes `symbolic+reduce_collapse+debuf`
(`rangeify.py:1085`): the lazy source collapses to `()` and the movement-op recompute
raises `bad reshape: () -> (4096,)` / `bad expand: () -> (4096, 1)`. The same failure
mode is documented at `rangeify.py:96,110,230` for the attention composite reduce. This
is the M3 reopen condition, still unbuilt: "the emitter indexes the producer's logical
shape and the scheduler preserves non-identity views through symbolic."

Consequence for ordering: L1's shape (a) epilogue absorption adds extra inputs to the q4k/
q6k/flash emitters. Under today's transport, every extra input is a copy (M4 measured the
result: -18.8%). L1 cannot land before the boundary, because the boundary is what makes
its fused variants economical. The design doc's "per-emitter opt-in is sufficient" was
written before M4's measurement; M4's measured copy-per-input falsifies it. Ordering
decision: **boundary first, then L1**, with M3/Path 3 norms reopening in between because
their machinery already exists and re-measures cheaply.

## 2. The design - view-preserving opaque boundary

Add an opt-in input mode to the opaque-kernel transport: when a consumer declares it
indexes inputs by logical shape, the transport binds the producer's base buffer with its
logical dims and the scheduler keeps the view alive through symbolic. Default behavior
(contiguous) is unchanged, so every existing emitter keeps its flat-index contract and
its pg3 hash.

### 2.1 What changes

- `tinygrad/uop/ops.py`: `UOp.custom_kernel` gains a view-preserving mode (new keyword,
  e.g. `preserve_views=True`) that skips the `.contiguous()` boundary for admitted
  sources and binds their logical shape instead. The flat-buffer default stays the
  default; this is the additive-variant pattern, not an ABI change.
- `tinygrad/schedule/rangeify.py`: the `symbolic+reduce_collapse+debuf` pass
  (`rangeify.py:1085`) must not collapse the movement-op chain over a lazy producer when
  the consumer is a view-preserving opaque kernel. The crash is a RESHAPE/EXPAND whose
  producer collapsed to `()`; the fix keeps the producer's logical shape alive for those
  roots (the view contract is a declared capability of the consuming kernel, read from
  its spec - not a global behavior change).
- `tinygrad/llm/decode_kernels.py`: `DecodeRMSNormSpec` already carries `x_rank` in
  (1,2,3); the view mode uses rank-3 (logical `(1,1,4096)`) and the emitter indexes the
  producer's logical dims. Other opted-in emitters (q4k/q6k/flash) declare the same
  contract when they take epilogue inputs (L1, section 4).
- Fail-closed admission: view mode is emitted only when the producer is
  buffer-backed/MEMORY_SEMANTIC with known strides (the probe shapes already proven in
  the Path 3 isolation tests); anything else falls back to the contiguous boundary.
  Undeclared targets keep the safe default automatically (the repo's declared-facts
  rule).

### 2.2 What does NOT change

The flat-buffer transport contract, `AFTER` semantics, the 72-copy consumer-owned
materialization at `decode_routes.py:78` (P0 verdict: consumer-owned, separate question),
the renderers, the packed storage layouts, the dequant math, the KV cache layout, the
prefill path, and the SDPA path.

### 2.3 Risks

1. Every opted-in emitter must be audited for flat-index assumptions (q6k coop is flat;
   the flash tile reads cache with identity - unaffected). The pg3 pins for opted-in
   emitters move deliberately and get new rows.
2. The symbolic pass change has blast radius across every JIT user; it must be provably
   inert when no view-preserving consumer is in the graph (unit-pinned, gate-off
   byte-identical pg3 hashes).
3. The "opaque = simple buffer" contract becomes two modes that must not drift; the
   view mode carries a validator that rejects unadmitted producers (the fail-closed
   pattern of the semantic RMSNorm lowering).

## 3. Then reopen the norm family (M3/Path 3) on the fixed boundary

The semantic RMSNorm machinery is committed and closed:
`Ops.RMSNORM` + fail-closed lowering + closed `decode-rmsnorm-native-lowering-route-policy.json`
(Path 3, `68020f7a0`). Its forced-open measurement proved byte-identical tokens at
d512/d2048/d4096 (sha `9d6b3787...` 3/3, first token `151936` 3/3) and a single
`rmsnorm_native_1_4096` kernel in isolation; only the boundary copies made it non-landing.
Reopen means: bind through the view-preserving mode (rank-3 logical input, no
contiguous), re-run the fixed-depth protocol, and attach a measured record with decode
tok/s >= the M2 baseline at all three depths and unchanged sha pins before the promotion
record flips. Expected census: norm family ~145 kernels with no copy/materialization
kernels around the norms (the task's original no-copy target), i.e. the llama-shaped
`rms_norm_f32` topology the reviewer amendment required (section 9.3 of
`decode-norm-fusion-paths-forward-20260802.md`).

## 4. Then L1 - plumbing fusion on the fixed boundary

The L1 design (`l1-decode-plumbing-fusion-design-20260802.md`) is accepted as the shape:
(a) epilogue absorption into the custom emitters, new kernel names, legacy untouched,
capability gate as a decode route fact row with a CLOSED default. Its nine open questions
stand; this design adds the boundary as the delivery precondition and sequences its
pieces by measured mass:

1. GEMV/flash epilogue absorption (M4/M5 variants already built and measured under the
   old transport; re-measure with the boundary - the ~0.4ms node-sum epilogue mass).
2. Q6K in-kernel merges: the coop merge already landed (M2, `910f538c5`); the partial
   single-pass variant remains under the substrate scope (its 466.6us anomaly is
   recorded, not chased).
3. Norm pair-fusion lands via section 3's reopen (the norm half of L1's claim,
   ~0.58ms node-sum of the ~0.9-1.0ms total).

Reduce-order discipline applies unchanged: in-kernel merges must preserve the generic
reduce's summation order or the decode sha moves; if a digit delta appears, STOP and
report the exact diff.

## 5. Beyond parity - B3 prefill host overhead (separate, after decode)

Decode parity is the realistic beat target (P0: llama's int8 IMMA mechanism is 25-40%
faster at raw GEMM rate than the fp16-mma ceiling this campaign can reach). Beyond
parity lives in two places:

- Prefill pp512: the measured busy ceiling is 512/24.1ms = 21.2k tok/s - ABOVE llama's
  14,250. The entire remaining prefill gap is the host factor (B3: 23.7ms polling in a
  44-46ms wall, `HcqView` memoryview-per-poll at `hcq.py:285`, three ranked fix shapes).
  Landing B3 pushes prefill past llama. It requires the AMD control run before landing
  (the submit path is shared HCQ code).
- Decode beyond parity: L2 single-pass partial, L4 vocab substrate fusion, and the flash
  tile structure are substrate-parked with measured masses; each is a separate scope
  after L1 lands, not part of this design.

## 6. Expected outcomes (node-sum-derived, re-measured in place)

| state | decode d512 tok/s (est.) | evidence |
| --- | ---: | --- |
| today (closed gates) | 172.8 | measured, final parity record |
| boundary + norm reopen | ~178-184 (est.) | M3-family -144 launches / ~-0.16ms node-sum, directional |
| + L1 GEMV/flash epilogue | ~+0.4ms node-sum recovered | M4/M5 class mass, delta-capped |
| + L1 full (incl. norm) | ~195-210 target | corrected budget 4.37-4.93 ms/token vs llama 4.07 |
| + B3 prefill | pp512 18-21k tok/s (est.) | busy ceiling 21.2k, host factor removed |

Every estimate is an upper bound from measured censuses until re-measured in wall time at
the fixed-depth protocol (d512/d2048/d4096, same-session llama rows pinned in
`nv-decode-parity-final-20260802.md`). Recovery numbers are not additive across levers
(the 60-80% haircut rule in the decode-gap scope section 8).

## 7. Controls and verification

- pg3 decode render equality (`scratchpad/pg3_decode_rendered_source_equality.py`): the
  legacy 10-kernel HIP baseline must stay byte-identical at every step; each fused/view
  variant gets new pinned rows; the Metal arm runs on the macOS box.
- NV pins: first-token digits, decode sha256
  `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe`, census row
  `prefill_overlay_promotion: candidate_set:sha256:1b8ea95d...`, fixed-depth token sha
  `9d6b3787...` per harness.
- New unit gates: view-mode admission (fail-closed on unadmitted producers), a unit graph
  probe for contiguous `RESHAPE(AFTER)` through the view mode, and a decode census
  asserting the norm family reaches the no-copy target.
- AMD control: pg2 six-route hashes byte-identical on every commit; B3 additionally
  requires the AMD runtime control run.

## 8. Sequencing and delivery

1. `[codegen]` view-preserving opaque boundary + fail-closed admission + unit gates.
2. `[nn]` reopen semantic RMSNorm through the view mode behind the closed record;
   `[docs]` measured reopen record at d512/d2048/d4096 with sha pins.
3. `[nn]` + `[test]` L1 epilogue absorption (M4/M5 re-measurement first, then the full
   design's pieces), each additive and capability-gated.
4. `[docs]` endgame measurement record vs the pinned llama rows; verdict: parity met or
   not, per-lever attribution.
5. B3 prefill host overhead as a separate scope with its AMD control.

Each piece is one owning-prefix commit on `nvidia-bringup-20260731` only. No promotion
to `dev`/`exp`/`master`.

## 9. Open questions for review

1. Is "boundary first, then L1" accepted, given M4's measured copy-per-input? (This
   design's load-bearing claim; the alternative - L1 with the current transport - is
   measured -18.8%.)
2. Is the view-preserving mode as an opt-in `custom_kernel` keyword the right shape, or
   should the logical-shape contract be a distinct builder so the two modes cannot
   drift?
3. Is the symbolic-pass fix scoped correctly (keep the producer's logical shape alive
   only for view-preserving consumers), and is the gate-off inertness proof sufficient?
4. Does the norm reopen need the 72 output materializations resolved first, or does the
   consumer-owned P0 verdict (decode_routes.py:78) make that independent?
5. Is B3 correctly sequenced after decode (section 5), or should prefill "beyond" work
   proceed in parallel given its AMD control requirement is independent?

HARD STOP after this section. No implementation beyond this design until it is reviewed.
