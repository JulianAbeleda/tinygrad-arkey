# NV GEMV substrate landing scope - Q8_1 + DP4A four-warp package for decode parity

Date: 2026-08-08
Branch boundary: tinygrad `nvidia-bringup-20260731`, HEAD `fb522cd17` (M4 resadd
promoted, census-fix stack in tree)
Status: **implementation scope. Authorizes the cooperative shared-Q8 attention
lease as a promotion candidate with a section-6 full gate, plus three bounded
reopen arms, each with its own HARD STOP. The isolated substrate wins are
re-verified at this HEAD (`nv-gemv-substrate-reverify-record-20260808.md`).**

## 1. Why this scope exists

The reconciled parity ledger (`nv-decode-parity-campaign-reconciled-ledger-20260805.md`)
closes the Q4/Q6 quant-core column at **302.788 us/token attribution** with 13
bounded NO-GOs, and its ranked next step 1 is:

> A distinct exact native Q4/Q6 DP4A substrate ... a third physical
> representation with an independent oracle, PTX/resources, and a material
> included-cost win before one real family.

The third physical representation now exists and is re-verified at this HEAD:
the llama MMVQ package consumed in tinygrad UOps = **Q8_1 packed activation
provider + DP4A (`int8x4_dot`) + four-warp/128-thread per-row ownership**,
consuming the production packed Q4_K/Q6_K storage directly (no Q8 weight
representation, no MMQ/MMA, no fp16 activation). Isolated included-cost wins at
this HEAD: Q4 cooperative **-56.28 us/replay (-45.6%)**, Q6 flat four-warp
**-0.81 to -0.94 us/replay** vs installed. The model-level cooperative Q4
progression already booked g12 (-24.676 us) and the max17 subset (-12.462 us)
as causal-ledger credit; g18 was a semantic stop, not a wall failure.

The M4 landing just promoted the residual_add half of the old rejected
Attention-O composition and cleared the census/union-resolve/transport stack.
This scope takes the GEMV half of that same composition - the shared-Q8
attention group - through the same landing ladder: production wiring, section-6
full gate, promotion record, and booking.

## 2. The substrate, precisely

### 2.1 Provider (already qualified, unchanged)

`_emit_q8_provider` and `_emit_rmsnorm_q8_provider` in
`tinygrad/llm/shared_q8_attention.py`. One program writes the llama-CUDA Q8_1
ABI: 1024 int8x4 packets + 128 d|s metadata half2 words. The RMSNorm fused
variant is the exact `REDUCE_OUTPUT` 1x4096 affine recipe with the 16-warp
reduction association. Both are byte-exact against the pinned live llama CUDA
Q8 cubin and are checked-in qualified producers.

### 2.2 Consumers (already built, closed-lease)

- `_emit_q4_cooperative`: four-warp Q4_K consumer, one flat LOCAL=128 launch,
  two logical words/lane/block, exact two-logical-word ownership, llama's
  `d*sc*dot - dmin*mn*sum(x)` affine correction in the Q8 domain, four partials
  reduced by an external tensor sum (included in every gate).
- `_emit_q6_warp_direct`: four-warp Q6_K V consumer, four blocks/warp, 16-byte
  shared publish + barrier + lane-0 merge (the `PostBarrierRegion`-free direct
  spelling that won; the 384-byte llama-stage spelling measured +0.185 us
  slower and is NOT reopened here).
- `_emit_q4`/`_emit_q6` row_tile=2 two-warp spellings exist as research arms;
  the Q6 row_tile=2 spelling re-measured **+0.405 us** at this HEAD and is
  closed for promotion (2.3 below).

### 2.3 What is NOT the substrate

- Exact fp16 four-warp (`q4k_exact_four_warp`): +2.87 us NO-GO, closed.
- Exact one-warp factorized: -0.21 us wall-neutral, closed.
- Q6 row_tile=2 single-kernel Q8+DP4A: +0.405 us at this HEAD, closed.
- Q6 384-byte lane stage with producer-warp retirement: +0.185 us vs flat
  control, closed as a next wall lever (the `PostBarrierRegion` primitive
  itself is landed and stays).
- Integer MMA / s8-s8 descriptor path: not d512 decode causality; not reopened.
- Q6-only shared-Q8 group: the authority model has zero all-Q6 groups; dead
  route, not reopened.

## 3. Production shape (three deltas, one record)

### 3.1 New per-route promotion record

`tinygrad/llm/generated/decode-shared-q8-attention-route-policy.json`
(schema `boltbeam.route_policy.v1`, same family as the M4 resadd record),
`promoted_targets: []` initially, gaining `NV sm_120` ONLY after the section-6
gate passes. Loader + predicate in `model_route_plan.py` following the
`load_decode_q4k_epilogue_resadd_promotion` pattern exactly:
`load_decode_shared_q8_attention_promotion` +
`_DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS` +
`decode_shared_q8_attention_promoted(target)`.

### 3.2 Loader-installed admission (harness -> policy)

Today `SharedQ8AttentionAdmission` is installed only by a qualification
harness; the model loader never constructs it (`model.py:687`). This scope
adds the loader-side install: when
`decode_shared_q8_attention_promoted(("NV","sm_120"))` is true, each
TransformerBlock with an exact Q4/Q4/{Q4,Q6} tuple and the exact
`REDUCE_OUTPUT` RMSNorm source receives
`SharedQ8AttentionAdmission(block_index, cooperative_q4=True)` on blocks
**1-12 and 14-18** (the booked max17 lease; block 0 embedding boundary stays
ordinary, block 13 is the precision boundary at rel L2 `1.002240e-3`). Block
19-35 stays ordinary: tail expansion is NO-GO at 0 credit.

The lease is installed only for the admitted blocks; every non-admitted block
keeps the three ordinary primitive calls verbatim. `q6_direct_output` stays
False at load (the Q6 direct lease is a reopen arm, section 5, not part of
this promotion).

### 3.3 Wiring deltas

- `model_route_plan.py`: record loader + predicate (section 3.1).
- `model.py`: at load, after `_decode_q4k_epilogue_resadd_promoted`, set
  `_shared_q8_attention_admission` from the new predicate + block index +
  norm weight (the exact `_K` fp16 weight for the fused provider).
- `shared_q8_attention.py`: no emitter change. The admission dataclass and
  call stay; only the install site changes.
- `decode_routes.py`, `qk_primitives.py`, `decode_kernels.py`: untouched.

### 3.4 What does NOT change

- M4 resadd record, combined M4 epilogue record, w1w3, kv-store, M5 typed-view
  contracts: untouched.
- No emitter change to any installed legacy kernel: pg3 legacy sha
  `27857cb8ca03` for `q4k_g3_lanemap_gemv_4096_4096` must stay byte-identical.
- No Q8 quantize tax scope: the provider already exists and is included in
  every measured number; no new quantize pass is authored.
- No default flip on any other target: the record is empty for AMD/Metal/etc.

## 4. Section-6 full gate (before the record gains NV sm_120)

Same-session, lock-held (`flock -w 600 /tmp/gpu-bench.lock`), Qwen3-8B-Q4_K_M,
nmeas 20, reps 3, median tok/s, fused prefill attention disabled. Open mode =
gate forced open via the module override
(`mrp._DECODE_SHARED_Q8_ATTENTION_PROMOTED_TARGETS =
frozenset({("NV","sm_120")})`) + loader-installed 17-block lease. Closed mode =
default records (lease dormant). Record mode = checked-in policy JSON.

1. **Wall** d512/d2048/d4096, open mode vs the M2-on baseline (172.80/161.50/
   149.00 tok/s, `nv-decode-parity-final-20260802.md`) and positive delta vs
   the same-session closed control. The 08-05 composed reference for the same
   lease: g12 -24.676 us, max17 -12.462 us incremental vs g12.
2. **Census** (d512, open mode): fused provider count 34 (17 blocks x two
   captures), cooperative Q4 consumer count 86, zero legacy shared-Q4
   consumers, zero duplicate providers, per the max17 record oracle.
3. **Semantic contract** (predeclared, shared-Q8 is an intentional Q8 change):
   exact token stream, equal argmax, ordered top-10 sets, aggregate relative
   L2 <= `1e-3`, `2*max_abs/min_top1_margin < 1.0`, all finite.
4. **Pins 3/3** at every depth, both modes; control and variant streams stay
   identical.
5. **Unit tests**: `test_shared_q8_attention*` + M4/M5 sets green on the same
   tree.
6. **pg3 legacy sha** `27857cb8ca03` unmoved.

The record gains `NV sm_120` only after 1-6 pass. After promotion, re-run the
closed-vs-open token-stream equality (pins 3/3 both modes) to prove the
record-open state equals the forced-open state.

## 5. Bounded reopen arms (each its own HARD STOP)

### 5.1 Q6 V direct-output lease (`q6_direct_output`)

The isolated four-warp Q6 win reproduces at this HEAD (-0.81 to -0.94 us vs
installed). The 08-05 model-level g12 arm was WALL NO-GO (+7.009 us/token)
measured at `a1a51c349` on the older composition. This scope authorizes ONE
fresh same-session g12 bracket at this HEAD with the identical settled
protocol; the expected count is 12 direct Q6 consumers (real Q6 V blocks).
Promotion follows ONLY if the fresh wall is negative; the 08-05 NO-GO stands
until then. No max17/FFN-down expansion in this arm.

### 5.2 Q4 FFN-down singleton re-bracket

08-05: primitive wins isolated, first passing singleton layer 8 regresses
settled wall by +6.205 us/token. The census-fix stack changed the composition;
this scope authorizes ONE re-bracket of the layer-8 singleton at this HEAD with
the same settled protocol. A second layer is NOT advanced from this arm alone;
the predicted two-layer subset stays closed pending a positive singleton
result.

### 5.3 Block-13 precision localization (CPU-only)

The max17 lease stops at block 13 (rel L2 `1.002240e-3`, marginal). This arm
authorizes CPU-only localization of the cumulative llama-Q8 error across the
g12->g18 boundary (per-block signed logit-delta vectors, no GPU time). It does
not authorize any lease expansion: block 13 + blocks 19-35 remain NO-GO until
the localization identifies a numerically independent block or an arithmetic
correction, and any such candidate returns through a fresh semantic + settled
wall gate.

## 6. Booking rules

- Book the section-6 same-session delta as a fresh ledger row, incremental on
  the composed P1/P2/P5/Q4-g12/max17 baseline, cited to this scope and its gate
  record. Do NOT add the isolated -56.28 us microgate number or the 08-05
  composed g12/max17 rows again.
- Do NOT combine this recovery with the rejected Attention-O/FFN-down
  composition row (0 credit) or with any FFN-down arm.
- The ledger's "claims that remain unsupported" rules apply unchanged: no
  synthetic microgate extrapolation, no subtraction-from-fixed-authority
  absolute claims, no fresh parity ratio from booked rows.

## 7. Ranked continuation (NOT authorized here)

Beyond this landing, the parity path remains: (1) the quant-core DP4A
substrate is THIS scope; (2) the norms population (495 us attribution) is
capability-blocked by C1 (no generic cooperative reduction-to-output primitive)
per `nv-substrate-capability-vs-ledger-scope-20260807.md`; (3) host/overlap
remain attribution. Each is a separate scope with its own HARD STOP.

## 8. HARD STOP

This scope authorizes exactly: the shared-Q8 attention promotion record
(section 3.1), the loader-installed 17-block admission (section 3.2), the
section-6 gate run, the promotion decision to `NV sm_120` after the gate
passes, and the three section-5 reopen arms. It does NOT authorize: changing
the combined M4 record, w1w3, kv-store, M5, any emitter change, the Q6
384-byte stage, the row_tile=2 spelling, the exact-fp16 constructions, tail
expansion beyond the 17-block lease, FFN-down two-layer advance, Q8 quantize
tax, or any promotion on another target. No GPU probe outside
`flock -w 600 /tmp/gpu-bench.lock`.

## 9. References

- `nv-gemv-substrate-reverify-record-20260808.md` (fresh isolated gates at this HEAD)
- `nv-decode-parity-campaign-reconciled-ledger-20260805.md` (booking rules, rank 1)
- `nv-q4-warp-cooperative-dynamic-static-record-20260805.md` (Q4 construction history)
- `nv-q4-cooperative-shared-q8-g1-integration-record-20260805.md` (g1/g4/g8/g12 progression)
- `nv-q4-cooperative-subset-precision-budget-record-20260805.md` (max17 lease)
- `nv-q6k-mmvq-instruction-delta-and-one-change-record-20260805.md` (Q6 package)
- `nv-q6-direct-shared-q8-g12-record-20260805.md` (Q6 model NO-GO, reopen arm)
- `nv-q6k-post-barrier-region-implementation-record-20260805.md` (post-barrier, closed)
- `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md` (FFN-down)
- `m4-resadd-landing-scope-20260806.md` (landing-ladder precedent)
- `nv-substrate-capability-vs-ledger-scope-20260807.md` (C1/C2/C3 capability map)
