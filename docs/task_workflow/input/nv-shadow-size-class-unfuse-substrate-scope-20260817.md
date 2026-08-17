# NV shadow size class: exhaustive scope for the unfuse substrate (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731` (HEAD `bad558740`, llama-arch envelope landed)
Status: **scope record. Answers "do we have the substrate to unfuse the
quantize shadow at the size class that co-schedules on native?" with a
per-piece inventory, the measured envelope it must satisfy, and the gated
order of work. No production change in this record.**

## 1. Why this scope exists

The llama-arch envelope probe (`nv-llama-arch-native-envelope-20260817.md`)
flipped the 08-17 wait-boundary verdict: at production decode sizes (1024/2048,
~7-13 us kernels) the native two-GPFIFO pair co-schedules the anchor+shadow
topology across a single cross-queue wait:

| arm | shape | median overlap |
| --- | --- | ---: |
| `pipeline_same` 1024 | producer signals; q0 continuation runs; q1 consumer waits | +9.6% |
| `pipeline_same` 2048 | same | +11.9% |
| `rejoin_same_n4` 1024 | 4 same-size shadows, one join at flash_score | +8.0% |
| `shadow_same` 1024 (control) | head-wait, same-size shadow | +12.4% |
| `shadow_llama_order` 1024/2048 | tiny rope + tiny kv + medium quantize | -5.2% / -1.1% |

The load-bearing cell: **same-size-class shadows co-schedule; all-tiny or
mixed tiny shadows still lose.** Llama's real overlap mass is
`quantize_q8_1` (549.8 us) + `rope` (127.3) + `kv_set_rows` (74.6) = 752 us of
shadow kernels pipelined behind the mmq anchor chain. We fused that work into
our GEMV epilogues (node sum 496 us below llama), so at HEAD there is no
shadow mass left to place. The question this scope answers: **what substrate
does "unfuse" require, and is it already built?**

## 2. The substrate definition this scope builds against

From `nv-substrate-definition-20260815.md`: a substrate is a capability in the
compile/lower/emit/runtime stack that makes a target construction expressible
as one valid, replayable program. Capability-blocked (cannot render) routes to
build work; wall-blocked (renders but loses) routes to values work.

The target construction here is precise: **one steady decode token whose
per-layer quantize (the Q8_1 provider) and/or kv support render as separate
kernel programs of the SAME SIZE CLASS as the GEMV anchor chain, placed on the
aux GPFIFO behind the primary chain, joined once at flash_score, with
`overlap_mass > 0` and bitwise-identical tokens.**

## 3. Inventory: what already exists in the tree (verified at HEAD)

The unfused quantize is NOT missing. Every hardware and runtime half is
present:

| piece | where | state at HEAD |
| --- | --- | --- |
| **Q8_1 provider as a separate kernel program** | `shared_q8_attention.py` `_emit_q8_provider()` / `_emit_rmsnorm_q8_provider()` -> `q8_1_llama_provider_4096` / `rmsnorm_q8_1_llama_provider_4096` | built; **PROMOTED NV sm_120** via `decode_shared_q8_attention_promoted` (blocks 1-12, 14-18; max17 lease) |
| **Q4/Q6 consumers of the packed Q8_1 packets** | `_emit_q4` / `_emit_q4_cooperative` / `_emit_q6` / `_emit_q6_warp_direct` | built; PROMOTED NV sm_120 (`decode_shared_q8_attention_promoted`, `decode_q6_direct_shared_q8_attention_promoted`) |
| **Two native compute GPFIFOs** | `HCQ_NUM_COMPUTE=2`, `hw_compute_queues()` | qualified construction (rank2 verdict), default off |
| **Generic readiness placement (S2)** | `hcq.py` `HCQ_NV_READY_PLACEMENT` + `_pick_compute_queue` + `DepsTracker.peek_access_resources` | built + unit-tested, gated off (`=0` default), S2 wall A/B slightly negative (205.88 vs 207.75) |
| **Single-join cross-queue signal machinery** | `Job`/`run_jobs` + `make_queues` + timestamp signals in the probe harness | proven; the envelope positive arms use exactly this |
| **The join point (flash_score)** | `decode_routes.py` `flash_decode_attention_route` | present; flash body at parity |

So the "unfuse substrate" is **not a new primitive build**. It is a
composition of pieces that already exist, which is exactly the boundary the
envelope probe proved expressible. The correct classification: the remaining
blocker is **wall-blocked (values), not capability-blocked (substrate)** — the
geometry co-schedules, and the wall A/B decides if it converts to tok/s.

## 4. What the shadow must look like (the size-class contract)

From the envelope matrix, the shadow that co-schedules on native satisfies all
of:

1. **Same size class as the anchor**: ~7-25 us kernels (1024/2048 class).
   The 512-class (~4 us) is overhead-dominated (negative in every arm).
2. **Enough mass to amortize the wait**: `rejoin_same` with only 2 shadow
   kernels was negative (-8 to -13%); `rejoin_same_n4` (4 kernels) flipped
   positive (+6.6 to +9.6%). The shadow chain needs >= ~4 same-class kernels,
   or one producer-continuation pipeline.
3. **One join, not per-kernel events**: `rejoin_multi_wait` (event-per-kernel,
   llama's per-kernel event pattern) was the worst arm (-14.7% median). The
   native pair amortizes ONE wait on the shadow's end; llama's CUDA executor
   amortizes per-kernel events, and we do not have that executor.
4. **No tiny tail on the same channel**: `shadow_llama_order` (tiny+tiny+
   medium) was negative at both sizes. Tiny kernels on the waiting channel pay
   the runqueue-switch penalty and drag the whole shadow negative.

llama's own shadow violates rule 4 (rope/kv_set_rows are tiny) and rule 3
(per-kernel events), yet co-schedules on CUDA because CUDA's graph executor
does not pay the native runqueue-switch cost. **The native 1-to-1 transfer of
llama's exact shape is not available; the transferable shape is the
same-size-class single-join shadow.** This is the honest difference this scope
exists to name.

## 5. The two candidate compositions (and which one the envelope supports)

### Candidate A: quantize provider as shadow behind the GEMV anchor chain

Per layer, the shared-Q8 provider quantizes the normed activation to Q8_1
packets; the q4k/q6k consumers then read those packets. Today the provider
runs immediately before its consumers in the same serial chain. Composition:
place the provider (and its kv-store siblings) on the aux GPFIFO behind the
primary GEMV chain with one rejoin before flash_score.

Envelope support: the provider at 4096-wide is a single small kernel
(~2-6 us, the `shadow` head-wait class), and its consumers are 1024/4096-class
(~4-13 us). This is the `shadow_llama_order` shape (tiny+medium), which
measured NEGATIVE. **Candidate A as-is does not satisfy rule 4.**

### Candidate B: merge the per-layer support into ONE same-size-class kernel

Merge rope + norm + quantize (+ kv_set_rows) into a single ~1024-class kernel
per layer (~7-13 us, matching the GEMV anchor class), so the entire shadow is
one hideable unit behind the anchor, joined once at flash_score.

Envelope support: this is the `rejoin_same_n4`-adjacent shape (medium shadow,
single join) at +6.6 to +9.6%, and the `pipeline_same` shape at +9.4 to
+15.4%. **Candidate B is the shape the envelope says co-schedules.**

The honest arithmetic is unchanged by this scope: even at the optimistic
envelope rate, the transferable shadow mass is llama's 752 us, and
`752 * 0.08 = 60 us` hidden at the rejoin rate ~ +2.8 tok/s, up to
`752 * 0.32 = 240 us` at the opportunistic spike ~ +11 tok/s (219.9). This
stays below the 233.8 non-overlap ceiling and does not by itself reach 240.
The value of this work is the substrate + the wall A/B answer, not a 240
claim.

## 6. Order of work (gated; test -> pass -> do)

1. **Compile-only capability probe (no GPU):** render Candidate B's merged
   per-layer support kernel as one program (rope+norm+quantize+kv_set_rows in
   a single 1024-class kernel) using the existing `ReduceOutputSpec` +
   `KernelProgram` machinery. If it renders with the Q8 provider ABI intact,
   the primitive is present; if it hits a `CONSTRUCTION_GAP`, the missing
   primitive is named and built first.
2. **Microbench the merged kernel's size class** (decode-shaped, ~1024):
   confirm it lands in the 7-25 us band the envelope says co-schedules. If it
   renders at 2-6 us (too small), widen it (merge more support, or batch
   multiple layers) until it is anchor-class. This is the size-class gate.
3. **Envelope probe with the REAL merged kernel** (not synthetic fp32 GEMVs):
   reuse `scratchpad/nv_llama_arch_native_probe.py` with the merged kernel as
   the shadow behind a real q4k/q6k GEMV anchor, single join. Gate: overlap
   > 0, repeatable across >= 3 runs.
4. **Production wall A/B** (same-session, flocked, bitwise-identical tokens):
   `HCQ_NUM_COMPUTE=2` + `HCQ_NV_READY_PLACEMENT=1` + the Candidate-B shadow
   route, vs the single-queue control. Promote only if wall is positive AND
   tokens sha match. This is the only authority that converts geometry to
   tok/s.
5. **Close the row honestly**: record the wall delta, the hidden-mass ledger
   (NV HCQGraph profiler `overlap_mass`), and whether it moves the 240
   arithmetic. If wall is flat or negative, the row closes as wall-blocked
   with the envelope as the explanation (geometry positive, production shape
   cannot reach the transferable mass), not as a substrate gap.

## 7. What this scope explicitly does NOT do

- Does not build a new primitive unless step 1 finds a `CONSTRUCTION_GAP`
  (then the gap is scoped separately with its own gate).
- Does not re-litigate the 08-17 wait-boundary verdict (superseded by the
  envelope record).
- Does not claim 240 tok/s. The honest ceiling of the unfuse row is the
  already-priced +3.7 to +11 tok/s, below the 233.8 non-overlap ceiling.
- Does not touch the CUDA route; this scope is native-only, matching the
  production route.

## 8. Acceptance test

The unfuse substrate is "present" when, in one session:

1. The merged Candidate-B support kernel renders (compile gate, step 1).
2. Its microbench size class is 7-25 us (size-class gate, step 2).
3. The real-kernel envelope probe shows repeatable positive overlap (step 3).
4. The production wall A/B is positive with bitwise-identical tokens (step 4).

Every claim is a measurement gate; a row is not "landed" until its gate passes
with bitwise-identical tokens in a flocked same-session A/B.

## 9. Evidence map

- envelope matrix: `nv-llama-arch-native-envelope-20260817.md` + session
  JSONs (positive pipeline/rejoin_n4 cells, negative tiny/multi-wait cells)
- wall account + arithmetic: `nv-240-exact-wall-account-20260817.md`,
  `nv-240-climb-stress-test-20260817.json`
- substrate definition: `nv-substrate-definition-20260815.md`
- S2 placement: `nv-substrate-exhaustive-scope-20260817.md` section 4,
  `hcq.py` (`HCQ_NV_READY_PLACEMENT`, `_pick_compute_queue`),
  `test/unit/test_hcq_nv_ready_placement.py`
- shared-Q8 provider/consumers: `tinygrad/llm/shared_q8_attention.py`,
  promotion records `decode-shared-q8-attention-route-policy.json`,
  `decode-q6-direct-shared-q8-attention-route-policy.json`
