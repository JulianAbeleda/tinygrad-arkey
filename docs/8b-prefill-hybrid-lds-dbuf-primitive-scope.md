# 8B Prefill Hybrid LDS/DBUF Primitive Scope

Date: 2026-07-09.

## Question

Can we handcode the hard LDS/DBUF parts and still make progress without turning the route back into a hand-tuned
kernel?

## Answer

Yes, if the handcoded part is a reusable compiler/backend primitive with a declarative input contract, and the generated
route still owns the kernel lifecycle around it.

No, if we copy `build_gemm_lds2` as a route-local instruction list or fixed-register full kernel body.

The viable compromise is:

```text
machine/search/spec owns:
  role selection
  shape
  tile geometry
  pipe-vs-LDS selection
  DBUF on/off
  primitive composition
  correctness/perf promotion

hand-authored backend primitive owns:
  one narrow hard operation, such as:
    packed global->LDS staging
    LDS two-slot cadence metadata
    DS offset proof/folding
    targeted wait/barrier template
```

This is a hybrid compiler primitive, not strict pure generation. It is also not a hand-tuned kernel unless it owns the
full model/shape-specific lifecycle.

## Boundary

| Candidate | Allowed hybrid primitive? | Why |
|---|---:|---|
| `GLOBAL_LOAD_B128`, `DS_STORE_B128`, `DS_LOAD_B128` lowerers | yes | ISA-local, reusable, already backend-owned. |
| `LDSAddr` / DS offset proof | yes | Reusable correctness primitive for LDS addressing. |
| Packed LDS store grouping | yes | Reusable transformation from proven lane/byte windows to b128 stores. |
| Two-slot DBUF slot identity | yes, if declarative | Primitive can encode slot identity and ordering, while route/search chooses when to use it. |
| Targeted `vmcnt/lgkmcnt` policy | yes | Backend scheduling primitive, not a full kernel. |
| Full `build_gemm_lds2` prologue/body/tail instruction list | no | This is the full hand kernel lifecycle. |
| Fixed physical register script for `512x12288x4096` | no | Shape-specific register choreography equals hand tuning. |
| Route-local `UOp(Ops.INS)` full GEMM body | no | Bypasses compiler ownership. |

## Current Feasibility

Feasible now:

- `WMMALDSSpec` can describe the `ffn_gate_up` LDS shape.
- `PREFILL_WMMA_LDS_PRIMITIVE=1` can divert before the raw oracle.
- Ordinary generated matmul transport can compile the single-buffer LDS route.
- Generated structural compile proof exists:

```text
global_load_b128 = 24
ds_store_b128    = 24
ds_load_b128     = 64
wmma             = 16
global_load_u16  = 0
```

Not solved:

- route-bound GPU correctness for the generated LDS transport,
- same-clock timing against the raw LDS oracle,
- DBUF body cadence/overlap,
- final byte-window proof for rematerialized LDS addresses,
- DS immediate folding selection policy.

So the hybrid route is possible, but the first hand-authored primitive should be small and proof-bounded. Do not start
with full DBUF lifecycle cloning.

## Prior Issues To Avoid

| Prior issue | What caused it | Hybrid guardrail |
|---|---|---|
| `UNROLL(STACK)` verifier failure | packed carrier inserted too early in `postrange.py` | insert only after expansion, or expose a target primitive whose inputs are verifier-clean. |
| wrong result `rr=2.2e-01` | unsafe DS `offset0` folding under DBUF both-stage | materialized offsets remain default; immediate folding requires proof. |
| B-side LDS aliasing | B staging keyed only by WARP lane, missing output-column tile identity | preserve bounded B tile-key contract. |
| no-spill pressure | scalar LDS/address temps live too broadly | primitive must stream packed stores and kill temps promptly. |
| fake generated ownership | route still called `build_gemm_lds2` through `Ops.INS` | generated route must execute ordinary compiler transport or a backend primitive, never raw full instruction list. |

## Primitive Options

### H1. Packed LDS Stage Primitive

Contract:

```text
packed_lds_stage(
  operand: A|B,
  global_base,
  lds_base,
  tile_key,
  row,
  frag_const,
  gate,
  order
) -> staged_lds_window
```

Hand-authored part:

- proof that the source span is exactly 16 bytes,
- select `global_load_b128 -> ds_store_b128`,
- preserve dependency order.

Generated-owned part:

- which operand to stage,
- tile/window key,
- shape,
- whether to use A-only, B-only, or both.

Feasibility: high. This aligns with existing `PREFILL_LDS_PACK_WITHLOCAL_B128` and `GATED_STORE_B128` substrate.

### H2. LDS Slot Identity Primitive

Contract:

```text
lds_slot_ref(
  operand: A|B,
  slot: 0|1,
  tile_key,
  row,
  frag_const
) -> LDSAddr
```

Hand-authored part:

- canonical slot/key byte-window representation,
- proof that store and load refer to the same LDS window,
- reject aliasing/overflow.

Generated-owned part:

- DBUF enable,
- slot schedule,
- which WMMA consumes which slot.

Feasibility: medium-high. Existing `LDSAddr` and lifecycle probes are close, but final remat equivalence is still weak.

### H3. DBUF Micro-Cadence Primitive

Contract:

```text
for each k_pair:
  stage next slot
  barrier/visibility edge
  load current slot
  compute current WMMA group
```

Hand-authored part:

- small reusable ordering template over symbolic slot references,
- wait/barrier placement rules.

Generated-owned part:

- tile shape,
- operand selection,
- number of slots,
- use/not-use decision.

Feasibility: medium. This is the performance lever, but it is closest to becoming a hand kernel. It must not hardcode the
full `ffn_gate_up` instruction stream or physical registers.

### H4. DS Offset Folding Primitive

Contract:

```text
fold_ds_offset(addr = base + const, op) -> (base, offset0) only if proof says equivalent
```

Hand-authored part:

- ISA legality,
- byte-window equivalence,
- opt-in unsafe repro remains fenced.

Generated-owned part:

- whether the route wants materialized or folded offsets based on search/perf.

Feasibility: high for correctness, mixed for performance. Prior data shows safe fold can slow some shapes and help others,
so it should be a selectable primitive, not a global default.

## First Implementation Slice

After the generated single-buffer LDS route passes route-bound GPU correctness, the first hybrid patch should be only the
smallest H1/H2 correctness slice:

- H1: typed packed `global_load_b128 -> ds_store_b128` staging for one operand window at a time, selected from
  `WMMALDSSpec`/tile metadata rather than role-local constants.
- H2: typed `LDSAddr` slot identity for the same staged window, proving the later `ds_load_b128` consumes the exact
  stored byte window with materialized offsets as the default.

The slice is acceptable only if `ffn_gate_up` still routes through generated transport, does not call `build_gemm_lds2`,
has zero scalar LDS fallback, and passes the route-bound fp32 sampled correctness gate for the 8B shape. It does not need
DBUF cadence, DS immediate folding, or same-clock oracle timing. Those remain later promotion work.

Current status: this first slice is established by
`bench/prefill-pipe-mvp/ffn-gate-up-lds-primitive.json`: generated single-buffer LDS transport is selected, the route
sample correctness gate passes, scalar LDS fallback is zero, and `wmma-lds-slot-identity-proof.v1` proves the active
single-buffer A/B byte windows. DBUF cadence is still explicitly unproven and remains the next promotion layer.

DBUF promotion definition:

- **Candidate**: generated ordinary-matmul transport compiles with `PREFILL_DBUF=1`, uses no raw `build_gemm_lds2`,
  keeps packed `global_load_b128 -> ds_store_b128 -> ds_load_b128 -> WMMA`, has zero scalar LDS fallback, exposes
  future-slot staging between WMMAs, and passes route-bound sampled correctness.
- **Promoted**: candidate plus strict dynamic D2 identity: both WMMA operands must observe at least two LDS load
  address families, or a stronger normalized byte-window proof must cover both operands.

Current DBUF status: promoted structurally, performance still unmeasured. The generated DBUF route compiles for the
8B `ffn_gate_up` shape, passes sampled correctness, and D2/D3/D7 are true. The prior apparent B-side D2 failure was a
proof-key bug: both B `ds_load_b128` halves used `offset0=0` through the same physical address VGPR, but that VGPR was
redefined between loads. The D2 key is now def-sensitive, so it distinguishes those byte windows:
`src0_lds_family_count=2`, `src1_lds_family_count=2`.

Performance comparison status: structural DBUF promotion does not yet hit the prefill target. The pinned smoke run in
`bench/prefill-whole-synced/lds-dbuf-promoted-smoke.json` reports pp512 `205.43 tok/s`, versus the stored path1 smoke
`218.12 tok/s` and the stored hand-path authority `4413.2 tok/s`. This means the DBUF primitive correctness/cadence
work is no longer the e2e blocker; the remaining gap is lifecycle/performance integration.

## Recommended Sequence

1. **Keep current generated single-buffer LDS transport.**
   This proves the route can avoid raw `build_gemm_lds2`.

2. **Run route-bound correctness for generated LDS transport.**
   If this fails, fix layout/correctness before touching DBUF.

3. **Promote H1 packed LDS stage primitive as the first hybrid primitive.**
   It is already close to existing substrate and does not encode the full lifecycle.

4. **Add H2 slot identity proof.**
   This is the prerequisite for DBUF without reliving the unsafe address-fold bug.

5. **Only then add H3 DBUF micro-cadence.**
   Keep it symbolic and shape-parametric. It must consume `WMMALDSSpec`/`LDSAddr`, not a copied instruction list.

6. **Gate H4 DS offset folding separately.**
   Materialized offsets stay the correctness baseline.

## 100% Definition

Hybrid LDS/DBUF primitive is acceptable only when:

| Gate | Required proof |
|---|---|
| G0. No raw full kernel | selected path does not call `build_gemm_lds2` and does not emit a route-local full `Ops.INS` body. |
| G1. Spec-owned | primitive consumes `WMMALDSSpec` or a smaller typed sub-spec, not hardcoded `512x12288x4096` constants. |
| G2. Generated transport | route executes ordinary compiler transport or backend primitive composition. |
| G3. Correctness | route-bound GPU correctness passes for `ffn_gate_up`. |
| G4. Address proof | LDS store/load windows are proven equivalent or conservatively materialized. |
| G5. No scalar fallback | promoted path has no scalar per-half LDS staging. |
| G6. No spill | native path compiles without spills or register-allocation escape hatches that corrupt correctness. |
| G7. Perf movement | same-clock timing moves toward the raw LDS oracle. |
| G8. Search ownership | search/spec can choose enable/disable; primitive is not unconditionally forced for one role. |

## Stop Conditions

Stop and classify as hand kernel if:

- the implementation copies the `build_gemm_lds2` instruction list,
- the primitive hardcodes physical registers for one shape,
- the primitive owns prologue, full K-loop, wait schedule, WMMA issue order, and epilogue together,
- route attribution cannot separate compiler primitive use from raw oracle use.

Stop and keep DBUF deferred if:

- generated single-buffer correctness fails,
- materialized-offset DBUF cannot pass,
- final address proof cannot distinguish unknown from alias,
- timing does not move after packed staging and slot identity are correct.

## Conclusion

The hybrid path is possible and likely useful, but the primitive must be scoped below full-kernel lifecycle ownership.

The best first hybrid primitive is not "handcode DBUF." It is:

```text
typed packed LDS stage + typed LDS slot identity
```

Once those pass correctness, DBUF cadence can be added as a small symbolic scheduling primitive. That gives us progress
without pretending a copied hand-tuned kernel is generated.
