# 14B MMQ logical-vocabulary and emission scope

Status: implementation scope, research-only, direct-packed default unchanged

Date: 2026-07-15

Target: AMD gfx1100, Qwen3-14B prefill, Q4_K weights with Q8_1/DS4 activations

## Objective

Replace the duplicated, partially contradictory MMQ contracts with one
descriptor-driven logical vocabulary that can be lowered to tinygrad UOps and
then to AMD code. The first usable target is one bounded `ffn_gate_up`
canary. No MMQ route may be promoted by this scope.

The intended lifecycle is:

```text
canonical ABI/layout contract
  -> logical MMQ vocabulary
  -> generated candidate descriptor
  -> UOp emitter
  -> explicit physical launch/lane lowering
  -> compile/resource gate
  -> guarded AMD correctness gate
  -> owner/identity/evidence gate
  -> same-session timing
  -> research artifact; direct-packed remains default
```

## Current hard-coded or contradictory assumptions

### Contracts that must be unified

- Q4_K block size, Q8_1 block size, DS4 grouping, packed-word count, and
  scale/minimum semantics are duplicated across ABI, reference, Tensor, and
  atom code.
- `mmq_abi.py` currently derives Q8 activation sizes from `n` while the
  generated emitter uses activation shape `m`; activation ownership must use
  `m` consistently.
- The ABI names Q8 sums, while the generated Tensor emitter accepts values and
  scales and recomputes sums. This must be explicit: either sums are derived by
  the logical operation or they are an operand, never both implicitly.
- `tokens_rows`, flattened Q4 weights, `[m,k]` activations, and scale layout
  are assumed by the emitter even when descriptor layout fields differ.

### Schedule and lowering assumptions

- The bounded atom fixes `16x16x256`, wave32, `lidx0`, lane-zero ownership,
  warp reduction width 32, DS4 indexing, LDS layout, and `32x16x1` launch
  metadata.
- The descriptor separately defaults to `16x16x256`, workgroup 64,
  four accumulator slots, register staging, owner writeback, and no lowered
  schedule options. These are not one contract.
- Candidate generation only varies tile M/N/K and rejects other schedule axes
  as inert.
- The current generated emitter is a wrapper around generic int8 WMMA/Tensor
  lowering. It fixes WMMA dimensions and lifecycle choices and does not yet
  lower the declared MMQ staging/writeback/launch fields.
- The emitter has fixed fallback dimensions (`wmma_n=16`, `wmma_k=16`, and
  shape-dependent M) and materialization choices that are not candidate data.

### Backend, role, and evidence assumptions

- gfx1100, wave32, AMD device naming, 64 KiB LDS, and workgroup limits are
  treated as universal rather than capability facts.
- The harness is effectively fixed to `ffn_gate_up` and the direct-packed
  rollback route.
- Static owner coverage describes intended stores but does not prove emitted
  lane ownership.
- Candidate identity is derived from incomplete axis payloads rather than the
  complete canonical descriptor, vocabulary version, lowering version, and
  backend identity.
- The evidence gate exists but must be the mandatory consumer before timing or
  promotion, not merely an independently testable helper.

## Required logical vocabulary

The vocabulary is semantic and JSON-safe. It must not encode one final kernel
body. It must represent:

- tensor roles and logical axes: tile M, tile N, reduction K, quant group,
  activation block, and edge predicates;
- Q4_K block decode, including scale/minimum fields and nibble ownership;
- Q8_1/DS4 values, scales, sums, and their derivation/operand policy;
- legal dot/WMMA operations and their operand dtypes;
- staging locations and lifetimes: registers, LDS, or direct loads;
- synchronization requirements and uniform barrier scope;
- accumulator ownership and exactly-one-owner writeback;
- candidate-selected physical wave/workgroup mapping;
- ABI, output layout, backend capability, provenance, and rollback identity.

The vocabulary must distinguish semantic facts from candidate decisions and
backend facts. A semantic descriptor may be reused across Q4 roles and later
Q6, while Q4/Q6 decode grammars remain separate.

## Required implementation changes

### A. Canonical contract

- Fix the M-versus-N activation ownership mismatch.
- Centralize Q4/Q8 constants and layout helpers.
- Make Q8 sum handling explicit and test both derived and supplied forms if
  both are retained.
- Make edge tiles and divisibility requirements explicit in the descriptor.
- Ensure every layout field consumed by validation is consumed by lowering.

### B. Logical descriptor and candidate identity

- Add typed logical axis, operation, staging, synchronization, and ownership
  descriptors.
- Serialize them deterministically.
- Include complete canonical descriptor, vocabulary schema, emitter/lowering
  version, target capability, and backend in candidate identity.
- Enumerate legal geometry and schedule choices from descriptor axes; do not
  reject all non-tile choices merely because the old emitter cannot lower them.

### C. Emitter and physical lowering

- Make the emitter accept the logical descriptor/candidate only.
- Lower logical axes to UOps; physical `gidx`/`lidx` usage must be generated
  from declared mapping data.
- Reject any candidate whose local dimensions are unused, whose barriers are
  non-uniform, or whose owner map is not one-to-one.
- Remove hidden fixed WMMA dimensions and hidden materialization/staging
  choices from the promoted candidate path.
- Keep raw ISA and inline assembly out of the implementation.
- Keep the bounded atom as a diagnostic oracle only until the generated path
  has independent correctness evidence.

### D. Harness, evidence, and route safety

- Compare descriptor ABI/launch metadata against final program metadata.
- Record source identity, binary identity, final launch geometry, resources,
  owner coverage, guard status, GPU health, correctness, fallback status, and
  candidate identity in one artifact.
- Require the complete evidence gate before timing; require timing plus the
  same-session direct-packed comparator before any research winner is recorded.
- Preserve direct-packed as default and rollback. No route branch or manifest
  promotion is allowed from this scope.

## Ownership boundaries for parallel work

- Contract owner: `extra/qk/mmq_abi.py`, `extra/qk/layout.py`, reference-facing
  Q4/Q8 helpers, and ABI/layout tests.
- Vocabulary owner: new logical MMQ descriptor modules and their tests.
- Emitter owner: `extra/qk/q4k_q8_mmq_emitter.py` and logical-to-UOp tests.
- Physical-lowering owner: generated harness, launch/metadata validation, and
  bounded emitted-candidate tests.
- Evidence owner: search, compile evidence, owner coverage, regression gates,
  and artifact schema/tests.
- Route-safety owner: registry, manifest, model route plan, and route tests.

Agents must not modify another owner's files without coordination. The route
owner must not promote MMQ. The physical-lowering owner must not silently fix
the probe by selecting a new handwritten schedule.

## Completion gates

The scope is complete only when all are true:

1. One canonical ABI/layout contract passes positive, negative, and edge-tile
   tests; M/N ownership is unambiguous.
2. The logical vocabulary can represent the bounded Q4_K/Q8_1 operation without
   embedding `lidx0`, lane zero, fixed workgroup shape, or fixed WMMA geometry.
3. A descriptor-generated emitter lowers the vocabulary to UOps and preserves
   identity, ABI, and launch metadata.
4. Static validation proves every physical local dimension is consumed,
   synchronization is uniform, and every output has exactly one owner.
5. The generated candidate compiles with complete resource metadata and no
   spills/scratch beyond policy.
6. On real AMD, the guarded candidate has no corruption or NaNs, matches the
   canonical reference, and passes GPU health checks.
7. Candidate and direct-packed timings are measured in the same session with
   all required preparation costs included.
8. The artifact is replayable and fail-closed; no incomplete result can be
   timed or promoted.
9. Existing 8B routes and direct-packed defaults remain unchanged.
10. MMQ remains research-only unless a later, separately authorized promotion
    decision satisfies the 14B scope.

If AMD hardware is unavailable, gates 5-7 remain blocked rather than being
replaced by host-only claims.
