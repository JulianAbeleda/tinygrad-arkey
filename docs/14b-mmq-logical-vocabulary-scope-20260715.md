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

## Beyond-parity continuation

The ten gates above establish a valid generated path; they are not the finish
line. After the first real-AMD canary reaches correctness parity, continue in
this order:

### P1 — Q4 role coverage

Search and validate independent candidates for `ffn_gate_up`, `ffn_down`,
`attn_qo`, and `attn_kv`. Each role needs its own full-output correctness,
physical/resource, identity, and same-session timing evidence. A passing
`ffn_gate_up` candidate cannot be reused as proof for another role.

### P2 — Q4 aggregate policy

Select one generated policy through the shared route manifest. Measure the
mixed Q4 route with all required Q8 preparation, packing, reductions, and
synchronization included. Compare against direct-packed in the same session;
individual kernel parity is insufficient if aggregate prefill regresses.

### P3 — Q6 vocabulary reuse

Reuse the common axes, staging, ownership, evidence, and route layers for Q6_K,
but define a separate Q6 decode grammar and reference. Q6 must not materialize
the full dequantized weight tensor as an unaccounted fallback. Validate Q6 roles
independently before mixed-quant integration.

### P4 — End-to-end 14B validation

Run the exact mixed-quant Qwen3-14B workload and record prefill tok/s, decode
tok/s, per-role timings, route census, output/token parity, memory use, Q8
preparation cost, GPU health, and fallback count. Compare with the authoritative
direct-packed baseline and the separately defined llama workload measurement.

### P5 — Beyond-parity performance gate

The generated route is beyond parity only when it is reproducibly faster than
direct-packed on the target role and improves the aggregate 14B prefill result
under the same measurement definition. Correctness parity, a faster isolated
contraction that excludes preparation, or a host-only result does not qualify.

### P6 — Promotion and closeout

Only after P1-P5 pass may the route be considered for promotion. Promotion must
record the machine-generated candidate identity, source/binary identity, final
resource and launch facts, role policy, rollback route, and reproducible
end-to-end artifact. Until then, direct-packed remains authoritative and every
MMQ failure must fail closed to research-only status.

## 2026-07-15 execution update

The shared logical candidate now lowers through the generated emitter, and the
host MMQ suite passes 350 tests. A real AMD gfx1100 bounded `16x16x256` canary
is finite and matches the canonical reference with maximum absolute error
approximately `2.44e-4`.

The first full-role-size `attn_kv` run (`M=512,N=1024,K=5120`) also completes
with finite output and maximum absolute error approximately `1.46e-3` against
the CPU reference. This establishes a correctness canary, not a promotion.

The same-session timing comparison currently measures approximately `155.7 ms`
median for generated scheduler plus Q8 preparation versus `28.8 ms` median for
direct-packed on that role. The generated path is therefore still below parity
and the next owning layer is the fused packed Q4_K/Q8_1 tile producer. No MMQ
route or default changed.

The next iteration adds an explicit `packed_ds4` logical candidate. Its mapping
declares a four-row micro-tile, one wave per workgroup, and supplied Q8 group
sums; the emitted graph rejects the scheduler candidate and cannot infer this
ABI. The GPU Q8 producer, transpose, and packed dot graph pass the bounded
reference gate and remain research-only.

On AMD gfx1100, the packed candidate including GPU Q8 preparation was measured
in the same session against direct-packed operands on all four Q4 roles:

| role | shape `(M,N,K)` | packed DS4 ms | direct-packed ms | ratio |
|---|---:|---:|---:|---:|
| `attn_kv` | `(512,1024,5120)` | 13.1 | 34.2 | 2.6x |
| `attn_qo` | `(512,5120,5120)` | 12.9 | 127.6 | 9.9x |
| `ffn_down` | `(512,5120,17408)` | 12.9 | 406.9 | 31.6x |
| `ffn_gate_up` | `(512,17408,5120)` | 12.9 | 421.6 | 32.6x |

All measured outputs were finite. This is role-level beyond-parity evidence,
not yet the P4 exact-model result: direct-packed timing still uses its existing
prepacked operand contract, so the aggregate gate must record preparation and
route census under one final workload definition before any promotion decision.

The installed `/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf` metadata confirms the
mixed-quant boundary: in layer zero, `attn_k`, `attn_q`, `attn_output`,
`ffn_gate`, and `ffn_up` are Q4_K while `attn_v` and `ffn_down` are Q6_K; the
inventory also records later-layer transitions and the Q6_K output head. A
real-weight Q4 canary using `blk.0.attn_k.weight` (`512x1024x5120`) is finite;
the first 16 output rows match the DS4 reference with maximum absolute error
about `1.05e-5`. The Q6 vocabulary therefore remains a separate required P4
workstream rather than an implicit Q4 fallback.

The real mixed tensors also execute through their existing direct Q6 grammar:
`blk.0.ffn_down.weight` (`512x5120x17408`) is finite at approximately `7.8 ms`
median on the existing Q6 packed direct-out kernel. A real `blk.0.ffn_gate`
Q4 run is finite at approximately `12.8 ms` median including DS4 preparation.
These are component measurements only; the exact mixed model route still lacks
full output/token parity, decode evidence, and lm-head coverage.

The new selector is route-bound as `prefill_q4k_packed_ds4_research` and was
run through the authoritative Qwen3-14B smoke with the existing Q6 route left
unchanged. The 512-token smoke completed at `10.04 s` (`50.99 tok/s`) versus
the direct-packed smoke at `9.43 s` (`54.28 tok/s`). This is a valid mixed-route
execution and route census, but it is below whole-model parity. On the same
real Q4 tensors, the direct kernel is about `6.1 ms` while packed DS4 including
Q8 preparation is about `13.2 ms`; the next owning layer is therefore a fused
Q8 producer with a new physical ownership proof. The opt-in route remains
default-off and no promotion is claimed. A one-entry graph-identity cache
reuses the packed activation for adjacent FFN gate/up consumers; the repeat
smoke improved to `9.85 s` (`51.96 tok/s`), but remains below the `9.43 s`
direct-packed smoke. The cache is therefore an aggregate mitigation, not a
parity result.

## 2026-07-15 descriptor geometry audit update

The packed DS4 lowering now has one shared geometry boundary. Q4 metadata
words, Q4 packed payload words, Q8 group width, packed DS4 width, groups per
DS4 block, lane packing, Q4 block strides, Q8 value indices, and metadata
indices are derived from the logical descriptor before UOp emission. The atom
rejects non-canonical descriptor grammars rather than silently interpreting
them with the old `4`/`8`/`32`/`128`/`36` assumptions. The generic emitter also
requires explicit ABI role, shape, weight/activation layouts, tile layouts,
staging, and writeback fields; it no longer reconstructs those from backend
defaults.

The audit found one producer correctness defect: supplied DS4 sums were being
computed from original FP32 activations while the Q4 minimum correction
requires weighted dequantized Q8 sums (`q * scale`). The producer now derives
those sums from the quantized values and scales, and a regression test pins
that invariant. Unsupported descriptor geometry is covered by a fail-closed
negative test.

Validation after the cleanup: 77 focused contract/emitter/route tests and 411
tests across the MMQ/Q4K-Q8/generated-harness slice pass. The descriptor-fed
AMD gfx1100 `16x16x256` generated canary is finite and matches the CPU
reference with maximum absolute error `2.44e-4`. This validates the lowering
contract and producer correctness only; Q8 preparation remains the measured
whole-model bottleneck, so the route stays research-only and direct-packed
remains default.

## 2026-07-15 row-major producer continuation

The next bounded optimization keeps the logical DS4 operation and atom but
declares Q8 activation storage explicitly as `row_major`. This removes the
values/scales DS4 transpose materializations; the supplied weighted sums remain
an explicit operand, so the proven minimum-correction algebra is unchanged.
At a representative `(M,N,K)=(512,512,5120)` shape, preparation decreased
from about `9.54 ms` to `7.86 ms`, while the contraction remained about `5.2`
to `5.6 ms`. The row-major candidate is available only through the separate
`PREFILL_Q4K_Q8=packed_row_major` research selector.

The first same-definition 14B smoke was `6.44 s` (`79 tok/s`) for row-major,
`6.71 s` (`76 tok/s`) for packed DS4, and `4.93 s` (`104 tok/s`) for the
direct-packed rollback. The local producer improvement therefore did not reach
parity. It is retained as an isolated research candidate while the next
performance owner remains a fused/reused Q8 producer plus a multi-output tile
mapping; no route promotion is implied.

## 2026-07-15 fused producer continuation

The fused producer was extended to a compact metadata ABI with staged
cross-lane reduction and a single-owner lane-0 store for scales and weighted
sums. Its bounded output matches the stable row-major producer within
`3.1e-5`, and the compact producer measures about `3.55 ms` with the packed
contraction at `5.28 ms` for `(512,512,5120)`.

The same-definition 14B smoke measured `6.14 s` (`83 tok/s`) for this fused
candidate versus `4.93 s` (`104 tok/s`) for direct-packed. It is the fastest
MMQ research variant tested in this continuation, but still below parity; the
selector remains default-off and direct-packed remains authoritative. The
next beyond-parity owner is now the output-tile/activation-reuse mapping, not
another unmeasured Q8 sum representation.
