# NV Q6_K deterministic optimization path review

Date: 2026-08-31  
Scope: fixed `M=512, N=4096, K=12288`, 170-owner persistent MMQ route  
Reference model: `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`

## Executive decision

Do not start by merging the two generated compute bodies. The next experiment is a corrected full-route direct-`dA` A/B. It keeps the 170-CTA owner geometry, the two existing physical bodies, the all-partials ABI, and the standalone fixup unchanged. It changes only where `dA` enters the FP32 recurrence.

This reverses the current experiment order for three reasons:

1. The experiment called "no-dA" in the depth ledger does not omit `dA`. It implements the mathematically equivalent direct form `acc = fma(dot, round(dA*dB), acc)`. The current full route is hard-coded to the factored form `tmp = fma(dot, dB, tmp); acc = fma(tmp, dA, acc)`.
2. The direct depth arm reduced the robust K256 slope from `5.7417` to `5.2064 us/K256`, and reduced the depth-37 stack from `256 B` to `32 B`. That is measured evidence that accumulation placement, rather than static body duplication alone, is creating much of the local-memory pressure.
3. The current factored full route is not a valid correctness anchor against the trusted direct reference: maximum absolute error is `0.1871337890625`, mean absolute error is `0.01368770468980074`, and `allclose(rtol=2e-5, atol=2e-3)` is false. Structural optimization against that result can preserve the wrong recurrence perfectly.

Direct and factored `dA` placement are equal over real arithmetic, but not in FP32. Direct placement can repair, leave unchanged, or worsen the current `0.1871` error. No existing artifact answers which: the depth fixtures use power-of-two scales (`dA=0.03125`, `dB=0.0625`) that conceal the rounding difference. Gate 0 below measures model weights and adversarial non-power-of-two scales, and reports the signed change in maximum error, mean error, and failing-element count.

The "one logical body" idea is only partly reflected in llama. The pinned source has one logical `mul_mat_q_process_tile` template, but the cubin contains two distinct physical SASS bodies. A one-physical-body kernel is therefore a tinygrad compiler-pressure hypothesis, not a llama parity requirement. It is Gate 2, after correctness and direct-`dA` resource effects are known.

## Evidence notation

- **[P] Proven:** directly established by source, disassembly, exact output comparison, or schedule arithmetic.
- **[M] Measured:** established for the named harness and artifact, but not automatically transferable to another route.
- **[I] Inferred:** best explanation consistent with the measurements; requires an isolating A/B.
- **[U] Unknown:** no qualifying experiment or reproducible provenance exists.

## Corrections to the existing narrative

1. **[P] The depth "no-dA" arm is direct `dA`, not missing `dA`.** Its output is exact on the synthetic depth fixtures. Rename the mode in code and artifacts before drawing further conclusions.
2. **[P] One physical SASS body is not a llama property.** The pinned cubin has a direct body at approximately `0x0d80..0xeb50` and a partial body at approximately `0x12880..0x20780`. They contain 3,550 and 3,568 instructions respectively, including their backedges.
3. **[P] The existing one-body screen does not preserve geometry.** It launches `grid=(170,2)` and therefore 340 CTAs, with each CTA selecting at most one segment. The production route launches 170 owner CTAs and lets an owner process its one or two segments. The screen's `413.696 us` versus `284.320 us`, with `0/31` wins, rejects that 340-CTA implementation only.
4. **[P] The pinned llama source stores Q6 `d` converted to FP32 in the shared Q6 word.** The current broad oracle stores the raw FP16 bits zero-extended and converts at consumption. The normalized contract that describes these as identical is incorrect.
5. **[P] Llama's standalone fixup walks scratch contributors backward and then adds the destination value.** The broad oracle walks its slots forward. With at most three total contributors, those implementations have the same effective binary association for the ordinary finite case: two partials are added to each other before the destination. This literal order difference is therefore not a persuasive explanation for `0.1871`.
6. **[P] Some saved artifacts have schema or prose drift.** One ownership artifact labels segment lengths as owner lengths, and the saved single-body result has a `passed` field inconsistent with its prose. Gate 0 must regenerate results and derive verdicts from the new JSON, not copy old verdict fields.
7. **[U] The source-to-cubin build is not reproducible from the repository alone.** The main `mmq.cuh` hash agrees with the pinned copies, and the instruction structure strongly corroborates the mapping, but the repository lacks the complete MMA include provenance, transformation path for the `dense_mul_mat_q*` symbols, compiler flags, and toolchain recipe.

## Fixed-shape ownership facts

These are invariants for every geometry-preserving experiment in this review:

| Quantity | Value | Status |
|---|---:|---|
| Output tiles | `4 * 32 = 128` | [P] |
| K256 epochs per tile | `12288 / 256 = 48` | [P] |
| Total tile-epochs | `128 * 48 = 6144` | [P] |
| Owner CTAs | `170` | [P] |
| Owner interval | `[floor(b*6144/170), floor((b+1)*6144/170))` | [P] |
| Owner lengths | 146 owners of 36; 24 owners of 37 | [P] |
| Owners touching one/two tiles | 46 / 124 | [P] |
| Total owner-tile segments | 294 | [P] |
| Tiles with two/three contributors | 90 / 38 | [P] |
| Destination-ending segments | 128 | [P] |
| Preceding partial segments | 166 | [P] |

The two slot ABIs currently in the tree are not interchangeable:

- The broad oracle uses tile coordinates `(tile % 4, tile // 4)` and plane-major slots `slot = segment * 170 + owner`.
- The generated owner route uses tile coordinates `(tile // 32, tile % 32)` and owner-major slots `slot = owner * 2 + segment`.

Every result artifact must record the coordinate convention, slot layout, contributor order, and direct-versus-partial output policy.

## Current hop-by-hop execution map

### Tinygrad broad route

| Hop | Current operation | Evidence and consequence |
|---|---|---|
| 1. Dispatch | Launch 170 owner CTAs. CTA `b` owns the exact interval above and processes one or two tile segments serially. | [P] Correct persistent geometry. |
| 2. Segment routing | The generated kernel contains two statically duplicated K256 bodies for segment 0 and segment 1. | [P] Whole-kernel selected census is 512 IMMA and 64 LDSM. |
| 3. Q6 publication | Per physical body: 69 global loads and 35 shared stores for the exact Q6_K publisher. `d` is stored as raw FP16 bits and converted by consumers. | [P] Publication count matches the intended packed traffic, but `d` representation differs from llama. |
| 4. Q8 half 0 | Publish 18 loads / 18 shared stores, barrier, then consume the first K128 half. | [P] The initial Q6 and Q8 publications use separate barriers. |
| 5. Q8 half 1 | Panel-1 global loads are issued before all half-0 compute, remain live for a long interval, then are stored after the half-0 barrier; consume the second half. | [P] This is nominal prefetch but not a short, late overlap. Prior normalized span is about 1,400 instructions. |
| 6. Arithmetic | Current full route is factored: four `tmp = fma(float(dot_p), dB_p, tmp)` updates, followed by `acc = fma(tmp, dA, acc)` for each affected accumulator. Epochs are visited in ascending K order within each segment. | [P] Per physical body selected census is 640 FFMA. |
| 7. Synchronization | Five barriers per physical body: Q6 publish, Q8 half 0 publish, half 0 consume, Q8 half 1 publish, half 1 consume. | [P] Whole-kernel count is 11 because of surrounding control. |
| 8. Main output | Every segment, including each destination-ending segment, writes a 128x128 FP32 tile to the partial buffer. Total: 294 tile writes. | [P] Plane-major slot ABI. |
| 9. Fixup | A separate kernel reads all 294 partial tiles in the CPU-map order and writes 128 destination tiles. GPU fixup is bit-exact to the CPU slot recurrence. | [P] Correct implementation of its declared recurrence, but that recurrence is not close enough to the trusted reference. |

Measured whole-route state:

| Metric | Tinygrad broad | Pinned llama | Gap |
|---|---:|---:|---:|
| Main median | `285.600 us` | `201.216 us` | `84.384 us` |
| Fixup median | `25.600 us` | `8.640 us` | `16.960 us` |
| Pair median | `311.360 us` | `209.856 us` | `101.504 us` |
| Main static instructions | 8,328 | 8,648 | Tinygrad is smaller |
| Registers | 255 | 255 | equal ceiling |
| Stack | 288 B | 72 B | +216 B |
| LDL / STL | 251 / 377 | 31 / 29 | +220 / +348 |
| IMMA / LDSM | 512 / 64 | 512 / 64 | equal selected work |
| LDG / STS | 210 / 142 | 210 / 143 | effectively equal selected traffic |
| FFMA | 1,280 | 1,280 | equal selected count |
| BAR | 11 | 9 | +2 |

The main gap is therefore not explained by total static instruction count, IMMA count, LDSM count, or selected global/shared traffic. Local-memory pressure and instruction scheduling remain the leading differences.

### Pinned llama route

| Hop | Pinned operation | Evidence and consequence |
|---|---|---|
| 1. Ownership | The fixed-shape launch realizes the same 170-way Stream-K ownership and 128 destination tiles. | [P] Schedule/cubin launch evidence; retain the arithmetic formula rather than depending on the cubin. |
| 2. Logical body | One `mul_mat_q_process_tile` source template is instantiated for direct and partial output paths. The relevant template/calls are in `mmq.cuh` around lines 3446, 3710, and 3779. | [P] One source template does not imply one compiled body. |
| 3. Physical body | Direct and partial paths compile to separate, non-identical SASS regions. Each selected region has 256 IMMA, 32 LDSM, 105 LDG, 71 STS, 4 BAR, 176 LDS, 512 I2FP, 640 FFMA, 80 PRMT, 205 LOP3, and 1,083 IMAD. | [P] The partial body uniquely contains 13 LDL and 6 STL in the selected census. |
| 4. Publication | Q6 and the first Q8 panel are published before one common barrier. Q6 `d` is converted to FP32 before its shared store. | [P] Four barriers per physical body rather than five. |
| 5. Prefetch | The second Q8 panel is loaded late enough to overlap the tail of half-0 compute; normalized load-to-store distance is about 122 instructions. | [I] Strong disassembly evidence; exact source-to-binary scheduling recipe is unavailable. |
| 6. Arithmetic | The selected body has the factored 640-FFMA shape. | [P] This is reflected in the pinned cubin, but it does not make factored arithmetic the correct reference contract for tinygrad. |
| 7. Main output | The 128 destination-ending segments write destination tiles directly. The 166 preceding segments write scratch partials. | [P] This is the principal output-policy difference from the broad route. |
| 8. Fixup | The separate fixup walks preceding block IDs downward, accumulates their scratch values, then adds the destination value. Relevant source is around `mmq.cuh:3823-3853` and `mmq.cuh:3888`. | [P] Safe standalone reduction, with no in-main global spin. |

Llama's shared layout also contains a 512-byte ID region that the broad oracle omits. Its dynamic payload is 57,856 bytes and its physical total includes a separate 1,024-byte static region. The broad oracle uses a 57,344-byte payload plus 1,024 static bytes. Copying llama's physical offsets without copying its ABI would be incorrect.

## Ranked causal gaps

### 0. Unqualified numerical anchor

- **Magnitude:** current maximum absolute error `0.1871337890625`; mean `0.01368770468980074`; trusted `allclose` false.
- **Confidence:** [P] for the failure; [U] for its dominant cause.
- **Why first:** an exact structural A/B can faithfully optimize a recurrence that is already outside the reference contract.
- **Isolation:** Gate 0 direct-versus-factored full route, followed only if necessary by the 128-CTA tile-aligned and captured-partial association diagnostics.

### 1. FP32 `dA` placement and resulting live ranges

- **Magnitude:** depth robust slope `5.7417 -> 5.2064 us/K256`, about 9.3%; depth-37 stack `256 -> 32 B`; representative local operations fall from roughly 64 LDL / 128 STL to roughly 9 LDL / 14 STL.
- **Confidence:** [M] causal on depth kernels; [U] on the full 170-owner route.
- **Potential:** extrapolating the slope delta across 48 epochs gives about 25.7 us, but this is only a prioritization estimate, not a predicted full-route win.
- **Challenge to prior assumption:** if the full direct route reaches a small stack and local-op census while retaining two physical bodies, body deduplication may no longer be the highest-value compiler change.

### 2. Full-route local-memory pressure and schedule quality

- **Magnitude:** broad main has 216 more stack bytes, 220 more LDL, and 348 more STL than llama, while selected work/traffic counts are essentially equal. The main timing gap is 84.384 us.
- **Confidence:** [P] for the resource/count gap; [I] that it dominates latency.
- **Isolation:** first direct `dA`; then a true 170-CTA one-body A/B; then late-prefetch placement. Do not combine them.

### 3. Long-lived second Q8 panel preload

- **Magnitude:** normalized load-to-store span is approximately 1,400 instructions in the broad body versus approximately 122 in llama; selected Q8 load/store counts are already equal.
- **Confidence:** [P] for broad dependence structure and counts, [I] for exact llama overlap, [U] for timing benefit in tinygrad.
- **Isolation:** move only the dependency placement of the same 18 loads. Keep arithmetic, four/five-barrier mode, and all traffic counts fixed.

### 4. Reduction/output policy

- **Magnitude:** the measured final-participant experiment changes `311.936 us` control to `294.368 us` embedded plus `1.504 us` counter reset, or `295.936 us` total. It wins 31/31 and recovers 15.744 us including reset. It removes `128 * 65536 * 2 = 16,777,216` bytes of final-partial write/read traffic.
- **Confidence:** [M] for the measured candidate; [P] that the current parent still fails the trusted reference; [U] for deadlock-free deployment without an explicit residency/cooperative-launch guarantee.
- **Remaining gap:** `295.936 - 209.856 = 86.080 us`; reduction alone cannot close the route.
- **Safe alternative:** llama-shaped direct-destination main plus standalone fixup. It removes final-partial scratch traffic but substitutes destination read/write traffic, so it does not automatically reduce aggregate bytes. It does avoid counters and spin waits.

### 5. Initial publication barrier and Q6 `d` representation

- **Magnitude:** combined publication changes five body barriers to four. The combined-initial factored depth slope was `5.6739` versus `5.7417`, about 1.18% better, while an isolated one-epoch 170-CTA case regressed about 1.7%.
- **Confidence:** [M] and explicitly workload-dependent.
- **Q6 `d`:** moving FP16-to-FP32 conversion to publication is [P] llama-shaped and may shorten/reduce consumer conversion live ranges, but its full-route benefit is [U]. It should not change numerical values because every FP16 value is exactly representable in FP32.

## Arithmetic contracts and pseudocode

### Direct versus factored `dA`

`dot[p]` below denotes the existing, unchanged integer correction/IMMA result for one output accumulator and one `p` group. Its construction order must not change in Gate 0.

```text
# Direct dA placement: current factor_dA=False depth arm.
# round32 is explicit FP32 rounding; fma32 is one rounded FP32 FMA.
for epoch in ascending_segment_k256_order:
  for half in [0, 1]:
    for cg in [0, 1, 2, 3, 4, 5, 6, 7]:
      for n in [0, 1]:
        for p in [0, 1, 2, 3]:
          scaled = round32(dA * dB[p])
          for r in [0, 1, 2, 3]:
            acc[cg,n,r] = fma32(float32(dot[cg,n,p,r]),
                                scaled,
                                acc[cg,n,r])
```

```text
# Factored dA placement: current full-route parent and llama-shaped census.
for epoch in ascending_segment_k256_order:
  for half in [0, 1]:
    for cg in [0, 1, 2, 3, 4, 5, 6, 7]:
      for n in [0, 1]:
        for r in [0, 1, 2, 3]:
          tmp = +0.0f
          for p in [0, 1, 2, 3]:
            tmp = fma32(float32(dot[cg,n,p,r]), dB[p], tmp)
          acc[cg,n,r] = fma32(tmp, dA, acc[cg,n,r])
```

Do not let algebraic simplification rewrite either contract. Express the intended fused operations with the existing `Ops.MULACC`/FMA representation and make the intervening FP32 multiply in direct mode explicit. Expected normalized selected counts per physical body are:

| Mode | Selected FP arithmetic per body |
|---|---:|
| Direct | 512 FMUL + 512 FFMA |
| Factored | 640 FFMA |

The A/B must census the emitted SASS rather than trusting these expectations.

### Geometry-preserving one-physical-body kernel

This is the Gate 2 target, not the existing 340-CTA screen:

```text
kernel q6_streamk_one_body(...):
  owner = blockIdx.x                         # grid = (170, 1, 1)
  lo = floor(owner * 6144 / 170)
  hi = floor((owner + 1) * 6144 / 170)

  tile0 = floor(lo / 48)
  segment_count = 1 + int(floor((hi - 1) / 48) != tile0)

  for segment in runtime_range(0, segment_count):
    tile = tile0 + segment
    begin = max(lo, tile * 48)
    end = min(hi, (tile + 1) * 48)
    kb0 = begin - tile * 48
    depth = end - begin

    reset_all_register_accumulators_to_positive_zero()

    for e in runtime_range(0, depth):
      kb = kb0 + e

      publish_exact_q6_panel(kb)
      block_barrier()

      publish_exact_q8_half0(kb)
      block_barrier()
      consume_half0_with_gate0_arithmetic()
      block_barrier()

      publish_exact_q8_half1(kb)
      block_barrier()
      consume_half1_with_gate0_arithmetic()
      block_barrier()

    # Gate 2 retains the broad plane-major all-partials ABI.
    slot = segment * 170 + owner
    store_partial_tile(slot, tile, accumulators)
```

Important invariants:

- The outer launch is exactly 170 CTAs. Do not encode `segment` in `blockIdx.y`.
- A CTA with two segments executes the same physical body twice and resets its accumulators between segments.
- Gate 2 retains five body barriers. Combining initial Q6/Q8 publication is a later, separate A/B.
- Gate 2 retains the exact Gate 0 arithmetic and output/fixup policy, so its partial buffer and final output must be bit-identical to the anchor.

### Nested runtime RANGE compiler fix

The failure is not missing `Ops.END` support: `Ops.END` is already represented in the range-start machinery. Two unsafe transforms interact:

1. Flattening an ended dynamic inner range traverses into its enclosing outer `RANGE` and mistakes the outer range for part of the inner lifecycle.
2. Adjacent-range merging permits extents that depend on a candidate range. Merging a non-rectangular inner extent with its outer range creates a self-dependent product/divmod expression and later a `KeyError`.

The minimal conceptual fix in `tinygrad/codegen/simplify.py` is:

```text
function ranges_for_ended_region(end_uop):
  work = operands_that_define_end_value(end_uop)
  found = ordered_set()
  while work not empty:
    u = work.pop()
    if u.op == RANGE:
      found.add(u)
      continue                 # RANGE is a lexical traversal boundary
    work.extend(u.sources)
  return found

function ranges_are_rectangular_merge_candidates(r0, r1):
  if r0 occurs in ranges(r1.extent): return false
  if r1 occurs in ranges(r0.extent): return false
  if lifetimes_are_nested_or_separated_by_END(r0, r1): return false
  return existing_axis_and_adjacency_checks(r0, r1)
```

The regression test must construct an outer runtime segment range, an inner runtime depth whose extent depends on the segment, a register accumulator reset once per outer iteration and carried across the inner range, and explicit nested `END`s. Acceptance is no `KeyError`, two preserved loop lifecycles in rendered code, and exact output. Assigning distinct `AxisType.LOOP`/`AxisType.REDUCE` axes can be a fail-fast prototype, but it is not a substitute for making dependent ranges safe.

### Reduction choices

The all-partials recurrence is the conservative Gate 0/Gate 2 anchor:

```text
for tile in 0..127:
  out = +0.0f
  for slot in declared_contributor_order(tile):
    out = add32(out, partial[slot])
  destination[tile] = out
```

The llama-shaped standalone policy is the default production candidate:

```text
# Main
if segment_ends_at_tile_boundary:
  destination[tile] = segment_accumulator
else:
  scratch[segment_slot] = segment_accumulator

# Separate fixup, exact declared order
for tile in 0..127:
  sum = +0.0f
  for slot in preceding_scratch_slots(tile) in descending_owner_order:
    sum = add32(sum, scratch[slot])
  destination[tile] = add32(sum, destination[tile])
```

The final-participant candidate is admitted only with a scheduling-safety proof:

```text
# Each owner computes/publishes every non-final segment before any wait.
for segment in owner_segments_reordered_nonfinal_first:
  acc = compute(segment)
  if not segment_ends_at_tile_boundary:
    scratch[fixed_slot(segment)] = acc
    device_release_fence()
    atomic_increment(ready_count[tile])
  else:
    live_final = acc
    wait_until_acquire(ready_count[tile] == predecessor_count[tile])
    for slot in fixed_predecessor_order(tile):
      live_final = add32(live_final, scratch[slot])
    destination[tile] = live_final
```

Reordering non-final work before waits removes owner-local dependency cycles, but it does not prove grid-wide progress. A spinning final CTA can occupy resources needed by an unscheduled producer CTA. Promotion therefore requires a cooperative/residency guarantee, a nonblocking work protocol, or a proof that all producer CTAs required by every resident waiter can run. Counter reset time and graph ownership are part of end-to-end timing.

## Sequential experiment decision tree

### Common qualification protocol

Every gate uses:

- One semantic variable per A/B.
- Fixed shape, weights, ownership formula, coordinate ABI, and named arithmetic/reduction contract.
- Same-process alternating `AB/BA` timing with warmup, 31 paired rounds, GPU lock, and recorded launch order.
- Repeated-run bit stability and finite-output checks.
- Trusted direct-reference metrics: maximum absolute error, mean absolute error, maximum relative error, failing-element count, and `allclose(rtol=2e-5, atol=2e-3)`.
- When the recurrence is intentionally unchanged: bit-exact partial buffers and bit-exact final output versus the admitted anchor.
- Cubin hash, full resource record, whole-kernel SASS census, per-physical-body normalized census, and branch-span report.
- Timing promotion only after correctness. The default investment threshold is paired median improvement at least `3.0 us` (and greater than `3 * paired MAD` if that is larger), at least 24/31 wins, with no unexplained stack/local-op regression.

Final performance admission is stricter than intermediate investment:

- Main median at most `1.05 * 201.216 = 211.2768 us`.
- Main+fixup/reset median at most `1.05 * 209.856 = 220.3488 us`.
- No runtime or build dependency on a llama cubin.

### Gate 0: corrected full-route direct-`dA` A/B

Control and candidate both use `grid=(170,1,1)`, two static segment bodies, five body barriers, plane-major all-partial output, 294 partial writes, and the same standalone fixup. The sole code difference is the direct versus factored recurrence shown above.

Qualification inputs:

1. The real Qwen model inputs used by the trusted reference.
2. A deterministic adversarial fixture with non-power-of-two, mixed-magnitude `dA`/`dB` values. Do not reuse the power-of-two depth fixture as the discriminator.

Required report:

- Direct and factored maximum/mean/relative error and failure count.
- `direct_max_abs - 0.1871337890625` and `direct_mean_abs - 0.01368770468980074` on the matching model case.
- Candidate-versus-control difference metrics; they are not required to be bit-equal because this gate deliberately changes FP32 association.
- CPU slot-recurrence exactness for each mode independently.
- Per-body expected/actual direct versus factored FP instruction census, plus stack, registers, LDL/STL, and timing.

Decision:

- If only direct passes the trusted reference, direct becomes the anchor.
- If both pass, admit the faster mode only when it clears the timing threshold; otherwise choose the lower-stack/lower-local-op mode and record the tie.
- If direct reduces error but still fails, continue to Gate 0A. Do not promote a tolerance change silently.
- If direct worsens error, keep both artifacts and continue to Gate 0A; the result disproves direct placement as the complete fix, not the depth resource measurement.
- If neither passes, no structural performance candidate can be promoted.

### Gate 0A: separate body arithmetic from Stream-K association

Run a 128-CTA tile-aligned direct route, one full 48-epoch tile per CTA, with no scratch/fixup.

- If it fails, localize one real-scale K256 recurrence against the trusted implementation. The fault is in publisher/layout/integer correction/arithmetic, not cross-CTA reduction.
- If it passes while the 170-owner route fails, capture the 2/3 contributor partials and evaluate all three FP32 binary associations for a three-contributor tile offline: `(p0+p1)+d`, `p0+(p1+d)`, and `(p0+d)+p1`.
- If one association passes, encode that exact contributor order in the ABI and rerun Gate 0.
- If none passes, segment subtotal rounding is the mismatch. Evaluate a two-component/compensated partial contract or fall back to tile-aligned ownership. Changing the trusted tolerance requires an explicit product decision.

This branch also tests the prior fixup-order hypothesis. Because current forward and llama backward-then-destination order already reduce to `(p0+p1)+d` for three contributors, merely reversing the two partial reads is expected to be a no-op numerically.

### Gate 1: nested runtime RANGE substrate

Add the dependent-range regression first, then make only the `simplify.py` lifecycle/dependency fix. Run the targeted range tests. Stop if independent rectangular merge behavior regresses or if the generated program does not retain two distinct loops.

### Gate 2: true 170-CTA one-physical-body A/B

Generate the pseudocode above against the exact Gate 0 anchor.

Promotion requirements:

- Both launch exactly 170 CTAs.
- Static selected body census is 256 IMMA and 32 LDSM, not 512/64.
- Partial buffer and fixed output are bit-identical to Gate 0.
- Same dynamic epochs, traffic, arithmetic, five-barrier contract, and slot ABI.
- Timing clears the common threshold.

Branch:

- If direct Gate 0 already has stack at or below roughly 86 B and LDL/STL near llama's 31/29 whole-kernel range, deprioritize Gate 2 unless compile size itself is a product problem.
- If Gate 2 is exact but not faster, stop treating one-body deduplication as a performance path. Keep the compiler fix because nested runtime ranges are independently valid substrate.
- If it wins, it becomes the structural anchor.

### Gate 3: late second-panel Q8 prefetch

Keep the Gate 2 body and counts fixed. Issue the same 18 panel-1 global loads only after a named late half-0 accumulator dependency, finish independent half-0 work, then join the load token and half-0 completion before the shared stores/barrier.

```text
half0_token = consume_half0_through_cg6()
panel1_regs = load_q8_panel1(after=half0_token)    # exactly 18 LDG
half0_done = consume_remaining_independent_half0()
store_q8_panel1(after=join(panel1_regs, half0_done)) # exactly 18 STS
block_barrier()
consume_half1()
```

Require exact anchor output, unchanged LDG/STS/IMMA/LDSM counts, normalized load-to-store span below 160 instructions, no stack/LDL/STL regression, and the timing threshold. If the renderer cannot retain the dependence placement, stop and fix scheduling observability rather than adding CUDA text.

### Gate 4: Q6 `d` publication representation

Convert the FP16 `d` value once in the publisher and store its FP32 bits in the shared Q6 word; consumers load FP32 directly. Change no offset, traffic count, arithmetic association, or barrier.

Require bit-exact anchor output, conversion movement visible in normalized SASS, no increased global/shared traffic, and lower or equal local operations. Admit on the common timing threshold; otherwise retain the simpler/lower-resource representation.

### Gate 5: combined initial Q6+Q8 publication

Publish Q6 and Q8 half 0 before a common first barrier, reducing five body barriers to four. This is a separate A/B because prior depth and one-epoch measurements disagree.

Require exact output, per-body BAR reduction from 5 to 4, unchanged data traffic and arithmetic census, no resource regression, and the timing threshold.

### Gate 6: reduction policy

Against one exact main anchor, compare:

1. Current all-partials plus standalone fixup.
2. Llama-shaped direct-destination plus standalone backward-contributor fixup.
3. Final participant, including counter reset and all synchronization state.

All arms use the same declared contributor association. Require exact identical destination bits, R31 end-to-end timing, reset included, and graph-owned state. Prefer the standalone policy if it is within 3 us of final participant because it has no spin/residency hazard. Admit final participant only if it wins materially beyond that margin and has an explicit progress guarantee.

### Gate 7: counter-guided residual work

Only after the normalized selected counts, correctness, and resource gates pass, collect dynamic instruction, local-memory, and cache counters. Continue only on the measured dominant category. Do not reopen previously rejected generic publication, LDSM, cache, buffering, or owner-count hypotheses without a new causal discrepancy.

## Smallest backend-neutral substrate

### Required core changes

| File | Minimal responsibility |
|---|---|
| `tinygrad/codegen/simplify.py` | Treat nested `RANGE` as a lexical boundary while flattening ended regions; prohibit merging mutually/dependently sized or lifecycle-nested ranges. |
| `test/unit/test_nested_runtime_range_lifecycle.py` | Reproduce the outer-segment/inner-dependent-depth accumulator lifecycle and require two loops plus exact output. |
| `tinygrad/codegen/opt/stream_k.py` | Express owner interval and one/two-segment descriptors with UOp-compatible runtime integer formulas, not only Python integers. |
| `tinygrad/codegen/opt/persistent_accumulator.py` | Make slot layout, tile-coordinate convention, contributor order, and direct/partial output policy explicit data rather than implicit list order. |

### Experiment and integration targets

| File | Change boundary |
|---|---|
| `extra/llm_research/prefill/nv_q6_oracle_broad_cta.py` | Add an explicit arithmetic enum; later add the runtime segment loop, late-prefetch dependency split, and FP32-`d` publication as independently selectable options. |
| `extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py` | Own Gate 0 same-process A/B, trusted-reference reporting, artifact schema/versioning, and normalized SASS capture. |
| `extra/llm_research/prefill/bench_nv_q6_oracle_single_body_experiment.py` | Replace the 340-CTA screen with a 170-versus-170 comparison after Gate 1. |
| `extra/llm_research/prefill/bench_nv_q6_final_participant_fixup.py` | Add the safe llama-shaped standalone arm and state the final-participant progress precondition. |
| `extra/llm_research/prefill/nv_q6_streamk_fixup.py` | Consume an explicit map ABI and predicate/gate invalid slots before constructing loads. |
| `extra/llm_research/prefill/bench_nv_generated_q6k_streamk_owner.py` | Wire generated main and generated fixup into one trusted-reference route rather than comparing only 170-CTA partials with a 340-CTA descriptor control. |

No new CUDA-only matrix fragment primitive is justified. The current renderer already emits the required IMMA/LDSM path; dependency grouping, barriers, and `Ops.MULACC` exist. The durable changes are range lifecycle, runtime schedule description, and explicit accumulator/reduction ABI. Remove post-render CUDA source splicing only after these pieces can express the full route.

The generated fixup needs special care: its test coverage is currently structural, its expected tile-to-slot map differs from the main route's slot-to-tile IDs, and an invalid slot load may be constructed before a `where` guard. It is not a qualified end-to-end replacement until those contracts are reconciled and executed against the trusted reference.

## Low-effort-agent task packets

The commands below define the intended CLI contract after each task adds its named flag. Each task changes one semantic variable and emits a self-contained result directory.

### Task A: Gate 0 direct-`dA` full-route A/B

- **Inputs:** broad source/harness, Qwen GGUF, current 170-owner schedule, trusted direct reference.
- **Files:** `nv_q6_oracle_broad_cta.py`, `bench_nv_q6_oracle_full_streamk.py`.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py --rounds 31 --arithmetic-ab direct,factored --model /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf --out docs/task_workflow/input/nv-q6-direct-da-full-route-20260831/result.json --artifacts docs/task_workflow/input/nv-q6-direct-da-full-route-20260831/artifacts'
```

- **Outputs:** versioned JSON, both CUDA sources/cubins/SASS, per-body normalized census, full error vectors/statistics, alternating samples.
- **Acceptance:** both arms use `(170,1,1)` and identical non-arithmetic contracts; independent CPU-recurrence exactness; explicit trusted-reference verdict; delta from current max/mean error; no stale copied verdict.

### Task B: dependent nested RANGE regression/fix

- **Inputs:** minimized two-level runtime loop reproducer.
- **Files:** `tinygrad/codegen/simplify.py`, new range lifecycle test.
- **Command:**

```bash
PYTHONPATH=. .venv/bin/python -m pytest -q test/unit/test_nested_runtime_range_lifecycle.py test/unit/test_generic_tc_split_range_axis.py test/unit/test_rangeify_multireduce.py
```

- **Outputs:** focused patch and test result.
- **Acceptance:** no `KeyError`; two loop lifecycles render; exact accumulator reset/carry output; existing selected range tests pass.

### Task C: true one-body 170-versus-170

- **Inputs:** admitted Gate 0 mode and Gate 1 compiler support.
- **Files:** broad source and single-body harness.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_single_body_experiment.py --rounds 31 --geometry 170-owner --anchor docs/task_workflow/input/nv-q6-direct-da-full-route-20260831/result.json --out docs/task_workflow/input/nv-q6-one-body-170cta-20260831/result.json'
```

- **Outputs:** launch trace, partial/final exactness hashes, artifacts/census, paired timing.
- **Acceptance:** 170 CTAs in both arms; 256 IMMA/32 LDSM static candidate; bit-exact anchor; at least 3 us and 24/31 wins to promote.

### Task D: late Q8 prefetch

- **Inputs:** admitted structural anchor.
- **Files:** broad source/harness only.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py --rounds 31 --prefetch-split-ab current,late-cg6 --out docs/task_workflow/input/nv-q6-late-prefetch-20260831/result.json --artifacts docs/task_workflow/input/nv-q6-late-prefetch-20260831/artifacts'
```

- **Outputs:** exactness, paired timing, normalized load/store positions and resource census.
- **Acceptance:** exact anchor; exactly 18 panel loads/stores; span below 160 instructions; no resource regression; timing threshold.

### Task E: Q6 `d` representation

- **Inputs:** admitted scheduling anchor.
- **Files:** broad source/harness only.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py --rounds 31 --d-publish-ab raw-f16-bits,fp32 --out docs/task_workflow/input/nv-q6-d-publish-20260831/result.json --artifacts docs/task_workflow/input/nv-q6-d-publish-20260831/artifacts'
```

- **Outputs:** exactness, conversion-location census, resource/timing samples.
- **Acceptance:** bit-exact anchor, unchanged traffic/offset ABI, conversion moves to publisher, common timing threshold.

### Task F: combined initial publication

- **Inputs:** admitted body anchor.
- **Files:** broad source/harness only.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_oracle_full_streamk.py --rounds 31 --combined-publish-ab separate,combined --out docs/task_workflow/input/nv-q6-combined-publish-full-20260831/result.json --artifacts docs/task_workflow/input/nv-q6-combined-publish-full-20260831/artifacts'
```

- **Outputs:** exactness, BAR/census/resources, timing.
- **Acceptance:** five-to-four body BAR change only; exact anchor; timing threshold.

### Task G: reduction policy A/B/C

- **Inputs:** one exact main kernel, explicit contributor map/order.
- **Files:** final-participant harness, fixup, persistent accumulator ABI.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_q6_final_participant_fixup.py --rounds 31 --reduction-ab all-partials,llama-standalone,final-participant --include-reset --out docs/task_workflow/input/nv-q6-reduction-policy-20260831/result.json'
```

- **Outputs:** exact destination hashes, scratch/output byte ledger, reset-inclusive timing, counter/progress contract.
- **Acceptance:** identical output bits across arms; explicit standalone order; final-participant progress proof; choose standalone within 3 us, otherwise require material final-participant win.

### Task H: generated-route end-to-end qualification

- **Inputs:** explicit schedule/slot ABI and admitted arithmetic/reduction contract.
- **Files:** generated owner harness, generated fixup, their focused tests.
- **Command:**

```bash
flock /tmp/nv-q6-oracle-gpu.lock -c 'PYTHONPATH=. .venv/bin/python extra/llm_research/prefill/bench_nv_generated_q6k_streamk_owner.py --full-route --trusted-reference --rounds 31 --out docs/task_workflow/input/nv-q6-generated-full-route-20260831/result.json'
```

- **Outputs:** generated main+fixup result, ABI dump, invalid-slot audit, cubin/SASS/timing artifacts.
- **Acceptance:** no hand-spliced CUDA compute/fixup, no invalid load before predicate, trusted-reference pass, exact admitted recurrence, final 5% performance thresholds.

## Stop list

Do not retry these as standalone optimizations without a newly measured causal gap:

- Generic contiguous Q8 publication.
- Generic lifetime separation or straight-line K256 rewrites.
- Spill elimination without preserving arithmetic/schedule semantics.
- LDSM substitution alone.
- Persistent Q6 cache.
- Double buffering.
- Owner-count tuning before the 170-owner route is qualified.
- The existing 340-CTA one-body selector.
- Reversing two partial reads as a proposed numerical fix.
- A tolerance increase presented as an implementation fix.

## Proven, inferred, and unknown conclusions

### Proven or directly measured

- The fixed shape decomposes into the exact ownership/segment counts listed above.
- Current broad main/fixup timing, error, SASS/resource, and traffic counts are measured for the named artifacts.
- Direct `dA` is exact on the synthetic depth fixtures and materially lowers their slope/local pressure.
- Current GPU fixup exactly implements its CPU slot recurrence.
- Final participant saves 15.744 us including reset in its measured harness and wins 31/31.
- The pinned llama cubin has two physical compute bodies despite one logical source template.
- Llama uses factored arithmetic, combined initial publication, FP32 shared `d`, direct destination-ending output, and standalone backward scratch fixup.
- Existing one-body timing is confounded by 340-CTA geometry and cannot reject the 170-CTA runtime-loop design.

### Inferred and deliberately tested by the sequence

- FP32 accumulation placement is a major source of broad-route local-memory pressure.
- A late panel-1 prefetch can shorten live ranges and reproduce the useful part of llama's schedule without copying its cubin.
- One physical body may help tinygrad only if pressure remains after direct `dA`.
- Direct-destination standalone reduction may retain most of the output-policy benefit with less risk than final participant.

### Unknown until the gates run

- Whether direct `dA` repairs or worsens the current `0.1871` direct-reference error on real model scales.
- Whether the full direct route reproduces the depth kernel's stack/local-op improvement.
- Whether segment subtotal association, rather than body arithmetic, prevents trusted-reference qualification.
- Whether a true 170-CTA one-body kernel is faster.
- Whether tinygrad can retain a useful late-prefetch schedule through rendering.
- Whether FP32 `d` publication or combined initial publication wins in composition.
- Whether final-participant waiting is safe on every target launch configuration.
- Whether the exact pinned llama cubin can be rebuilt from the available source; it currently cannot be reproduced from repository evidence alone.

## Deterministic path forward

The critical path is:

```text
Gate 0 direct-dA correctness/resource A/B
  -> if needed, Gate 0A tile-aligned and association localization
  -> freeze exact arithmetic/reduction anchor
  -> Gate 1 dependent nested RANGE fix
  -> Gate 2 true 170-CTA one-body A/B
  -> Gate 3 late Q8 prefetch
  -> Gate 4 FP32 d publication
  -> Gate 5 combined initial publication
  -> Gate 6 safe standalone versus final-participant reduction
  -> Gate 7 counter-guided residual only
  -> generated end-to-end route and 5% promotion gate
```

This path does not guess from wall time, does not treat llama's cubin as a runtime dependency, and does not conflate mathematical equivalence with FP32 equivalence. Each admitted step has one changed cause, a stable correctness anchor, a normalized binary signature, a resource signature, and an explicit branch or stop condition.
