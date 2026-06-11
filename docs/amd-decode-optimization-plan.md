# AMD Decode Optimization — Execution Plan

Executable plan for closing tinygrad's decode-speed gap vs llama.cpp/ROCm on
gfx1100. Hypothesis derivation and measured baselines live in
`docs/amd-rocm-llamacpp-research.md` (H-OPT section); this doc is the test plan
and action checklist. Goal: ROCm-parity-class decode (tg128), reached by
letting the machine (fusion + BEAM) do the work, minimizing hand-written code.

## Hypothesis (H-OPT, condensed)

tinygrad's ~7x decode gap is ~4.4x bytes-moved (fp32 dequant + broken fusion)
x ~1.6x scheduling (BEAM=0). Both are machine-reachable: the fused-Q4-GEMV is
expressible in existing ops, so it's a fusion+search problem, not a hand-kernel
problem. Predicted post: 8B from 15.8 -> ~60-100 tok/s (mid-to-parity).

## Baselines (measured, 2026-06-11, Qwen3 Q4_K_M, single 7900 XTX)

| model | tinygrad BEAM=0 | llama.cpp ROCm | gap |
|---|---|---|---|
| 4B | 18.8 | 152.8 | 8.1x |
| 8B | 15.8 | 101.2 | 6.4x |
| 14B | 9.1 | 65.8 | 7.2x |
| 32B | 4.4 | 30.8 | 7.0x |

Primary working model: **8B** (fits comfortably, big enough to be bandwidth-
bound). All tests on Ubuntu native (DEV=AMD, local PCIe) unless noted.

## Test plan

Each test: purpose / method / expected / falsifier / gate.

### T0 — BEAM sweep (wall-locator, delivers the floor)
- **Purpose**: how much is pure scheduling (machine, free)? Locate the wall.
- **Method**: `DEV=AMD BEAM=2 JIT=1 python -m tinygrad.llm --model ~/models/Qwen3-8B-Q4_K_M.gguf --warmup --benchmark 128` then BEAM=4. First run pays search cost (slow); BEAM cache persists. Repeat on 14B.
- **Expected**: 8B -> ~22-30 tok/s (per decomposition's 1.6x).
- **FALSIFIER**: if BEAM alone > ~50 tok/s (>50% of llama), the bytes-bloat
  thesis is WRONG — scheduling was the gap. Re-derive.
- **Gate**: always run first. Cheap, no code.

### T1 — Byte-bloat diagnostic (confirm the 4.4x and its source)
- **Purpose**: prove bytes-moved >> Q4 weight size, and find where.
- **Method**: `DEV=AMD DEBUG=2` on one 8B decode step; sum kernel bytes
  (time x GB/s per kernel) for one token; compare to Q4 weight bytes (~4.68GB).
  Identify whether a fp32 weight tensor is materialized between dequant and
  matmul (look for a large write+read pair).
- **Expected**: ~3-5x more bytes than 4.68GB/token; a visible fp32 weight
  materialization.
- **FALSIFIER**: if bytes/token ~= Q4 size, dequant already fuses and the gap
  is elsewhere (scheduling/occupancy) — pivot to T0-style work only.
- **Gate**: run alongside T0; it interprets T0's result.

### T2 — fp16 dequant (cheapest byte fix)
- **Purpose**: stop emitting fp32 from dequant.
- **Method**: in `tinygrad/llm/gguf.py` Q4_K path (ggml_type 12), output
  fp16/bf16 instead of float32 (the `.cast(dtypes.float32)` chain). Confirm the
  matmul accumulates in half where safe.
- **Expected**: measurable tok/s gain if accumulation was fp32-bound.
- **CORRECTNESS CHECK (mandatory)**: fixed-prompt completion before/after must
  stay coherent; spot-check a few generations. fp16 dequant of Q4_K should be
  lossless-ish (scales are fp16 already) but verify.
- **Gate**: after T0/T1 confirm bytes are the lever.

### T3 — dequant fusion (the main lever, machine's job)
- **Purpose**: make tinygrad fuse dequant into the GEMV so Q4 weights are read
  once, no fp32 spill.
- **Method**: identify what breaks fusion in the Q4_K dequant expression
  (the `.contiguous()` on blocks, the transpose in q_to_uint8, expression
  complexity). Try: simplify/restructure the dequant graph; test whether BEAM
  + the scheduler then fuse it; measure bytes/token (T1 method) drop toward Q4
  size. This is "gardening the graph", not writing a kernel.
- **Expected**: bytes/token -> ~1.2x Q4 size; 8B -> ~50-70 tok/s.
- **FALSIFIER**: if no graph restructuring makes it fuse (scheduler refuses),
  the fusion is a tinygrad capability gap — escalate to improving tinygrad's
  fusion (still machine-side) or, last resort, T5.
- **Gate**: the campaign's center of mass; after T2.

### T4 — llama.cpp kernel inspection (decides if T5 is ever needed)
- **Purpose**: name llama.cpp's actual gfx1100 decode primitive.
- **Method**: read `ggml/src/ggml-cuda/mmvq.cu` + HIP path / `mul_mat_vec_q`;
  determine plain vectorized FMA vs dp4a/v_dot4 for the quantized matvec.
- **Expected**: identifies whether a packed-dot instruction is in play for
  DECODE (vs only GEMM/prefill).
- **Gate**: do before contemplating any renderer work (T5).

### T5 — hand-added primitive (LAST RESORT, likely unneeded for decode)
- **Purpose**: only if T0-T3 wall well below parity AND T4 shows a needed
  instruction. Add packed-dot emission to the RDNA3 renderer as a templated
  primitive + BEAM tune (the AutoTVM/CUTLASS blend).
- **Gate**: only if T3 falsifier fires and T4 justifies it. Decode-unlikely.

### T6 — Mac transport-neutrality (H7 confirmation)
- **Purpose**: confirm kernel wins transfer to the Mac over USB4.
- **Method**: same model + winning BEAM-cached schedules on the Mac
  (DEV=AMD via TinyGPU); compare decode tok/s to Ubuntu native.
- **Expected**: within ~10% of native (decode is on-card; transport carries
  only per-token dispatch).
- **FALSIFIER**: if Mac >10% slower after JIT warmup, dispatch/roundtrips
  matter — separate dispatch-amortization work (TinyJit/graph batching).
- **Gate**: after a kernel win exists worth deploying.

## Action items (ordered; machine; cost; dependency)

1. [ ] **T0 BEAM sweep** — Ubuntu native, 8B+14B, BEAM=2/4. Free. → records the
   floor and tests the central falsifier. (no deps)
2. [ ] **T1 byte diagnostic** — Ubuntu, DEBUG=2 bytes/token on 8B. Free.
   (parallel with T0)
3. [ ] **Decision gate A**: if T0 falsifier fires (BEAM alone >50% llama) →
   thesis wrong, pivot to scheduling-only. Else continue.
4. [ ] **T2 fp16 dequant** — edit gguf.py; bench + correctness check. ~hours.
   (after gate A)
5. [ ] **T3 dequant fusion** — graph-restructure for fusion; bytes/token + tok/s.
   ~days; the main work. (after T2)
6. [ ] **Decision gate B**: if T3 reaches parity-class → done, go to T6. If it
   walls → run T4.
7. [ ] **T4 llama.cpp kernel read** — name the primitive. ~hours. (only if T3 walls)
8. [ ] **T5 primitive (if justified)** — renderer + BEAM. ~weeks. (only if gate B + T4)
9. [ ] **T6 Mac neutrality** — deploy winning schedules to Mac, compare. 1 power
   cycle. (after any real win)
10. [ ] **BEAM cache for deployment** — tune once on Ubuntu, ship cached schedules
    to Mac ("separation in time" — search cost paid once). (with T6)
11. [ ] Record every result + falsification in amd-rocm-llamacpp-research.md.

## Decision tree (one line)

T0 big win → scheduling was it (thesis wrong, easy). T0 small + T2/T3 big →
bytes were it (thesis right, machine took it, no kernel). T3 walls + T4 shows
instruction → one templated primitive (T5). Each branch has a measured gate;
no building without a number.

## Correctness / safety (do not skip)

- Any dtype change (T2) requires a generation-quality check, not just speed.
- BEAM runs are nondeterministic in search but deterministic in output; verify
  output unchanged across BEAM levels.
- Keep a frozen baseline tag before kernel edits for A/B and rollback.

## Audit + external validation (2026-06-11)

Checked the plan's load-bearing claims against outside sources. Three
corrections, one confirmation.

### CONFIRMED: fusion is the right lever (field-validated, not just our reasoning)
External literature is unambiguous that fused dequant+matmul is THE known
solution to quantized-decode speed: "kernel fusion is essential to developing
a quantized model with superior throughput to FP16"; the SplitK W4A16 work is
a one-step fused dequant+matmul kernel built precisely for this. So T3 aims at
the field-proven target. Good.

### CORRECTION 1: the 4.4x / 1.6x split is ONE measurement + a residual, not two
We measured the ~4.4x bytes ratio (DEBUG=2: kernels ~355 GB/s vs 81 eff) and
then ASSIGNED the remaining 1.6x to scheduling to reach 7x. BEAM's actual
contribution is unmeasured (no public quantitative BEAM speedup found; tinygrad
docs only say BEAM makes it "competitive with PyTorch"). So the decomposition
is suggestive, not established. **T0 is what actually tests the split** — treat
it as a hypothesis test, not a confirmation. Don't cite 4.4x/1.6x as if both
were measured.

### CORRECTION 2: "parity with llama.cpp" ceiling is likely too optimistic
tinygrad's own claim is BEAM makes it "competitive with PyTorch" — and PyTorch
quantized decode is itself below llama.cpp. llama.cpp's 567 GB/s (59% of peak)
is a years-tuned kernel. Realistic best case for tinygrad fusion+BEAM is more
like PyTorch-class = ~50-70% of llama.cpp; full parity is aspirational, not
expected. Down-weight the "ceiling ~100 tok/s 8B" scenario; treat "mid
~50-70 tok/s" as the realistic success target.

### CORRECTION 3: every fast quant-decode in the wild is a HAND-WRITTEN fused kernel
llama.cpp MMVQ, the SplitK Triton kernel — the existence proofs of fast
quantized decode are all hand-fused, not compiler-auto-fused. Nobody has shown
auto-fusion MATCHING hand-fusion for this workload. So T3 (tinygrad auto-fuses
the dequant) is the OPTIMISTIC bet, and T5 (hand-add a fused primitive) is more
likely than "last resort, decode-unlikely" implied. Reweight: T3 success would
be a mildly novel result, not the expected one; budget for T5. The "machine
takes most" path is worth trying FIRST (it's the goal, and a clean win if it
lands) but the field's prior favors needing some hand-fusion.

### ADDED to T1: rule out CPU/dispatch overhead
At 15 tok/s on 8B, part of the gap could be Python/dispatch per token, not GPU
kernel bandwidth. T1 must check whether summed GPU kernel time accounts for the
full per-token wall time; if there's a large gap, that residue is dispatch
(TinyJit/batching work), which fusion won't fix.

### Net audit verdict
Structure sound, central lever (fusion) externally validated. But: the gap
decomposition is one measurement not two (T0 tests it), the parity ceiling is
optimistic (target ~50-70% of llama as realistic), and hand-fusion (T5) is more
probable than framed (auto-fusion is the hopeful path, not the safe one). The
plan's ordering still holds — try the machine-first path first — but with
honest expectations, not the optimistic ones this session has repeatedly shown.

## REVISED PLAN (post-audit, 2026-06-11) — supersedes T2/T3 framing

Independent audit (Codex) falsified the fp32-spill thesis: tinygrad ALREADY
fuses Q4_K dequant into a fp16 GEMV (no fp32 materialization; HALF=1 default).
The gap is the QUALITY of that fused kernel (scalarized, poor vectorization/
occupancy/access pattern) vs llama.cpp's tuned packed MMVQ. Revised tests:

- **T2 (fp16 dequant): MOOT** — already default (model.py:329, HALF=1).
- **T3 (make dequant fuse): MOOT** — already fuses (REALIZE=0 default).
- **T0 (BEAM): still run, now the primary MACHINE-side lever**, not a floor.
  BEAM tunes the existing fused kernel's schedule (tiling, vectorization,
  occupancy) within the move set. Expectation tempered: it cannot add a
  packed-dot primitive the renderer lacks. Measures how far layer-1 search
  gets on the already-fused kernel.
- **T1' (NEW — kernel-quality profile): the new diagnostic.** Inspect the
  generated fused GEMV kernel: scalar vs vectorized loads, occupancy, memory
  access pattern; diff against llama.cpp MMVQ behaviour. Names the specific
  deficiency. (Audit already showed: fused, scalarized, GPU-bound not dispatch.)
- **T1b (NEW — cheap A/B): REALIZE / recompute.** Default recomputes dequant
  every token (model.py:385 NOTE). Test REALIZE=1 (materialize fp16 weights
  once): does avoiding per-token recompute help, or does the larger fp16 read
  hurt? One env var, no code.
- **T-SPECIALIZE (was T5, now PRIMARY hard lever): specialized packed Q4_K
  GEMV lowering** — vectorized/packed loads + dot, better occupancy. This is
  layer-2 (expand the representation). Could be a tinygrad lowering improvement
  (machine-general-ish) or a hand-written/templated kernel. The field prior and
  the audit both say this is likely REQUIRED to approach llama.cpp, not optional.
- **T6 (Mac neutrality): unchanged**, after any real win.

Revised ceiling: OPEN. 50% of llama.cpp = 3.2x over current; audit lowers
confidence that BEAM alone reaches it since fusion is not the missing piece.
Honest framing: BEAM measures the layer-1 ceiling on the existing kernel;
the gap beyond that needs the specialized lowering (layer-2), effort unknown.

Revised "machine takes most" assessment: PARTIALLY ALREADY TRUE — the machine
did the fusion. The REMAINING gap is the vectorized packed GEMV primitive,
which is the layer-2 residue the field hand-writes. So the honest order is:
(1) BEAM to harvest layer-1 on the existing kernel [machine], (2) profile to
size the residual, (3) specialized GEMV lowering for the rest [human-shaped,
possibly templated so BEAM tunes its params].

## T-SPECIALIZE refined: "BEAM + Q4_K primitive" (2026-06-11, agreed with Codex audit)

Verified: BEAM's action set (search.py:13-22) is 8 schedule transforms
(UPCAST/UNROLL/LOCAL/GROUP/GROUPTOP/THREAD/SWAP/TC); the ONLY hardware
primitive is TC (tensor cores), and it is HAND-ADDED (search.py:20,22). So
BEAM provably cannot synthesize a packed quant GEMV — confirming the wall is
the action set's span, not search depth. The plan is "BEAM + a Q4_K primitive",
modeled exactly on how TC was added. Harness exists: extra/q4_k_bench.py.

Two tracks (Codex), with refinements:

### Track 1 — BEAM containment (safety, do first; it currently faults the GPU)
- Run BEAM on the q4_k_bench microbench, NOT the full model.
- PARALLEL=0, lower candidate count, strict timeout, BEAM_DEBUG=2.
- Identify the faulting Opt candidate; blacklist/constrain its shape on AMD.
- The HW fault (memory_lost=1) is itself a bug: a candidate kernel should never
  hard-fault the GPU — add a guard. REPORT it.
- HARD RULE: BEAM must NEVER run on the Mac remote path — a faulting candidate
  would drop the TinyGPU bridge / PCIe tree. Tune on Ubuntu native ONLY, cache
  schedules, ship the cache to the Mac ("separation in time").

### Track 2 — the primitive (the real lever)
- REFINEMENT (try the cheap machine-side shot FIRST): attempt to vectorize the
  Q4_K dequant by RESTRUCTURING the gguf.py:57 expression (bitcast blocks to a
  wider dtype, unpack with vector bit-ops) so codegen emits vector loads instead
  of scalar uint8 — no new Opt needed. If codegen still scalarizes (the gather
  pattern forces it), THEN add the primitive. (Skip if Codex already determined
  the gather forces scalarization.)
- PRECISION: the primitive must introduce VECTORIZED PACKED LOAD + DOT
  capability (like TC introduces WMMA), not merely a tuned schedule/heuristic —
  a heuristic within the existing action set cannot add vectorization. Add it
  near the matvec heuristic (heuristic.py:63) as a new candidate.
- Then let BEAM tune around it: rows/thread, group size, local shape, unroll.
- Only then re-run full decode; then T6 Mac deploy with cached schedules.

### Honest expectation
This is the right architecture (templated-autotuning, the field standard) and
the only path with a real shot at llama.cpp parity. But the primitive's QUALITY
is itself the hard, uncertain part — reaching parity depends on whether
tinygrad's codegen can express packed loads + dot efficiently. Right plan, not
a guaranteed win. "BEAM + primitive" >> "BEAM harder" is correct; do not expect
plain BEAM to contribute beyond tuning the primitive's parameters.

## Search-literature reference point (2026-06-11)

Corrected framing: schedule search and graph search are not novel. TASO searches
verified DNN graph substitutions; Tensat uses equality saturation for tensor
graph superoptimization; Welder uses tile-graph and tile-traffic cost modeling
for memory-access scheduling; Mirage uses µGraphs to search across algebra,
kernel, thread-block, and thread levels.

What remains relevant here is one layer lower: packed sub-byte quantized
representations are not exposed as clean dense tensor algebra or tile objects.
Q4_K is a packed struct layout: fp16 scale words, 6-bit packed scales/mins,
nibble weights, and interleaved sub-blocks. The project framing is therefore:

> Make packed quantized formats expressible to tensor search/scheduling systems
> by exposing Q4_K as a verified packed tile primitive rather than opaque scalar
> byte math.

Practical implications:

- Welder/Mirage are the right mental models for the final abstraction: search
  wants tile-level objects plus a memory-traffic objective, not orphan kernels.
- The immediate engineering path remains unchanged: make Q4_K word storage and
  primitive lowering correct, then give search scheduler-safe knobs.
- BEAM's role is local tuning after representation exists. It is not expected
  to discover Q4_K packing semantics from scalar byte arithmetic.

References:

- TASO: https://github.com/jiazhihao/taso
- Tensat: https://arxiv.org/abs/2101.01332
- Welder: https://www.usenix.org/conference/osdi23/presentation/shi
- Mirage: https://www.usenix.org/conference/osdi25/presentation/wu-mengdi

## FINAL agreed plan (2026-06-11) — execute in order

Ordering (Codex, agreed): microbench -> expression-vectorization probe ->
primitive if needed -> BEAM tunes primitive -> full decode. BEAM native-Ubuntu
only (never the Mac bridge). This is the plan of record.

1. [x] Native Ubuntu only for BEAM (Mac bridge would drop on a faulting
   candidate). Current execution has used no BEAM and no Mac bridge.
2. [x] Build the Q4_K GEMV microbench at the dominant decode shapes.
   Implemented in `extra/q4_k_bench.py`; representative Qwen3-8B FFN/attention
   shapes are selected from GGUF metadata.
3. [x] Cheap expression-vectorization probe FIRST: rewrite `gguf.py` Q4_K path
   to encourage wider loads; inspect DEBUG=4; accept only if scalar uint8 loads
   become vectorized AND microbench improves. Result: NO-GO. `GGUF_Q4K_WIDE=1`
   is bit-exact, but still emits scalar `unsigned char` loads and regresses the
   microbench.
4. [x] Primitive load-width viability probe. `extra/q4_k_primitive_probe.py`
   confirms a custom UOp kernel can emit `unsigned int` loads over a word-typed
   Q4_K buffer.
5. [x] Representation staging probe. Opening the GGUF file as
   `Tensor(path, dtype=dtypes.uint32)`, slicing the aligned Q4_K tensor range on
   DISK, then copying that slice to AMD avoids the scalar byte-pack kernel and
   copies only the target tensor range. This is the viable input path for the
   primitive.
6. [x] Add the first correctness-only primitive scaffold: packed Q4_K
   `uint32` load + scale/min unpack + fp16 dot over a small GEMV slice.
   Implemented in `extra/q4_k_gemv_primitive.py`; it consumes the word-typed
   storage path from step 5 and passes the frozen reference gate on 2 and 16
   rows of `blk.0.ffn_gate.weight`.
7. [x] First tunable parallel primitive pass: `--mode partial --parts N` splits
   the K-block reduction into per-row/per-part partial sums and reduces those
   partials with tinygrad. Correctness passes on the full FFN shape. Result:
   no speed win yet; `parts=1` is best, and the kernel still tops out around
   38-39 Q4-GB/s device time on `blk.0.ffn_gate.weight`, below the existing
   fused graph path's ~80 Q4-GB/s device time.
8. [x] Harden primitive correctness gates: deterministic random fp16
   activations replace the earlier all-ones vector, and a direct unpacked-weight
   comparison checks the primitive's decoded Q4_K weights element-wise against
   `q4_k_reference`. Result: unpack max_abs `0`; random-GEMV max_abs
   `0.00123835` on the full FFN shape.
9. [x] Scheduler-safe parallelization: explicit opt sweep found the first
   compiling, correct, fast primitive schedule. `LOCAL:0:32` keeps exact unpack
   correctness, random-GEMV correctness, emits `unsigned int` packed Q4 loads,
   and reaches ~369 Q4-GB/s device time over 10 iterations on the full FFN
   shape. Broad `--schedule auto` still fails AMD compilation, so BEAM remains
   gated until compile-failure containment is isolated.
10. [x] Wire the tuned primitive into the microbench behind a flag and compare
   against the existing fused expression. `extra/q4_k_bench.py --primitive`
   now runs the `LOCAL:0:32` primitive under JIT, uses deterministic random
   fp16 activations by default, checks exact primitive unpack against
   `q4_k_reference`, and checks GEMV against the decoded matmul reference before
   timing. On Qwen3-8B representative shapes, the primitive wins on the large
   FFN and attention-Q shapes; the tiny KV projection is device-time weaker, so
   model integration should be shape-aware rather than blanket replacement.
11. [x] Tune the primitive's exposed parameters with search on native Ubuntu
   only. Added `extra/q4_k_policy_sweep.py`, a subprocess-contained,
   shape-aware sweep over explicit primitive opts. Result: `LOCAL:0:64` wins
   `12288x4096` and `4096x4096`, `LOCAL:0:32 --parts 4` wins
   `4096x12288`, and the existing fused graph remains best for `1024x4096`
   KV projection by device time. This is the first usable shape policy.
12. [x] Wire the selective primitive policy into the real model path behind a
   flag. `Q4K_PRIMITIVE=1` now preserves GGUF tensor metadata, installs
   `Q4KPrimitiveLinear` wrappers after normal state loading, and dispatches
   custom packed Q4_K GEMV only during rollout/decode for policy-selected
   roles. Prefill, batched/symbolic non-decode paths, bias cases, and small KV
   projections fall back to the existing tinygrad graph.
13. [x] Run full 8B decode after the model-path primitive flag compiles.
   Sustained `--benchmark 128` result: baseline averages `15.44 tok/s`;
   `Q4K_PRIMITIVE=1` averages `28.74 tok/s`, a 1.86x full-model decode gain.
   DEBUG=2 confirmed rollout emits `q4k_gemv_partial_*` kernels.
14. [x] Repeat the validated path on 14B. Sustained `--benchmark 128` result:
   baseline averages `8.88 tok/s`; `Q4K_PRIMITIVE=1` averages `14.90 tok/s`
   over all samples, with last-32 median `15.72 tok/s`. The same role policy
   generalizes, though 14B has visible outliers.
15. [ ] Contain BEAM faults: reproduce on microbench (PARALLEL=0, BEAM_DEBUG=2),
   find the faulting Opt sequence; it's a tinygrad-arkey bug (candidates must
   fail safe, not hard-fault). Report it.
16. [ ] Mac neutrality/deployment: never run live BEAM on the Mac/TinyGPU path;
   ship only native-Ubuntu-proven primitive policies/schedule cache.

### Two additions (both mandatory, easy to miss)

A. CORRECTNESS GATE on step 3/4: any dequant rewrite must match the current
   ggml_data_to_tensor output BIT-EXACTLY on a test block before any speed
   number counts. Q4_K layout (6-bit scales packed across bytes, nibble
   interleave) is fiddly; a fast-but-wrong kernel silently corrupts weights.
   The microbench must assert numerical equality vs the reference, not just time.

B. GO/NO-GO NUMBER for the probe (step 3), set before running so it can't
   drift: microbench effective bandwidth must move from ~75 GB/s toward
   >=200 GB/s to justify carrying the expression rewrite to full decode. Below
   that, the gather forces scalarization -> go to the primitive (step 4).
   A microbench win must also be confirmed to translate to a full-decode win
   (step: full decode last) before claiming success.

### Current next action

Step 15: contain the remaining BEAM/search fault path on the microbench. The
primitive policy is now useful without broad BEAM, but search candidates still
need fail-closed behavior before live BEAM can be trusted again.

## Feedback on Codex step 3-6 probes (2026-06-11) — correctness testing gap

Good: expression-vectorization probe correctly FAILED (bitcast inside the graph
does not change load width) and was correctly concluded — we are now in
primitive/layer-2 territory as planned. The word-typed disk-u32 storage staging
is elegant. The primitive issues real uint32 loads and the layout arithmetic is
plausibly correct.

TWO CORRECTNESS-TESTING FIXES NEEDED before trusting the primitive (the current
gate is weaker than the locked "bit-exact dequant" requirement):

1. **Test uses x = Tensor.ones (q4_k_gemv_primitive.py:88).** A dot against ones
   is a SUM — permutation-invariant. A Q4_K layout bug that misorders the 256
   elements within a block (wrong sub-block/scale assignment) is INVISIBLE under
   ones. Use RANDOM fp16 activations so element-ordering errors surface.

2. **Gate is GEMV max_abs < 1e-2 (line 101), not bit-exact dequant.** A loose
   end-to-end tolerance over k=4096 can mask a few wrong weights. Add a SEPARATE
   gate that compares the primitive's UNPACKED WEIGHTS element-wise against
   q4_k_reference (exact, or fp16-ULP if dtypes differ), BEFORE the dot. That is
   the gate that catches the fiddly 6-bit-scale / nibble-interleave layout bug;
   the GEMV tolerance does not.

The observed max_abs ~2.9e-4 is consistent with fp16-rounding (reference casts
weights to fp16; primitive keeps fp32), i.e. probably NOT a layout bug — but
"probably" is exactly what the per-weight exact gate exists to remove.

Also: no PERFORMANCE number yet. The primitive is serial per row; whether wide
loads beat the ~75 GB/s scalar path is still unproven. Parallelize + tune, then
compare vs q4_k_bench — and keep an eye on integration (the primitive must be
wireable into the model's real matvec / a codegen lowering, not a standalone
orphan).
