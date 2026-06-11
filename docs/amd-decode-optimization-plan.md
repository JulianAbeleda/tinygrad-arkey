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
