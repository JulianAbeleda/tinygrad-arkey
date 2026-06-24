# Prefill AMD GEMM — Which Path Is Traceable, and the Clock-Matched Tensile Gap

Date: 2026-06-20

## The question

Two paths remained to close the last gap to Tensile: (1) **full A+B PLR** (hand-asm), (2) the **vendored
Tensile `.co`**. Which is traceable — and, by timing the `.co`, what is the *real* clock-matched gap?

## Answer: the vendored `.co` is the traceable one

The Tensile `.co` is **already loadable, launchable, and correct from tinygrad's HCQ** — no HIP runtime, no
copies:

- `extra/qk_tensile_hcq_launch.py` (`NamedAMDProgram`) unbundles the `.co` → ELF (host `clang-offload-bundler`),
  resolves the `<kernel>.kd` descriptor, substitutes tinygrad buffer VAs into the captured 128-byte kernarg, and
  dispatches at grid (4,96,1)/wg (128,1,1). Proven correct: rel_err 3.5e-4, stable, PASS.
- This pass **times** it: load via `NamedAMDProgram`, run interleaved with our kernel + the authority.

By contrast, **full A+B PLR is not built and is VGPR-squeezed**: a 2nd A+B fragment buffer needs ~64 VGPR, but
only ~52–57 are dead during compute (CTA/CTB + unallocated) — short ~7–12. It would need a smaller tile (e.g.
WN=3) or deeper register reuse (Tensile's both-operand pool). Tractable in principle, uncertain, unbuilt.

**So the `.co` is traceable now; A+B PLR is a harder maybe.** Timing the `.co` is therefore the decisive move.

## The decisive clock-matched measurement (pinned high, interleaved, reproduced 4×)

| kernel | TFLOPS | vs authority |
|---|---:|---:|
| **Tensile `.co`** (vendored) | **~62** (60.8–63.5) | ~1.15–1.19× |
| ours (`BK32+PAD16+A-PLR`) | ~57 (56.6–57.3) | ~1.06× |
| LLVM authority | ~53.4 | 1.0× |

- **ours / Tensile = 0.92 (0.89–0.93), reproduced 4×.** The gap is **real even at a matched clock** (not a
  clock artifact), and it is **~8%**.

## What this corrects — Tensile is ~62, not 66, at this clock

The widely-cited "Tensile ~66" is an **auto-boost** number — exactly like our own earlier 60.7/61 boost
readings. **Clock-matched (pinned), Tensile is ~62 and ours is ~57.** So:

- The earlier "~92% of Tensile" (our boosted 61 ÷ Tensile boosted 66) and this clock-matched "92%" (57 ÷ 62)
  agree — the **~8% gap is consistent and real**, now measured directly in one process, not inferred.
- Neither our 61 nor Tensile's 66 was the "true" number; both were boost. The honest, reproducible,
  clock-matched picture is **ours ~57, Tensile ~62, authority ~53**.

## Final standing of the dependency-free arc

| kernel | clock-matched TFLOPS | note |
|---|---:|---|
| LLVM authority | ~53 | tinygrad's own WMMA |
| **ours (dependency-free)** | **~57** | BK32 + PAD16 bank-fix + A-prefetch PLR, correct (2.08e-4) |
| Tensile `.co` (vendored dep) | ~62 | the target |

**Dependency-free, we reach ~92% of Tensile and +6–7% over tinygrad's LLVM authority, clock-matched and
reproduced — correct, zero dependencies.** The remaining ~8% to Tensile is the **B-operand prefetch** (we do
A-only PLR) plus Tensile's full both-operand register-pool scheduling.

## Which to pursue for the last ~8%

- **Vendored `.co`** — *traceable and timed now* (~62, +8% over ours), but it is the **dependency** the project
  declined. It exists and works if the policy ever changes; this pass proves the integration end-to-end.
- **Full A+B PLR** — *dependency-free but unbuilt and VGPR-squeezed*; the A-only result (+9–11%) shows the
  direction pays, but fitting B' needs a smaller tile or Tensile-style both-operand register overlap. This is
  the only **dependency-free** route to close the last ~8%.

## Honesty

- Absolute TFLOPS still carries the measurement-context wobble (interleaving the `.co`'s `wait=False` launch
  with our `run_linear` shifts ours to ~57 vs ~61 in the ours-only pinned run); the **clock-matched ratio
  (ours ≈ 0.92 × Tensile), reproduced 4×, is the robust claim.**
- Single shape; `.co` correctness re-verified this session (loads + dispatches + matches its oracle).
- perflevel set `high` for the measurement, reset to `auto` after.

## Verdict

`BELOW_TENSILE_CLOCK_MATCHED` — the dependency-free kernel is a reproducible **~92% of the vendored Tensile
`.co`** at a matched clock (Tensile ~62, ours ~57, both well under the boosted ~66 myth). The `.co` is the
*traceable* path (proven, timed); full A+B PLR is the *dependency-free* path to the last ~8% (unbuilt). Arc
lands: dependency-free Tensile-class GEMM at ~92% of Tensile, +6–7% over the LLVM authority, correct.
