# Step 2 — machine search lowers the batched-decode plateau (scope)

Date: 2026-06-15. The ceiling probe found: batching amortizes the weight read (~2.4–3.5×) but the
per-token cost plateaus at ~14 ms (16 GB/s = 2% of peak) — i.e. the plateau is tinygrad's UNTUNED batched
GEMM/attention, NOT a memory floor. The verification GEMMs (Q4_K linears at speculative batch N) are
small-N matmuls — the exact substrate our validated loop (N1/N2/L0/L1) drives to 33–98% of peak. This step
measures whether machine search lowers that plateau, i.e. whether the loop improves decode-verification.

## What it tests (and why it's the mission goal)
The forward currently runs the batched verification matmuls with the DEFAULT (no-opt / heuristic) schedule
— that's the 14 ms/tok plateau. If the loop finds a schedule that's k× faster on those exact shapes, the
plateau drops to ~14/k ms/tok, raising the batched-decode ceiling. That is **machine search measurably
improving decode**, in the reachable (batched/speculative) regime.

## Constraints honored
- **Raw BEAM hangs gfx1100** (S1) → tune via the GPU-SAFE curated 277-config loop (qk_loop_live /
  qk_beam_log path, `_time_program`), NOT JITBEAM/beam_search.
- **Held-out**: use Qwen3-8B FFN shapes whose (M,K) are ABSENT from the N0 corpus so the loop isn't
  memorizing (corpus M,K ∈ {4096,5120,8192,11008,13824,14336}; 12288 is absent).

## The shapes (8B Q4_K linears as matmuls M=out, K=in, N=speculative batch)
- ffn_gate / ffn_up: **(12288, 4096, N)** — held out, dominant FFN weight read.
- ffn_down: **(4096, 12288, N)** — held out.
- N ∈ {8, 16} (realistic speculative batch; N=16 is in the loop's small-N trained range, N=8 tests
  extrapolation just below it).

## Method (reuse qk_loop_live, GPU-safe)
Per shape: live-time all 277 opt-schedules (`_time_program`), then report
- **no_opt tflops** (config []) — the untuned default the forward pays now (the plateau),
- **oracle tflops** (best of 277) — the tunable ceiling,
- **loop guided top-8 tflops** (train on the N0 corpus, rank, take the model's top-8 live-timed) — what
  machine search actually delivers without trying all 277.
Key metrics: `best/no_opt` (how much faster tuning is than the untuned default → the plateau lever) and
`guided/oracle` (does the loop find it cheaply on these decode shapes).

## Pre-registered gate
- **best/no_opt ≥ 1.5×** on the FFN shapes → tuning the verification GEMMs is a real plateau lever →
  machine search improves batched decode; proceed to wire it in (Step 3) + the speculative scaffold.
- **guided/oracle ≥ 0.95** → the loop finds the good schedule cheaply on decode shapes (transfer holds).
- best/no_opt < 1.2× → the default is already near-optimal for these shapes → the plateau is intrinsic
  (attention / kernel-launch within the batched forward), not matmul-schedule → re-diagnose.

## Connect to the ceiling
guided tflops vs no_opt tflops projects the plateau drop: new ms/tok ≈ 14 × (no_opt/guided). Combined with
the ~2.4–3.5× memory-amortization from batching, this is the realizable batched-decode speedup the loop
delivers — the first end-to-end instance of machine-search improving decode (in the batched regime).

## Out of scope (later steps)
The speculative scaffold (draft model / Medusa), wiring matmul_decoded as the T>1 Q4_K verification kernel,
and the full e2e speculative tok/s. This step isolates the ONE question: does the loop lower the plateau?
