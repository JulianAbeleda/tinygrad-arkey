# Per-layer isolation — where the e2e 5× penalty actually lives (the clock-ramp confound, killed)

Date: 2026-06-15. Goal: distinguish H1 (per-layer GEMV saturates standalone → e2e penalty is JIT-graph/
launch/integration overhead) from H2 (small per-layer kernel can't sustain bandwidth → the kernel itself).
Method: `extra/qk_cold_perlayer.py` — vdot int-dot + fp-dequant GEMV at the *real* per-layer sizes, COLD
(2 GB backing buffer, each rep reads a different region), launch-amortized (200 reps), and — critically —
at **forced full memory clock** (`rocm-smi --setperflevel high`) with a ~2 s warmup.

## The confound this run uncovered and removed

The first pass (perf=auto, no warmup) showed bandwidth scaling with kernel size — 8 MB=27%, 28 MB=54%,
300 MB=77% — which *looked* like H2 (small kernels don't saturate). It was an artifact: the small/short
measurements run during tinygrad's slow memory-clock ramp (96→1249 MHz over ~4 s), the large/long one
reaches full clock. **Force the clock high + warm up, and the size-scaling vanishes:**

| size            | vdot (int-dot) | fp (dequant) |
|-----------------|---------------:|-------------:|
| attn  (8 MB)    |    **64.6%**   |    28.8%     |
| ffn   (28 MB)   |    **79.9%**   |    56.1%     |
| large (300 MB)  |    **76.1%**   |    18.7%     |

The int-dot kernel **saturates at every per-layer size** (64–80%), above llama.cpp's 57%. **H2 is false** —
the per-layer kernel is not the wall. (fp's 56% at 28 MB is residual cache benefit; its true cold ceiling is
the 300 MB number, **18.7%** — the serial fp-add dequant chain, compute-bound.)

## The e2e number is NOT clock-limited either

Sustained `generate()` for 400 tokens at forced full clock, measured at steady-state (tokens 200–380, long
past the 4 s ramp): **21.5 tok/s = 101 GB/s = 12% of peak.** Fully boosted, fully warmed — still 12%. The
clock ramp does not explain the e2e gap. (cli short-benchmark reports ~30; both are the 12–16% band.)

## The decisive synthesis (H1, with the mechanism)

Three facts that only fit one story:
1. The int-dot per-layer GEMV **saturates standalone** at full clock (64–80%) — kernel throughput is solved.
2. e2e is **12% at full clock** — and the weight read is **95% of the token** (V1: 4.68 GB at 113 GB/s = 42 ms).
3. **D1/E0**: wiring the 76%-standalone vdot kernel into decode gave the *same* e2e as fp (30 = 30, null).

Fact 3 is the hinge: swapping a **4× faster** standalone kernel into the graph **did not move e2e**. So e2e
is **not bandwidth-bound by the GEMV kernel.** The default decode runs the fp-dequant kernel (true ceiling
~19%), and e2e's 12–13% is that kernel in-graph minus per-launch/non-GEMV overhead. The int-dot kernel that
*would* lift the ceiling needs per-token activation quantization + sustained occupancy across ~252 single-
shot launches — and that **integration** (not the kernel) is what D1/E0 failed to amortize.

**Verdict: H1.** The e2e penalty lives in the graph/integration layer, not the per-layer kernel:
- NOT the clock (forced high, both ends still pinned: kernel 76%, e2e 12%).
- NOT kernel throughput (int-dot saturates standalone; swapping it in is e2e-neutral).
- NOT Amdahl (the weight read is 95% of the token).
- It is the **e2e integration of int-dot**: amortized quant + occupancy sustained across launches — exactly
  the structure llama.cpp's fused `mmvq` has and tinygrad's per-launch decode graph does not. That is the
  lever, and llama.cpp's 57% proves it is addressable, not a fundamental wall.

## What this closes / opens
- CLOSES: "small per-layer kernels can't saturate" (H2) and "the e2e gap is the clock ramp" — both false.
- CLOSES: "we need a faster standalone GEMV kernel" — the int-dot kernel already saturates above llama.cpp.
- OPENS (the real remaining lever): the int-dot **e2e integration** — fuse the q8 activation-quant into the
  GEMV (one quant feeding q/k/v and gate/up) and keep occupancy across the launch chain (megakernel /
  persistent / horizontal fusion), i.e. adopt llama.cpp's mmvq structure. D1/E0 attacked half of this;
  neither amortized the quant *and* sustained occupancy together.

Repro: `rocm-smi --setperflevel high && DEV=AMD PYTHONPATH=. .venv/bin/python extra/qk_cold_perlayer.py`
(kernel table); sustained `generate()` 400-token steady-state for the e2e number. Flags default-off.
