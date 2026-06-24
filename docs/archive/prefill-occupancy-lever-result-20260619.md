# RESULT — Prefill "occupancy lever" → it was a HARNESS BUG (gfx1100, 2026-06-19)

Executes `prefill-occupancy-lever-scope-20260619.md` P0 (the decisive kernel-identity gate). **The P0 gate found the
premise itself was false: there is NO per-process WMMA bimodality / boost / occupancy lottery. The entire effect was a
flag-leak bug in the `qk_tensile_ab_measure.py` measurement harness.** This RECONCILES the whole multi-turn prefill
arc.

Probe: `extra/qk_prefill_kernel_identity.py`. Artifacts: `bench/qk-prefill-boost/p0_kernel_identity_{stuck,fast}.json`.

## The bug
`TinyJit` **captures the graph on the 2nd call** (the 1st only traces). The "fast" reference harness
(`qk_tensile_ab_measure.py`) built `joff=build(False)` (WMMA), then `jon=build(True)` — and `build(True)` sets the
process-global `Mod.PREFILL_TENSILE_GEMM=True` and **never resets it**. So when `joff` was *captured* on its later
2nd call, `model.forward` re-ran with the flag **True** → `joff` silently **routed through Tensile**. The harness's
`assert not rc_off` only checked `joff`'s 1st trace (flag False), so the leak was invisible.

## The proof (P0)
Kernel-identity dump of the `joff` ("OFF"/WMMA) captured graph + a controlled flag test:
| case | tensile kernels in graph | tok/s |
|---|---:|---:|
| `ab_measure`-repro "OFF" (flag leaked True at capture) | **3** (`tensile_qo/gateup/down`, lds=25088, vgpr~256) | **2626** |
| clean WMMA (flag held False through capture) | 0 | **1433** |
| clean Tensile (flag True) | 3 | **2673** |
- Warmstart fired **identically** in stuck vs "fast" (match=5, apply=5, error=0) — the warmstart-miss hypothesis was
  also wrong; the difference was the 3 Tensile kernels, not a TC-schedule miss.
- Fresh-process WMMA (flag held False): **1437 / 1434 / 1433** — consistent, NO lottery.
- `ab_measure` after the fix (capture each jit before changing the flag): **OFF=1431, ON=2629, SPEEDUP=1.838×**,
  `rel_err(ON vs OFF)=0.0` (byte-identical).

## The true, reconciled picture
- **tinygrad WMMA prefill = ~1433 tok/s = ~47% of llama (3070). Consistent. No bimodality, no boost state, no
  occupancy lottery, no clock dependence.**
- **Tensile prefill = ~2673 tok/s = ~87% of llama. Consistent. Byte-identical greedy (rel_err 0). A real,
  reproducible ~1.84× win.**
- The **original reconciliation was right all along** (`prefill-RECONCILIATION-source-of-truth`: 1449 WMMA / 2633
  Tensile / 1.83×). The `0.997× / "Tensile no in-model win"` (tensile-land, ab_measure) was THIS flag-leak bug — both
  arms were Tensile. Every "fast ~2674 WMMA" chased over the last 3 turns was leaked-Tensile.

## What is RETRACTED (all the same bug)
- "prefill WMMA is bimodal per-process ~1438↔~2674" — **FALSE**; ~2674 was always leaked-Tensile.
- "boost-state lottery / latched at init / ROCm #6289" (`prefill-boost-resolution-result`) — **FALSE** (artifact).
- "clock-authority: auto maxes sclk / Tensile robust / SOLVED" and its later "not clock, it's occupancy/power-grant"
  refinements (`prefill-clock-dpm-authority-result`) — **all artifacts of the same leak.**
- "Tensile 0.997x no in-model win" — **FALSE** (flag-leak).

## Production impact: NONE
The shipped path reads `PREFILL_TENSILE_GEMM` from the env once (default 0) and never toggles it, so production WMMA
prefill is the honest ~47% llama and Tensile only fires under the research flag. The leak existed **only** in the
research A/B harness that toggled the flag mid-process. Fixed in `qk_tensile_ab_measure.py` (capture each jit before
changing the flag + asserts that OFF has no Tensile kernels and ON does).

## The real prefill lever (reframed, finally clean)
- **Dependency-free:** WMMA prefill is genuinely ~47% llama, bounded by the pure-tinygrad WMMA codegen ceiling
  (~42 TFLOPS, POWN-walled — the SW-pipelined K-loop capability tinygrad can't express). Plus the shipped concrete-KV
  1.24× for the first chunk. There is no hidden "fast WMMA" state to unlock.
- **With the vendored rocBLAS Tensile `.co` (dependency):** ~87% llama, byte-identical, a real ~1.84× win — gated
  **only** on the dependency-policy decision (`PREFILL_TENSILE_GEMM=0` default; conflicts with the standing
  no-external-deps preference).
- So the prefill decision is purely: **accept the Tensile artifact dependency for ~87% llama, or rest at ~47%
  dependency-free.** No occupancy/clock/codegen investigation changes that.

## Methodology bankable (the meta-lesson)
A `TinyJit` A/B that toggles a routing global must **capture each jit (2 calls) before changing the global**, and
should **assert on the captured graph's kernels** (not just the 1st-trace route count). The 3-turn detour
(clock-authority → boost-resolution → occupancy) all traced to trusting one buggy harness's number over the clean
reconciliation. When two "identical" harnesses disagree, dump the actual compiled kernels before theorizing about
hardware.
