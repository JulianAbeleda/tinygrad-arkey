# decode_attention_v3 — RESULT: REFUTED on performance (decode regime) — 2026-06-17

Isolated build/benchmark arc for `decode_attention_v3` (high-occupancy WMMA flash + cooperative GQA V-reuse).
Hard gate: isolated v3 must beat the **current shipped `hoisted` flash** by **≥1.3×** before any model integration.
**Verdict: REFUTED at decode shapes** — the cooperative-LDS/WMMA lever is measured *slower* than the
Infinity-Cache-served baseline. No model integration, no default changes (correct outcome: hoisted stays).

Hardware: RX 7900 XTX (gfx1100), 24 GB. The WMMA custom-kernel capability itself is **revived and working** (the
prerequisite arc); this refutation is about **performance in the decode regime**, not expressibility.

## Phase 0 — baseline (the number v3 had to beat by 1.3×) [measured]

Isolated current `hoisted` flash (`flash_decode_attention(variant=hoisted, L=128)`), DEBUG=2 device tm, per-token
per-layer attention, T=1, Hq=32, Hkv=8, Hd=128:

| KV | hoisted GPU µs | gate (≤ hoisted/1.3) | rel_err vs SDPA |
|---:|---:|---:|---:|
| 512 | 189.5 | 145.8 | 0.0012 |
| 1024 | 317.7 | 244.4 | 0.0010 |
| 2048 | 574.7 | 442.1 | 0.00095 |
| 4096 | 1093.7 | 841.3 | 0.0014 |

## The decisive measurement — cooperative LDS attention tile is SLOWER [measured, this session]

`extra/lds_attention_tile.py` IS the v3 core lever (cooperatively stage K/V into LDS once, 2-pass softmax with
1s-augmented denom, reuse across query rows) — exactly what a fused v3 does minus WMMA. Re-run at
decode-relevant shapes (T = query rows, L = KV tile, Hd=128), LDS vs global-reread (IC-served) baseline:

| L | T | LDS speedup vs global |
|---:|---:|---:|
| 64 | 16 | **0.71×** |
| 64 | 32 | **0.75×** |
| 128 | 16 | **0.50×** |
| 128 | 32 | **0.52×** |

**LDS staging is 0.50–0.77× — slower everywhere**, and T=16/32 is *more* favorable than decode's T=4 per GQA
group. The Infinity Cache (64 MB) serves the K/V re-reads so well that explicit LDS staging only adds a
load+barrier tax. This is the heart of v3, and it loses.

## Why v3 cannot clear the gate at decode shapes

1. **Decode attention is bandwidth-bound but Infinity-Cache-served.** K+V at decode KV fit the 64 MB IC, so the
   "global-reread" baseline is *not* HBM-bound; the theoretical "read once = 32 µs / 34× headroom" is
   **unreachable** because you cannot beat a cache-served baseline by staging into LDS (measured: 0.5–0.77×).
2. **WMMA cannot help.** (a) Compute is not the bottleneck (BW-bound), so tensor-core throughput is wasted.
   (b) WMMA operands must come from LDS/registers — so WMMA *inherits* the LDS-staging tax that's already
   slower. (c) High-occupancy WMMA needs GEMM-size M; decode has Hq=32 total and **M=4 per kv-group under GQA
   (group=4)** — structurally low-M, the regime where WMMA underfills.
3. **The "best-case 2.35×" bound was the wrong regime.** scores+P@V at M=128 = 466 µs vs hoisted 1094 µs looked
   like 2.35×, but M=128 is the GEMM regime (and that P@V was a 1-block underfilled measurement). Real decode is
   M=4–32; the GEMM bound does not transfer.
4. **Convergent with prior measurements:** decode-block map (`flash_partial` occupancy/IC-bound at 33 GB/s,
   not HBM-bound); register-blocking refuted (1.07–1.08×, regresses); L-sweep flat; LDS Phase-5 (prior arc,
   same 0.5–0.74×). Five independent measurements agree.

## Gate result

| gate | result |
|---|---|
| isolated ≥1.3× vs hoisted @ KV1024 or 4096 | **FAIL** — the core (cooperative LDS) lever is 0.50–0.77× |
| build full WMMA kernel | **not built** — would confirm a 5th convergent negative; violates "don't keep complexity that loses" |
| in-model gate | **not reached** (isolated failed) |
| default change | **none** — hoisted stays the default |

## Verdict

**REFUTED on performance for the decode regime — for the levers we explored.** The high-occupancy-WMMA /
cooperative-LDS levers that win in the GEMM/prefill regime do **not** transfer to low-M decode: the
GEMV-structured `hoisted` flash already exploits split-K occupancy and lets the Infinity Cache serve K/V, and
explicit LDS staging is measured 0.5–0.77× (slower). NOT blocked by codegen — WMMA is revived and correct; it's
a regime mismatch for *these approaches*.

**CORRECTION (do not over-claim a "floor"):** freshly-measured llama.cpp on this same XTX
(`qk-llama-baseline-xtx-20260617.md`) is **~context-flat decode (99.5→92.2 tok/s, −7% to ctx4096)** while
tinygrad decays −43% — i.e. llama's decode attention is *cheap at long context*. So decode attention is **NOT at
a fundamental floor**; an efficient context-flat kernel demonstrably exists on this hardware (llama's tuned
FA2-style flash-decode). What is refuted is **our specific LDS/WMMA levers**, not the existence of a faster
kernel. Matching llama's flash-decode structure (efficient KV streaming, not LDS re-tiling) remains an **open,
harder** kernel arc — closed only for the approaches tried here.

## Lasting assets (not wasted)

- **WMMA custom-kernel revival** (`spec_tensor` rule + authoring) — a real capability unlock, committed. It is
  the right tool for the **PREFILL** regime (large M, compute-bound, where WMMA + LDS tiling wins — cf. the
  Phase-5 GEMM probe 3.79× and `amd_copy_matmul` ~760 GFLOPS). The natural next home for this asset is a
  **prefill attention / GEMM** arc, not decode.
- The isolated harness + baseline (`extra/qk_decode_attention_v3.py`) and the decode-block map remain.

## Next recommendation

Decode-side: **stop** — every bounded and codegen-local decode-attention lever is now shipped (hoisted) or
refuted (register-blocking, cooperative-LDS, WMMA-at-decode). The residual decode gap to llama is structural
(low-M GEMV regime). Highest-value redirections: (a) apply the revived WMMA to **prefill** attention (large-M,
the regime where it wins); (b) revisit speculative decoding (algorithmic, the only large decode upside);
(c) the 14B/32B model matrix. Do **not** build the full v3 WMMA kernel — the regime mismatch is measured.
