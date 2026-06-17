# AMD decode — BANKED (2026-06-16)

Single-entry closeout for the decode-optimization arc. The decode result is **banked**: it is a
strong, exact, fully-documented win, and the remaining levers are mapped + honestly gated. Future
work should start here.

## Final state

Qwen3-8B-Q4_K_M, RX 7900 XTX (gfx1100), HBM peak 859 GB/s, llama.cpp = ~101–106 tok/s.

| | tok/s | % llama |
|---|---:|---:|
| start of arc (Q4_K primitive only) | 23 | 22% |
| **banked decode (default-on, ffn_down demote)** | **~64** | **~63%** |
| per-model (generated policy): 14B / 32B | 40.6 / 17.2 | 62% / 56% |

The standalone int-dot Q4_K GEMV kernel sustains **76% of HBM peak** vs llama's 57% end-to-end —
machine search *exceeds* the reference at the kernel level (banked, `KERNEL_BEATS_LLAMACPP`).

## What shipped (exact unless noted)

- **Q6_K primitive coverage** (default-on) — the dominant e2e win, byte-identical.
- **`Q6K_COVER_MORE`** (attn_v + lm_head, default-on), byte-identical.
- **ffn_down Q6→Q4 demotion** (`Q6K_DEMOTE_FFNDOWN` / `QK_DEMOTE_TENSORS`) — +14%, dNLL ≈ free.
- **Flash-decode** (`FLASH_DECODE=1`) — exact, long-context (2.4× at ctx 3072); default-off
  (crossover ~ctx 400).
- **Default-on flip** — primitives auto-on for AMD GGUF loads (12→55 tok/s out-of-box, exact).

## Lever map — why decode is banked, not abandoned

| lever | status |
|---|---|
| Q6_K coverage / COVER_MORE / ffn_down demote / flash | ✅ shipped |
| In-pattern Q6→Q4 demotion search (B3) | ✅ searched — **tapped at ~64 tok/s** within the dNLL budget; lm_head fast (74% llama) but rejected on quality (`amd-decode-demotion-search-20260616.md`) |
| Faster per-kernel GEMV (B1, int-dot in-graph) | ❌ refuted — batch-1 occupancy ceiling |
| Norm-into-GEMV fusion | ❌ refuted — single-accumulator kernel blocks it; non-exact; already lazily fused |
| **Overlap non-GEMV (B2, ~+30%)** | 🔒 **gated** — HBM ~58% idle during GEMV proves it's reclaimable, but tinygrad's AMD backend has ONE compute ring (`ops_amd.py:1001`); needs a 2nd hardware compute ring + per-ring submit routing (`[runtime]` surgery) before the cross-layer scheduler (`amd-decode-two-queue-probe-20260616.md`) |
| **Sub-4-bit (Q3/Q2 on the Q4 bulk)** | 🔒 **gated** — needs a new quantizer + new GEMV kernel (dangerous-power surface) |
| Prefill | ❌ separate problem, ~2% of llama — LDS cache-blocking codegen (`amd-decode-prefill-plan.md`); the largest absolute headroom left |

**The pattern:** every remaining decode lever is either refuted, already tapped, or requires adding
dangerous-power surface (2nd compute ring / new sub-4-bit kernel). Banking is the principled stop —
not capitulation. The two gated builds are scoped and killable when/if revisited.

## The lasting asset — a bounded machine-search system (shipped, dogfooded)

`search spec → candidate generator → isolated runner → quality-gated scorer → accepted policy`,
proven end-to-end on the B3 demotion lever (the quality gate demonstrably rejected a
tempting-but-degrading config). Reusable for future levers / models / search spaces:
- `extra/qk_search_spec.py` — typed schema authority (rows, constraints, accepted-policy records).
- `extra/qk_nll_eval.py` — teacher-forced decode-path dNLL quality gate.
- `extra/qk_demote_search.py` — the search orchestrator (pure reuse of the existing measure CLIs).

## Resume pointers (if decode is reopened)

1. **Prefill** (biggest untapped): `amd-decode-prefill-plan.md` — LDS-staged matmul tiling or call rocBLAS.
2. **Overlap build**: start at Milestone 0's gate — wire a 2nd compute ring, re-run
   `extra/qk_two_queue_probe.py` (A‖B should jump > 1.2×), then the cross-layer scheduler.
3. **Widen the search**: point `qk_search_spec` at 14B/32B or new search spaces (flash threshold,
   storage) — the loop is proven.

Banked-state anchors: `amd-decode-capstone.md`, `amd-decode-beyond-llama-roadmap.md`,
`amd-decode-arc-synthesis.md`, `amd-decode-measurement-confounds.md`, and the six
`amd-decode-*-20260616.md` Phase-2 docs.
