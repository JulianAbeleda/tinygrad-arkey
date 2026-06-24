# Prefill Per-Role Transfer Attribution — Result (2026-06-23)

## Verdict: `PREFILL_TRANSFER_ATTRIBUTED` — the gap is small-shape WG-starvation, much smaller than the stale headline, and is a bounded per-shape-config fix (NOT machine search)
Executed the synced per-role in-model prefill time-tax the owner asked for. It answers **"why does a parity-class GEMM
not transfer to whole prefill?"** — and revises the premise: on the concrete chunk the dependency-free graph-GEMM is
**within 2.5 % of the Tensile route** in whole-model GPU-busy. The residual is **small-N WG-starvation** (kv_proj),
not a fundamental transfer failure. Attribution-only (PROFILE); decode untouched; no kernels; no search; no defaults.

## Method (harness SOP)
`extra/qk_prefill_per_role_time_tax.py`: TinyJit prefill graph **captured under `Context(PROFILE=1)`** (so the HCQ
graph carries per-kernel signals — the bug that zeroed the first attempt was capturing *outside* PROFILE), median-of-5
ProfileGraphEvent GPU-busy, per role. Kernel name `prefill_graph_gemm_512_N_K` encodes the shape → per-role achieved
TFLOPS. Concrete start_pos=0 chunk. Attribution-only authority (not a promotion number), per `HARNESS_GUIDE.md`.

## Per-role result (graph-GEMM route, concrete chunk, total GPU-busy 152.1 ms)
| role | shape | ms | share | TFLOPS | vs parity (63) | vs Tensile |
|---|---|---:|---:|---:|---|---|
| ffn_gate_up | 512×12288×4096 | 58.73 | 38.6 % | **63.2** | **parity** | ours 58.73 < Tensile 60.64 → **ours faster** |
| ffn_down | 512×4096×12288 | 33.05 | 21.7 % | 56.1 | 89 % | Tensile 28.49 (65 TFLOPS) → **Tensile +14 %** (deep-K) |
| qo_proj | 512×4096×4096 | 22.56 | 14.8 % | 54.9 | 87 % | Tensile route doesn't cover (stays graph-GEMM) |
| kv_proj | 512×1024×4096 | 14.61 | 9.6 % | **21.2** | **34 % — WG-STARVED** | Tensile route doesn't cover |

GEMM = 84.7 % of GPU-busy; non-GEMM ~15 % (copy/cast `E_`, reductions `r_` = norm/rope/dequant, attention).

## Cross-route comparison (the decisive diff)
With `PREFILL_TENSILE_GEMM=1` the route sends **only ffn_gate_up + ffn_down** through Tensile (Tensile covers FFN
only); qo/kv stay on graph-GEMM. Whole-model GPU-busy: **graph-GEMM 152.1 ms vs Tensile 148.3 ms = Tensile +2.5 %**
(ours *wins* gate_up; Tensile wins down by 4.5 ms). **The "66 % vs 87 %" stale headline is not reproduced on the
concrete chunk.**

## Attribution against the owner's candidate causes
1. **Route coverage** — RULED OUT: all GEMM roles fire graph-GEMM (coverage complete).
2. **Graph integration / copies** — MINOR: the route inserts `a/c.contiguous()`; non-GEMM ~15 %, not dominant.
3. **Shape mismatch** — **CONFIRMED DOMINANT**: one kernel config (waves2×2 / bk32) is tuned for the large gate_up
   shape (parity) but degrades on small/deep shapes — **kv_proj N=1024 is WG-starved** (grid 8×4 = 32 workgroups ≪ 96
   CUs → 21 TFLOPS / 34 %), qo_proj 87 %, ffn_down deep-K 89 % (Tensile per-shape-tuned wins +14 %).
4. **Chunking** — separate axis (concrete vs whole multi-chunk); not measured here.
5. **Non-GEMM** — ~15 %, not the lever. 6. **Scheduling** / 7. **Host** — GPU-bound (PROFILE), not isolated.

## The bounded fix (non-search)
**Per-shape kernel config selection**, not a blind machine search:
- **kv_proj (N=1024):** the standout — use a smaller tile (bn=64 or 32) so the grid fills the GPU (32 → 64/128
  workgroups), fixing the 34 % → ~parity. Biggest relative win.
- **ffn_down (deep K=12288):** a deeper-K / per-shape config to recover the +14 % Tensile takes.
- gate_up is already at parity (even beats Tensile) — leave it.
This is a **small enumerated config-per-shape** (a handful of tile choices keyed on N/K), ISA-audited and gated on
**whole-prefill synced (W==P) transfer** — bounded, deterministic, not a search.

## The same lesson as decode
The winning kernel exists; the transfer gap is at the **lifecycle/shape boundary** — here, one-config-fits-all
WG-starves the small-N roles. The decode analogue was the buffer-identity slice materialization. Both are bounded
integration fixes, not new kernels or searches.

## Caveat / next
This is the **concrete start_pos=0 chunk**. The stale `1983 (graph-GEMM) vs 2673 (Tensile)` whole-prefill numbers
(which imply a far larger gap) were **not reproduced** here and need a **synced whole multi-chunk prefill
re-measurement** to confirm or retire — the symbolic later-chunk attention growth is the untested axis. If that
re-measure also shows near-parity, prefill is at rest pending the per-shape kv_proj config micro-fix.

## Files changed
New: `extra/qk_prefill_per_role_time_tax.py` (reusable synced per-role profiler) + this doc +
`bench/qk-prefill-post-decode-parity-frontier/per_role_time_tax.json`. README/handoff updated. **No `tinygrad/`
source, no kernels, no search, no default flips.** Decode untouched.

## Git status
Clean before; adds one tool + one doc + one artifact + doc updates.
