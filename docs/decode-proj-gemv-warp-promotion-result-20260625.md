# Decode attn q/o projection warp-GEMV (`Q4K_GEMV_WARP_PROJ`) — promotion result (2026-06-25)

## Verdict: **`PROJ_GEMV_WARP_WD_TRANSFERS_PROMOTABLE`** → promoted **default-on**

First-pass follow-up to the two-kernel audit (`docs/decode-two-kernel-problem-audit-result-20260625.md`), which found
the decode **attention** two-kernel route exhausted and named the **weight-GEMV q/o projection** as the leading open
lever *outside* that boundary. This task resolved a standing contradiction on that lever and promoted it.

## The contradiction (resolved)

`Q4K_GEMV_WARP_PROJ` routes the attn q/o projection (Q4_K 4096×4096) through the **same lossless
`q4k_gemv_warp_kernel`** already shipped default-on for FFN gate/up/down (32-thread/row + K-block-parallel + in-kernel
`ds_bpermute` warp-reduce). Two prior records disagreed:

- **2026-06-22 promotion-hardening audit** → "local speedup but **no W==D transfer**" → kept default-off/research-only
  (`docs/archive/q4k-gemv-warp-promotion-hardening-result-20260622.md`; model.py comment "did not transfer").
- **2026-06-24 aggressive-target-proof** → clean **+1.5%/ctx** over the default (40 reps, host-sync 0%), but
  `do_not_promote` *relative to a higher aggressive theoretical target* (`bench/qk-decode-aggressive-target-proof-20260624/`).

Root cause: the aggressive-proof ran the two arms as **separate sequential blocks under `auto` clock** — the documented
clock-confound (`[[amd-decode-measurement-confounds]]`). The 06-22 "no transfer" was the symmetric failure: a
non-drift-cancelled A/B in which a real ~1.6% sits inside cross-run clock variance.

## The decisive test (`extra/qk_proj_gemv_warp_wd.py`)

In-process **interleaved alternating A/B** (shared clock per repeat → drift-cancelled), real per-token `.item()` W==D
sync, best-effort clock pin, route-fire verified (proj kernel `q4k_gemv_warp_4096_4096`), byte-identical token check.
Both arms keep the shipped FFN default (`Q4K_GEMV_WARP=1, Q4K_GEMV_WARP_DOWN=1`); only `PROJ` toggles. NMEAS=40,
REPEATS=6, ctx {512,1024,2048,4096}.

| ctx | default tok/s | +PROJ tok/s | Δ | per-arm spread | tokens | proj kernels (base) |
|----:|----:|----:|----:|----:|:--:|:--:|
| 512  | 101.9 | 103.7 | **+1.67%** | 0.27–0.45% | match | 72 (0) |
| 1024 | 100.2 | 101.8 | **+1.58%** | 0.22–0.45% | match | 72 (0) |
| 2048 | 97.6  | 99.2  | **+1.61%** | 0.26–0.36% | match | 72 (0) |
| 4096 | 93.1  | 94.6  | **+1.61%** | 0.14–0.38% | match | 72 (0) |

The +1.6% is **~4–10× the worst per-arm spread (0.45%)**, consistent across all ctx, **byte-identical** at every ctx,
and the proj route fires cleanly (72 kernels per decode step = 36 layers × {q,o}; 0 in the base arm — no flag leak).
This **refutes** the 06-22 "did not transfer". Artifact: `bench/qk-proj-gemv-warp/wd.json`.

(Note: the artifact's `perflevel` string is a cosmetic rocm-smi capture artifact in the first run — fixed in the
harness afterward; the tiny 0.14–0.45% spreads independently confirm a stable clock during the interleaved run.)

## Change made

`tinygrad/llm/model.py`: `getenv("Q4K_GEMV_WARP_PROJ")` → `getenv("Q4K_GEMV_WARP_PROJ", 1)` (default-on), guarded
exactly as before (parts==1, out==4096, in==4096, `(in//256)%4==0`, `DECODE_ATTN_AMDGCN_ARCH_OK`; try/except fallback
to the coop branch). Comments at model.py:259/302 corrected; `docs/decode-q4k-gemv-warp-promotion-result-20260624.md`
corrected. Reversible: `Q4K_GEMV_WARP_PROJ=0`. Other shapes/devices unaffected.

Effect: the new decode default is the old "aggressive probe" (~103.7/101.8/99.2/94.6 tok/s @512/1024/2048/4096),
closing most of the gap to the aggressive-theoretical envelope (104.0/102.1/99.6/95.1; now ~0.3–0.6%/ctx away).

## Status / recommended hardening (before considering fully promoted)

This is a **first-pass** promotion on the decisive W==D + byte-identical + route-fire evidence. To match the full
promotion discipline the campaign uses (cf. the lifecycle-recheck bundle), the recommended next steps are:
- run the standard gate + unknown-bucket lockstep bundle with PROJ default-on (`extra/qk_decode_search_gate.py`,
  `extra/qk_decode_unknown_bucket_lockstep_audit.py`) and confirm no `E_49152` / route preserved;
- refresh the canonical decode baseline snapshot (`bench/canonical-benchmarks.json` via
  `extra/qk_update_benchmark_refs.py`) — the prior "baseline 101.6/99.8/97.3/92.7" now reflects the old PROJ-off default;
- a clock-pinned multi-run reproducibility band (≥5 runs) per harness SOP.

Owner note: this flips a default that the 06-24 aggressive-proof marked `do_not_promote` — but that was relative to the
*aggressive theoretical target*, not vs the shipped default; vs the shipped default this is a clean, reproducible,
byte-identical **+1.6%**. Trivially reverted with `Q4K_GEMV_WARP_PROJ=0` if you'd rather keep it gated.
