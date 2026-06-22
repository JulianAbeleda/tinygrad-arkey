# Decode Primitive Time-Tax Audit (after Route B attention closure) — Result

Date: 2026-06-22

Attribution/audit only — **no kernel optimization, no new primitives, no default change.** Builds a measured
one-token-decode decomposition so the next primitive target is chosen by token-share + W==D headroom, not intuition.
Authority: `extra/qk_decode_time_tax_audit.py` → `bench/qk-decode-time-tax-audit/latest.json` (ProfileGraphEvent
per-kernel GPU-busy, median-of-5; real per-token wall via the `.item()` W path). Qwen3-8B / RX 7900 XTX / gfx1100.

## Verdict: **`NEXT_PRIMITIVE_Q4K_GEMV_SCHEDULER`** — the FFN **Q4_K GEMV** (gate/up 24% + down 14% = **~38%** of decode) is the dominant, highest-headroom, **proven-transferring** tax. q8 FFN (+6%, lossy, already shipped opt-in) is the practical route; the larger prize is the **lossless** Q4_K GEMV schedule/codegen (our ~57% vs llama ~70% peak). The bounded int-dot path is closed, so the funded next step is a **focused FFN-GEMV re-diagnostic**, not blind deep codegen.

## Phase A — baseline reproduced
68.2 / 66.8 / 64.3 / 61.0 tok/s @ctx512/1024/2048/4096 (canonical 68.1/66.4/—/60.7 ✓). `gpu_busy_sum` (16.9–18.5ms)
**exceeds** the token wall (14.7–16.4ms) → decode is **GPU-bound; host/graph overhead ≈ 0** (kernels overlap;
confirms the prior GPU-bound finding). Commit/dirty stamped in the artifact.

## Phase B — one-token decode time-tax (share of GPU-busy)
| bucket | ctx512 | ctx1024 | ctx2048 | ctx4096 | path |
|---|---|---|---|---|---|
| **FFN gate/up** | 4.16ms / **25%** | 4.17 / **24%** | 4.18 / 24% | 4.08 / 22% | `q4k_gemv_partial_12288_4096` (Q4_K dequant-GEMV) |
| norm/rope/small ops | 4.04 / 24% | 4.04 / 23% | 3.99 / 22% | 3.96 / 21% | ~15 scattered `r_*`/`E_*` (rmsnorm, rope, kv-stack, softmax reduces) |
| FFN down | 2.52 / 15% | 2.50 / 14% | 2.43 / 14% | 2.17 / 12% | `q4k_gemv_4096_12288` + `q6k_coop_4096_12288` |
| attention compute | 2.19 / 13% | 2.52 / 15% | 3.17 / 18% | 4.36 / **24%** | flash decode (gqa_coop_vec) — grows with ctx |
| FFN activation | 1.55 / 9% | 1.56 / 9% | 1.55 / 9% | 1.55 / 8% | `E_49152` silu(gate)·up |
| attn q/o proj | 1.38 / 8% | 1.36 / 8% | 1.32 / 7% | 1.30 / 7% | `q4k_coop_4096_4096` |
| lm_head | 0.68 / 4% | 0.68 / 4% | 0.68 / 4% | 0.68 / 4% | `q6k_coop_151936_4096` |
| attn k/v proj | 0.40 / 2% | 0.40 / 2% | 0.40 / 2% | 0.40 / 2% | `q6k_gemv_1024_4096` |

**Aggregates:** FFN (gate/up + down + activation) = **48%@ctx1024 / 42%@ctx4096** (dominant). Attention (compute +
projections) = 25%→33% (grows with ctx). The single biggest **primitive kernel** is FFN gate/up (`q4k_gemv_12288_4096`)
at a **flat ~24%**.

## Phase C — controlled toggles (observed vs Amdahl; the transfer test)
| route (default-off) | Δ tok/s | targets | finding |
|---|---|---|---|
| q8 FFN handwritten (`Q8_FFN_HANDWRITTEN=1`) | **+6.7/+6.1/+5.4%** @512/1024/4096 (lossy, dNLL 0.0029) | ffn_gate/up | **+6% TRANSFERS** → FFN gate/up is **on the critical path**; implies a ~1.33× gate/up kernel speedup |
| B4/B5 attention AMDGCN (`hw128`) | +0.23/+1.98/+5.66% @1024/2048/4096 (exact) | attention | **SATURATES ~+5.7%@4096** despite a 2.4× cheaper combine → attention partly **OVERLAPS** (off critical path) |

**The decisive contrast:** an FFN-gate/up speedup (q8) **transfers** to W==D; an attention speedup (B5) **does not**
(it overlaps). So the lever is the **FFN GEMV**, not attention — exactly inverting where the bounded effort was.

## Phase D — primitive ranking
| primitive | share | lossless headroom | projected W==D | bounded action? | verdict |
|---|---|---|---|---|---|
| **FFN gate/up (Q4_K GEMV)** | 24% | ~57%→70% peak vs llama (~1.2×); q8(lossy) ~1.33× | **+4% lossless gate/up-only; +6% q8 (measured, lossy)** | re-diagnose schedule/occupancy (int-dot CLOSED) | **PRIMARY** |
| FFN down (Q4_K/Q6_K GEMV) | 14% | ~1.2× | gate/up+down together ≈ **+6% lossless** | same lane | SECONDARY (same lane) |
| norm/rope/small ops | 23% | ~1× (work-conserved) | ~0 (fusion CLOSED) | none | CLOSED |
| attention compute | 15→24% | 2.35× local | +5.7% SATURATED (overlaps) | none | CLOSED (B5: rest) |
| attn q/o/k/v proj | 10% | ~1.2× | +2% | folds into the GEMV lane | MINOR (same lane) |
| FFN activation | 9% | — | ~0 (fusion CLOSED) | none | CLOSED |
| lm_head (Q6_K) | 4% | ~1.2× | +0.8% (too small) | none | TOO SMALL |

**The only buckets with real W==D headroom are the Q4_K/Q6_K weight GEMVs** (FFN gate/up + down = 38%, projections
+10%). They are also the only ones **proven to transfer** (q8). Everything else is closed: attention (B5 saturation),
small-ops/activation fusion (decode-fusion-build ~0%, work-conserved), lm_head (4%, too small).

## Phase E — closed-lane review
| lane | status | new evidence? |
|---|---|---|
| Route B attention combine-only | CLOSED (B5: W==D saturates +5.7%, combine overlaps) | confirmed by this audit (attention off critical path) |
| FLASH_L=64 / scalar fused tiles / matmul-PV | CLOSED | none — no reopen |
| q4 int-dot MMVQ | CLOSED (whole-linear refuted; 57→70% residual = per-thread codegen) | time-tax shows the **share** (24%) is worth a re-diagnostic, but no NEW *bounded* action — the int-dot path stays closed |
| q8 FFN artifact route | SHIPPED opt-in (+6%, lossy) | this audit confirms it's the proven FFN lever; lossy → stays opt-in (no default promotion) |
| q8 native scheduler/codegen | CLOSED (ROADMAP_ONLY) | none |
| spec decode / prefill v2 default | CLOSED / decided | out of scope (decode audit) |

**Reopen test (data + blocker-relevance + bounded-action):** the FFN Q4_K GEMV has the **headroom** (38%, ~1.2× to
llama) and is **proven-transferring** (q8), but the prior blocker (per-thread codegen) **is still relevant** and the
*int-dot* bounded action is exhausted. → Reopen as a **diagnostic** (is there a non-int-dot schedule/occupancy/
dual-issue lever at the FFN shapes, given B5's standalone≠in-graph lesson?), **not** a blind deep-codegen commit.

## Phase F — answers
- **Top decode tax after attention closure:** the **FFN Q4_K weight GEMV** — gate/up (24%) + down (14%) = ~38%, the
  single dominant, proven-transferring share. (Attention is 25–33% but closed/overlapping; small-ops 23% but fusion-closed.)
- **What to work next:** the **Q4_K/Q6_K FFN GEMV schedule/codegen** (`NEXT_PRIMITIVE_Q4K_GEMV_SCHEDULER`). q8 FFN
  (+6%, lossy) is the practical already-shipped route; the *larger* target is the lossless GEMV reaching llama's ~70%
  peak.
- **Projected W==D upside:** lossless gate/up+down to llama parity ≈ **+6%** (clears +5%@1024); q8 (lossy) **+6%
  measured**. (Tempered by B5's overlap lesson — but q8's measured +6% confirms the FFN GEMV *does* transfer, unlike
  attention.)
- **Bounded next scope to fund:** a **FFN-GEMV re-diagnostic** — measure `q4k_gemv_12288_4096` / `4096_12288`
  in-graph vs llama at the exact FFN shapes, attribute the ~57→70% peak gap to schedule/occupancy/dual-issue vs
  per-thread codegen, and decide whether a **non-int-dot bounded lever** exists. Only commit to deep codegen if the
  diagnostic finds a bounded one; else the lossless ceiling is the tinygrad backend and **q8 opt-in is the practical cap**.
- **Do NOT pursue:** more attention combine/tile work (closed), small-op/activation fusion (closed), lm_head (too
  small), the q4 int-dot MMVQ path (refuted), or blind deep per-thread codegen before the diagnostic.

## Boundaries honored
Attribution/tooling/docs only. No kernel optimization, no new primitive, no default change, no q8/B4/B5 promotion.
`gqa_coop_vec` comparator SSOT; llama as the peak reference. Shares are of the GPU-busy sum (overlap not removed —
flagged); the q8/B5 toggles are the in-graph transfer ground-truth. Unrelated dirty work untouched.
