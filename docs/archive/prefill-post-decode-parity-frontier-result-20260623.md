# Prefill Post-Decode-Parity Frontier — Audit Result (2026-06-23)

## 1. Verdict: `PREFILL_FRONTIER_AUDIT_COMPLETE` + `PREFILL_TENSILE_GAP_ATTRIBUTED` + `PREFILL_MACHINE_SEARCH_NOT_READY` / `PREFILL_NEEDS_NONSEARCH_FIX_FIRST`
The prefill GEMM kernel is **already solved**: the dependency-free `build_gemm_lds2` is at **GPU-level parity-to-+10 %
with vendored Tensile** on the actual prefill shape. **Machine search is NOT justified** — the kernel isn't the
bottleneck, its tuning knobs are exhausted, and local kernel timing does **not** transfer to whole-prefill. The real
remaining lever is the **in-model integration penalty** (non-search). Decode untouched, no defaults flipped, no
kernels implemented, no search started.

## 2. Authority / config
HEAD `29e936f80`, gfx1100 (24 GB), Qwen3-8B-Q4_K_M, ROCm 7.2.4. Decode default = owned whole-cache tile (102–105 % of
llama) — **not touched**. Prefill: `PREFILL_V2` default-off (VRAM-auto, owner-decided off); `PREFILL_GRAPH_GEMM`
(dependency-free LDS GEMM) default-on **within** PREFILL_V2 on gfx1100; `PREFILL_TENSILE_GEMM` research-only.

## 3. Corpus reconciliation (`corpus_reconciliation.json`)
Trustworthy **synced** whole-prefill: symbolic V2 **1236** (~40 %) → graph-GEMM **1983** (~66 %) → Tensile **2673**
(~87 %) vs llama ~3020–3070. The dependency-free kernel is at **GPU-level parity-to-+10 %** with Tensile (gold-standard
74–78.6 vs 70.9; each-alone 63 vs 62). The historical "8 % behind / ~92 %" was **measurement artifact** (interleave
perturbation + batch-1 host overhead). **The entire bimodal/WMMA-boost/clock-lottery/SIA1/WGM8 saga was one flag-leak
bug — fully retracted.** WGM8 L2-locality and "latency-bound hidden by occupancy" were **refuted** (ours has higher
L2 hit; occupancy is an interior optimum).

## 4. Current prefill baseline (`baseline_prefill.json`) — harness SOP applied
Per `bench/qk-decode-eval/HARNESS_GUIDE.md` (Measurement-Authority: clean **synced** = authority; nosync = diagnostic):
- **Whole-prefill synced authority:** graph-GEMM **1983 tok/s (~66 % llama)** (reconciled arbiter).
- **Fresh synced check** (canonical `qk_prefill_tc_attn_concrete_gate.py`, burst K=8 + `dev.synchronize()`):
  concrete start_pos=0 chunk **3436 tok/s** (149 ms/512) — matches the documented 3394 headline but is a **single
  chunk, not whole-prefill** (subsequent symbolic-KV chunks are slower); not quoted as whole-prefill.
- **Excluded:** `qk_prefill_v2_measure` 4037 tok/s = the documented **nosync trap** (inflated ~2×), diagnostic-only.

## 5. Shape inventory (`shape_inventory.json`)
Primary GEMM shape **M=512, N=12288, K=4096** (ffn gate/up). Others (ffn down K=12288, q/o proj 4096², k/v proj 1024)
are tile-divisible. Attention compute (QK/PV) has **no** GEMM kernel either route (flash path). **Single-shape-family
caveat:** parity is exhaustively measured only on gate/up; other roles' in-model transfer not separately profiled.

## 6. Time-tax / bottleneck (`time_tax.json`)
`PREFILL_TAX_COMPUTE_BOUND_GEMM` at the kernel level — but the kernel is at Tensile parity, so the **real lever is the
in-model integration penalty**: isolated parity (63–78 TFLOPS) does not transfer (in-model gate/up ~22 TFLOPS;
concrete-chunk 3436 vs whole-prefill 1983). The fresh **synced per-role** in-model breakdown was **deferred** (the
prefill_v2 path is intricate; not ad-hoc-measurable) — that profiling is the first step of the non-search lane.

## 7. ISA audit (`isa_audit.json`) — `PREFILL_ISA_GAP_FOUND`
`build_gemm_lds2` static mix (751 insts): **32 v_dot (WMMA)** + 32 ds_load / 8 ds_store + 8 global loads + **557 VALU**
(dominated by `v_add_nc_u32`/`v_lshlrev`/`v_mul_lo` address arithmetic + `v_cvt_f16_f32` output convert). LDS 32768 B,
~256 VGPR (BK64 overflows the 256-VGPR wall). vs Tensile (MT128x128x16 DepthU=16, VGPR 256, LDS 25088): ours **+23 %
VALU** (8.66M vs 7.04M, PMC-exact), residual LDS bankcf 2.93 vs 0, but **higher L2 hit** (64.7 vs 56.6). The only real
residual is the VALU address arithmetic.

## 8. Tensile-class gap attribution (`tensile_gap_attribution.json`) — `PREFILL_TENSILE_GAP_ATTRIBUTED`
On the actual shape there is **no material kernel-GEMM headroom** (parity-to-+10 %). The only residual is **+23 % VALU
address arithmetic** — a **deterministic addressing-leanness** fix (hoist + strength-reduce per-iteration index math),
**not** a tuning knob. Prefetch (A and A+B PLR), WGM8 L2-locality, and occupancy were all **tried/refuted**. The
dominant whole-prefill gap (66 % → 87 %) is the **in-model integration penalty**, not the kernel.

## 9. Search-readiness decision (`search_readiness.json`, `next_action_decision.json`)
**`PREFILL_MACHINE_SEARCH_NOT_READY`.** Of the 6 search-justification criteria, **4 fail**: (1) not materially
bottlenecked at the kernel (Tensile parity), (2) local timing does **not** transfer, (5) tuning knobs **exhausted**
(BK/PAD/PLR/occupancy), (6) expected whole-prefill gain from a GEMM search ≈ 0. A search would optimize an
already-solved, non-transferring metric. → **`PREFILL_NEEDS_NONSEARCH_FIX_FIRST`.**

## 10. Next scope written
`docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md` (Phase 8B). Ranked non-search levers:
1. **In-model integration penalty** (the 66 %→87 % gap; needs a fresh synced per-role prefill time-tax first) — dominant, up to ~+20 pp llama.
2. **VALU address-leanness** micro-lever (the ISA-confirmed +23 % residual; deterministic, bounded, ISA-gated, must show whole-prefill transfer).
3. **Policy** (graph-GEMM default-on; PREFILL_V2 stays off) — owner calls, not perf work.

## 11. Files changed
New: this doc + `docs/prefill-frontier-rest-or-nonsearch-next-scope-20260623.md`; 9 artifacts under
`bench/qk-prefill-post-decode-parity-frontier/`. README + handoff updated. **No `tinygrad/` source, no kernels, no
default flips, no machine search.**

## 12. Git status
Clean before this task (HEAD `29e936f80`). Audit-only: 2 docs + 9 artifacts + doc updates. Decode untouched; defaults
unflipped.
