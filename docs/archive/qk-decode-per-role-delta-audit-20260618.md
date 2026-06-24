# Final decode per-role delta audit — where the ~1.45–1.5× llama gap lives (2026-06-18)

Audit (no build, no route) fulfilling priority-1 of `what-makes-a-performance-primitive-efficient-20260618.md`.
Quantitatively attributes the remaining llama-vs-tinygrad **decode** gap to each role, with Amdahl ceilings, and
assigns every residual a state. Authority: in-model **W==D** for tok/s; isolated role %peak for per-role efficiency;
eager DEBUG=2 only for anatomy. Every number tagged **[M]** measured / **[I]** inferred / **[H]** hypothetical.

## Authority numbers

- **Decode (current banked, W==D) [M]:** 68.2 / 66.4 / 60.7 tok/s @ ctx 512/1024/4096 = 14.67 / 15.06 / 16.48
  ms/token (`bench/qk-ctx-sweep-20260618/wd-result.json`, `qk-decode-banked-reproduce-20260618.md`).
- **llama.cpp decode [M]:** ~98.6 / 97.6 / 92.2 tok/s = ~10.14 / 10.25 / 10.85 ms/token
  (`qk-llama-token-primitive-accounting-20260617.md`). **Gap @ctx1024 = 15.06/10.25 = 1.47×.**
- **Decode is weight-bandwidth-bound [M]:** both read ~4.68 GB quantized weights/token; llama MMVQ ~626 GB/s
  (~70% HBM peak), tinygrad matvec ~349 GB/s aggregate pre-coop. Role time ∝ weight_traffic / achieved_%peak.

**Reconciliation note:** the accounting doc (44–48 tok/s) and block-map (43.5 tok/s) are **PRE-coop provenance**.
This audit uses **post-coop** isolated role %peak (the shipped `qk-mmvq-coop-*` + `gqa_coop_vec` line, banked in
`qk-8b-decode-banked-20260617.md`) with measured weight-traffic shares (`bench/qk-mmvq-coop-q4k-ffn/role_inventory.json`,
`…-ffn-down/`). Time-shares are **[I]** from the traffic/%peak model (decode is BW-bound), anchored to the W==D total.

## Per-role delta table

%peak: llama ~70% all roles [M, rocprof]. tinygrad = post-coop isolated [M]. Traffic% [M]. Decode-time share [I]
(traffic/%peak, normalized so matvec≈62% of decode, attention≈16%, norm/elementwise/reduce≈19%, tail≈3% — the
post-coop split inferred from block-map anatomy rescaled by the coop speedups). Amdahl = e2e ceiling **if that role
reached llama's ~70% peak** (requires the full MMVQ activation lifecycle unless noted).

| role/family | traffic% [M] | tg %peak [M] | llama %peak [M] | role gap | decode-time share [I] | Amdahl e2e ceiling [I] | limiting factor | status |
|---|---:|---:|---:|---:|---:|---:|---|---|
| **Q4_K ffn_gate/up** | 44.0 | 41 | ~70 | ~1.7× | **~25.7%** | **+11.9%** | fp-dequant ALU; full MMVQ needs q8 lifecycle | **open — deferred deep** (q8 side-channel D) |
| Q4_K attn_q/o | 14.5 | 29 (coop) | ~70 | ~2.4× | ~12.0% | +7.5% | coop coalescing shipped; rest = full MMVQ | **shipped** (coop); residual deferred deep |
| Q6_K ffn_down | 15.7 | 39 (coop) | ~70 | ~1.8× | ~9.7% | +4.5% | coop shipped; rest = full MMVQ | **shipped**; residual deferred deep |
| Q4_K ffn_down | 10.9 | 35.5 | ~70 | ~2.0× | ~7.4% | +3.8% | split-K fp; subordinate, same fp/int wall | **open — subordinate** (no standalone arc) |
| Q6_K lm_head | 10.8 | 51 (coop) | ~70 | ~1.4× | ~5.1% | +1.4% | coop shipped; near settled | **shipped** (mostly closed) |
| Q6_K attn_k/v | 1.3 | 14 | ~70 | ~5× | ~2.2% | +1.8% | coop works isolated but Amdahl tiny | **sub-gate** (not routed; ~+0.5% e2e) |
| **decode attention** | — | — | — | ~2× share | ~13–18% [M block-map] | **≤+3%** | `gqa_coop_vec`; slope closed | **shipped** (mostly closed) |
| norms/RoPE/residual/SwiGLU | — | — | — | — | ~12–19% combined [M] | <+3% (fusion only) | many small elementwise; GPU-bound | **audit-only** (below gate; see below) |
| graph/runtime (1000 vs 260 progs) | — | — | — | — | host ~0% [M, W==D] | low | symptom of per-role matvec granularity, not launch cost | **refuted as host issue** |

## Amdahl roll-up — why no bounded edit closes the gap

Summing the matvec Amdahl ceilings **if every role reached llama's ~70% peak** [I]: ffn_gate/up +11.9%, attn_q/o
+7.5%, Q6_K ffn_down +4.5%, Q4_K ffn_down +3.8%, lm_head +1.4%, attn_k/v +1.8% → compounded ≈ **+27–30% e2e**,
i.e. ≈ the entire 1.47× llama gap (attention ≤+3% on top, mostly closed). **The whole decode gap is the matvec
efficiency gap** (349→626 GB/s), exactly as the accounting doc concluded.

The decisive point for "no bounded edit": **every one of those role ceilings is gated behind the same wall** — the
full llama MMVQ requires **q8_1 activation + native dot4 + block-amortized affine**, and tinygrad's byte-identical
fp path is **ALU-ceilinged at ~41–51%** (handwritten fp also ~49%, zero spills — `q4k-fp-coop-codegen-quality-scope-20260618.md`).
The int-dot kernel is faster but the **q8 activation pack/reuse economics** make the whole linear lose or sub-gate
(`qk-mmvq-int-dot-closeout-20260618.md`). So the residual is **not N bounded edits** — it is **one deep
activation-lifecycle change** (q8 side-channel, deferred D, ~+3–4% EV for gate/up alone) that even if fully built
captures only a fraction, because the q8 reuse ceiling is 2 and k/v are Q6_K.

## Residual assignment (close criterion — every residual classed)

| residual | state | reason |
|---|---|---|
| Q4_K ffn_gate/up (the dominant ~25.7%) | **open → deferred deep** | reachable only via q8 side-channel (deep fused-norm, multi-output, lossy, ~+3–4% EV) → machine row `decode_q4k_ffn_q8_sidechannel` |
| Q4_K attn_q/o, Q6_K ffn_down, Q6_K lm_head coalescing | **shipped** | coop default-on; the only bounded wins, taken |
| Q4_K attn_q/o / Q6_K ffn_down residual to 70% | **deferred deep** | same q8/full-MMVQ wall as gate/up; folded into the side-channel row |
| Q4_K ffn_down | **open — subordinate** | smaller than gate/up, same wall, no standalone arc earned |
| Q6_K attn_k/v coop | **sub-gate** | isolated win, ~+0.5% Amdahl → not routed |
| ffn_gate coop routing (+1–2.3%) | **sub-gate** | measured in-model, below ≥5% gate → row `decode_q4k_ffn_coop_subgate` |
| decode attention | **shipped / mostly closed** | `gqa_coop_vec`; residual ≤+3% → audit row `decode_attention_residual_audit` |
| norms/RoPE/elementwise | **audit-only** | block-map ~12–19% combined but spread over ~310+73 tiny kernels; fusion refuted/low-EV; only reopen if a ≥5% fused target is named (none) |
| graph/runtime overhead | **refuted as host issue** | W==D host-sync ~0%; it is a *symptom* of matvec granularity, not a launch cost |

**Close criterion met:** the table is filled; every residual is shipped / refuted / deferred-deep / sub-gate /
open-with-a-row; and the summed ceilings (~+27–30%) show the gap is one deep MMVQ-lifecycle wall, not a bounded edit.

## Contradictions found / superseded
- The accounting doc and block-map state tinygrad ~43–48 tok/s and per-role 10–41% peak — **PRE-coop**; superseded
  by the post-coop banked line here (68/66/61; lm_head 51%, ffn_down 39%, attn_q/o 29%). They remain valid as
  *anatomy/provenance*; their absolute tok/s and worst-role %peak are stale.
- The existing machine-search rows doc ranks `mmvq_q6k`/`mmvq_q4k` "full work-decomp" as the open #1/#2 — those are
  now **shipped (coop) / deferred-deep (q8 lifecycle)**; superseded by `qk-machine-search-primitive-rows-20260618.md`.

## Provenance
`qk-llama-token-primitive-accounting-20260617.md` (+ bench), `qk-decode-banked-reproduce-20260618.md` (+
`bench/qk-ctx-sweep-20260618/wd-result.json`), `qk-8b-decode-block-primitive-map-20260617.md` (+
`bench/qk-decode-block-map/result.json`), `qk-mmvq-{q6k-lm-head,coop-ffn-down,coop-q4k-attn}-*`, `gqa_coop_vec`,
`q4k-fp-coop-codegen-quality-scope-20260618.md`, `qk-mmvq-int-dot-closeout-20260618.md`,
`q8-sidechannel-ffn-verdict-20260618.md`. No kernel/model/default changes.
