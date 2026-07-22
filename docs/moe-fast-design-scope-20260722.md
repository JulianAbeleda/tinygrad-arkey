# Design Scope: Fast MoE via the existing packed-WMMA substrate

**Audience:** a model/agent picking this up later, cold. Self-contained. **Thesis:** MoE-fast here is *not* a new kernel — it is **feeding MoE's per-expert static-Q4 GEMMs into the packed-WMMA substrate you already have**, behind a routing/permute layer. The expensive part is already solved; the work is shape + plumbing + one integration check.

**Principle spine (from `knowledge_base/principles/minimization-principles.md` + the four design principles):**
- **Never add an op you can compose (§II.4):** MoE = router + topk + gather/scatter + matmul, all existing primitives. Add no new op.
- **Generate, never hand-write (§III.9):** reuse the generic packed-WMMA GEMM; write zero expert kernels (the llama.cpp `mul_mat_id` counter-example is what we're NOT doing).
- **Rules-as-data (§III.10):** expert tile geometry is *searched data* (`PACKED_WMMA_GEOM`-style), not a hand-coded pass.
- **Don't fight the compiler (§III.12):** keep it in the AOT/static-shape lane (capacity routing) — no carve-outs, no `NOOPT`, no per-op exemptions.
- **AOT everything (§II.6):** capacity-based routing → static shapes → compile-once.
- **Orthogonality / centralization / modularization / abstraction:** routing, dispatch, GEMM, and combine are separable; the GEMM stays expert-agnostic (one central path); only `ExpertWeights` changes.

---

## 1. Current state (verified, with line refs — do NOT re-derive)

- **MoE modeling is COMPLETE and general** — `model.py:_feed_forward` (401+): router `ffn_gate_inp`, `pairwise_topk` (314), two gating styles (sigmoid+bias 414–417 / softmax 419–420), expert gate/up/down via `ExpertWeights`, weighted combine `(x_down * probs).sum(axis=2)`, optional shared expert. Multi-arch config from GGUF (qwen3moe/qwen35moe/deepseek, lines 1165/1173). A MoE GGUF routes correctly today.
- **The ONE slow spot is `ExpertWeights` (model.py:299–305):**
  - `self.weight = Tensor.zeros(num_experts, out, in)` — **dense, not packed-Q4, no `prefill_packed_weight()`**.
  - `__call__` does `self.weight[sel]` — a **data-dependent gather of weights to tokens** → `(B,T,k,out,in)` materialization (prefill VRAM blowup) and a **dynamic-weight matmul** the substrate can't touch.
- **The substrate is shape-agnostic and reusable AS-IS** — `PackedWmmaPrefillCandidate.run()` (`extra/qk/prefill/packed_wmma_prefill_candidates.py:167`): given a **static** packed weight + `x_batch`, builds the fp16-overlay view-chain (`bitcast/reshape/pad/expand/reshape/bitcast`) and does `x_batch @ b.T` on WMMA. It never asks *which* weight it is.
- **Packed bytes come from `lin.prefill_packed_weight()`** (candidate:181) — retained for `nn.Linear`, NOT for `ExpertWeights`.
- **Geometry is a frozen searched table** — `PACKED_WMMA_GEOM` (candidate:36), 6 `(quant, role)` combos for the *dense* model's shapes at pp512; `gate_combo` (105) declines anything else. Search tool = **BubbleBeam+FutureSight** (`extra/qk/bubblebeam_futuresight.py`).
- **Route coverage is nn.Linear-only** — `_prefill_v2_covered` (model.py ~828).

## 2. The design (target shape)

**Core reframe:** the expert weights are **static** (N per-expert Q4 tensors); only *routing* is dynamic. So replace *"gather weights to tokens"* with *"permute tokens to experts → per-expert static GEMM → combine."* Four orthogonal layers:

1. **Routing (index compute, data-dependent, pure primitives).** router logits → `pairwise_topk` → `sel` + `probs`. **Reuse `_feed_forward`'s existing routing verbatim.** Add: from `sel`, compute per-expert **capacity assignment** — for each expert a fixed `C` token slots, and the scatter/gather index tensors (token→slot, slot→token). Composes from `arange`/`scatter`/`gather` primitives. Orthogonal to the GEMM.
2. **Dispatch (permute, data-dependent indices, STATIC output shape).** Scatter token activations into a `(num_experts, C, dim)` buffer using the routing indices. Capacity `C = ceil(num_experts_per_tok * T / num_experts * capacity_factor)`. Overflow drops / underflow pads — **static shape**, AOT-friendly.
3. **Expert GEMM (THE SUBSTRATE, reused unchanged).** For each expert `e`: `run(expert_e_packed_weight, tokens_e_buffer, spec_e)` — the *existing* `PackedWmmaPrefillCandidate.run()`. Each expert's weight is a **static packed-Q4 slice**; each call is exactly a covered static GEMM. **Zero new kernel.** (gate→silu→up→down mirrors `_feed_forward`'s expert_h/x_down, now as three per-expert packed GEMMs.)
4. **Combine (unpermute, pure primitives).** Scatter expert outputs back to token order, weight by `probs`, sum over the k slots — the existing `(x_down * probs).sum(axis=2)` adapted to the permuted layout. Add shared-expert (dense, unchanged).

**Why capacity-static:** it is the single decision that keeps MoE in the AOT/static-shape lane so *everything above is reused untouched* (§III.12 don't-fight-the-compiler). Ragged/dynamic sizes would need a scheduler extension (see §6 fallback).

## 3. Reuse map (explicit — reuse, do not rebuild)

| Need | Reuse (file:symbol) | Change |
|---|---|---|
| router/topk/gating/combine | `model.py:_feed_forward` (401), `pairwise_topk` (314) | keep; only the expert-compute call changes |
| per-expert GEMM | `packed_wmma_prefill_candidates.py:PackedWmmaPrefillCandidate.run()` (167) | **none** — call per expert |
| packed bytes | `lin.prefill_packed_weight()` pattern | extend retention to 3D `*_exps` (per-expert slice) |
| geometry search | `bubblebeam_futuresight.py` (score/rank) | run for expert shapes → new table rows |
| geometry table | `PACKED_WMMA_GEOM` (candidate:36) + `gate_combo` (105) | add expert `(quant, role)` combos as **data** |
| route registration | `_prefill_v2_covered` / the covered-linear plumbing | register per-expert GEMMs as covered routes |
| GGUF quant load | `gguf.py:ggml_data_to_tensor` (40) — dequant; packed-byte retention path | route `*_exps` through packed retention, not dequant |
| dynamic-dispatch lessons (if ragged ever needed) | the composite-reduce substrate work (this session) | reference only |

## 4. Phased build (each phase = a committed artifact a reviewer re-runs)

- **P0 — DE-RISK FIRST (the one real unknown).** Does a `sel`-driven **capacity scatter/gather capture cleanly through `prefill_v2`'s JIT/AOT path**? Build the minimal permute (scatter tokens into `(num_experts, C, dim)`, per-expert GEMM stub, scatter back) and run **one forward pass** under the real prefill capture. **Artifact:** it captures + executes + numerically matches the current dense-gather path. **GO/NO-GO for the whole approach** — if the data-dependent scatter can't capture, this becomes a substrate-extension project (see §6), not a plumbing one. Do this before anything else.
- **P1 — Loader: per-expert packed bytes.** Give `ExpertWeights` a `prefill_packed_weight()`-equivalent that slices the 3D `*_exps` Q4 tensor per expert (mirror the nn.Linear retention). **Artifact:** `expert_e` packed bytes match the GGUF, dequant-round-trips to the dense weight.
- **P2 — Restructure `ExpertWeights.__call__`.** Replace `weight[sel]` with capacity-permute → per-expert `run()` → scatter+combine. **Oracle:** the *current dense-gather output is the correctness reference* — refactor to match it exactly (`max_rel_err ≤ 1e-2`). Behavior-preserving; only the mechanism changes.
- **P3 — Geometry (rules-as-data).** BubbleBeam/FutureSight search for the expert GEMM shapes (M=C, N/K=expert dims), across Q4_K/Q6_K and gate/up/down roles → populate the expert geometry table. **Artifact:** experts hit `run()` (not the declined fallback); per-kernel `__WMMA` dump shows tensor cores on the expert GEMMs.
- **P4 — Route registration.** Register per-expert GEMMs as covered `prefill_v2` routes / warmstart entries so the packed-WMMA path fires end-to-end in the model.
- **P5 — GATE (BoltBeam, model-level, NOT isolated kernel).** Full-model MoE prefill tok/s via BoltBeam A/B: restructured vs the dense-gather baseline, and vs llama.cpp `mul_mat_id`. **This is the real deliverable** — isolated-kernel speedups don't count (see the flash lesson: attention-in-isolation ≠ model win).

## 5. Principle compliance (map every choice)

- **Orthogonality:** routing / dispatch / GEMM / combine are separable; the GEMM is expert-agnostic; changing capacity or gating never touches the kernel.
- **Centralization:** ONE GEMM path (the existing packed-WMMA) serves dense linears *and* experts. No second regime, no MoE kernel.
- **Modularization:** `ExpertWeights` is the single module that changes; router + substrate + loader-core untouched. A new MoE arch = config, not code.
- **Abstraction:** `run()` is the abstract "static packed GEMM"; MoE is one caller alongside the dense FFN.
- **Generate-not-handwrite / rules-as-data:** zero expert kernels; geometry is searched data.
- **Don't-fight-the-compiler / AOT:** capacity-static shapes; no carve-outs, no `NOOPT`, compile-once.
- **Authored-vs-generated (§I):** the authored surface added is small (the permute/dispatch in one class + a geometry table populated by search); kernels + geometry stay on the generated/data side of the `sz.py` boundary.

## 6. Risks + fallbacks

- **The one real risk — P0.** Data-dependent capacity scatter through the specialized JIT. If it captures: the rest is plumbing. If it doesn't: you need to teach the capture path to handle data-dependent dispatch — the *same class of problem* as the composite-reduce work (make the machine handle dynamic structure natively, don't carve it out). **Interim fallback if blocked:** dense-compute-all-experts + mask (every token through every expert, masked by routing) — static, correct, wasteful, but reuses the substrate immediately and unblocks a first model-level number while the real dispatch is built.
- **Capacity waste** (padding/dropping) — a `capacity_factor` knob; report drop rate; it's the standard MoE tradeoff.
- **Decode vs prefill.** The current `weight[sel]` gather may be *fine for decode* (T=1, gather k experts cheaply). Target the restructure at **prefill** (where it blows up); optionally keep the gather path for decode. State which path each regime uses — don't silently regress decode.
- **Quant coverage.** Confirm the loader packs `*_exps` for both Q4_K and Q6_K (mirror the dense roles); a MoE model may mix.

## 7. Guardrails (for the executor)

- **No hand kernels.** Reuse `run()`. If you're writing a matmul kernel, you're off-design.
- **Per-artifact proof.** Per-kernel `__WMMA` dumps (not aggregate counts); numerics vs the dense-gather oracle and vs fp32, never vs another MoE path.
- **Model-level gate is BoltBeam**, not an isolated-kernel microbench (the flash lesson: isolated 1.6× ≠ model win).
- **Behavior-preserving where it's a refactor** (P2): the dense-gather output is the oracle.
- **Run the suite each commit; commit on master; honest fallback** — stop and report the exact blocker (file:line) rather than force/fake; never carve MoE out of the machine to "unblock."

## 8. One line
**Fast MoE = restructure `ExpertWeights` from per-token weight-gather into capacity-permute → per-expert calls of the existing packed-WMMA `run()` → combine, with searched geometry data and covered-route registration. The kernel is reused, not written; the design stays orthogonal/central/AOT-static; the only real unknown is whether the data-dependent capacity scatter survives the JIT (P0 de-risk first). Prove it at the model level with BoltBeam, not in kernel isolation.**
