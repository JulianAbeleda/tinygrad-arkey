# BUILD SPEC (deepseek): Fast MoE via the packed-WMMA substrate — exhaustive

**Companion to** `docs/moe-fast-design-scope-20260722.md` (the design + principle rationale — read it first). This is the *executable* spec: exact code anchors, exact artifacts, hard stops.

## 0. Task framing (READ — this is how you must work)
- **You are extending the existing substrate to MoE. You write ZERO new kernels.** The per-expert GEMM is the *existing* `PackedWmmaPrefillCandidate.run()`. If you're writing a matmul kernel or `__builtin_amdgcn`, you're off-spec — stop.
- **P0 is a HARD STOP before any scaffolding.** Do P0, report the artifact, wait for Claude to gate it. Do NOT scaffold the capacity-permute structure until P0 says GO — if P0 is NO-GO the whole skeleton changes (see §Fallback). You have charged ahead before; do not here.
- **Per-artifact proof, never prose.** Correctness = numbers vs an oracle. WMMA = per-kernel `__WMMA` call-site dumps, never aggregate grep counts. Model speedup = **BoltBeam**, never an isolated-kernel microbench (the flash lesson: isolated 1.6× ≠ model win).
- **Refactor steps are behavior-preserving.** Where a step restructures working code (P2), the *current output is the oracle*; match it, don't change it.
- **Honest fallback.** If a step is genuinely blocked, report the exact file:line and what it can't express — do NOT hardcode, carve out of the optimizer, or fake a number. (These are the exact failure modes from prior work.)
- **Reuse principles (minimization-principles):** never add an op you can compose; generate don't hand-write; rules-as-data (geometry is searched data); don't fight the compiler (capacity-static, no `NOOPT`/carve-outs); AOT. Orthogonality/centralization/modularization/abstraction: only `ExpertWeights` changes; the GEMM stays expert-agnostic.
- Single GPU lane; commit on master, trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`; run the suite each commit.

## 1. Verified starting facts (line refs — do NOT re-derive)
- MoE modeling complete: `model.py:_feed_forward` (401), `ExpertWeights` (299), `pairwise_topk` (314), router `ffn_gate_inp`, combine `(x_down*probs).sum(axis=2)`, shared expert, multi-arch config (1165/1173).
- Slow spot: `ExpertWeights.__call__` does `self.weight[sel]` over a **dense** `Tensor.zeros(num_experts,out,in)` (302) — dynamic-weight gather, no packed bytes, substrate can't touch it.
- Substrate (reuse verbatim): `extra/qk/prefill/packed_wmma_prefill_candidates.py:PackedWmmaPrefillCandidate.run()` (167) — static packed weight + `x_batch` → view-chain → `x_batch @ b.T` on WMMA. Shape/identity-agnostic.
- **Packed-weight contract (P1 anchor):** `prefill_routes.py:_is_q4k_linear` (76) — a packed linear has `q4k_storage` (bytes) **and** `prefill_packed_weight()` (method). Q6 = `q6k_storage`. The route calls `lin.prefill_packed_weight()` (174).
- **Route spec (P4 anchor):** `PrefillLinearRouteSpec("direct_packed", quant, role, m, n, k)` (`prefill_routes.py:230`); roles are `ffn_gate`/`ffn_up`/`ffn_down` (`_direct_packed_module_role`, `_direct_packed_role` 90/206); per-role opts `_direct_packed_opts` (104).
- **Geometry (P3 anchor):** `PACKED_WMMA_GEOM` (candidate:36) keyed `(quant, role)`; `gate_combo` (105) declines off-table shapes; search = `extra/qk/bubblebeam_futuresight.py`.
- **Fallback anchor:** `model.py:758` already builds a `dense_config = replace(config, num_experts=0, ...)` — a MoE→dense reduction path exists to lean on for the §Fallback interim.

## 2. P0 — DE-RISK (do FIRST; hard stop; this decides the whole skeleton)
**Question:** does a `sel`-driven **capacity scatter/gather** (static output shape) capture + execute + numerically match through `prefill_v2`'s real JIT path at prefill scale?

**Build the minimal probe (no packed weights, no WMMA yet — just the dispatch shape):**
1. Inputs: `x (B,T,dim)`, and `sel (B,T,k)`, `probs (B,T,k)` from the real `ffn_gate_inp`+`pairwise_topk` on random x.
2. Capacity `C = ceil(capacity_factor * k * T / num_experts)` (start `capacity_factor=1.25`).
3. Compute dispatch indices (standard one-hot/scatter dispatch): for each `(token, slot)` pair with expert `e=sel[...]`, assign a position `0..C-1` within expert `e` (prefix-sum/cumsum over the one-hot; overflow → drop). Produce `dispatch (num_experts, C)` (token index per expert-slot) and the inverse for combine.
4. Scatter `x` → `xe (num_experts, C, dim)` via `dispatch` (a `gather`).
5. Per-expert compute STUB — use a plain dense matmul or even identity (the GEMM is P2/P3; here you only test the *dispatch* survives).
6. Combine: scatter `xe_out` back to `(B,T,k,dim)` via the inverse indices, weight by `probs`, sum → `out (B,T,dim)`.
7. **Run it through the real prefill_v2 capture/JIT** (not a bare realize) at prefill shapes.

**Artifacts (all):** (a) it captures + executes without error through prefill_v2; (b) `max_rel_err ≤ 1e-2` vs a reference that computes the same routing densely (per-token, no permute); (c) confirm the intermediate shapes stay STATIC (no symbolic/dynamic-shape crash). **GO/NO-GO:** GO → capacity-permute skeleton is valid, proceed to P1. **NO-GO → STOP, report the exact capture-path blocker (file:line, what it rejects about the data-dependent scatter), and switch to §Fallback.** Do not scaffold past this.

## 3. The dispatch math (exact — don't guess)
- `one_hot = (sel.reshape(B*T,k,1) == arange(num_experts))` → `(B*T, k, num_experts)`.
- position within expert = `(one_hot.cumsum over the B*T axis) - 1`, masked to `< C` (overflow drop).
- `dispatch_idx`: for expert `e`, slot `p`, the flat token index → build via `scatter` of token ids into `(num_experts, C)`.
- `combine`: `out_token = Σ_slot probs[token,slot] * xe_out[sel[token,slot], pos[token,slot]]` — a `gather` from `xe_out` + weighted sum. All `gather`/`scatter`/`cumsum`/`arange` are existing primitives (§II.4 compose, add no op).

## 4. P1 — Loader: per-expert packed bytes
Give `ExpertWeights` the packed contract `_is_q4k_linear`/`_is_q6k_linear` expect:
- Add `q4k_storage`/`q6k_storage` holding the **3D** `*_exps` packed bytes, and `prefill_packed_weight(expert_idx)` returning expert `e`'s `(out, in)` packed slice — mirror how `nn.Linear` retains its packed bytes (find the nn.Linear `prefill_packed_weight`/`q4k_storage` construction and replicate per-expert).
- The GGUF `blk.N.ffn_*_exps.weight` is `(num_experts, out, in)` in Q4_K blocks; slicing expert `e` yields exactly the `(out,in)` packed weight `run()` consumes.
**Artifact:** `expert_e` packed bytes dequant-round-trip to the reference dense expert weight (`max_abs_err` ~ fp16 tol), for a couple of experts across Q4_K and Q6_K.

## 5. P2 — Restructure `ExpertWeights.__call__` (refactor; oracle = current dense-gather)
Replace `weight[sel]` with: **dispatch (P0 math) → per-expert `run()` → combine.** For gate/up/down, mirror `_feed_forward`'s chain but per expert:
- per expert `e`: `g = run(gate_e, xe[e], spec_gate_e)`, `u = run(up_e, xe[e], spec_up_e)`, `h = silu(g)*u`, `d = run(down_e, h, spec_down_e)`.
- Combine `d` back per §3, weight by `probs`. Shared expert unchanged.
- `spec_*_e = PrefillLinearRouteSpec("direct_packed", quant, role, m=C, n, k)` with role in {`ffn_gate`,`ffn_up`,`ffn_down`}.
**Oracle:** the **current** `ExpertWeights` (dense `weight[sel]`) output on the same inputs. `max_rel_err ≤ 1e-2`. This is a behavior-preserving refactor — match the oracle, don't change semantics. (Keep the old path available behind a flag until P5 passes.)
**Artifact:** correctness vs oracle + vs an fp32 numpy reference; full suite green.

## 6. P3 — Geometry (rules-as-data; GPU)
Expert GEMM shapes: gate/up `(m=C, n=hidden_dim, k=dim)`, down `(m=C, n=dim, k=hidden_dim)`, per quant. **First check whether existing `PACKED_WMMA_GEOM` `(quant, ffn_gate/up/down)` rows already cover these shapes** (if `C` and the expert dims match a dense entry, you may reuse). For uncovered shapes, run **BubbleBeam+FutureSight** to produce geometry and add rows (as *data*, mirroring the table's structure) — do not hand-tune.
**Artifact:** experts hit `run()` (not the declined fallback); per-kernel `__WMMA` dump shows tensor cores on gate/up/down expert GEMMs; correctness held.

## 7. P4 — Route registration
Register the per-expert GEMMs as covered prefill_v2 routes / warmstart entries (the `_prefill_v2_covered` + spec machinery) so the packed-WMMA path fires for experts end-to-end in the model, not just in a unit harness.
**Artifact:** in-model forward shows experts dispatching through `route_packed_wmma_prefill` (census/DEBUG), correctness held.

## 8. P5 — GATE (BoltBeam, model-level; GPU)
Full-model MoE prefill A/B via **BoltBeam**: (i) restructured vs the dense-gather baseline (kept behind the flag), (ii) vs llama.cpp `mul_mat_id` on the same MoE model.
**Artifact:** tok/s table + correctness + per-kernel WMMA confirmation. GO = faster than the dense-gather baseline with correctness held; report the gap to llama. **This is the deliverable** — not any isolated-kernel number.

## 9. Fallback (if P0 is NO-GO)
If the data-dependent capacity scatter can't capture through prefill_v2's JIT: **do not force it.** Interim = **dense-all-experts + mask** — every token through every expert (static, no dynamic dispatch), masked/weighted by the router. This reuses `run()` immediately (each expert is a static `(T, dim)@(dim,hidden)` GEMM, same as the dense FFN — reuse the `dense_config` machinery at `model.py:758`) and gives a first model-level number, at the cost of `num_experts/k`× wasted compute. Then the *real* project becomes teaching the capture path to handle data-dependent dispatch — the same class of work as the composite-reduce substrate extension. Report which regime you're in.

## 10. Sequencing + which phases need the GPU
- **No GPU (graph/correctness):** P1, P2 (correctness vs oracle is device-agnostic; confirm a CPU/CLANG backend is wired, else run cheaply on GPU without WMMA). Design/plumbing.
- **GPU required:** P0 (capture through the real path — ideally on device), P3 (WMMA geometry search on gfx1100), P5 (BoltBeam model-level). These serialize on the single GPU lane.
- **Order:** P0 (gate) → P1 → P2 → P3 → P4 → P5. Correctness track (P1/P2) can proceed off-GPU while the GPU lane is busy.

## 11. Reuse map
`run()` (unchanged) · `prefill_packed_weight`/`q4k_storage` contract (extend per-expert) · `PrefillLinearRouteSpec` + `ffn_gate/up/down` roles · `PACKED_WMMA_GEOM`+`gate_combo` (add rows) · `bubblebeam_futuresight` (search) · `_prefill_v2_covered` (register) · `_feed_forward`/`pairwise_topk` (keep) · `dense_config` @ model.py:758 (fallback) · `ggml_data_to_tensor` (loader).

## 12. One line
**P0 de-risk the data-dependent capacity scatter through prefill_v2 (HARD STOP) → P1 per-expert packed bytes → P2 restructure `ExpertWeights` to dispatch→per-expert `run()`→combine (oracle = dense-gather) → P3 searched geometry → P4 route registration → P5 BoltBeam model gate. Zero new kernels, reuse `run()`, capacity-static (AOT), per-artifact proof, honest fallback to dense-mask if P0 blocks.**
