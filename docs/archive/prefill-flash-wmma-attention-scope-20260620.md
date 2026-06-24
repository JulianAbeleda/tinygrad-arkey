# Scope: Real Flash/WMMA Attention for Prefill (the deferred big prize)

Date: 2026-06-20
Repo: `/home/ubuntu/tinygrad-arkey`, branch `qk-prefill-flag-leak-resolution`. GPU gfx1100 (RX 7900 GRE).
Model Qwen3-8B-Q4_K_M (n_heads=32, n_kv_heads=8 → GQA G=4, head_dim=128, 36 layers, rope_dim=128).
Predecessors: `docs/prefill-graph-route-attribution-result-20260620.md` (Step-1 audit → Branch B),
`docs/prefill-branch-b-tc-attention-result-20260620.md` (shipped fusion win).

## TL;DR — the prize moved, and it is cheaper than it looked

The Step-1 audit said "attention is 47%, llama is 4.4%, the gap lives in attention." That is still true, **but the
regime matters**, and the shipped Branch B fusion win already changed the picture:

| regime | attention share BEFORE | attention share NOW (after Branch B) | fires? |
|---|---|---|---|
| **concrete** (first chunk, start_pos=0) | ~18% | **~5%** (fusion already cut it — near llama's 4.4%) | ✅ default-on |
| **symbolic** (chunks 2+, long ctx) | **~47%+** | ~47%+ (unchanged — fusion can't fire) | ❌ `isinstance(int)` guard blocks it |

**So the concrete regime is essentially DONE (~5% ≈ llama). The entire remaining prize is the SYMBOLIC regime** —
subsequent prefill chunks and long context, which is most of a long prompt. Two facts make this the lever:

1. **Symbolic-start_pos codegen makes the SAME attention ~3× more expensive** than concrete (the 30% kernel
   `r_2_512_*start_pos*` vs the 10.7% concrete `r_16_32_*`). So a big chunk of the 47% is codegen inefficiency,
   not fundamental work — recoverable by making chunks concrete OR by a kernel that schedules symbolic-KV well.
2. **At long KV the explicit path materializes the full `Hq×T×KV` fp16 score tensor** (117 MB/layer at KV=3584).
   That HBM traffic grows with context and is exactly what llama's flash avoids with online (tiled) softmax.

### Payoff arithmetic (symbolic regime)
Symbolic forward ≈ 0.53 non-attention + 0.47 attention. Cut attention 0.47 → ~0.05 ⇒ forward × **~1.7×** on
symbolic chunks. The symbolic ladder is 66% of llama (= llama 1.5× faster); 1.7× on the bulk would bring symbolic
prefill to **~llama parity or better**. This is the whole "34%-to-llama gap." Confirmed: the gap lives in
symbolic-regime attention.

## Mechanism inventory (what exists, what blocks WMMA today)

- **Current symbolic path:** `q.scaled_dot_product_attention(k,v,mask,enable_gqa=True)` (model.py:887). Lowers to
  the `r_*start_pos*` reduce — no tensor cores, full-score implied, symbolic KV.
- **Current concrete path (shipped):** explicit `qg@kgᵀ → fp32 softmax → P@vg` (model.py:876–885), better-fused,
  byte-identical, but **still materializes scores** and **still no WMMA** (`tc_fired=False`).
- **Why WMMA doesn't fire:** TC is applied only via the warmstart table `_WARMSTART_OPTS`
  (postrange.py:337–376), keyed `(frozenset(concrete out-dims), concrete reduce)` → forced `Opt(OptOps.TC,...)`.
  It is populated for the FFN/projection linears (`_prefill_v2_opts`, model.py:51–56) but **not for the attention
  matmul shapes**. No BEAM (hangs gfx1100), so nothing else inserts TC. Add attention-shape keys → WMMA fires (in
  the concrete regime, where dims are int; symbolic reduce isn't an int so it won't key — see Increment 2).
- **Flash infra that exists:** `extra/qk_flash_decode.py` — a hand-written online-softmax flash that **already
  handles SYMBOLIC KV length** (symbolic split-count `S=cdiv(start_pos+1,L)`, Smax-sized buffers, bound/unbound
  start_pos twins; model.py:863–874). BUT it is **T=1 (decode), scalar (no TC, no LDS)** — one query row,
  128 threads = head_dim. A prefill flash needs 2D tiling over T×KV + causal masking + (for the prize) TC frags.
  The symbolic-length plumbing is directly reusable; the compute kernel is not.
- **WMMA-scheduling wall (POWN):** caps large-GEMM WMMA at ~42 TFLOPS (software-pipelined-K-loop codegen tinygrad
  can't express). **Largely sidestepped by flash:** flash tiles KV so each tile's Q@Kᵀ has small reduce (Hd=128)
  and P@V tiles the K=KV dim — neither is the deep-K pipeline the wall is about. The hand-asm LDS GEMM line
  (`amd-prefill-lds-gemm-not-refuted`, ~55–61 TFLOPS) is a separate, GEMM-shaped frontier; attention is not that
  shape. Net: the wall is a weaker constraint here than for the FFN matmuls.

## Increments (sequenced cheapest-first; each independently gated & shippable)

### Increment 0 — Force concrete + reuse the SHIPPED fusion path (NO new kernel) ★ DO FIRST
Hypothesis: `PREFILL_CONCRETE_KV=1` makes every chunk concrete → the already-default-on Branch B fusion fires on
all chunks → symbolic attention 47% collapses toward the concrete ~5%, capturing most of the prize with **zero
new kernel code**. Cost: one prefill jit per distinct start_pos (0,512,1024,…; ~8 jits for a 4096-prompt) +
score materialization per chunk.
- **Build:** none (paths exist: `PREFILL_CONCRETE_KV`, the fusion path). Pure measure + gate.
- **Measure (synced, same-process interleaved, the iron-law arbiter):** whole-forward ms at start_pos ∈
  {0,512,1536,3072} with `PREFILL_CONCRETE_KV=0` (today: first chunk concrete, rest symbolic SDPA) vs `=1` (all
  concrete + fusion). Plus end-to-end `generate()` on a long prompt: total prefill wall + per-distinct-start_pos
  **compile cost** (the real tax) + greedy-exact.
- **Risks:** (a) K-jit compile time may dominate for prompts with many distinct start_pos (the comment at
  model.py:1284 already flags "pays off only when cached / prompt≤512"); (b) score materialization at KV=3584
  (117 MB/layer) may eat the win at long context — this is the boundary where Increment 2 (flash, no
  materialization) becomes necessary.
- **Decision:** if Increment 0 captures most of the symbolic prize at small/moderate KV with acceptable compile
  cost → ship it (gated like the others) and only build flash for the long-KV tail. If materialization/compile
  kills it → go straight to Increment 2.

### Increment 1 — WMMA on the explicit attention matmuls (concrete regime)
Add warmstart TC-opt keys for the attention Q@Kᵀ and P@V shapes so the explicit path's matmuls use tensor cores.
- **Build:** find a forced-TC schedule (gate-validated, NO BEAM) for the two attention matmul shapes (Q@Kᵀ:
  out[T,KV] reduce Hd=128; P@V: out[T,Hd] reduce KV), add to `_WARMSTART_OPTS` keyed on concrete dims. Mirror
  `_prefill_v2_opts` / `extra/qk_prefill_gate.py` (the per-shape schedule finder).
- **Payoff:** bounded — still materializes scores; helps only if the matmuls (not materialization/softmax)
  dominate the explicit path. Likely small after Increment 0 already cut concrete attention to ~5%. **Probably
  skip unless Increment 0's per-kernel breakdown shows the matmuls dominating.**
- **Risk:** the per-shape TC schedule may not validate without BEAM; causal half-emptiness wastes half the WMMA.

### Increment 2 — Fused causal flash prefill kernel (the real long-context prize) ★ THE BIG BUILD
A hand-written (HIP-C or UOp) flash-attention kernel for prefill: 2D tiles over (T queries × KV keys), online
softmax (running max/sum, no full-score materialization), causal mask per tile, GQA broadcast, WMMA for the
per-tile Q@Kᵀ and P@V. Symbolic-KV-capable via the flash-decode symbolic-S plumbing (so it fires on symbolic
chunks directly — **no PREFILL_CONCRETE_KV, no K-jit tax**).
- **Build (substantial, multi-day):** new kernel modeled on `qk_flash_decode.py` but T-tiled + causal + TC.
  Reuse: symbolic-length twins, Smax buffers, the assemble→lib / UOp-custom-kernel path, the LSE-reduce structure.
  New: 2D Q/K/V tiling, per-tile causal masking, WMMA fragment loads (RDNA3 `v_wmma`, see
  `extra/gemm/rdna3_wmma_matmul.py` + `tinygrad/renderer/amd/elf.py` LDS-descriptor helper just added), online
  softmax rescale across KV tiles.
- **Payoff:** the full prize — attention → ~llama's ~4.4%, avoids the 117 MB/layer score traffic, no compile tax,
  works at any (incl. symbolic/long) context. Highest ceiling.
- **Risks:** correctness of online softmax + causal tiling (gate hard: rel RMSE + dNLL + greedy-exact); WMMA
  fragment layout for the skinny attention shapes (Hd=128); GQA broadcast in-tile; the partial WMMA-scheduling
  wall on P@V's K=tile; multi-kernel lib kernarg gotcha (flash-decode header note); single-shape best-of-N
  measurement discipline.

## Cross-cutting: measurement & gates (iron law — non-negotiable)
- **SYNCED only**, same-process interleaved arbiter (K forwards / one `dev.synchronize()` / total/K), clock pinned
  `high`. No cross-process clock comparison. No nosync `realize()` loops.
- Per change: rel RMSE < 1e-2 + sampled/chunked NLL dNLL ≤ 0.01 + greedy-exact + fallback + OOM.
- Flag-leak guard on every A/B: capture each jit BEFORE toggling the routing global; assert kernel-identity
  (the regression that produced the bogus Branch-B refutation — verify the env NAME the code reads AND that the
  regime satisfies the route guards).
- No BEAM (hangs gfx1100). Default-off unless owner-approved; gfx1100-restricted.
- Report every prefill number with its regime (concrete vs symbolic) AND clock provenance (prefill WMMA clock is
  volatile across sessions — see `amd-decode-measurement-confounds` / the prefill clock-authority note).

## Open questions the cheap probes resolve (before the big build)
1. **Does Increment 0 capture the symbolic prize?** Measure `PREFILL_CONCRETE_KV=1` + fusion at start_pos
   {512,1536,3072}. If symbolic attention collapses to ~5% like concrete → most of the prize is FREE (no flash).
2. **Where is the concrete/long-KV cliff?** At which KV does score materialization (117 MB/layer scaling) or
   K-jit compile cost overtake the win? That KV is the Increment-0→Increment-2 boundary.
3. **What dominates the explicit path's remaining ~5%** at moderate KV — matmuls (→ Increment 1) or
   softmax/materialization (→ Increment 2 only)? One per-kernel attribution answers it.
4. **Does a TC schedule validate for the attention shapes without BEAM?** A `qk_prefill_gate`-style schedule
   search on the two attention matmul shapes. Gates whether Increment 1 is even buildable.

## Recommended path
Do the **Increment 0 probe first** (cheap, no new code, possibly captures most of the prize). Branch on its
result: ship Increment 0 if it suffices at the prompt lengths that matter; build Increment 2 (flash) only for the
long-KV tail it can't cover; treat Increment 1 as an optional concrete-regime top-up gated on the attribution.
Sequencing: probe → (ship 0) → (flash 2 for long ctx) → (1 if attribution justifies).
