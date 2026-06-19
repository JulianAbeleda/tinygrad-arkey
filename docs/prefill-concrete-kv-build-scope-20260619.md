# BUILD SCOPE — concrete-KV prefill (1.24x validated) + Option-B TC-attention (unvalidated stack)

Lever (measured): symbolic-KV attention is the prefill bottleneck; concrete start_pos -> 1.24x byte-identical
(`prefill-l1-l2-result`); externally corroborated (strix-halo: attention 15-35%, rocWMMA FA). Build-readiness below.

## What's VALIDATED vs NOT
- VALIDATED: concrete start_pos=0 -> 1.24x e2e, rel_err 0 (standalone clock-controlled A/B, `qk_gateup_sched_ab`/
  the concrete-vs-symbolic A/B). Matmul/dispatch/graph-count NOT levers (exhausted).
- NOT validated: (a) the 1.24x in the REAL `model.generate` loop (only the isolated jit A/B); (b) the multi-chunk
  (start_pos>0) case + its recompile cost; (c) **Option-B TC-attention on concrete KV IN-MODEL** (probe: 2.56x
  standalone, 0.79x symbolic-in-model, concrete-in-model "blocked by jit arg plumbing" = NEVER MEASURED); (d) dNLL
  of Option-B's fp16 scores.

## The generate-loop reality (model.py:1221 `generate`)
Prefill uses `v_start_pos.bind(start_pos)` (BOUND symbolic) so ONE prefill_v2_jit replays across all 512-chunks
(start_pos = 0,512,1024,...). That binding is WHY it's symbolic — jit reuse across chunks. Concrete start_pos
breaks that reuse: one concrete jit PER distinct start_pos.

## Plan
- **P0 — validate 1.24x in the real loop (single-chunk):** add a concrete-start_pos prefill path for start_pos=0
  (the common prompt<=512 case): pass concrete `0` (not `v_start_pos.bind(0)`) -> a cached concrete-0 jit. Gate:
  generation byte-identical to symbolic; warm pp512 >= 1.2x. Confirms the lever in production, not just isolated.
- **P1 — multi-chunk strategy (prompt>512):** start_pos values are DETERMINISTIC + bounded (0,512,...,<=max_ctx ->
  ~K=ceil(max_ctx/512) values). Cache ONE concrete jit per start_pos value (K jits vs 1 symbolic). Tradeoff: +K
  compiles (one-time / disk-cached; amortized over server reuse) for 1.24x replay on every chunk. Measure: total
  prefill time incl. compile, single-shot vs warmed; decide concrete-all-chunks vs concrete-first-chunk-only.
- **P2 — VALIDATE Option-B on concrete KV in-model (THE un-done measurement):** wire `_explicit` (from
  `qk_prefill_tc_wr_softmax_probe.py`: Q@Kᵀ TC + fp16 scores + softmax + P@V TC, GQA broadcast) into `_attention`'s
  prefill branch GATED on concrete start_pos; measure pp512 (Option-B vs SDPA, both concrete, clock-controlled) +
  dNLL <= 0.01. This tests whether TC fires on concrete KV (the inference) and whether it stacks on the 1.24x.
  IF it doesn't fire / fails dNLL -> Option-B stays unwired (the 2.56x doesn't transfer); concrete-KV alone is the win.
- **P3 — wire Option-B** only if P2 passes (>= some gain, dNLL OK); flag-gated; SDPA fallback for symbolic/chunked.

## Gates
correctness: byte-identical generation (concrete-KV); dNLL <= 0.01 (Option-B fp16 scores). speed: pp512 >= 1.2x
(P0), Option-B stack measured (P2). fallback: symbolic path unchanged; decode W==D untouched (prefill-only).
multi-chunk: correct across all start_pos; compile cost characterized.

## Risks / open
- **Option-B-on-concrete may NOT fire TC in-model** (the core unvalidated assumption) -> P2 is the gate; if it fails,
  only concrete-KV's 1.24x stands.
- Multi-chunk: K concrete jits = K compiles (latency for one-shot; fine for server). Memory: K captured graphs.
- fp16 scores (Option-B) numerics -> dNLL gate.
- Concrete start_pos loses the single-jit-reuse; net win depends on replay-1.24x > amortized-compile-cost.

## Files
levers: `prefill-l1-l2-result-20260619.md`, `prefill-symbolic-kv-tc-attention-scope-20260619.md`,
`prefill-graph-ramp-benign-20260619.md` (exhaustions). probe: `extra/qk_prefill_tc_wr_softmax_probe.py`.
generate loop: `tinygrad/llm/model.py:1221`. external: `findings-external-verification-20260619.md`.
