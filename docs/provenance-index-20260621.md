# Docs Supersession Index (2026-06-21)

Mechanized map of the fork's `docs/*.md` (roadmap #5 consolidation). **No docs deleted/moved** — this only
classifies, so the ~251 canonical->dated-doc pointers stay intact. Regenerate: `PYTHONPATH=. .venv/bin/python bench/qk-active-surface-reduction/build_docs_index.py`. Backing data: `bench/qk-active-surface-reduction/docs_index.json`.

**Totals:** 647 fork docs — canonical **4** · current **286** · provenance **357**

## Authority order (read these; ignore the rest unless tracing provenance)
1. `docs/current-project-state-handoff-20260621.md` — canonical current state
2. `docs/README.md` — curated navigation map
3. `bench/README.md` — bench/evaluator map
4. `structure/Development/performance-primitive-research-principles.md` — method authority
5. `docs/project-north-star-llama-and-lifecycle-search-20260620.md` — completion definition

## Provenance (historical; superseded by the canonical syntheses) — by topic
These dated `*-result/-scope/-probe/-audit.md` are the chronological probe log; their verdicts are folded into
the canonical docs above. Kept for history, **not authority**.

| topic | provenance docs | CURRENT authority (read this instead) |
|---|---:|---|
| prefill | 126 | handoff §2-3 + docs/prefill-policy-integration-result-20260620.md |
| decode | 81 | handoff §4 + docs/post-matmul-pv-decode-strategic-scope-20260621.md (REST_DECODE) + docs/fused-flash-concrete-gate-result-20260621.md |
| q8 | 59 | handoff §2 (q8 opt-in, default-off) + docs/q8-mmvq-lifecycle-deep-result-20260619.md |
| other | 41 | docs/README.md map |
| mmvq | 24 | docs/decode-gap-is-attention-not-weight-gemv... (closed) + handoff §3 |
| tensile | 9 | docs/amd-prefill-lds-gemm... + handoff (prefill kernels closed) |
| wmma | 7 | handoff (WMMA decode not pursued; prefill Tensile path) |
| gemm | 3 | docs/amd-prefill-lds-gemm-not-refuted... (handoff) |
| harness | 2 | docs/harness-contract-audit-20260621.md |
| flash | 2 | docs/fused-flash-concrete-gate-result-20260621.md (FAIL_LOCAL_AB -> REST) |
| quant | 1 | handoff §2 |
| attention | 1 | docs/llama-flash-attn-tile-oracle-result-20260621.md (oracle) + decode pointers |
| spec | 1 | handoff (speculative decode rested) |
