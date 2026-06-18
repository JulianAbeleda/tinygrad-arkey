# docs/ — map

Single navigation source-of-truth for this fork's docs. The AMD-decode work produced a long
chronological probe log; the **verdicts are folded into the syntheses below** — start there, treat
the dated `*-plan/-result/-probe.md` files as provenance, not current state.

## ⭐ Start here (canonical, post-bank)

- **`amd-decode-banked-20260616.md`** — THE entry point. Final decode state (~64 tok/s / 63% llama),
  the full lever map (shipped / tapped / refuted / gated), the machine-search system, resume pointers.
- `amd-decode-beyond-llama-roadmap.md` — the lever map with live statuses (parity vs beyond-llama).
- `gpu-performance-first-principles.md` — **canonical** bytes/math/overhead + roofline reference;
  diagnose the bucket BEFORE optimizing.
- **`../bench/README.md`** — the benchmark results index: every current number, its artifact, and the
  exact command to reproduce it.
- `amd-decode-capstone.md` — the decode ledger (23 → ~64 tok/s arc).
- `amd-decode-arc-synthesis.md` — synthesis through the primitive lens.

## 8B decode-attention + MMVQ frontier (2026-06-17 → 18) — latest state

The work after the decode bank. Closeouts/results are canonical; the many dated `qk-*` arc docs are provenance.

- **Decode-attention wins SHIPPED (byte-identical greedy, default-on):**
  - `qk-8b-attention-fusion-result-20260617.md` — flash-decode threshold 1024→512 (+12.8% ctx520).
  - `qk-8b-flash-variant-result-20260617.md` — `hoisted` exp + L=128 default (+11–29% across ctx).
  - `qk-gqa-coop-vector-load-result-20260617.md` — `gqa_coop_vec` default → decode-attention slope gap CLOSED.
- **Q4_K MMVQ int-dot line — CLOSED:** `qk-mmvq-int-dot-closeout-20260618.md` (**read this**) — the
  consolidated bank. SHIPPED `_sdot4`→native signed dot4 via `__builtin_amdgcn_sudot4` (fixed a latent
  unsigned-bug; value-tested; used by no default path); 128-thread/row sudot4 kernel 57% correct (beats opaque
  52%) but whole-linear REFUTED by the q8-pack wall (reuse ceiling 2 + ~7µs pack floor); int-dot FFN refuted.
  - Key sub-arcs (provenance): `qk-dot4-isa-audit-20260618.md` (the sudot4 fix + RDNA3 dot4 ISA map),
    `llama-q4k-mmvq-scheduler-audit-20260618.md` (llama's MMVQ decomposition),
    `qk-mmvq-llama-scheduler-probe-verdict-20260618.md`, `qk-mmvq-sudot4-full-linear-arc-20260618.md`,
    `qk-q8-activation-lifecycle-verdict-20260618.md`, `qk-mmvq-{codegen,deep-linearizer,fused-coop-row}-*`.
- **Current decode standing:** ~66–69% of llama via the shipped coop + flash-decode routes. Residual MMVQ gap =
  per-thread codegen (tinygrad-internals). 14B/32B pivot deferred per standing preference.

## Active / open frontiers

- **`amd-decode-prefill-v2-increment1-20260617.md`** — **prefill v2 BUILT & WON: ~13x warm prefill** (189→2486
  tok/s, ~83% of llama) via concrete-ubatch + fp16 + realized-weights + warmstart-TC, gated `PREFILL_V2`,
  decode untouched. Quality gate PASSED (dNLL ~0, 8B). Corrects the Stage-0 gate's premise (lazy weights →
  realize/VRAM; per-shape opts; host-overhead confound). Gate: `amd-decode-prefill-v2-gate-20260616.md`.
- **`amd-decode-prefill-v2-increment2-20260617.md`** — **flash-prefill attention: GATED (banked)**. Attention
  is the next prefill bottleneck at long ctx (~51% @ sp=3072) but the tractable approaches are refuted.
- **`amd-decode-prefill-v2-increment2-phase5-correction-20260617.md`** — **CORRECTION + kernel-level
  confirmation**: a custom score-free fused attention kernel IS expressible/correct (bridge + capabilities +
  expressibility proven, `test_flash_prefill_custom_kernel*.py`), but **honest DEBUG=2 GPU time REFUTES it on
  perf (~170–760× SLOWER than SDPA**; the earlier ~2.7× were wall-clock artifacts). Score-free w/o LDS reuse =
  memory-bound; real flash-2 needs LDS tiling (BEAM-territory, hangs gfx1100). Flash-prefill banked; prefill v2
  rests at Increment 1. **Methodology lesson: GPU timing via DEBUG=2 `tm`, never wall-clock around `.realize()`.**
- `amd-decode-prefill-plan.md` — the original prefill diagnosis (~2% of llama; LDS cache-blocking). Superseded
  as the active plan by prefill v2 above, but still the canonical root-cause reference.
- Phase-2 decode docs (2026-06-16): `amd-decode-sequential-tax-profile`, `…-overlap-feasibility-spike`,
  `…-overlap-derisk`, `…-two-queue-probe` (**overlap GATED** on a 2nd compute ring), `…-demotion-search`
  (B3 done), `amd-decode-flash-attention-plan` (flash SHIPPED).
- Direction + status: `structure/Development/machine-search-decode-context-plan-2026-06-16.md`;
  running log `structure/Development/session-handoff.md`.

## Machine-search system (shipped this arc)

The bounded search loop, dogfooded on B3. Code: `extra/qk_search_spec.py` (schema authority),
`extra/qk_nll_eval.py` (decode-path dNLL gate), `extra/qk_demote_search.py` (orchestrator). Result:
`amd-decode-demotion-search-20260616.md`.

## Architecture references (live)

- `amd-decode-harness-architecture.md`, `amd-decode-qk-storage-architecture.md`,
  `amd-decode-primitive-v2-design.md`, `amd-decode-bandwidth-roofline.md`,
  `amd-decode-packed-{load-lowering,qk-tile-design,qk-semantic-op}.md`.

## Historical — the decode-arc probe log

Dated scope/result docs whose verdicts are now captured in the syntheses above. Kept for provenance;
**not current state** (several carry a SUPERSEDED header).

- *"current state" docs, now superseded by the bank:* `amd-decode-current-verdicts.md`,
  `amd-decode-methodology-and-roadmap.md`, `amd-decode-final-report.md`, `amd-decode-hypothesis-statement.md`,
  `amd-decode-consolidated-first-principles.md`, `amd-decode-optimization-plan.md`.
- *bottleneck diagnosis & probes:* `amd-decode-rootcause`, `…-fix-plan`, `…-perlayer-plan`,
  `…-validate-plan`, `…-memory-access-audit`, `…-dequant-instruction-count`, `…-latency-vocabulary`,
  `…-dp4a-vocabulary`, `…-prefetch-plan`, `…-mirage-probe`.
- *kernel/TC/GEMM probes:* `…-option1-result`, `…-option1-corrected`, `…-batched-tc-{plan,result}`,
  `…-warmstart-plan`, `…-verify-loop-plan`, `…-fusion-probe-plan`, `…-vdot-amort-plan`,
  `…-amortized-quant-plan`, `…-scale-and-vdot4-plan` (`amd-loop-…`), `…-semantic-family-b`,
  `…-lossy-quant-search`.
- *levers later synthesized:* `amd-decode-speculative-plan` (B5), `amd-decode-prior-art`.

## Flywheel sub-arc (model-to-kernel triage/generation) — concluded

Read the postmortem first; the learned model added no value at the current feature set, the
native-matmul loop substrate works (decoupled from the decode bar).

- `amd-decode-flywheel-postmortem.md` (read first), `amd-decode-loop-substrate.md`,
  `amd-decode-flywheel-proof-plan.md` (2.6k-line plan), `amd-decode-kernel-optimization-flywheel.md`,
  `amd-decode-ansor-direction.md`, `amd-decode-loop-live-plan.md`,
  `flywheel-judging-rewrite-scope.md`, `flywheel-rewrite-ubuntu-handoff.md`,
  `qwen-json-eval-objective-scope.md`, `research-paper-brief.md`.

## Other subsystems

- **PSP / boot** (separate from decode): `amd-kdb-root-cause.md`, `amd-linux-psp-good-trace.md`,
  `amd-ubuntu-boot-prompts.md`, `amd-remote-dropout-investigation.md`.
- **Reference research:** `amd-rocm-llamacpp-research.md` (llama.cpp/ROCm/MMQ deep dive).

## Upstream tinygrad docs (not fork-specific)

`index.md`, `quickstart.md`, `mnist.md`, `nn.md`, `dtypes.md`, `env_vars.md`, `runtime.md`,
`tinygpu.md`, `tinybox.md`, `showcase.md`.
