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

## Active / open frontiers

- `amd-decode-prefill-plan.md` — **prefill** = the largest untapped gap (~2% of llama; LDS
  cache-blocking codegen). Diagnosed, not solved.
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
