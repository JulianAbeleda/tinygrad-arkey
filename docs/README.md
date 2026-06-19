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
  exact command to reproduce it. **Includes "Which harness for decode tok/s — READ FIRST"** (use the clean
  `model.generate`-path CLI/W==D harnesses; the flash auto-bench's ~54 is contaminated, not a tok/s number).
- `qk-decode-banked-reproduce-20260618.md` — banked decode line reproduced on HEAD (68.2/66.4/60.7, W==D,
  host-sync 0%, whole stack default-on) + the harness lesson.
- `amd-decode-capstone.md` — the decode ledger (23 → ~64 tok/s arc).
- `amd-decode-arc-synthesis.md` — synthesis through the primitive lens.

## 8B decode-attention + MMVQ frontier (2026-06-17 → 18) — latest state

The work after the decode bank. Closeouts/results are canonical; the many dated `qk-*` arc docs are provenance.

- **`what-makes-a-performance-primitive-efficient-20260618.md` — READ THIS FIRST for the performance-primitive model and gap.**
  Consolidated source of truth for what makes a performance primitive efficient, using llama.cpp vs tinygrad as the
  case study: decode, lm_head, MMVQ, attention, spec, prefill, machine-search lessons, and every remaining path
  marked shipped/refuted/deferred/open.
- `performance-primitive-external-research-audit-20260619.md` — second-round external research audit across
  arXiv/OpenReview/ChinaXiv. Cites each paper/source, records the claim, checks whether it is true/applicable to this
  tinygrad gfx1100 project, and maps it to current or future primitive rows.
- `primitive-local-observability-search-scope-20260619.md` — scope for building primitive-local tooling instead of a
  generic profiler: read-only ledger first, then schema validators, runner wrappers, deterministic failure
  classifiers, guided search memory, and optional rocprof/SQTT counter plugins.
- `primitive-local-observability-search-result-20260619.md` — **executed PLO-1..PLO-6.** Adds
  `extra/qk_primitive_ledger.py`, a read-only primitive ledger/validator/classifier/search-memory/trace-plugin
  inventory that reconstructs current verdicts from existing artifacts without hardware execution.
- `primitive-local-observability-audit-20260619.md` — replay audit over the primitive ledger, including the TPE-7a
  rebindable-node artifact. Confirms graph-protocol prerequisite PASS while keeping in-model capture and artifact
  policy as remaining gates.
- `primitive-ledger-analysis-audit-20260619.md` — uses the primitive ledger for the intended analysis pass: decode is
  q8/MMVQ lifecycle-limited; prefill is graph/artifact-boundary-limited; broad kernel search is not supported by the
  current evidence.
- `primitive-pmu-observability-scope-20260619.md` — scope for using installed ROCm profiler tooling as the PMU oracle
  and building only the tinygrad primitive-local attribution layer needed around HCQ.
- `primitive-pmu-observability-result-20260619.md` — PMU-1..PMU-3 result: ROCm PMU works on HIP controls, but tinygrad
  HCQ is invisible to `rocprofv3` in the smoke; redirects to a tinygrad-native HCQ attribution adapter.
- `primitive-hcq-attribution-scope-20260619.md` — PMU-4 scope: tinygrad-native HCQ attribution for eager launches and
  graphs, producing Level-3 runtime/graph evidence without pretending to have PMU counters.
- `primitive-hcq-attribution-result-20260619.md` — PMU-4a..c result: probe-local attribution captures eager HCQ
  launches, HCQGraph construction/replay, and a Tensile runtime row; classifies `rocprof_hcq_visibility_gap` +
  `graph_rebind_ok`.
- `prefill-tensile-tpe7a-rebindable-node-result-20260619.md` — TPE-7a result: one extracted Tensile kernel object
  can be rebound to current buffers through graph-style kernarg filling; correctness/protocol proof only.
- **`performance-frontier-exhaustion-20260619.md` — latest exhaustion checkpoint.** Bounded decode primitives are
  exhausted; q8/RMSNorm is codegen-deferred; hand-LDS WMMA is refuted; external BLAS ceiling is measured; the bounded
  no-deps prefill WMMA sweep is refuted; EBT-1 kills the HIP-runtime bridge; the only material prefill route left is
  Tensile primitive extraction through HCQ or a codegen/Tensile-class rewrite.
- `qk-decode-per-role-delta-audit-20260618.md` — the quantitative per-role decode gap table (traffic/%peak/time-share/
  Amdahl/status); summed ceilings ~+27–30% ≈ the whole 1.47× llama gap, all behind one q8/full-MMVQ wall.
- `qk-machine-search-primitive-rows-20260618.md` — current machine-search rows (live + closed); supersedes the
  06-17 rows doc. Live/deferred: q8 side-channel, ffn coop sub-gate, attention residual audit, LDS flash-prefill,
  external/raw-HIP boundary/control; closed: quant-weight-reuse-8b, broad mmvq_q4k/q6k, decode_block_fusion,
  hand-LDS WMMA as the prefill lever, and bounded pure-tinygrad WMMA issue/occupancy.
- `q8-mmvq-lifecycle-deep-scope-20260618.md` — deep scope for the only remaining decode MMVQ lifecycle reopening:
  producer-side q8 from fused RMSNorm/apply into Q4_K ffn_gate/up int-dot. Explains what "q8/MMVQ lifecycle"
  means, what is already refuted, phase gates, and why this is low-EV/deep rather than a kernel tweak.
- `q8-mmvq-lifecycle-deep-result-20260619.md` — **executed it: Q8L-0/1 pass, Q8L-2 KILL.** The fused
  per-row→per-32 multi-output producer is NOT expressible via the store-group idiom (needs an LDS-reduction
  flash-style kernel); q8 side-channel is **deferred behind a codegen capability**, not a buildable arc — closes
  the last bounded decode research question.
- `llama-kernel-residual-primitive-audit-scope-20260619.md` — scope for auditing llama.cpp's **own** remaining
  primitive headroom: MMVQ residual-to-peak, q8 quant, attention, small-op fusion, graph boundaries, and prefill.
  Separate from the tinygrad-vs-llama gap explanation.
- `llama-kernel-residual-primitive-audit-20260619.md` — result of that audit. llama is not theoretically optimal,
  but fresh `rocprofv3` traces show prompt-free decode is 85.6% MMVQ; q8/RMSNorm is the only moderate non-MMVQ
  decode lifecycle candidate, graph launch overhead is already solved by HIP graphs, and pp512 prefill is 74.4%
  quantized MMQ/matmul rather than attention-limited.
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

- `prefill-wmma-lds-tiling-scope-20260619.md` — provenance for the now-refuted Branch A. After decode closed, the surviving high-EV arc:
  PREFILL_V2 forward is ~74% fp16 WMMA matmul emitted with LDS=0; the lever is WMMA operand LDS-tiling (~1.6× pp).
  Decision-first: Phase PWLT-0 is the authority call — Branch A (tinygrad hand-LDS, **triple payoff**: also unblocks
  q8 producer + flash-prefill attention) vs Branch B (external hipBLASLt/rocBLAS, prefill-only). Both feasible
  (assets/libs present); recommendation A-first, B as fallback control.
- `prefill-wmma-lds-tiling-result-20260619.md` — **executed Branch A: PWLT-A1 pass, PWLT-A2 KILL.** Hand-LDS WMMA
  = 1.02× the default matmul (both ~34% peak) → **LDS-tiling is NOT the lever** (IC-served on gfx1100, like decode
  attention). Real headroom is dense WMMA issue / Tensile-class scheduling, not LDS staging.
- `prefill-external-blas-result-20260619.md` — **ceiling/control measured.** Host-only C++ avoids the split-HIP
  compile issue; hipBLASLt reaches 69.8 TFLOPS on ffn_gate/up (1.71× tinygrad) and rocBLAS reaches 70.9/76.7 TFLOPS
  on ffn_down/attn_q/o. This proves a higher GEMM ceiling, but routing remains an external-dependency + HCQ-vs-HIP
  runtime boundary.
- `prefill-external-rawhip-tensile-boundary-scope-20260619.md` — broad external/raw-HIP/Tensile boundary scope
  before EBT-1.
  Starts with the authority decision, then EBT-1 tinygrad-buffer pointer interop, EBT-2 bridge/shape overhead,
  EBT-3 one-block transfer, EBT-4 full warm pp, and fallback lanes for Tensile HSACO or raw-HIP kernels. It also
  states the key gate conflict: strict >=1.5x full pp likely stops because the measured ceiling caps around
  1.4-1.45x before overhead. Superseded as the active plan by the Lane B scope below after EBT-1 killed Lane A.
- `prefill-external-bridge-ebt1-result-20260619.md` — **executed EBT-1: Lane A KILL.** HIP runtime and tinygrad
  HCQ/KFD are mutually exclusive in one process, so in-process rocBLAS/hipBLASLt on tinygrad pointers is closed.
- `prefill-tensile-primitive-extraction-and-codegen-scope-20260619.md` — **current Lane B scope.** Extract the
  selected Tensile primitive and its full launch contract (solution, HSACO, symbol, `.kd`, kernargs, launch geometry,
  workspace) and run it through tinygrad HCQ. Also scopes option 2: only after a working extracted contract exists,
  use it as the target for a tinygrad codegen/Tensile-class schedule transfer.
- `prefill-tensile-tpe4-perf-result-20260619.md` — **executed TPE-4: PASS.** The extracted rocBLAS Tensile
  ffn_gate/up primitive runs through tinygrad HCQ at 66.91 TFLOPS median (0.7703 ms), correct, no copies, no HIP
  runtime in-process. Lane B is now runnable and fast for one fixed shape.
- `prefill-tensile-tpe5-shape-matrix-result-20260619.md` — **executed TPE-5: PASS.** The extracted Tensile primitive
  generalizes: ffn_gate/up 66.8, ffn_down 68.9 (StreamK, no workspace), attn_q/o 58.9 TFLOPS through HCQ — all correct,
  stable, no workspace/aux/layout-copies, one code object + one pointer convention. Weighted model predicts **~1.40×
  full warm pp512** (→ ~2920 tok/s ≈ 95% of llama) if all three are routed, above the 1.25× gate.
- `prefill-tensile-tpe6-block-transfer-result-20260619.md` — **executed TPE-6: REDIRECT.** A whole FFN block
  (gate+up+silu·up+down) routed through the kernels is **exact** (rel 4.8e-4) and copy-free (weights stay natural
  `[out,in]`, run in `[feature,T]` space, zero per-matmul transposes), and the block matmuls hit 61 TFLOPS = **1.53×
  the PREFILL_V2 plateau on GPU time**. But naive per-op routing adds ~6.2 ms host sync overhead (a JIT-less probe
  artifact) that swamps the win end-to-end → realizing it needs a **single-dispatch graph (HCQGraph/TinyJit) runtime
  helper**. Next: build that helper, re-run the block gate, then TPE-7 (no model default; external-artifact policy pending).
- `prefill-own-wmma-kernel-scope-20260619.md` — pure tinygrad/no-deps scope. Key learning: tinygrad's
  WMMA matmul (41 TFLOPS) only *matches* the non-WMMA ALU matmul (40) — it gets **none** of the tensor-core 2×, so
  WMMA units are **stalled, not the bottleneck**. POWN-0 diagnose (occupancy / accumulator-chain / issue-rate) →
  POWN-1 config sweep (LDS-off since IC-served, chase dense WMMA issue + occupancy) gated ≥1.5×. The result below
  banks the bounded no-deps ceiling.
- `prefill-own-wmma-kernel-result-20260619.md` — **executed POWN-1: KILL.** Best config is the existing
  B128x128x16/W2x2 at 42.0 TFLOPS; more waves, bigger tiles, BK32, and noLDS all regress. No bounded no-deps
  prefill WMMA knob reaches the 62 TFLOPS gate.
- `prefill-external-blas-scope-20260619.md` — **DECLINED (no external deps).** rocBLAS/hipBLASLt ceiling-first plan;
  kept as provenance for the bridge analysis (DEV=AMD HCQ vs HIP-runtime). Its PXB-1 ceiling has now been measured
  in `prefill-external-blas-result-20260619.md`.
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
