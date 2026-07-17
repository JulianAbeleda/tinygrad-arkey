# docs/ - Current Map

Closed campaign scopes and superseded probe logs live in git history, not the current tree. Every doc listed below is
either a current authority or a retained refutation/lesson. If a doc is not listed here, it has been pruned — recover
it from git history.

**Single-number rule:** for any performance figure, the authority doc named below owns it. A number quoted in a
non-authority doc without a pointer here is presumed stale. (The `4413`/`2549` 8B baselines were exactly this failure;
they are refuted — see `prefill-current-state.md`.)

## Start Here

- `../README.md` - repo purpose, current performance snapshot, and run commands.
- `prefill-current-state.md` - **authority for shipped 8B prefill state and 8B pp512 tok/s** (pinned 3561.32 @ `8045efcef`).
- `pure-machine-search.md` / `pure-machine-search-roadmap.md` - search classification contract and promotion gates.
- `prefill-lessons-ledger.md` - distilled verdicts of closed scopes (qualitative lessons; defers all live numbers to authorities).
- `prefill-flag-graveyard.md` - per-flag REMOVED/verdict/why-safe ledger.

## 8B fitting / full-overlay route

- `prefill-memory-fit-architecture-20260715.md` - model-agnostic fit-vs-non-fit routing architecture (why 8B takes the FP16 overlay).
- `automatic-prefill-route-planner-repair-scope-20260716.md` - fact-derived route planner replacing the `PREFILL_GRAPH_GEMM` boolean (current planner authority; 3503-3509 smoke).

## 14B non-fitting / MMQ route

- `qwen3-14b-generated-prefill-completion-scope-20260714.md` - **canonical phase/gate spec** (frozen llama.cpp comparator, promotion gates).
- `qwen3-14b-generated-prefill-claude-handoff-20260716.md` - live status/blocker layer (read for current blocker; scope doc for frozen gates).
- `non-fitting-prefill-foundation-performance-prune-scope-20260716.md` - F0-F10 ordering authority for non-fitting work.
- `non-fitting-prefill-f0-baseline-20260716.md` - frozen F0 baseline (llama.cpp 14B pp512 1,889.41 tok/s; tinygrad direct-packed ~366).
- `14b-direct-packed-prefill-authority-baseline-20260710.md` - frozen direct-packed baseline (per-context tok/s).
- `14b-mmq-pure-machine-search-scope-20260714.md` - oracle-first-then-search doctrine; pins the llama 128x128 MMQ geometry.
- `14b-mmq-llama-kernel-reduction-roadmap-20260710.md` - R0-R7 MMQ-to-atoms roadmap.
- `14b-mmq-logical-vocabulary-scope-20260715.md` - descriptor-driven MMQ vocabulary + live measurement log.
- `14b-mmq-wave-process-deconstruction-20260710.md` - wave/LDS/writeback ownership law for llama's MMQ kernel (ground truth for other MMQ docs).
- `14b-role-facts-inventory-20260710.md` - Qwen3-14B GGUF tensor/role structural inventory.

## Compiler / resource evidence

- `amd-wmma-resource-adapter.md` - WMMA resource-evidence adapter contract.
- `mmq-resource-evidence-contract-20260715.md` - `mmq_resource_checks.py` fail-closed contract (no reconstructed numbers).
- `mmq-lowering-resource-audit-20260715.md` - cooperative MMQ resource-audit tool.
- `mmq-mfma-lowering-path-20260715.md` - WMMA->MFMA lowering chain (CDNA).
- `14b-mmq-renderer-asm-proof-introspection-20260710.md` - AMD ISA proof-manifest tool (`AMD_ISA_PROOF_MANIFEST`).
- `cooperative-mmq-integration-blocker-20260715.md` - cooperative MMQ gate blocker record.
- `q4k-fused-q4-correctness-owner-20260715.md` - fail-closed Q4 correctness gate ownership.
- `q6k-lm-head-prefill-authority-20260712.md` - **authority for valid lm_head measurement** (whole-prefill harness only).
- `q6k-staged-wmma-role-boundary-20260715.md` - which Q6_K roles the staged WMMA path is admitted for.
- `wave32-geometry-compile-sweep-20260715.md` - compile-only WMMA geometry validation tool.

## Measurement regime

- `measurement-regime-audit-llama-prefill-20260715.md` - **guardrail**: keeps counterfactual/modeled ceilings from being cited as measured.
- `harness-consolidation.md` - canonical-harness decision record ("don't rebuild a measurement harness").
- `authored-core-loc-consolidation-scope-20260716.md` - authored-LOC budget tracker.

## Runtime Boundary

- `tinygrad-runtime-operational-policy-r10.md` - runtime operating policy.
- `tinygrad-runtime-client-separation-roadmap-20260630.md` - client/runtime separation plan.
- `tinygrad-runtime-client-separation-implementation-status-20260630.md` - implementation status.

## Local References (upstream tinygrad, retained)

- `quickstart.md`, `env_vars.md`, `dtypes.md`, `nn.md` - upstream tinygrad reference docs, still useful for the fork.

## Retained refutations / historical results

These are outdated but hold a refutation or measurement worth keeping until its lesson is durably in the knowledge base
(`/home/ubuntu/knowledge_base`). Do not cite their numbers as current. Candidates for the next prune once their lessons
are confirmed rescued:

- `8b-prefill-lifecycle-compression-audit-20260709.md` - density gap (12.51 vs ~58 TFLOPS) + slot-only LDS suppression NaN trap.
- `owned-tile-buffer-identity-kv-read-result-20260623.md` - decode +18.7% from real-KV read; root cause was materializing SLICE, not purity.
- `split-kv-economics-audit-result-20260621.md` / `b4-split-kv-combine-tax-result-20260621.md` - combine-tax dominates; latency- not bandwidth-bound.
- `decode-vector-flash-tile-realigned-result-20260621.md` - whole-decode W==D win below gate despite standalone 1.08x.
- `lane-a-lm-head-packed-refutation-20260712.md` - packed Q6_K lm_head is a prefill regression (+87ms).
- `q4-q4-owner-comparison-20260715.md` - sudot4 beats WMMA only on a tiny bounded tile, not whole-linear.
- `ctx512-device-time-attribution-20260712.md` / `hcq-graph-profile-attribution-20260712.md` - device-time attribution caveats.
- `pure-baseline-vs-exact-candidate-kernel-only-20260712.md` / `pure-two-buffer-whole-prefill-benchmark-20260712.md` - kernel-only vs whole-model scoping caveats.
- `14b-mmq-r4-r7-theory-matrix-20260710.md`, `14b-role-measurement-owner-20260715.md`, `handoff-14b-mmq-boltbeam-tinygrad-20260711.md`, `qwen3-14b-integrated-loop-compile-gate-20260715.md`, `role-baseline-14b-q4km.md`, `s9-regression-ab-blocker-20260712.md`, `s9-resource-gate-boundary-20260712.md`, `amd-dynamic-tile-owner-validation-20260715.md`, `q4k-fused-q4-role-sweep-20260715.md`, `q4k-packed-q4-symbolic-loop-validation-20260715.md`, `q4k-q6k-prefill-path-alignment-audit-20260715.md`, `pure-pipe-policy-centralization-20260712.md`, `pipe-transport-abi-inventory-20260712.md`, `prefill-harness-profile-generalization-20260710.md`, `primitive-space-learning-loop-lora-first-result-20260623.md`, `egpu-usb4-link-keepalive.md`, `bubblebeam-futuresight-terminology-20260625.md` - see each doc's header for its retained lesson.

## Authorities

- BoltBeam owns route policy, candidate/search schema, evaluation policy, roofline reports, and ledgers.
- tinygrad owns runtime execution, compiler/backend lowering, and hardware gates.
- The durable-principles knowledge base is `/home/ubuntu/knowledge_base`; project docs apply its principles.
- `tinygrad/llm/route_policy.py` consumes `boltbeam.route_policy.v1`.
- `tinygrad/llm/route_ops.py` is the runtime adapter for QK route implementations.
- `bench/qk-search-spaces/default_route_manifest.json` is the local route-state manifest.
- `extra/audit/pure_machine_search_default_path_census.py` is the local generated-default census.

## Boundary Rule

tinygrad hot paths must not import `extra.qk.*`, `extra.qk.quant.*`, or `extra.audit.*` directly. Use
`tinygrad/llm/route_ops.py` for LLM routes and `tinygrad/codegen/experimental.py` for default-off codegen probes.
`test/unit/test_tinygrad_boundary.py` enforces this.
