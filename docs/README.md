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
- `pure-pipe-policy-centralization-20260712.md` - compiler policy boundary (`tinygrad.codegen.opt.compiler_policies`).
- `pipe-transport-abi-inventory-20260712.md` - pipe-transport insertion chain (rangeify SINK -> transport).

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

## Hardware

- `egpu-usb4-link-keepalive.md` - USB4/Thunderbolt eGPU idle link-drop root cause + keepalive fix. Referenced by
  `extra/remote/amd_power_cycle.py`. OPEN: the fix may not be ported into the current tree — verify before trusting.

## Pruned closed campaigns (2026-07-17)

25 closed result/probe logs were removed to git history in the doc prune. Their **transferable** lessons were first
rescued into the knowledge base (`/home/ubuntu/knowledge_base`, commit `cb78bbc`): counterfactual/refuted numbers must
be labelled at every citation; late-stage destructive suppression is invalid once identity is gone; scheduling cannot
manufacture overlap construction never created; resource gates fail closed on missing facts; AMD regalloc/occupancy is
ours because the toolchain doesn't hide it; matched evidence classes or "faster" doesn't transfer; a positional rewrite
pass corrupts silently on a skipped element. Project-specific measurements from those docs (decode +18.7% real-KV,
+87ms packed lm_head regression, the 12.51-vs-58 TFLOPS density gap, split-KV combine-tax, etc.) remain in git history —
recover the doc by name if a specific number is needed again.

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
