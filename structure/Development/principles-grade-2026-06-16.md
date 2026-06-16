# Principles Grade — 2026-06-16

Graded analysis of the repo against [coding-principles.md](coding-principles.md)
(esp. "Reducing Code The Right Way") + [tinygrad-coding-overrides.md](tinygrad-coding-overrides.md),
after the hard-fork prune (~−240k LOC) and the Round-2 consolidations (C1–C10).

## Overall: **A− (Strong)**

The repo went from a sprawling upstream fork + ~7k-LOC unmaintainable flywheel to a
disciplined AMD-only fork: byte-proven consolidations, documented + enforceable
principles, dangerous-power surface reduced, ~240k LOC of dead/non-AMD code removed.
Two real detractors: stringly-typed env-flag config (encode-invariants) and an
uncommitted *refuted* consolidation lingering in the tree (hygiene).

## Scorecard

| dimension | grade | evidence |
|---|:--:|---|
| Centralize (single source of truth) | **A−** | hubs `llm_eval_common` (in≈25), `qk_layout→gguf` dequant delegation (C7), `qk_modes`/`qk_paths`, one `LLAMA_REFS`; C1–C3 routed dups to SSoT. Remaining: adapter `_load_json` (gated on goldens), a few validation-divergent `_load_json`. |
| DRY = knowledge (Reducing Code Right) | **A** | genuine dups consolidated **NFC, byte-proven** (23/23 golden+phase gates pass); divergent extractors AND the per-prober `q4k_bench._classify` correctly **left duplicated** (different signatures/logic = not a violation); C9 honestly **refuted, not forced**. Exemplary "duplication cheaper than wrong abstraction." |
| Contain dangerous power | **A** | `q4_k_safety` risky-search gate, `_dp4a` gated emit, custom kernels/asm isolated, flash gated; the hard-fork **removed a privileged core op** (`Ops.QK_BLOCK_DOT`) — a net reduction in dangerous surface. |
| Anti-re-sprawl / structure | **A** | −240k prune; `coding-principles` + `overrides` + audit/manifest/script-map docs recorded; the rule is now machine-checkable (portability golden) and human-documented. |
| Orthogonalize | **A−** | no import cycles in `extra/`; decode/prefill orthogonal; backends were *cleanly removable* (proves boundaries held). |
| Abstract / boring surfaces | **A−** | `qk_flywheel_cli` dispatcher, `llm_generate` deep module, `gguf` as dequant authority. |
| Test at the boundary | **A−** | golden + reproduce-from-artifact gates (23/23), fork suite **239 green**, every consolidation byte-proven. |
| Commit discipline | **A−** | Round-2 commits all `[test] NFC`, one owning prefix, byte-proven; `tinygrad/` core never `[test]`. (Round-1 codex shipped 1 red + 1 mixed-prefix; both already fixed.) |
| Errors at boundaries | **B+** | scorer Wilson CIs, `runtime_contract` table-gate, `_render_arg_format` actionable errors. |
| Modularize / deep modules | **B+** | cohesive modules; large leaves remain (`qk_flywheel_shadow` 978, `qk_policy_pipeline` 674) — defensible orchestrators, watch for internal sprawl. |
| Encode invariants | **B−** | `qk_modes` enums + `QKPrimitiveBudget` are exemplary, **but ~24 stringly-typed `getenv("Q4K_*/Q6K_*/FLASH_*/QK_*")` flags in `model.py`** with scattered `if x not in (...)` validation — no typed `QKConfig` (audit §D, deferred). The biggest standing principle gap. |
| Working-tree hygiene | **C+** | uncommitted, **refuted** C9 change (`llm_json_rejection_sample.py` routed to `llm_generate`) sits in the tree; the handoff records C9 as token-parity-refuted. It changes rejection-sample tokens and is unverified — should be **reverted** (the suite stays green only because the sampler needs GPU+model and isn't in `test/external`). |

## What improved (since the audits)
- **Knowledge-duplication collapsed where genuine**: C1 (`_load_json`/id-jsonl → `llm_eval_common`), C2 (`_fmt`/`LLAMA_REFS` single-source), C3 (opt-string/load-width parse rule), C4 (shadow jsonl → SSoT), C7 (Q4_K/Q6_K dequant → `gguf` authority), C10 (dead `_majority`/branches). All NFC, byte-proven.
- **Honest negatives**: C5 (staged-shadow table) + C8 (test fixtures) skipped-and-reported; C9 refuted, not forced — this *is* the principle working ("abstract only what's earned").
- **Dangerous surface shrank** (privileged op removed; non-AMD backends gone).

## What remains (ranked)
1. **Revert the uncommitted refuted C9** in `llm_json_rejection_sample.py` (hygiene; it's an unverified behavior change). Keep the harmless `llm_generate` `seed=None` enhancement.
2. **`QKConfig`** — fold the ~24 `model.py` env flags into one validated typed config (audit §D). The top "encode invariants" win; a real refactor, gate behind token-parity.
3. Adapter `_load_json`/dataset-scaffold dedup — **needs a golden test added first** (audit §E), then byte-provable.
4. Optional: C5 staged-shadow batch table, C8 shared `_qk_testutil.py` (modest, byte-provable).

## Verdict
Principle adherence is high and, crucially, **disciplined** — consolidations are
byte-proven, wrong abstractions are refused, and the structure now *prevents*
re-sprawl. Close the env-flag `QKConfig` gap and tidy the working tree and this is
solidly an **A**.

## Update — 2026-06-16 (later): QKConfig + QK runtime invariants

The top "encode invariants" gap is now addressed.

- **`QKConfig` landed (NFC).** `[nn] NFC centralize QK runtime config` folds the scattered
  QK primitive *install* env reads (strict/cap/storage/debug/demote/fuse) into one typed
  authority built at the top of the active-primitive block; byte-proven by
  `test/external/test_qk_config.py` (9 tests). Activation gating + forward-pass probe flags
  stay at their sites by design (runtime-coupled / per-call), documented on the class.
- **DEV=AMD guard (functional).** `[nn] require DEV=AMD for QK quant primitive paths` makes
  an explicit `Q4K_PRIMITIVE`/`Q6K_PRIMITIVE`/`QK_GENERATED_POLICY` on a non-AMD backend
  fail fast with an actionable error instead of an obscure later kernel failure (covered by
  `test_qk_amd_guard.py`). This is the "main actionable bug" from the flag scan.

Bug-scan disposition (the rest):
- **Already enforced:** GGUF-path requirement (existing `isinstance(gguf, Tensor)` guard);
  shared-storage-needs-metadata (shared mode is only reachable inside the `q4k_meta`
  block, which requires a GGUF path + `gguf_load_with_metadata`).
- **No change (correct):** `Q4K_FUSE`/`Q4K_VDOT`/`Q6K_DEMOTE_FFNDOWN`/`FLASH_DECODE` stay
  default-off (probes, not accepted paths).
- **Deferred (needs sign-off + AMD validation):** make `QK_GENERATED_POLICY_STRICT`
  default-on, or warn when generated linears are silently skipped under a storage cap.
  This is a real operational footgun *with caps set*, but: (a) default-strict would turn
  currently-passing capped runs (e.g. the 32B capped policy) into hard `MemoryError`s, and
  (b) an under-install warning needs the install functions to return their `skipped`
  counter (a return-contract change across both installers + 4 call sites). Both are
  behaviour changes on the AMD decode path, which the no-GPU-test constraint means cannot
  be validated here. Recommend a dedicated, AMD-validated commit. Operationally, until then:
  generated-policy runs should set `QK_GENERATED_POLICY_STRICT=1 QK_GENERATED_POLICY_DEBUG=1`
  when using `QK_PRIMITIVE_MAX_STORAGE_MB`.
- **Operational note (not a bug):** accepted generated-policy wins (14B/32B especially) are
  NOT active unless `QK_GENERATED_POLICY=...` is set; the AMD default is explicit primitives.
  Intentional per `docs/amd-decode-current-verdicts.md`; the most likely reason someone
  "doesn't see" the expected wins.
