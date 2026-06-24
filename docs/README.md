# docs/ — map

Navigation source-of-truth for this fork's docs. **Current state only** — the long chronological probe log
(decode-attention arc, prefill WMMA/Tensile, fused-flash, flywheel) was archived on 2026-06-24 under
`docs/archive/` and indexed by `provenance-index-20260624.md`. Read the canonical docs below; treat anything
in `docs/archive/` as provenance, not current state.

## ⭐ Start here (canonical)

- **`current-project-state-handoff-20260624.md`** — ⭐⭐ CANONICAL CURRENT STATE (read first). Current numbers
  (decode **101.6/99.8/97.4/92.9 tok/s** @ctx512/1024/2048/4096 ≈ **100.6–104% of llama** on the `Q4K_GEMV_WARP*`
  default stack; prefill **3597/3505/3263/2784/2217**), decided policies, and the parity win (owned attention tile +
  buffer-identity fix). Guardrail: `extra/qk_policy_consistency_check.py` fails if a canonical doc re-opens a closed question.
- **`decode-campaign-final-synthesis-20260623.md`** — how decode reached llama parity (attention not exhausted;
  buffer identity was the wall; owned AMDGCN tile + `Q4K_GEMV_WARP` weight-GEMV).
- **`gpu-lifecycle-primitive-coverage-tracker-20260624.md`** — per-primitive coverage vs llama (decode ~101.7–105%,
  prefill ~114.5% pp512).
- **`prefill-decode-next-workstreams-codex-scope-20260624.md`** — current next-work map.
- **`provenance-index-20260624.md`** — the archive map (what moved, where current authority lives).

## Decode (current)

- **`decode-q4k-gemv-warp-promotion-result-20260624.md`** — `Q4K_GEMV_WARP*` promoted default-on (weight-GEMV at/below llama).
- **`decode-parity-no-regression-audit-result-20260623.md`** — parity reconciliation + the flag-stack caveat
  (102+ requires `Q4K_GEMV_WARP*` on; a fresh default-off run reads below llama).
- **`post-owned-attention-promotion-synthesis-20260623.md`** — owned hand-AMDGCN decode-attention tile promoted into the decode path.
- **`three-lane-completion-result-20260623.md`** — the three-lane completion.
- **`decode-aggressive-target-proof-scope-20260624.md`** — aggressive-target proof (planning).

## Prefill (current)

- **`prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md`** — confirmed prefill baseline + aggressive bound.
- **`prefill-eightwave-promotion-result-20260624.md`** — `eightwave` promoted default (~+3%).
- **`prefill-long-context-no-regression-audit-result-20260623.md`** — long-context no-regression confirm.
- **`prefill-aggressive-target-proof-scope-20260624.md`** — aggressive-target proof (planning).

## Decided policies (do not re-open — see handoff §2)

Global `PREFILL_V2` default **OFF**; `PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1` / q8 FFN are **opt-in**;
`Q4K_GEMV_WARP*` and `eightwave` promoted **default-on**. Enforced by `extra/qk_policy_consistency_check.py`.

## Live tooling

- **Machine-search:** `extra/qk_decode_eval.py` (lifecycle evaluator), `extra/qk_lifecycle_search_loop.py`
  (generate→evaluate→prune), `extra/qk_search_spec.py` (schema authority), `extra/qk_nll_eval.py` (dNLL gate),
  `extra/qk_demote_search.py` (demotion orchestrator). Benches: `bench/README.md`.
- **Benchmark harnesses + numbers:** `bench/README.md` (read its "Which harness READ FIRST" before quoting decode tok/s).

## References

- **Architecture (still useful, in archive):** `archive/amd-decode-harness-architecture.md`,
  `archive/amd-decode-qk-storage-architecture.md`, `archive/amd-decode-primitive-v2-design.md`,
  `archive/amd-decode-bandwidth-roofline.md`.
- **Other subsystems:** PSP/boot — `archive/amd-kdb-root-cause.md`, `archive/amd-linux-psp-good-trace.md`,
  `archive/amd-remote-dropout-investigation.md`; reference research — `archive/amd-rocm-llamacpp-research.md`.
- **tinygrad developer docs:** `developer/` (am, hcq, runtime, speed, uop) and `tensor/`.
- **Upstream tinygrad docs:** `index.md`, `quickstart.md`, `mnist.md`, `nn.md`, `dtypes.md`, `env_vars.md`, `showcase.md`.

## History

The full chronological probe log lives in `docs/archive/` (797 docs) and the dated `bench/<arc>/` dirs.
Start from `provenance-index-20260624.md` to trace any historical cluster to its current authority.
