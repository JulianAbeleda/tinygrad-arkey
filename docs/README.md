# docs/ — map

Navigation source-of-truth for this fork's docs. **Current state only** — the long chronological probe log
(decode-attention arc, prefill WMMA/Tensile, fused-flash, flywheel) was archived on 2026-06-24 under
`docs/archive/` and indexed by `provenance-index-20260624.md`. Read the canonical docs below; treat anything
in `docs/archive/` as provenance, not current state.

## ⭐ Start here (canonical)

- **`pure-machine-search-roadmap.md`** — stable roadmap for replacing hand-owned decode kernels with BubbleBeam-selected generated routes.
- **`claude-active-work-audit-and-agnostic-search-scope-20260630.md`** — Claude-ready consolidated handoff for active-work
  audit, residual tracing/probe tooling, and quant/shape/target agnostic search.
- **`current-project-state-handoff-20260624.md`** — ⭐⭐ CANONICAL CURRENT STATE (read first). Current numbers
  (decode **103.9 / 102.0 / 99.7 / 94.4 tok/s** @ctx512/1024/2048/4096 on the G3/owned-equivalent decode stack;
  prefill **4291 / 4089 / 3711 / 3137 / 2423 tok/s** @ctx512/1024/2048/4096/8192 on the promoted `pipe_tm2_tn2`
  graph-GEMM route), decided policies, and the closed decode weight-kernel result. Guardrail:
  `extra/qk_policy_consistency_check.py` fails if a canonical doc re-opens a closed question.
- **`decode-campaign-final-synthesis-20260623.md`** — how decode reached llama parity (attention not exhausted;
  buffer identity was the wall; owned AMDGCN tile + `Q4K_GEMV_WARP` weight-GEMV).
- **`gpu-lifecycle-primitive-coverage-tracker-20260624.md`** — per-primitive coverage vs llama (decode ~101.7–105%,
  prefill ~114.5% pp512).
- **`prefill-decode-next-workstreams-codex-scope-20260624.md`** — current next-work map.
- **`provenance-index-20260624.md`** — the archive map (what moved, where current authority lives).

## Decode (current)

- **`decode-q4k-gemv-warp-promotion-result-20260624.md`** — `Q4K_GEMV_WARP*` promoted default-on (weight-GEMV at/below llama).
- **`amd-isa-g3-weight-promotion-hardening-scope-20260629.md`** — generated G3 LaneMap is the speed-equivalent Q4_K route under BubbleBeam/FutureSight.
- **`amd-isa-q6k-direct-route-full-scope-20260629.md`** — Q6_K direct-route work; later Q6K-3 refuted the half-warp direct route, so current Q6_K stays on coop/default.
- **`decode-two-kernel-problem-audit-result-20260625.md`** — attention tile+combine audit; combine/fused lifecycle is exhausted for the current route.
- **`decode-parity-no-regression-audit-result-20260623.md`** — parity reconciliation + the flag-stack caveat
  (102+ requires `Q4K_GEMV_WARP*` on; a fresh default-off run reads below llama).
- **`post-owned-attention-promotion-synthesis-20260623.md`** — owned hand-AMDGCN decode-attention tile promoted into the decode path.
- **`three-lane-completion-result-20260623.md`** — the three-lane completion.
- **`decode-aggressive-target-proof-scope-20260624.md`** — aggressive-target proof (planning).

## Prefill (current)

- **`prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md`** — confirmed prefill baseline + aggressive bound.
- **`prefill-pure-machine-search-roadmap-20260629.md`** — prefill P0-P8 roadmap and authority chain.
- **`prefill-eightwave-promotion-result-20260624.md`** — historical `eightwave` promotion (~+3%); superseded as the prefill default by `pipe_tm2_tn2`.
- **`prefill-long-context-no-regression-audit-result-20260623.md`** — long-context no-regression confirm.
- **`prefill-aggressive-target-proof-scope-20260624.md`** — aggressive-target proof; `pipe_tm2_tn2` was later hardened and promoted.

## Decided policies (do not re-open — see handoff §2)

Global `PREFILL_V2` default **OFF**; `PREFILL_V2=auto` / `PREFILL_SERVER_PROFILE=1` / q8 FFN are **opt-in**;
Q4_K decode uses generated G3 where eligible, Q6_K direct is refuted/default-off, and `pipe_tm2_tn2` prefill is promoted
**default-on** with rollback `PREFILL_GEMM_PIPELINE=0`. Enforced by `extra/qk_policy_consistency_check.py`.

## Live tooling

- **Machine-search:** `extra/qk_decode_eval.py` (lifecycle evaluator), `extra/qk_lifecycle_search_loop.py`
  (generate→evaluate→prune), `extra/qk_search_spec.py` (schema authority), `extra/qk_nll_eval.py` (dNLL gate),
  `extra/qk_demote_search.py` (demotion orchestrator). Benches: `bench/README.md`.
- **Benchmark harnesses + numbers:** `bench/README.md` (read its "Which harness READ FIRST" before quoting decode tok/s).

## References

- **tinygrad docs (current):** `developer/` (am, hcq, runtime, speed, uop), `tensor/`, and upstream
  `index.md` / `quickstart.md` / `mnist.md` / `nn.md` / `dtypes.md` / `env_vars.md`.
- **History & subsystem docs** (architecture, PSP/boot, reference research, the full 797-doc probe log):
  `docs/archive/` — start from `provenance-index-20260624.md` to trace any cluster to its current authority.
