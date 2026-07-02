# docs/ — map

Navigation source-of-truth for this fork's docs. **Current state only** — the long chronological probe log
(decode-attention arc, prefill WMMA/Tensile, fused-flash, flywheel) was archived on 2026-06-24 under
`docs/archive/` and indexed by `provenance-index-20260624.md`. Read the canonical docs below; treat anything
in `docs/archive/` as provenance, not current state.

## ⭐ Start here (canonical)

- **`pure-machine-search-roadmap.md`** — roadmap for replacing hand-owned decode kernels with generated/search-owned routes.
- **`tg-p12-resolution-verify-and-land-scope-20260701.md`** — current active handoff: verify and either land or reject Claude's default-off manual-accumulator widening fix.
- **`tg-p12-manual-end-accumulator-fold-guard-scope-20260701.md`** — prior TG-P12 scope; its scalar-fold-first strategy is superseded by the resolution handoff after the baseline showed expander-created vector accumulator state.
- **`tg-p11-reduce-upcast-accumulator-widening-scope-20260701.md`** — provenance scope that diagnosed the generic reduce/upcast accumulator-lowering invariant.
- **`claude-active-work-audit-and-agnostic-search-scope-20260630.md`** — older consolidated handoff for active-work audit and quant/shape/target agnostic search; useful background, but not the current route-state authority.
- **Current route-state authority:** `bench/pure-machine-search-default-path-census/summary.md`,
  `bench/tg-p10-reg-scalar-combine-lowering/summary.md`, and `extra/qk_route_manifest.py`. These supersede the
  June 24 state handoff for default-route and purity status.
- **`decode-campaign-final-synthesis-20260623.md`** — how decode reached llama parity (attention not exhausted;
  buffer identity was the wall; owned AMDGCN tile + `Q4K_GEMV_WARP` weight-GEMV).
- **`gpu-lifecycle-primitive-coverage-tracker-20260624.md`** — per-primitive coverage vs llama (decode ~101.7–105%,
  prefill ~114.5% pp512).
- **`prefill-decode-next-workstreams-codex-scope-20260624.md`** — current next-work map.
- **`provenance-index-20260624.md`** — the archive map (what moved, where current authority lives).

## Decode (current)

- **`amd-isa-g3-weight-promotion-hardening-scope-20260629.md`** — generated G3 LaneMap is the speed-equivalent Q4_K route under BubbleBeam/FutureSight.
- **`amd-isa-q6k-direct-route-full-scope-20260629.md`** — Q6_K direct-route work; later Q6K-3 refuted the half-warp direct route, so current Q6_K stays on coop/default.
- **`tg-p8-generated-8b-attention-parity-scope-20260701.md`**, **`tg-p9-pure-attention-primitive-route-scope-20260701.md`**,
  **`tg-p11-reduce-upcast-accumulator-widening-scope-20260701.md`**, and
  **`tg-p12-manual-end-accumulator-fold-guard-scope-20260701.md`**, and
  **`tg-p12-resolution-verify-and-land-scope-20260701.md`** — the current 8B attention purity chain:
  generated live-split solved the short-context tile gap; the remaining blocker is a manual accumulator/load-fold lowering fix.
- **`decode-two-kernel-problem-audit-result-20260625.md`** — historical attention tile+combine audit; still useful provenance, but superseded for current purity work by TG-P8/TG-P9/TG-P10/TG-P11.
- **`post-owned-attention-promotion-synthesis-20260623.md`** — provenance for the owned HIP decode-attention tile that remains the 8B default/rollback oracle.

## Prefill (current)

- **`prefill-baseline-confirmed-aggressive-bound-handoff-20260624.md`** — confirmed prefill baseline + aggressive bound.
- **`prefill-pure-machine-search-roadmap-20260629.md`** — prefill P0-P8 roadmap and authority chain.
- **`prefill-eightwave-promotion-result-20260624.md`** — historical `eightwave` promotion (~+3%); superseded as the prefill default by `pipe_tm2_tn2`.
- **`prefill-long-context-no-regression-audit-result-20260623.md`** — long-context no-regression confirm.
- **`prefill-aggressive-target-proof-scope-20260624.md`** — aggressive-target proof; `pipe_tm2_tn2` was later hardened and promoted.

## Decided policies (do not re-open — see handoff §2)

Q4_K decode uses generated G3 where eligible. Q6_K direct half-warp is refuted; Q6_K generated coop is the default generated route with shipped kernels retained as rollback. The generated G=5 K-only attention route is promoted for the validated 14B shape. 8B long-context attention still uses the owned HIP two-kernel route because the generated route is close but below the 98% promotion bar. Prefill uses the generated role-selective schedule by default with rollback via `PREFILL_GENERATED_SCHEDULE=0` or the older pipeline flags described in the route manifest.

## Live tooling

- **Machine-search:** `extra/qk_decode_eval.py` (lifecycle evaluator), `extra/qk_lifecycle_search_loop.py`
  (generate→evaluate→prune), `extra/qk_search_spec.py` (schema authority), `extra/qk_nll_eval.py` (dNLL gate),
  `extra/qk_demote_search.py` (demotion orchestrator). Benches: `bench/README.md`.
- **Benchmark harnesses + numbers:** use the phase-specific authority artifacts before quoting a number. The committed multi-model table under `bench/models/qwen/` is useful provenance but may lag the latest route changes.

## References

- **tinygrad docs (current):** `developer/` (am, hcq, runtime, speed, uop), `tensor/`, and upstream
  `index.md` / `quickstart.md` / `mnist.md` / `nn.md` / `dtypes.md` / `env_vars.md`.
- **History & subsystem docs** (architecture, PSP/boot, reference research, the full 797-doc probe log):
  `docs/archive/` — start from `provenance-index-20260624.md` to trace any cluster to its current authority.
