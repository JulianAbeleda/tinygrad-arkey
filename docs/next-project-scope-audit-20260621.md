# Next-Project Scope Audit (2026-06-21)

Audit only — no kernels, no defaults, no flag tuning, no reopened lanes. Picks the highest-value next project
after the bounded-decode closure.

## Canonical state (inputs)

- Prefill: solved, fast paths shipped opt-in, **global `PREFILL_V2` stays OFF** (do not flip).
- Decode: **bounded work rested** (~67% llama steady-state).
- `FLASH_L=64` validated the T=1 split principle locally but failed W==D promotion → **do not promote**.
- Closed decode lanes (do not reopen): FFN activation fusion, attention microfusion, raw fused flash tile, scalar
  LDS+GQA fused tile, **WMMA decode (refuted — llama decode is a non-WMMA vector tile)**, **MMVQ (wall parity)**,
  bounded vector-tile tuning.
- Remaining decode lever: the **north-star llama-style `flash_attn_tile` lifecycle** (efficient many-KV-split /
  stream-k combine) — **not a bounded patch**.
- North star = **beat llama + closed lifecycle-search system + clean tinygrad-v2 execution repo**
  (`docs/project-north-star-llama-and-lifecycle-search-20260620.md`).

## What the artifacts show

- **The lifecycle-search "system" is a static read-only seed ledger.** `bench/qk-lifecycle-search/` holds
  `candidates.json`, `generated_candidates.json`, `refutations.json` (20 entries), `runner_bindings.json`,
  `policy_exports.json`, `summary.md`. Its own summary says: *"Read-only seed ledger. It does not run hardware or
  route a model path."* Candidates and refutations are **hand-added** (as in every experiment this session); there
  is **no closed generate→evaluate→prune loop**.
- **The evaluation ladder exists only as scattered, bespoke scripts.** Each experiment this session re-implemented
  the same ladder by hand: structural/identity (`qk_decode_current_route_attribution.py`), correctness
  (per-`*_ab.py` numpy refs), clock-controlled local A/B (`qk_clock_pin.py` added mid-session after auto-clock
  volatility burned several runs), W==D promotion (`qk_decode_runtime_overhead.py`), policy/guardrail
  (`qk_policy_consistency_check.py`). The gate **thresholds** (≥1.05× local, ≥5%@1024 / ≥7%@4096 W==D, host-sync,
  dNLL) are written in prose across result docs, not encoded as a callable contract.
- **Measurement fragility was the recurring tax.** Auto-clock cold-context volatility (2–3× swings) forced
  clock-pinning; the `87.6` headline was a unit/ctx confusion needing a full reconciliation; "promotion vs
  diagnostic vs attribution" timing classes had to be re-litigated repeatedly. The measurement layer is the most
  fragile, most re-built, least durable part of the stack.

## Ranked recommendation

| # | candidate | expected value | risk | prerequisite | first gate | stop condition | why now / why not now |
|---|---|---|---|---|---|---|---|
| **1** | **Decode evaluation & benchmark hardening** — a durable, automated, reproducible evaluator (`decode_eval(candidate)` runs the full ladder) + machine-readable artifact contract + canonical gate/authority encoding | **HIGH leverage** — unblocks machine-search and makes every future experiment trustworthy + cheap; directly de-risks the measurement fragility this session paid for | **LOW** — assembles existing pieces; measurement-only, no kernels/defaults | none (pieces exist: W==D harness, clock-pin, attribution, guardrail, ledger) | one `decode_eval` call runs structural→correctness→clock-controlled local A/B→W==D→policy and emits a schema'd result that **reproduces a known verdict** (FLASH_L=64: local-pass / W==D-fail) **within noise across re-runs** | the existing harnesses already meet a durable callable contract (they do not — bespoke per experiment) | **WHY NOW:** the project pivots to machine-search, which *requires* a durable evaluator; the ladder + gates already exist but only as prose+scripts; lowest-risk, highest-leverage, ready today |
| 2 | **Lifecycle-search system** — close the generate→evaluate→prune loop on top of the evaluator (route/fusion/layout/schedule/policy templates → automated ladder → machine-readable results → refutation pruning) | **HIGHEST** (the "Method" completion pillar) | MEDIUM — without a durable evaluator it optimizes noise; template/search design risk | **#1** (durable evaluator + artifact contract) | generator emits ≥1 template candidate, the loop runs it through `decode_eval` and records result + applies a refutation **with no human running scripts** | evaluator not durable yet → do #1 first; or candidates too bespoke to template | **WHY NEXT, NOT FIRST:** the seed ledger exists but is static; it needs #1's evaluator as its execution substrate before the loop is meaningful |
| 3 | **North-star decode `flash_attn_tile` codegen** — efficient many-split / stream-k combine vector flash-decode | HIGH (the "beat llama" pillar) | **HIGH** — multi-week deep codegen, the walled lever; done before the search system it is another one-off hand patch (anti-north-star) | trustworthy evaluator (#1) + ideally search templates (#2) so it is a *route template*, not a one-off | a bounded sub-capability spike (efficient many-split combine) beats `gqa_coop_vec`'s combine standalone, measured by #1 | no interim sub-gate is tractable without the full multi-week port | **WHY NOT NOW:** bounded work just closed; a multi-week hand kernel before the evaluator/search exists contradicts "the win is not a one-off hand patch" and risks unmeasurable churn |
| 4 | **tinygrad-v2 clean execution repo** | MEDIUM (the "clean repo" pillar) | MEDIUM — premature; migrates a system that does not exist yet | #1 + #2 (and a winning path) must exist to migrate | v2 runs W==D and reproduces current numbers within noise | nothing stable to migrate | **WHY NOT NOW:** downstream packaging; there is no closed search system or winning path to carry over yet |
| 5 | **Stop / rest** | — | — | — | — | — | **NOT recommended** — there is ready, high-leverage work (#1). Rest applies only to *bounded decode kernels*, not to the Method/measurement pillars |

## Chosen next project: **#1 — Decode evaluation & benchmark hardening**

It is the prerequisite that gates #2 (and any trustworthy attempt at #3), it is **ready today** (the pieces all
exist from this session), it is **low-risk** (measurement-only), and it directly removes the fragility that taxed
every experiment. It is the concrete first deliverable of the north-star "Method" pillar: the evaluation ladder
made first-class and automatable. Full execution scope:
`docs/decode-evaluation-harness-hardening-scope-20260621.md`.

Why it wins over scoping #2 directly: a machine-search loop is only as trustworthy as its evaluator; building the
loop on the current bespoke/prose measurement layer would optimize against noise (exactly the auto-clock trap).
#1 is the de-risked foundation; #2 is its immediate, well-defined follow-on.

## Explicit anti-drift guards (this audit does NOT recommend)

- **No `PREFILL_V2` default flip** — global default stays OFF.
- **No `FLASH_L=64` promotion** — it failed W==D; optional short-context owner knob only.
- **No bounded decode fusion** — closed (work-conserved / occupancy-harmful).
- **No WMMA decode claim** — refuted; llama decode is a non-WMMA vector tile.
- **No MMVQ reopen** — wall parity; structural int8/q8 difference is not a wall-time gap.
- **No new bounded hand-tile or flag sweep** — bounded decode is rested.

The guardrail `extra/qk_policy_consistency_check.py` already fails on these patterns; the chosen scope is
measurement infrastructure, orthogonal to all closed lanes.
