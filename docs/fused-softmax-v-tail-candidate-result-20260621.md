# Path A — Fused Softmax+V Tail Candidate — Result

Date: 2026-06-21

Scope: `docs/native-fused-flash-linearizer-scope-20260621.md` (the bounded Path-A first gate).

## Decision: **`FUSED_SOFTMAX_V_TAIL_FAIL_LOCAL_AB`**

The fused softmax+V tail candidate is **value-correct** but **0.725× @ctx1024 / 0.876× @ctx4096** vs `gqa_coop_vec`
(throughput) — it **loses** the local A/B. Per the discipline (a tail-only improvement isn't enough unless it moves
the whole comparator), the local gate missed → **stopped before any W==D route**, banked. coop's hoisted-exp split
stands; the 5–6× gap to llama is unchanged and is the in-kernel q·k codegen-quality problem (which Path A does not
attack).

## Phase 0 — design reconciliation

- The "linearizer wall" was **refuted** (the coupled online-softmax+V kernel verifies/runs today).
- Path A is **softmax+V tail fusion only**; it **keeps coop's matmul q·k** (near-optimal among tinygrad primitives).
- Path A does **not** attack llama's full in-kernel-q·k codegen-quality gap (the real 5–6×).
- First gate = local A/B vs `gqa_coop_vec`; W==D only after a local pass; **stop if local misses** (it did).

## What was built

`extra/qk_fused_softmax_v_tail_ab.py` — total attention = coop's **matmul q·k** (scores) + a **fused inline-exp
softmax+V partial** (`inline_exp_partial_kernel`: coop_vec's GQA-reuse structure + `flash_max` precomputed, but
`p = exp(score − pm)` computed **inline per d-lane** instead of reading a precomputed `prob` buffer) + reused
`flash_gmax/den/combine`. This fuses `flash_prob` **into** the partial (one fewer kernel, no `prob` materialization),
using the proven register-accumulator idiom. Single output (`pout`) → no two-granularity wall.

**The FULL online-max removal (a single kernel that also fuses `flash_max`) is `BLOCKED_BY_IDIOM`** at the multi-split
decode shape: it must output **per-split `pm`** (for the cross-split combine) **and per-d `pout`** from one kernel —
two different store granularities, which is the Q8L-2 two-granularity store wall (`AFTER(STORE,STORE)` / store-group
of different range nests fails verify). The agent's expressiveness probe avoided it (single output, single split).
So at the real decode shape, removing `flash_max` needs the same deep idiom/codegen work the scope flagged — and it
would not change the verdict (the cost below is the `exp` redundancy, not `flash_max`).

## Correctness (vs numpy reference)

| ctx | rel_rmse | max_abs | gate (≤1e-3) |
|---:|---:|---:|---|
| 512 | 7.4e-4 | 3.9e-4 | PASS |
| 1024 | 7.0e-4 | 1.9e-4 | PASS |
| 4096 | 8.0e-4 | 1.7e-4 | PASS |

(≤1e-5 is unattainable: inline `exp` reorders fp vs hoisted; ~7e-4 matches coop's own ~2e-4 fp-reassoc error, so
1e-3 is the justified threshold. No layout mismatch.)

## Local A/B (throughput, clock-pinned, vs gqa_coop_vec)

| ctx | candidate µs | gqa_coop_vec µs | **speedup** |
|---:|---:|---:|---:|
| 512 | ~118 | ~85 | ~0.72× |
| 1024 | 118.4 | 84.7 | **0.725×** |
| 4096 | 163.2 | 142.9 | **0.876×** |

**Gate FAIL** (needs ≥1.05× @ctx1024; got 0.725×). (Per-call ProfileGraphEvent GPU-busy did not capture on the warm
JIT replay → throughput is the authoritative metric, consistent with the oracle/dispatch-probe method.)

## Interpretation — why it loses, and for the llama gap

Fusing `flash_prob` into the partial makes each of the **W=Hd+1=129 output-dim lanes recompute `exp`** for every
key, vs coop's `flash_prob` computing `exp` **once per key** (no redundancy) and the partial just reading `prob`.
The ~129× redundant `exp` **outweighs** the saved `prob` materialization (~128 KB ≈ ~0.3 µs) + one launch. This is
the same per-lane-redundancy that sank the fused-LDS tile (0.21×) — coop's hoisted-exp split is **near-optimal**, and
fusing it back is harmful (a fusion that re-introduces redundant work, per the decode T=1 principle).

So Path A confirms: **tail fusion does not help** (the tail is already cheap and coop's split is optimal). The 5–6×
gap to llama remains the **in-kernel q·k codegen quality** — which Path A deliberately did not touch (it keeps coop's
matmul q·k). That is the deep, separate problem (no bounded gate found; warp tile already showed tinygrad's UOp
codegen does not match llama's hand-tuned kernel).

## Lifecycle verdict

decode_eval candidate `fused_softmax_v_tail` (family `attention_split`, `ab_script` runner) → **`FAIL_LOCAL_AB`** →
`refute_candidate`, banked through the system. Refutation added.

## W==D: NOT reached (local A/B failed — discipline = stop).

## Acceptance gates

| gate | result |
|---|---|
| G1 fused kernel runs or precise idiom blocker | PASS (inline-exp runs; full online-max = precise BLOCKED_BY_IDIOM two-granularity wall) |
| G2 correctness measured | PASS (rel_rmse 7–8e-4) |
| G3 local A/B vs gqa_coop_vec measured | PASS (0.725×/0.876×) |
| G4 candidate through decode_eval/lifecycle | PASS (`FAIL_LOCAL_AB`) |
| G5 no W==D unless local passes | PASS (not added) |
| G6 no default/model change unless gated | PASS (no `tinygrad/`, no model route) |
| G7 no closed lane reopened | PASS |
| G8 policy guard passes | PASS |
| G9 tree clean after commit | PASS (commit below) |

## Next action

**Do not iterate bounded tile variants** (Path A is banked as a measured negative). The remaining lever is the deep
**in-kernel q·k codegen quality** (matching llama's hand-tuned kernel) — a separate, large project with no bounded
first gate yet — or **rest decode**. The llama oracle stays the validated target / non-promotable reference.

## Changed files
`extra/qk_fused_softmax_v_tail_ab.py` (new), `bench/qk-decode-eval/candidates.json`, this doc,
`bench/qk-lifecycle-search/refutations.json`, handoff/READMEs.

## Boundary
No `tinygrad/` change, no model route/default, no W==D route, no closed lane reopened, no performance claim (it lost).
Clock-pinned diagnostic; perf-state restored to `auto`.
