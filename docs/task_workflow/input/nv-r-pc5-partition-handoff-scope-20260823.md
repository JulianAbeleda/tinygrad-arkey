# NV R P-C5 partition handoff scope (2026-08-23)

This scope is Tier 0 of the master iteration driver
`nv-remaining-route-iteration-scope-20260823.md`.

Status: measurement only. No production model, renderer, scheduler, runtime,
or route code is authorized. This scope is the efficient continuation of
`nv-r-residual-pdl-concurrency-adjudication-scope-20260822.md`.

## 1. Frozen authority (do not redo)

Repository `/home/ubuntu/tinygrad-arkey`, branch `nvidia-bringup-20260731`,
HEAD `6570abc025514273faa100c66b979e531585a1e1`. GPU RTX 5090 sm_120, model
`/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`, depth 512, `DEV=NV`.

- Q occurrence-0 closed: `P-C5 = +0.008 us`, prefix lever `C5-C6 = +0.664 us`.
  Authority `docs/task_workflow/output/nv-r-predecessor-conditioned-exact-result-20260823.md`.
- K/V occurrence-0 closed: `P-C5 = +0.520 us/call`, `C5-C6 = -0.112 us`,
  count-weighted `+13.520 us/token`. Authority
  `docs/task_workflow/evidence/nv-r-predecessor-conditioned-exact-kv-20260823/closure.json`.
- O occurrence-0 closed: `P-C5 = +1.032 us/call`, `C5-C6 = +0.016 us`,
  count-weighted `+37.152 us/token`. Authority
  `docs/task_workflow/evidence/nv-r-predecessor-conditioned-exact-o-20260823/closure.json`.
- Both rows have zero timing-identity residual, zero installed overlap, and
  matching output/token SHAs. The launch order for the K/V chain is
  gate/up -> FFN down -> provider -> cooperative Q -> cooperative K/V.

The next agent must not regenerate these numbers. The only open question is
what the two positive installed remainders are made of.

## 2. Objective

Partition the two positive `P-C5` remainders into three named, additive terms
with zero residual:

```text
P - C5 =
  (admission_P - admission_C5) +
  (wait_P      - wait_C5)      +
  (body_P      - body_C5)
```

where `admission = grid_start - predecessor_end`, `wait = wait_exit -
grid_start`, and `body = exit - wait_exit`. Each term must close in the same
session, forward and reverse, with locked clocks and a token-SHA gate. This
directly names the mechanism instead of inferring it from command intervals.

## 3. Reusable assets

All of the following already exist and must be reused, not rebuilt:

- Live exact capture: `extra/llm_research/decode/nv_r_predecessor_conditioned_exact_kv.py`
  and `nv_r_predecessor_conditioned_exact_o.py`.
- In-kernel `%globaltimer` injection (grid_start, wait_exit, trigger):
  `extra/llm_research/decode/nv_edge_aware_pdl_render_policy.py` and
  `nv_edge_aware_pdl_stage3_capture.py`. This is the proven, numerics-neutral
  wait-exit machinery.
- HCQ profile overhead and per-kernel duration:
  `extra/llm_research/decode/nv_hcq_profile_per_kernel.py`.
- Counters: `nv_row_counter_bridge.py` (NCU exact-image), and
  `nv_full_token_dram_counters.py` (single-token DRAM window).
- QMD dependence semantics: `nv_qmd_dependence_counter_probe.py`.
- Capture source of truth: `/tmp/decode_calls.json` if present, otherwise
  regenerate with `full_token_dag_capture.py`.

## 4. Gated plan

### Phase A: instrument and partition (mandatory, one process per row)

1. Extend the Stage 3 render policy to arm the installed K/V and O target
   occurrences and their immediate producers with `grid_start`, `wait_exit`,
   and `exit` slots. Use the existing `%globaltimer` injection; never touch
   kernel numerics.
2. In the same fresh decode process, capture both the installed path and the
   C5 replay for that row, `PROFILE=1`, locked clocks.
3. Compute the three-term partition above. Require the sum to equal `P-C5`
   within one 32 ns timestamp quantum; otherwise stop and diagnose the
   harness before interpreting anything.
4. Run forward and reverse arm orders and merge medians.
5. Emit a row with `admission_us`, `wait_us`, `body_us`, and the residual.

Efficiency: K/V, Q, and O all occur in one decode token, so the hook can arm
all three families in a single model load and record all rows in one pass
instead of three separate decode runs.

### Phase B: conditional scheduler microgate

Run only if `wait_P - wait_C5` is the dominant term and count-weighted
`wait` exceeds `+50 us/token`.

- Reconstruct the smallest producer-to-join span from retained production
  cubins: Q branch overlapped with K/V, control/candidate/control.
- Report wall, union, node_sum, useful_body, and overlap.
- Promotion requires a positive token-SHA reverse wall bracket.
- Do not run if the dominant term is admission or body; those go to Phase C.

### Phase C: counter attribution for the dominant term

For the winning term, run the exact-image NCU bridge or the DRAM counter
window under a validated hot/cold eviction control with a positive
cache-sensitive control. Label L2, TLB, DRAM, and instruction-state
contributions separately. No `cache_state` label without all three of:
positive eviction control, reverse bracket, and direct counter evidence.

### Phase D: final partition report

Write findings-first under `docs/task_workflow/output/` with:

- named partition and residual;
- whether admission, wait, or body explains each remainder;
- invalidated prior claims;
- terminal verdict below;
- ranked next actions with expected wall leverage.

## 5. Decision rules

- `wait` dominant: dependency serialization; scheduler route.
- `body` dominant: kernel residence; occupancy/body route.
- `admission` dominant: QMD/launch dispatch route.
- Residual nonzero after calibration: harness failure, do not interpret.
- `C5 == C6` is expected; it means the real prefix is neutral, not that the
  installed remainder is absent.
- Do not extrapolate occurrence 0 to all 26/36 calls without a count-weighted
  model, and do not book projected wall recovery.

## 6. Artifacts and integrity

- One analysis script per phase under `extra/llm_research/decode/`.
- One findings report under `docs/task_workflow/output/`.
- One evidence directory with closure JSON, raw profile/timestamp logs, and a
  `sha256.txt` manifest covering every retained artifact.
- Labels `[MEASURED]`, `[INFERRED]`, `[UNMEASURED]`, `[INVALIDATED]`.
- `py_compile` and `git diff --check` must pass. Reset GPU clocks at the end.

## 7. Terminal outcomes

- `PARTITIONED_RECOVERABLE`: a named term is positive, reproducible, and above
  the `+50 us/token` promotion bar after a token-SHA wall bracket.
- `PARTITIONED_NOT_RECOVERABLE`: the partition closes but no term clears the
  bar.
- `PARTITION_INCOMPLETE`: a term cannot be measured; report exactly which
  observable is missing and stop, do not infer.

The overall 240 tok/s verdict stays `240_UNMEASURED` until a fresh
token-SHA reverse wall bracket demonstrates recovery. No parity or
wall-recovery claim is permitted from this scope.
