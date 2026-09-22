# B3 tuned-schedule characterization record - same-run cause instruments (NV session)

Date: 2026-08-03
Status: measurement record. Authorized by
`b3-prefill-host-overhead-scope-20260803.md` sections 1.1 and 4.1: run the same-run cause
instruments on the production tuned pp512 schedule and resolve the UNRESOLVED
wall-minus-busy residual split (submit vs polling vs overlap) on NV. No runtime code
changed; all probes are monkeypatch-only; nothing lands from this NV-only session. The AMD
leg is UNRESOLVED (no AMD GPU exists on this machine; section 7). Branch boundary: tinygrad
`nvidia-bringup-20260731`.

## 1. Protocol

Session: 2026-08-03 NV-only, RTX 5090 (sm_120), machine idle at every run acquisition
(foreign M5 probe absent), all timing runs serialized under `flock /tmp/nv_gpu.lock`.

Commit under test: `05b1e977484af04ef66ab3f69f42bf474cae1ea8` (current HEAD, tree clean for
tracked files). Pinned schedule tree (`04e500079d8f9615725736a99734b3a217ad0cb6`) measured
via worktree `/tmp/b3-pinned-tree` for the pinned-vs-HEAD control. The prefill runtime files
(`prefill_graph_gemm.py`, `hcq.py`, `fused_attention.py`, `flash_prefill.py`, `spec.py`
PACKED_FRAGMENT_LOAD rules) are identical pinned-vs-HEAD; only decode-side files differ.

Config for every run: `Q4K_PRIMITIVE=1 Q6K_PRIMITIVE=1 HALF=1 prefill_v2=true
prefill_concrete_kv=true PROFILE=0`, model `/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf`,
pp512 prompt `[(i % 1000) + 1 for i in range(512)]`, max_context 2048. Fused prefill
attention disabled per the B3 protocol:
`tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()`.

Evidence classes: OBSERVED = measured in this session under the lock; INFERRED = attributed
by arithmetic/subtraction or by promotion-record provenance, flagged inline.

## 2. Instruments (all in the same run on the tuned pp512 schedule)

Probe: `/tmp/b3_split_probe_20260803.py` (monkeypatches only; output JSON). Discard and
pass1 are the one-time graph-instantiation passes (6.8 s / 2.9 s); the steady state is
pass2/pass3 (DEBUG=0). GPU busy requires `DEBUG>=2` (per-graph end-wait added, changing the
wait count but not the kernels), so one DEBUG=2 pass follows for busy.

| instrument | pattern | result (pass3 steady) |
| --- | --- | --- |
| (a) poll count per wait / total | spin iterations counted inside `HCQSignal.wait` via the `HCQSignal.value` getter trace | 10 waits: 9 tiny waits at 1 poll each + 1 final wait at 97,440 polls; total 97,449 polls |
| (b) exclusive polling cost | time inside `HCQSignal.value` only (getter), excluding the rest of `wait()` | 95.05 ms total poll time; ~0.98 us/poll (97,449 polls); per-poll median 2.48 us, p90 95.0 ms (the final wait) |
| (c) submission latency | time inside `HWQueue.submit` per graph group and total | 9 submits, 0.17 ms total; 0.01-0.04 ms per graph group (6 groups) |
| (d) wall-minus-busy residual split | wall - busy, split between (b), (c), and overlap | residual 22.70 ms = submit 0.17 ms + non-overlapped poll 0.00 ms + other 22.53 ms (INFERRED by subtraction); overlap = min(wait, busy) = 137.26 ms |

Signal writes (value setter) measured for completeness: 12 writes, 0.02 ms. No polls occur
outside `wait()` (outside_polls = 0).

## 3. Same-run measured rows (HEAD `05b1e9774`, fused attention OFF)

| pass | wall ms | busy ms | waits | wait ms | polls | poll ms | submit ms | groups |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| discard | 6842.67 | - | 60 | 33.43 | 23,788 | 22.92 | 4.99 | 0 |
| pass1 | 2880.91 | - | 10 | 72.15 | 50,791 | 50.16 | 0.21 | 6 |
| pass2 | 161.31 | - | 10 | 137.34 | 97,255 | 94.96 | 0.15 | 6 |
| pass3 | 160.70 | - | 10 | 137.26 | 97,449 | 95.05 | 0.17 | 6 |
| busy (DEBUG=2) | 176.03 | 138.0 | 18 | 137.83 | 97,428 | 95.71 | 0.19 | 6 |

Wait structure (pass3): 9 waits at 1 poll each (~0.003-0.009 ms) + 1 final wait of 137.2 ms
at 97,440 polls. Per-group host cost 0.04-0.13 ms/group; the final wait is the only
non-trivial one and it tracks GPU busy, i.e. it is overlapped with execution, not a submit
gap.

## 4. Wall-minus-busy residual split (section 1.1 resolution)

| component | pass2 ms | pass3 ms | share of residual | evidence |
| --- | ---: | ---: | ---: | --- |
| wall | 161.31 | 160.70 | - | OBSERVED |
| busy | 138.0 | 138.0 | - | OBSERVED (DEBUG=2 pass) |
| residual = wall - busy | 23.31 | 22.70 | 100% | OBSERVED arithmetic |
| (c) submission latency | 0.15 | 0.17 | 0.7% | OBSERVED |
| (b) non-overlapped polling | 0.00 | 0.00 | 0.0% | OBSERVED: poll time is 95 ms but sits inside a wait that is 99.6% overlapped with busy (137.3 vs 138.0); non-overlapped poll = max(0, wait - busy) x poll/wait |
| overlap = min(wait, busy) | 137.34 | 137.26 | (not residual) | OBSERVED |
| other = residual - submit - non-overlapped poll | 23.16 | 22.53 | 99.3% | INFERRED by subtraction: host work outside `wait()` and `submit()` |

Verdict: the residual is NOT submit-dominated and NOT poll-dominated. Submits measure 0.17
ms (0.7%); non-overlapped polling is 0.00 ms because the single 137 ms wait overlaps GPU
busy almost exactly. The residual is dominated by host work outside `wait()`/`submit()` -
graph-group dispatch, per-chunk host prep between groups, and first-token post-processing -
at ~22.5 ms (99.3%, INFERRED by subtraction). This resolves section 1.1's UNRESOLVED split
on NV: the "submit cost" reading of the residual is refuted by the measured split.

## 5. Recommendation (section 2 fix shapes, ranked by the measured split)

The measured split does not support naming the residual "submit cost" (submit is 0.7%), so
the shape ordering rests on what the residual actually is:

1. **(a) Cache the signal view per wait** - smallest first probe, cheap, and buys the AMD
   runtime row, as the scope recommends. Measured here: ~0.98 us/poll exclusive. Polls are
   fully overlapped with busy, so (a) removes CPU spin but will NOT collapse single-pass
   wall by itself (the measured split gives it 0% of the residual). Still the right first
   probe: closed-default, both targets, smallest blast radius.
2. **(c) Whole-schedule graph replay** - the endgame shape, but NOT for submit removal:
   submits are only 0.17 ms. Its value is removing the inter-group host gap, which is where
   the ~22.5 ms "other" likely lives. Re-scoped as "remove the generate-loop host gap
   between graph groups", it is the only shape that targets the measured residual mass.
3. **(b) Real blocking wait** - removes CPU spin only; wait is overlapped with busy, so it
   does not cut the residual either, and it is the riskiest shape (driver-facing,
   NV-specific, shared HCQ surface). Not indicated by this split; keep closed unless (a)'s
   measured wait collapse proves insufficient.

Next instrument before choosing: split the ~22.5 ms "other" into per-chunk host prep vs
graph dispatch vs first-token post-processing (e.g. trace the generate loop between graph
calls), so (c)'s re-scope is evidence-backed.

## 6. Baseline pins re-verified where applicable

The scope's pinned rows (warm wall 44-46 ms, busy 24.1 ms, wait() 23.7-23.8 ms across 10
waits, 8 graph groups, wall/busy ~1.9x) were measured on the **fused-prefill-attention-ON**
schedule (promotion present in the pinned tree; fused port `d8bac6914`). The B3 protocol
disables fused attention, which switches to a different schedule. At both trees, fused ON
crashes deterministically in pure Python: `UOp verification failed ... Ops.PACKED_FRAGMENT_LOAD
dtypes.half 5 ... native_abi='nv_sm120_packed_fragment_hd128_loop_v1'` (verified at HEAD via
the e2e bench prefill arm, and at the pinned tree in this session); pinned fused ON with
`SPEC=0` crashes at the renderer emitting the AMD ABI
(`amd_gfx1100_attention_loop_state_v1`). The 44-46 ms pins therefore CANNOT be re-measured
in this environment; they are attributed to the fused-ON schedule (INFERRED from the
promotion record + fused port timeline) and the B3-protocol rows above are the measured
rows. Recorded as a deviation (section 8).

Pinned-tree control (fused OFF, worktree at `04e500079`): wall `[6.7609, 2.8615, 0.1607,
0.1591]` s, busy 137.93 ms (DEBUG=2), 6 groups, first token 151936 - identical to HEAD,
so the fused-OFF measurements are schedule-valid at both trees.

Correctness pins (verified at HEAD in this session via `model_e2e_bench.py`):

- First-token digits `[50994, 82, 31109, 3508, 692, 2, 11162, 100, 254, 30317, 2655, 12080,
  25, 576, 35264, 5624]` - MATCHED exactly (decode stream; prefill first token is 151936, a
  different quantity, and holds in every row above).
- Decode sha256 `0721c16fbf70779cb6cebd5cf64eab50a1f61c7882d402c60c27d22597548ebe` - MATCHED.
- Census row `prefill_overlay_promotion: candidate_set:sha256:
  1b8ea95d50bb55962474721cf013a6c3a704038916856353c65281112a166c7f` - MATCHED at HEAD,
  pinned tree, and every probe run; strategy `FULL_RESIDENT_OVERLAY`.

Wall/busy ratio on the measured fused-OFF schedule: 160.70/138.0 = **1.16x**, inside the
llama 1.15-1.35x envelope by ratio, but the absolute wall is ~160 ms because busy is 138 ms
(SDPA path), vs the pinned fused-ON 24.1 ms busy. The ratio row is therefore not comparable
to the pinned 1.9x; the fused-ON rows are unreproducible (section 8).

## 7. AMD leg - UNRESOLVED

No AMD GPU exists on this machine (RTX 5090 only, no ssh target), so the AMD runtime
control required by the scope (section 3) cannot be run. The submit/poll path is shared HCQ
code that also serves AMD (`HCQSignal.wait`/`value` -> `cpu_view().view()` -> `to_mv().cast()`
is target-independent), so a blind change to the shared polling path risks the shared
target. The AMD leg must be runtime-measured on AMD hardware in the same session family
before any change lands; nothing lands from this NV-only session. The compile-only AMD
render equality (pg3) remains required per the campaign guardrail.

## 8. Deviations

- The scope's fused-ON pins (44-46 ms wall, 24.1 ms busy, 8 graph groups, 23.7-23.8 ms
  wait(), ~1.9x) are not re-verifiable in this environment: fused prefill attention is
  broken at BOTH trees (deterministic `PACKED_FRAGMENT_LOAD` UOp verification failure; the
  pinned tree additionally emits the AMD gfx1100 ABI on NV when forced). The B3 protocol
  (fused OFF) measures a different schedule: ~160 ms wall / ~138 ms busy / 6 groups / 10
  waits. Recorded as OBSERVED pinned-tree fused-OFF = 160 ms; the pinned 44-46 ms row is
  attributed to fused-ON (INFERRED) and cannot be re-measured.
- The first probe run (HEAD `44725ad41`) was contaminated by a foreign M5 probe at ~88% CPU
  during part of the run; that result is excluded. All numbers above are clean runs at
  `05b1e9774` under the lock (runtime files identical between the two commits).
- The DEBUG=2 busy pass adds an end-wait per graph call (18 waits vs 10), changing the wait
  count but not the kernels; busy (138.0 ms) is unaffected.
- The e2e bench's default prefill arm (fused ON) errors at HEAD with the same
  `PACKED_FRAGMENT_LOAD` verification failure; the decode sha / first-token pins are taken
  from the decode arm, which uses the SDPA path and holds.
- GPU busy was 0% at every lock acquisition; no un-locked timing runs were made.

## 9. HARD STOP

Nothing beyond this scope. This record authorizes no implementation, no promotion to
`dev`/`exp`/`master`, and no push; the parent pushes after review. The runtime change is
env/record opt-in (closed-default) per scope section 4.4, gated on the NV+AMD same-session
measurement with the three pins, and never lands from this NV-only session.

## 10. References

- `b3-prefill-host-overhead-scope-20260803.md` (sections 1.1, 2, 3, 4.1, 4.3)
- `nv-campaign-forward-review-amendment-20260803.md` (sections 3 Q4, 4.2)
- `nv-performance-campaign-scope-20260801.md` (sections 8.4, 8.5, 11.1-11.3, 13.1-13.2)
- Probes: `/tmp/b3_split_probe_20260803.py`, `/tmp/b3_control_wall.py`,
  `/tmp/b3_control_busy.py`, `/tmp/b3_pin_control.py` (all /tmp, not committed)
