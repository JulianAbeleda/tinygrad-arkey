# NV substrate exhaustive scope: the anchor+shadow topology that unlocks 240 (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731`
Status: **scope record; S1 re-measured at HEAD (runtime width present).** Turns the
08-15 substrate definition and the 08-17 stress test into one exhaustive,
buildable inventory: every construction that must exist for the native NV
decode route to render `overlap_mass > 0` (the anchor+shadow topology), its
gate, its test, and its honest ceiling. This is the plan the next GPU sessions
execute against.

## 1. The definition the scope builds against

From `nv-substrate-definition-20260815.md`: **substrate = a capability in the
compile/lower/emit/runtime stack that makes a target construction expressible
as one valid, replayable program.** Capability-blocked (cannot render) is a
substrate problem; wall-blocked (renders but loses) is a values problem. They
route to different work.

The target construction for 240: one steady decode token on the **native NV
production route** (the one at 208.84 tok/s in the 08-17 wall account) that
renders with `overlap_mass > 0` and the promoted kernel-work rows intact.

## 2. Why the scope exists (the arithmetic envelope)

08-17 exact wall account (zero residual), same session, d512, RTX 5090:

| term | tinygrad | llama | delta |
| --- | ---: | ---: | ---: |
| wall | 4788.3 us (208.84 tok/s) | 4058.9 us (246.37 tok/s) | +729.4 |
| GPU busy (union) | 4519.3 | 3890.5 | +628.8 |
| host gap | 269.0 | 168.3 | +100.6 |
| overlap mass | 0.0 | 1125.1 | -1125.1 |
| node sum | 4519.3 | 5015.7 | -496.3 |

Stress test (`nv-240-climb-stress-test-20260817.json`, committed): llama's
overlap is real GPU concurrency (865 same-stream pairs/replay, negative min
inter-kernel gap, worth 53.5 tok/s to llama under a serialization
counterfactual); tinygrad is exactly serial (0 pairs, union == node sum).
Sensitivity: perfect kernel-row parity caps at 228.45 tok/s, plus host parity
at 233.83 tok/s; clearing 240.0 needs at least ~110 us of hidden kernel time,
and matching llama's union needs 628.8 us.

**Conclusion this scope executes:** the last ~110-621 us to 240 is overlap.
Overlap needs the anchor+shadow topology, which is CONSTRUCTION-REQUIRED on
native. This document inventories every construction that path needs.

## 3. Why the current DAG has zero overlap (measured root cause)

`nv-overlap-planner-serialization-root-cause-20260815.md` (runtime dep
capture, not a reconstruction): the **memory planner aliases independent
fan-out live ranges into one arena slot**, adding WAR/WAW edges that collapse
the DAG to width 1. With the planner on, every node after index 3 has exactly
one predecessor: `q -> k -> v -> rope/kv -> flash -> ...` is one serial chain.
With `NO_MEMORY_PLANNER=1`, q/k/v and gate/up become true siblings (9
fan-in/fan-out joins in the first 32 nodes).

The probe A/B on the CUDA route (same record): planner-on 178.95 tok/s,
planner-off 4 streams 187.53 tok/s = **+8.5 tok/s (+4.8%)**, bitwise identical
tokens. The gain is real but bounded: the only independent work is GEMV
siblings (q/k/v, gate/up) which contend for HBM bandwidth, and the
rope/kv/flash/norm support is a fan-in on the critical path that cannot hide.

This is the structural reason the 08-13 "launch hiding exhausted" account and
the 08-14 "anchor does not transfer" verdict both measured flat: the scans ran
on a planner-serialized DAG that had nothing to hide.

## 4. The substrate inventory (the stack, bottom to top)

Every row is a construction (or an already-built capability that gates a later
construction). Rows are ordered by dependency: S1 must land before S2 can be
measured, S2 before S3, etc.

### S1. DAG width — MEASURED PRESENT AT HEAD (no planner change required)

**What.** The memory planner (`tinygrad/schedule/memory.py`,
`memory_plan_rewrite`) must stop aliasing dependency-independent fan-out live
ranges (q/k/v, gate/up) into one arena slot. Product change, not the
`NO_MEMORY_PLANNER=1` probe (which pins every buffer and spikes VRAM). The
probe proves the geometry; the product change must restore sibling edges
without the VRAM cost.

**Re-measured at HEAD (2026-08-17).** The 08-15 chain finding
(`nv-overlap-planner-serialization-root-cause-20260815.md`) was captured on the
stale CUDA route at 08-15 HEAD. On the current native NV production route the
runtime-recorded dependencies (HCQGraph profiler replay JSONL, steady token)
show the decode DAG already has width: q/k/v GEMVs are true siblings (identical
predecessor sets, neither depends on the other), the max-ready width is 10, and
the critical path is 189 levels over 594 nodes. Evidence:
`nv-substrate-s1-runtime-width-head-20260817.json`. S1 is therefore not a
construction anymore; it is a measured gate that already passes at HEAD.

**Why it still gates the stack.** Without width there is nothing to co-schedule;
the measurement confirms width is present, so the next construction (S2) is
unblocked. The S1 gate becomes a regression guard: keep q/k/v (and gate/up)
siblings under future planner changes.

**Gate (pass criteria, in order):**
1. Runtime dep capture shows q/k/v with the same predecessor (siblings) — PASS
   at HEAD (width 10, evidence above).
2. Ledger (NV HCQGraph profiler) shows `overlap_mass > 0.0` on steady tokens —
   this moves to the S2 gate, because width without multi-queue placement still
   executes serially on one GPFIFO.
3. Same-session canonical A/B wall: candidate not slower than control, tokens
   bitwise identical (existing authority, fixed-depth d512).

**Honest ceiling.** On the CUDA route the width restoration alone is +4.8%
(~8.5 tok/s off a ~179 baseline). On the NV production route the baseline is
208.84, so the same fractional gain is ~+10 tok/s (to ~219) if bandwidth
physics transfers; this is a measurement, not a promise. Full 240 needs the
stack above, not S1 alone.

**Test.** Pin the runtime-width finding as a regression test: assert the decode
q/k/v triple are siblings in the runtime dependency capture.

### S2. Native NV multi-GPFIFO execution — BUILT + QUALIFIED; now the first real construction

**What.** The native hardware substrate exists and passed: two compute GPFIFOs
under one async ctxshare, constructed before the group's first schedule,
co-schedule light kernels at 9.7% interval-union overlap (repeatable, clears
the 5% gate). See `nv-rank2-native-concurrency-construction-verdict-20260805.md`.
`HCQ_NUM_COMPUTE=2` is the qualified construction; `hw_compute_queues()`
exposes `COMPUTE:{i}` and the HCQ scheduler already has a name-pinned
multi-queue cut policy (`NV_MULTI_QUEUE_CUT_POLICY`,
`HCQ_NV_MULTI_QUEUE_INDICES`, `scratchpad/nv_multi_queue_probe.py`,
`test/unit/test_nv_multi_queue_probe_construction.py`).

**Why it is now first.** S1 is measured present at HEAD (width 10), so the
remaining gap between "DAG has width" and "overlap_mass > 0" is execution
placement. The scheduler already exposes a name-pinned NV admission policy
(`HCQ_NV_MULTI_QUEUE_PROGRAMS`, `HCQ_NV_MULTI_QUEUE_INDICES`,
`HCQ_NV_MULTI_QUEUE_CUT_POLICY`, `_pick_compute_queue` in `hcq.py`), and
`HCQ_NUM_COMPUTE=2` is the qualified native construction. The gate is: does
placing the width-10 DAG across two NV compute GPFIFOs convert to wall on the
production route, with tokens bitwise identical?

**Gate.**
1. `HCQ_NUM_COMPUTE=2` + a cut policy that places q/k/v (and gate/up) on
   distinct GPFIFOs runs end-to-end on the production decode route, tokens
   bitwise identical.
2. NV ledger shows `overlap_mass > 0` AND the wall A/B is positive (not
   merely non-negative): same-session control (S1, 1 queue) vs candidate
   (S1, 2 queues).
3. The +4.8% CUDA-route number is reproduced or beaten on the native route.

**Gate result (measured 2026-08-17, generic readiness placement at HEAD).**
The name-pinned cut policy was replaced by a generic dependency-readiness
placement (`HCQ_NV_READY_PLACEMENT=1` in `hcq.py`: a node goes to the
least-loaded GPFIFO unless it directly depends on the primary queue's current
tail). The gate FAILS on the production route as composed:
1. Placement engages but correctness holds: the census records 17-27% of
   steady-token nodes on the aux GPFIFO, including the q/k/v GEMV siblings
   (`q4k_g3_lanemap_gemv_1024_4096`, `q6k_v_four_warp_fp16_direct_1024_4096`,
   `q4k_warp_coop_q8_dp4a_partial_1024_4096`, `q6k_q8_warp_direct_1024_4096`),
   and the A/B tokens are bitwise identical (`5ede6924...` both arms).
2. Hardware overlap does not appear: candidate NV ledger shows
   `overlap_mass = 2.1us` over a 4580us node sum (vs llama's 1125us), so the
   aux kernels serialize behind the join instead of co-running.
3. The wall A/B is slightly NEGATIVE: candidate 205.88 vs control 207.75
   tok/s (+44us/token), i.e. the cross-queue handoff costs more than the
   (absent) overlap.
Evidence: `nv-substrate-s2-ready-placement-wall-control-20260817.json`,
`...-candidate-20260817.json`, `...-census-20260817.jsonl`,
`...-ledger-candidate-20260817.json`.

**Why it is flat (mechanism).** The DAG width is temporally sandwiched: each
ready set (q/k/v siblings, gate/up siblings) sits between a primary producer
(rope) and a primary consumer (attention score / ffn resadd), so while the aux
GPFIFO runs the siblings the primary has no independent continuation to
overlap against; the extra cross-queue signal is pure overhead. This is
exactly the stress-test prediction (`bf57ead51`): non-overlap levers cap at
233.8 tok/s and every prior arm measured FLAT. The generic readiness primitive
is kept (gated off by default) as the scheduler that S3 needs once a
shadow-timed anchor exists to hide work behind.

**Honest ceiling.** The measured +8.5 tok/s on CUDA is the current best
evidence for what width + multi-queue is worth on this DAG; the native number
is unmeasured. HBM contention between GEMV siblings caps the gain below the
serialized-sum naive estimate.

### S3. Anchor+shadow composition — CONSTRUCTION-REQUIRED (the layer-2 wall)

**What.** The composed topology: one long fused-quant GEMV anchor per token
with support work hidden behind it (`overlap_mass > 0`), exactly as defined in
`nv-substrate-definition-20260815.md` section 3. On the current DAG the GEMV
chain IS the anchor; what is missing is the shadow — support kernels whose
durations stop landing on the critical path.

**The dependency reality (measured, not guessed):**
- q/k/v and gate/up GEMV siblings: hideable behind each other (S1+S2).
- rope/kv/flash/norm: fan-in on the critical path (each consumes a GEMV
  output), so they cannot hide behind an anchor *without a body change*. This
  is why the flash pair is the only llama-exposed mass left on our side, and it
  is at body parity (4.16 vs 4.10 us isolated).
- reduce_output and vocab_aux: epilogue folds, not overlap; they are the
  kernel-work rows in the wall account (fusion row, measured FLAT for the
  body-free fold at 08-13).

**Gate.** NV ledger on a steady token shows `overlap_mass > 0` with the
promoted body rows (Q6 four-warp, reduce-output fusion) intact, and the wall
A/B at HEAD is positive with bitwise-identical tokens.

**Honest ceiling.** This row's transferable mass is bounded by the DAG's real
independence (the S1 siblings) plus any prologue/epilogue hiding the execution
mechanism (S4) adds. It is NOT llama's 1125 us: ~571 us of llama's hidden mass
is quantize/norm/rope work tinygrad already fused away (08-16 PDL trace), and
the flash pair is at body parity with an installed gap priced at ~+122 us.

### S4. Programmatic dependent launch (llama's mechanism) — half wired, economics-negative at HEAD

**What.** llama's overlap is single-stream PDL: `cudaTriggerProgrammaticLaunchCompletion`
at kernel start + `cudaGridDependencySynchronize` before consumer reads
(`ggml-cuda/common.cuh`, verified 08-16). Two halves on native:

- Launch-gap half: **already wired.** `NVComputeQueue.exec()` chains
  consecutive same-queue kernels via `dependent_qmd0_*` (QMD_SCHEDULE),
  eliminating CPU round-trips (`ops_nv.py:166-171`; hcq.py:390-393 NV chain
  optimization).
- Programmatic half: **CONSTRUCTION-REQUIRED.** No renderer emits
  `griddepcontrol`-class PTX (zero hits in `ptx.py`/`cuda.py`), and the QMD v05
  latch fields (`DEPENDENCE_COUNTER`, `WAIT_ON_LATCH_ID`, `ARRIVE_AT_LATCH_ID`)
  are present in `autogen/nv_570.py` but never programmed. Full PDL needs
  renderer emission + QMD latch programming + a per-pair scheduler policy.

**Why not first.** The 08-16 trace priced full PDL at ~18-33 us recoverable
(the launch-gap half is already active; the programmatic half mostly overlaps
work we have already fused). It is the mechanism that maps 1:1 to llama, so it
is the fallback if S1+S2+S3 do not reach 240, but it is not the first lever.

**Gate (if pursued).** A two-kernel microbench (producer/consumer, no data
dependency between prologue and body) shows next-kernel prologue overlap on
one NV GPFIFO with clean numerics; then a decode wall A/B.

### S5. Host gap (269.0 vs 168.3 us) — separate row, graph substrate exhausted

**What.** 100.6 us of the wall delta is host-gap: eager/JIT handoff and graph
install behavior. The 08-13 account found the graph substrate for this
exhausted (replay factor 0.917, 95.6% busy). Not part of the anchor+shadow
stack; tracked separately in the wall account.

## 5. What this scope explicitly does NOT build

- Not llama's quantize pass (we fused it; no node to pipeline).
- Not the CUDA multi-stream route (Route B) as the production target: it runs
  a degraded graph (no NV reduce-output fusion, ~179 baseline) and is a
  benchmark oracle for S1/S2 ceilings, not the delivery route.
- Not per-shape GEMV tuning (Q4 FFN-down, Q6 attention-V sweeps closed NO-GO).
- Not the flash single-stage body change (structural NO-GO at 08-13).
- Not `NO_MEMORY_PLANNER=1` as a product (VRAM spike; probe only).

## 6. Order of execution and gates (what the next sessions run)

1. **S1 gate recorded at HEAD (done):** runtime deps show q/k/v siblings, width
   10; no planner change needed. Pin as a regression test.
2. **S2 (the real first lever):** `HCQ_NUM_COMPUTE=2` + name-pinned cut policy
   for the q/k/v (and gate/up) GEMV programs on the production decode route;
   same-session ledger + wall A/B vs single-queue control, tokens bitwise
   identical. **MEASURED 2026-08-17:** the generic readiness placement engages
   (census) but overlap stays ~0 and the wall is slightly negative; S2 as
   composed does not convert to wall. The primitive is kept gated-off; the
   blocker for wall now is S3 (anchor+shadow temporal alignment), not placement.
3. **S3 after S2:** compose the anchor+shadow and measure the transferable
   overlap mass against the S1+S2 baseline.
4. **S4 only if the stack stalls below 240:** PDL microbench, then decode A/B.
5. **S5 in parallel where cheap:** host-gap measurement at HEAD (269.0 vs
   168.3) is already in the account; any eager/JIT handoff reduction is a
   separate tracked row.

## 7. Acceptance test (one steady token at HEAD, native NV)

The substrate is "present" when, in one session:

1. Runtime dep capture shows q/k/v (and gate/up) siblings — PASS at HEAD
   (no `NO_MEMORY_PLANNER`).
2. NV HCQGraph ledger shows `overlap_mass > 0.0` on steady tokens with the
   promoted body rows intact.
3. Same-session canonical A/B: candidate wall <= control, tokens bitwise
   identical, and the wall beats the 208.84 baseline by the measured S1+S2
   margin.

Every claim above is a measurement gate; a row is not "landed" until its gate
passes with bitwise-identical tokens in a flocked same-session A/B.

## 8. Evidence map

- wall account + stress test: `nv-240-exact-wall-account-20260817.md`,
  `nv-240-climb-stress-test-20260817.json`
- substrate definition: `nv-substrate-definition-20260815.md`
- DAG width root cause: `nv-overlap-planner-serialization-root-cause-20260815.md`
- native co-schedule: `nv-rank2-native-concurrency-construction-verdict-20260805.md`
- PDL mechanism + economics: `nv-llama-pdl-launch-hiding-trace-record-20260816.md`
- multi-queue policy code: `ops_nv.py` (`HCQ_NUM_COMPUTE`),
  `tinygrad/runtime/graph/hcq.py` (`NV_MULTI_QUEUE_CUT_POLICY`),
  `scratchpad/nv_multi_queue_probe.py`
