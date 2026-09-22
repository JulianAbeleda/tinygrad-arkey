# NV overlap substrate: exhaustive build scope (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731` (HEAD `4e80f32fe`)
Status: **decisive probe run. The llama-shaped shadow pattern co-schedules on
the native pair ONLY when the shadow kernels are the same size as the anchor
kernels. The production shadow (tiny aux: rope/norms/kv-store at 0.5-1.5 us,
one 15.7 us q6k_v) does not co-schedule; every faithful arm measures
flat-to-negative. The subgraph-partition substrate is therefore not buildable
to a positive overlap on native; this scope records what was tried, why it
fails, and what would be needed.**

## 1. Goal and context

The 240 tok/s line needs ~110 us of hidden kernel time beyond the
non-overlap levers (`nv-overlap-substrate-arithmetic`). Llama hides 1125 us by
running per-layer support kernels (quantize, rope, kv_set_rows) on a second
CUDA stream, dep-free against the mmq anchor. We fused all that support work
into our GEMV epilogues (node_sum 4519.3 us is 496 us BELOW llama's 5015.7),
so at HEAD there is no dep-free kernel mass left to co-schedule at all.

The substrate question: can the native two-GPFIFO construction host the
llama-shaped pattern at decode kernel sizes? If yes, we build a subgraph
partitioner that moves the K/V subgraph (rope(k/v) + q6k_v + kv-store) to the
aux queue behind the primary chain. If no, the native 1-to-1 route is closed
and the ledger must say so.

## 2. Decisive probe: decode-shaped shadow composition matrix

Probe: `scratchpad/nv_decode_shaped_overlap_probe.py` on the two bootstrap
compute GPFIFOs (`HCQ_NUM_COMPUTE=2`), fp32 GEMVs + tiny elementwise aux
kernels, timestamp signals via the same `run_jobs` machinery as the R3/R5
probe. All runs on the bench lock; evidence JSONs in
`docs/task_workflow/evidence/nv-shadow-probe-*.json`.

### 2a. Shadow chain = same-size GEMVs as the anchor (`shadow` arm)

| m (anchor+shadow) | reps | overlap (3 runs) |
| --- | ---: | ---: |
| 512 (~4 us) | 16 | -17.8% / -10.3% / -3.3% |
| 1024 (~9 us) | 16 | **+9.6% / +10.8% / +15.9%** |
| 2048 (~20 us) | 12 | -0.5% / +2.0% / +0.6% |

At 1024 the equal-size shadow co-schedules reliably (+9.6-15.9%). At 512 it is
overhead-dominated (negative). At 2048 it is DRAM-bound (flat). The
co-scheduling window exists but is narrow: it requires the shadow kernels to
be large enough to amortize the wait boundary and small enough to stay
latency-bound.

### 2b. Shadow chain = production mix (`shadow_real` arm)

1 big GEMV (q6k_v-like) + 3 tiny elementwise (rope/norm-like), head wait.

| m | auxsize | overlap (3-4 runs) |
| --- | ---: | ---: |
| 1024 | 4096 | +0.9% / +0.8% / +1.6% / +3.6% |
| 2048 | 4096 | +3.0% / +2.2% / +5.3% |
| 4096 | 4096 | +11.2% / +2.1% / +6.7% |

At the real decode anchor sizes (1024-2048) the production mix is
flat-to-weakly-positive (+0.8 to +5.3%). The one big shadow kernel hides a
little, the tiny kernels add serialization overhead that eats most of it.

### 2c. Shadow chain = all-tiny aux (`shadow_mix` arm)

The production rope/norm/kv-store shadow is all-tiny kernels (0.5-1.5 us).

| m | auxsize | shadown | overlap |
| --- | ---: | ---: | ---: |
| 1024 | 4096 | 2 | -9.3% / -10.2% / -10.6% / -11.2% |
| 1024 | 4096 | 16 | -32.5% / -37.1% / -37.2% |
| 1024 | 262144 | 2 | -7.4% / -6.5% / -4.3% |

All-tiny shadows are strongly NEGATIVE: each tiny kernel on the waiting
channel pays the runqueue-switch penalty and contributes no overlap. More tiny
kernels makes it strictly worse (-37% at 16).

### 2d. Production direction: rejoin (no head wait, join at end)

The real decode DAG has q/k/v ready at the same instant (`nv-decode-dag-width`),
so the kv subgraph is a SIBLING of the q anchor, not a waiter. The join is at
flash_score. This is the `rejoin` / `rejoin_real` arms.

| arm | m | auxsize | overlap (3-4 runs) |
| --- | --- | ---: | ---: |
| `rejoin` (all-tiny shadow) | 1024 | 4096 | -13.5% / -17.0% / -15.7% / -17.8% |
| `rejoin_real` (1 big + tiny) | 1024 | 4096 | -5.4% / -8.7% / -7.9% / -9.4% |
| `rejoin_real` (1 big + tiny) | 2048 | 4096 | -0.1% / -0.4% / -1.9% |

The production join direction is flat-to-negative at every size. Even the
production mix (one big q6k_v plus tiny rope/norm) does not co-schedule when
the anchor is the one that waits on the shadow.

## 3. Verdict

| configuration | measured | buildable to positive overlap? |
| --- | --- | --- |
| same-size big shadow, 1024 anchor | +9.6 to +15.9% | YES but NOT the production shape |
| production mix, head-wait direction | +0.8 to +5.3% | marginal, not reliable |
| production mix, rejoin direction (real DAG) | -0.1 to -9.4% | NO |
| all-tiny aux shadow | -4 to -37% | NO |

The llama-shaped shadow pattern co-schedules on the native pair only when the
shadow kernels are the same size class as the anchor. The real decode shadow
is one 15.7 us q6k_v plus a tail of 0.5-1.5 us kernels, and the real DAG joins
at flash_score. Every faithful configuration measures flat or negative.

**Conclusion: the subgraph-partition substrate is CONSTRUCTION-BLOCKED on the
native construction at the real DAG shape. Building a partitioner would move
the same ~0 overlap the S2 generic placement already measured (2.1 us ledger,
slightly negative wall). This closes the native 1-to-1 llama-overlap route.**

## 4. What was already built (reusable if a shape ever qualifies)

The generic readiness placement (S2, `b2431266a`/`1443d8011`) is the
scheduler-side half of the substrate and stays in the tree, gated off by
`HCQ_NV_READY_PLACEMENT=0`:

- `_pick_compute_queue` with `DepsTracker.peek_access_resources` (read-only
  probe) places a node on the least-loaded GPFIFO unless it depends on the
  primary queue's current tail.
- Unit tests pin the peek semantics and the placement rule
  (`test/unit/test_hcq_nv_ready_placement.py`).
- The two bootstrap GPFIFOs (`HCQ_NUM_COMPUTE=2`) are the construction; the
  probe queues and `run_jobs` machinery prove cross-GPFIFO signals work.

What the probe proves is that the WAIT BOUNDARY is the blocker: any
cross-queue semaphore wait degrades the pair to serial interleave (pipeline
-11%, subgraph -8.3%, rejoin -13 to -18%). The only co-scheduling is dep-free
or equal-size work, which the production DAG does not provide after fusion.

## 5. What would be needed to reopen this row

1. A shadow chain made of kernels that are the same size class as the anchors
   (8-25 us). The real aux subgraph (rope 0.9 us, norms 0.5-1 us, kv-store)
   is not, and we deliberately fused the work that would be (quantize, gate/up
   fusion, reduce-output fusion). Unfusing just to co-schedule would ADD the
   node_sum back; the arithmetic already prices that at +3.7 to +11 tok/s max
   (`nv-overlap-substrate-arithmetic`), which does not reach 240.
2. A driver/runtime behavior where the waiting channel does not yield the
   runqueue. Measured channel flags already match CUDA (0/0x10); the yield
   behavior is in the driver, not our code.
3. A different construction: CUDA-stream-style co-scheduling via a different
   engine family (copy/DMA) for the tiny support kernels. The decode support
   work is compute (rope needs exp, norms need rsqrt), so this is not a clean
   fit and is not measured.

## 6. Where this leaves the 240 ledger

Overlap remains the only route past 233.8 tok/s (the non-overlap ceiling).
The native 1-to-1 route is now measured closed for the production shape. The
remaining honest paths:

- Non-overlap levers to the 233.8 ceiling (host gap parity 100.6 us, any
  residual body wins) - already scoped in prior rows.
- The CUDA-route question: our probe proves the native pair yields on any
  wait. Whether a CUDA-style graph/stream construction can express the llama
  overlap is a separate experiment on the CUDA backend, out of scope for the
  native branch.

## Evidence

- `docs/task_workflow/evidence/nv-shadow-probe-shadow_{512,1024,2048}_*.json`
- `docs/task_workflow/evidence/nv-shadow-probe-shadow_mix_1024_*.json`
- `docs/task_workflow/evidence/nv-shadow-probe-sr_1024_r1.json` (production mix, head wait)
- `docs/task_workflow/evidence/nv-shadow-probe-sr_2048_r1.json`
- `docs/task_workflow/evidence/nv-shadow-probe-rr_1024_r1.json` (production mix, rejoin)
- `docs/task_workflow/evidence/nv-shadow-probe-rr_2048_r1.json`
- `docs/task_workflow/evidence/nv-shadow-probe-rejoin_1024_r1.json`
- `scratchpad/nv_decode_shaped_overlap_probe.py`

Prior rows this supersedes: `nv-native-cosched-wait-boundary-20260817.md`,
`nv-overlap-substrate-arithmetic` (the rate estimate is now shape-gated).
