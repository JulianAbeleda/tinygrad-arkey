# NV 240 audit - reconciled gap ledger (2026-08-15)

Date: 2026-08-15
Branch: `nvidia-bringup-20260731` (HEAD `df3dca075`)
Status: **audit record. Read-only: no runtime change, no GPU session, no lock.**
Reconciles the four competing gap decompositions committed 08-12 through 08-15
into one ledger, re-anchors at the post-Q4-fp16 production point, and names the
two remaining paths to 240 tok/s with their measured status. Supersedes the
"220 composition" claim where it relied on an unverified census-to-wall map.

## 1. Current position (re-anchored)

| side | tok/s | ms/token | source |
| --- | ---: | ---: | --- |
| tinygrad production (post Q4 fp16 geometry) | 198.86 census / ~200.3 bracket | 5.029 / 4.993 | `/tmp/census_prod_promoted_20260815.json`, `/tmp/q4k_fp16_geom_timing_20260815.json` |
| llama same-session | 245.45 pair / 247.98 fresh | 4.074 / 4.037 | `nv-decode-gap-attribution-same-session-20260812.md` |
| 240 target | 240 | 4.167 | |

Wall gap to 240 = ~830-870 us. Wall gap to llama = ~930-1000 us. The Q4
FFN-down four-warp fp16 promotion moved production from 193.49 -> 198.86 census
(-100.3 us reverse bracket), so that shape is no longer a lever.

## 2. The four competing views reconciled

| view | what it priced | measured disposition |
| --- | --- | --- |
| 08-12 gap attribution | +652 us class deltas, "220 at 1:1" ceiling | kernel-work view; the 1:1 ceiling is unverified |
| 08-14 gap correction | +936 us overlap mass is the dominant term | correct on llama's side; overlap is real there |
| 08-15 Route B wall | multi-stream overlap is FLAT | correct at `319241408` (1021 kernels); stale at HEAD - width is 4 now |
| 08-15 composition review | body-free fusion folds map ~0 wall | correct: "fusion to 220" is falsified |

Resolution: the gap is three independent axes, and the four docs each price a
different subset.

| axis | llama | tinygrad | delta |
| --- | ---: | ---: | ---: |
| kernel work (node sum) | 4766 us | ~5164 us (post-fp16 ~5064) | ~+300-400 us |
| overlap mass (node sum - union) | 936 us | 0 us | -936 us |
| internal launch gaps | ~10 us | ~211 us | +211 us |

Views 1 and 4 price the kernel-work axis and show fusion caps ~207-222 tok/s.
Views 2 and 3 price the overlap axis and show the overlap lever is real in
llama. The width-1 justification for our side was stale: see section 6.

## 3. The overlap axis is topology, not a knob

llama runs up to 4 concurrent kernels: it hides quantize (391 us), rms_norm
(156 us), rope (33 us), and flash (143 us) behind its long `mul_mat_vec_q`
anchor. tinygrad runs 1 concurrent kernel at wall time: `overlap_mass_us = 0.0`.
The 08-15 Route B stream-distribution probe found 1020 of 1021 graph calls on
stream 0, but that was at `319241408`; at HEAD the same scheduler distributes
258 of 668 calls off stream 0 (see `nv-decode-dag-width-verdict-20260815.md`).

Root cause of the stall: fusion traded overlap for kernel count. tinygrad is
ahead of llama on norm/rope/kv because it fuses them into GEMV epilogues, but
that same fusion serializes the support into the critical path and removes the
independent kernels llama overlaps. The 08-15 composition review then measured
that the fused bodies are launch-cost-bound, so re-fusing more support maps to
~0 wall. llama captures the same support mass by hiding it, not by removing it.

Correction (2026-08-15, HEAD `c74567a24`): the decode DAG is not width 1. It
has width 4 in the attention head (q/k/v GEMVs plus one support eltwise are
simultaneously ready) and a 425-442 kernel critical path out of 668. The open
question is no longer "is there parallelism to distribute"; it is whether that
width converts to wall, given q/k/v are all memory-bound and contend for the
same HBM bandwidth.

## 4. Open kernel-work levers (the second axis)

Current production census class sums (`/tmp/census_prod_promoted_20260815.json`):

| lever | measured (us) | llama (us) | excess | status |
| --- | ---: | ---: | ---: | --- |
| Q6 GEMV core (V partial + FFN-down) | ~820 | ~566 | ~+240 | open; Q6 substrate unbuilt; Q4 analog proven |
| reduce-output epilogue (q/k + ffn-down) | 378 | 0 (in-kernel) | +378 | body-free fold measured FLAT |
| input norm r_16_256 + E_32_32_4 | 238 | 0 (in-kernel/overlapped) | +238 | body-free FLAT, body-adding NO-GO |
| flash score | 242 | 114 | +128 | structural; floor ~90 us; tile sweep NO-GO |
| vocab aux | 52 | 0 | +52 | single-pass max unbuilt |

Q6 detail from the current census: 10 slow V blocks at 17.92 us/node vs llama
4.90 us/node (3.65x), and 18 FFN-down blocks at 34.92 vs 28.75 (1.22x). The
other 8 V blocks are already at/below llama (4.22 us). The Q4 four-warp fp16
promotion proved this exact mechanism (geometry fix on the anchor) is worth
~100 us, so Q6 is the highest-confidence remaining kernel-work lever.

These excesses must not be summed at 1:1. llama's norm/flash/reduce rows are
partly hidden behind its anchor, and the census harness measures kernel-sum,
not wall. The realistic capture is smaller: Q6 ~240 us (if it lands like Q4),
flash ~90 us (structural floor), plus ~50 us partial reduce/vocab. That is
~380-500 us -> ~215-222 tok/s.

## 5. Two paths to 240

**Path A, kernel work.** Close the Q6 GEMV core (~240 us) with the same
four-warp geometry / folded-quant substrate that worked for Q4, plus the flash
score floor (~90 us), plus partial support removal (~50 us). Lands ~215-222
tok/s. Sufficient to beat 200 and reach the low-220s; NOT sufficient for 240.

**Path B, anchor shadow.** Recover llama's overlap by giving each layer one long
GEMV anchor with the next-layer support (norm/reduce/rope/vocab) co-scheduled
behind it. This is the only mechanism that reaches 240+. Its two sub-routes are
both blocked today:

- native multi-compute channel (Route A): `CONSTRUCTION_BLOCKED` (driver-private
  native-channel activation gap);
- CUDA multi-stream graph (Route B): FLAT at `319241408` (width-1 graph); the
  width-4 graph at HEAD has not been wall-tested. The scheduler now distributes
  39% of calls, so the FLAT conclusion no longer transfers.

The remaining ~650 us of support work is the target: fusion has been measured to
capture ~0 of it, and overlap is the only measured mechanism that does capture
it (llama proves it). So 240 = Q6 core (~240) + flash floor (~90) + ~500 us of
that support mass captured via anchor shadow.

## 6. The question that decides 240 (corrected 2026-08-15)

The width-1 hypothesis is falsified at HEAD: the decode DAG has intrinsic width
4 (q/k/v siblings) and `plan_multi_stream` distributes 39% of calls. See
`nv-decode-dag-width-verdict-20260815.md`. The remaining question is whether
that width converts to wall:

1. q, k, and v are all memory-bound GEMVs; concurrency may saturate HBM and buy
   ~0 wall even though the scheduler distributes them.
2. Production NV uses `HCQGraph`, whose multi-queue path is native (Route A)
   and opt-in via `HCQ_NUM_COMPUTE` + `HCQ_NV_MULTI_QUEUE_*`, not
   `CUDA_GRAPH_STREAMS`. That path still needs a wall test at HEAD.

If neither converts, the width-4 parallelism is bandwidth-bound and the honest
240 lever returns to kernel work (Q6 GEMV core + flash score floor), not
anchor shadow.

## 7. Next measured steps (no implementation until these clear)

1. Fresh wall census at HEAD `df3dca075` to confirm the JIT capture fix is
   perf-neutral on decode replay (capture-only change, replay should prune).
2. Q6 GEMV core: replicate the Q4 four-warp geometry fix on Q6 V (3.65x) and
   Q6 FFN-down (1.22x). Mechanism already proven; highest-confidence lever.
3. Re-run the Route B wall A/B at HEAD (`CUDA_GRAPH_STREAMS` 1..4) to test
   whether width 4 alone converts to wall.
4. Re-test NV native multi-queue at HEAD (`HCQ_NUM_COMPUTE=2` plus a pinned
   support tail) and measure decode wall with token-identity against serial.
5. Do not re-litigate: F3 norm folds (FLAT/NO-GO), DP4A Q4 down (NO-GO +45),
   the pre-HEAD multi-stream overlap result (FLAT on a width-1 graph), vocab
   two-program chain (NO-GO), the "220 at 1:1" composition (falsified).

## Evidence

- `/tmp/census_prod_promoted_20260815.json` (current production census)
- `/tmp/q4k_fp16_geom_timing_20260815.json` (Q4 promotion reverse bracket)
- `nv-q4-down-fp16-geometry-promotion-20260815.md`
- `nv-overlap-route-b-wall-outcome-20260815.md` (Route B FLAT, width 1)
- `nv-decode-dag-width-verdict-20260815.md` (width-1 falsified at HEAD: width 4)
- `nv-220-composition-review-outcome-20260815.md` (fusion-to-220 falsified)
- `nv-gap-audit-correction-20260814.md` (overlap axis, +936 us)
- `nv-gemv-core-recovery-status-20260813.md` (Q6 core deficit, DP4A successor)
- `nv-decode-gap-attribution-same-session-20260812.md` (same-session pair)
- `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
- `docs/task_workflow/evidence/nv-tinygrad-d512-node-ledger-20260813.json`
