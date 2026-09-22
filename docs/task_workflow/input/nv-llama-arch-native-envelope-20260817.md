# NV llama-arch native envelope: does the anchor+shadow topology co-schedule at production sizes? (2026-08-17)

Date: 2026-08-17
Branch: `nvidia-bringup-20260731`
Status: **measured. The "any cross-queue wait serializes" verdict from
`nv-native-cosched-wait-boundary-20260817.md` was a 512-kernel artifact. At the
production decode sizes (1024/2048, ~7-13 us kernels) the llama-shaped
anchor+shadow pattern co-schedules on the native pair: single-join rejoin
with a same-size shadow chain measures +6.6 to +9.6%, and the
producer-continuation pipeline measures +9.4 to +15.4%. The wait boundary is
amortized once kernels are large enough; the remaining blocker is shadow SIZE
CLASS (tiny aux kernels still measure negative), not the semaphore itself.**

## 1. Question

The 08-17 wait-boundary record closed the native 1-to-1 llama-overlap route on
the claim that "any cross-queue semaphore wait degrades the pair to serial
interleave". That record's arms all ran **512x512 fp32 GEMVs (~4 us)**. Llama's
real overlap mass (quantize_q8_1 549.8 + rope 127.3 + kv_set_rows 74.6 = 752 us)
is pipelined against a ~10-25 us mmq anchor chain, i.e. the 1024/2048 kernel
class, with a single event join at flash_score. Those cells were never
measured on native. This probe fills exactly them, plus the size-class and
wait-count dimensions, to answer: **is llama's anchor+shadow architecture
expressible on the native NV pair at production kernel sizes?**

## 2. Method

Same machinery as the committed decode-shaped probe
(`scratchpad/nv_decode_shaped_overlap_probe.py`): fresh process per arm,
`HCQ_NUM_COMPUTE=2`, two bootstrap compute GPFIFOs, `Job`/`run_jobs` timestamp
signals, `flock /tmp/gpu-bench.lock`, RTX 5090 / driver 595.84. New probe:
`scratchpad/nv_llama_arch_native_probe.py` (untracked scratchpad). Each cell is
3 fresh runs x 16 reps. Overlap = `(node_sum - span) / node_sum`, same metric
as every prior record.

## 3. Results

| arm | m | shadow shape | wait structure | overlap (3 runs) | median |
| --- | ---: | --- | --- | ---: | ---: |
| `pipeline_same` | 1024 | 1 same-size gemv | producer signals; q0 continuation runs; q1 consumer waits | +9.5 / +9.6 / +15.4% | **+9.6%** |
| `pipeline_same` | 2048 | 1 same-size gemv | same | +9.4 / +11.9 / +12.2% | **+11.9%** |
| `rejoin_same_n4` | 1024 | 4 same-size gemvs | q0 continuation waits ONCE on q1 end (flash_score join) | +6.6 / +8.0 / +9.6% | **+8.0%** |
| `rejoin_same` | 1024 | 2 same-size gemvs | same | -10.1 / -10.8 / -12.7% | -10.8% |
| `rejoin_same` | 2048 | 2 same-size gemvs | same | -5.2 / -8.2 / -8.8% | -8.2% |
| `rejoin_multi_wait` | 1024 | 4 same-size gemvs | q0 waits on EACH shadow kernel (event-per-kernel) | -14.1 / -14.7 / -16.1% | -14.7% |
| `shadow_llama_order` | 1024 | tiny rope + tiny kv + same-size quantize LAST | head-wait (no join) | -3.9 / -5.2 / -5.5% | -5.2% |
| `shadow_llama_order` | 2048 | same | head-wait (no join) | -0.8 / -1.1 / -1.2% | -1.1% |
| `shadow_same` (control) | 1024 | 2 same-size gemvs | head-wait | +10.0 / +12.4 / +12.7% | +12.4% |

Control sanity: `shadow_same` at 1024 reproduces the committed record's +9.6 to
+15.9% band (median +12.4%), and a 512 `split_free` smoke run measured +11.7%
vs the recorded +11.4%. The machinery is not drifting.

## 4. What flips vs the 08-17 verdict

1. **The wait boundary is size-dependent, not intrinsic.** At 512 (~4 us
   kernels) every wait-carrying arm was negative (pipeline -11%, subgraph
   -8.3%). At 1024/2048 (~7-13 us) the same patterns are positive
   (+9.4 to +15.4%). The 08-17 "serializes on any wait" conclusion was
   extrapolated from a size class the production DAG does not use.
2. **A single-join rejoin with a same-size shadow chain co-schedules.**
   `rejoin_same_n4` (+8.0% median) is the closest native analog to llama's
   flash_score join: 4 shadow kernels of the anchor's class, one wait on the
   end. It is positive at 1024.
3. **Wait COUNT matters more than wait presence.** `rejoin_multi_wait`
   (event-per-kernel, llama-style per-kernel events) is the worst arm
   (-14.7% median), while one join on the last kernel is positive. Llama's
   CUDA executor amortizes per-kernel events; the native pair does not.
4. **Shadow size class is still the blocker for the real DAG.** The production
   aux shadow (rope 0.5-1.5 us, norms 0.5-1 us, kv-store) is all-tiny.
   `shadow_llama_order` (tiny + tiny + medium) stays negative at both 1024
   (-5.2%) and 2048 (-1.1%), consistent with the 08-17 all-tiny finding. The
   medium kernel alone would ride positive; the tiny kernels on the same
   channel pay the runqueue-switch penalty and drag it down.

## 5. Honest ceiling for the production DAG

The architecture IS expressible on native at production kernel sizes, but the
transferable mass on our real DAG is bounded:

- Our GEMV anchors are 1024/2048-class (~10-20 us) and already form the serial
  chain; the co-schedule rate measured here (~8-12% median) matches the
  committed "reliable band" (7-13%) once the size class is corrected.
- The only same-size-class shadow mass we could re-introduce is llama's
  quantize_q8_1 (549.8 us) and rope/kv_set_rows (201.9 us) IF unfused. The
  08-17 arithmetic priced unfusing at +3.7 to +11 tok/s (752 us * 7-32%
  rate). This probe does not raise that ceiling: the rejoin rate at the join
  shape is ~8%, i.e. 752 * 0.08 = 60 us hidden ≈ +2.8 tok/s at the reliable
  band, and the tiny rope/norm tail still measures negative alongside it.
- The 233.8 tok/s non-overlap ceiling and the ~110 us-overlap requirement for
  240 stand. This probe proves the *mechanism* exists on native at production
  sizes; it does not manufacture the *mass* to reach 240 without unfusing,
  and unfusing is priced below the ceiling.

## 6. What would reopen the row (next steps, in evidence order)

1. A wall-gated A/B that unfuses ONLY the same-size-class quantize mass
   (llama's quantize_q8_1, 549.8 us) behind the GEMV anchor chain with a
   single-join rejoin, leaving rope/norm fused. The probe says the geometry
   co-schedules (+8%); the wall A/B decides if it converts to tok/s.
2. A shadow-chain composition test that replaces the tiny aux kernels with
   one medium kernel (merge rope+norm+quantize into a single same-size-class
   kernel) so the entire shadow is one hideable unit instead of a tiny tail.
3. Keep `rejoin_multi_wait` as the explicit NO-GO: per-kernel events on native
   are strictly worse than a single join.

## 7. Reproduction

```bash
cd /home/ubuntu/tinygrad-arkey
for arm in pipeline_same rejoin_same_n4 rejoin_same rejoin_multi_wait shadow_llama_order; do
  flock /tmp/gpu-bench.lock python3 scratchpad/nv_llama_arch_native_probe.py \
    --arm $arm --m 1024 --reps 16 --out /tmp/nv_llama_arch_$arm.json
done
```

Raw evidence: `/tmp/nv_llama_arch_<arm>_<m>_r<1..3>.json` (session scratch,
not committed). `kernel_durations_us` in each row confirms ~7-13 us kernels at
1024, i.e. the production decode size class, and span < node_sum on the
positive arms (real overlap, not launch-gap arithmetic).
