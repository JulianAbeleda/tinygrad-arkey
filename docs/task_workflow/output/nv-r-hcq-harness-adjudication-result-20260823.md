# NV R-residual HCQ harness adjudication

Date: 2026-08-23  
Branch: `nvidia-bringup-20260731`  
HEAD: `6570abc025514273faa100c66b979e531585a1e1`  
GPU: RTX 5090, sm_120

## Findings first

1. **[MEASURED] The reported Xid/block-0 blocker is not caused by HCQ
   timestamps or by `active_qmd`.** The original flush reproduced Xid 13 on
   the supported one-kernel `NVProgram(..., wait=True)` path with
   `grid=(1,1,1)`, where the probe performs no manual QMD reset and builds no
   multi-kernel chain. A direct native-HCQ readback then showed that an NVRTC
   kernel launched with `block=(256,1,1)` observes `blockDim.x == 0` in all
   four launched CTAs. The probe kernels used `blockIdx.x * blockDim.x`; all
   CTAs therefore addressed block 0. When the same value participated in the
   grid-stride increment, the increment was zero and the kernel did not
   terminate.

2. **[MEASURED] Baked launch geometry fixes the blocker without changing the
   production runtime.** A measurement kernel using
   `blockIdx.x * 256 + threadIdx.x` and the constant stride
   `4096 * 256` wrote all 128 MiB in 58.976 us. Ten spread words, including
   indices 256 and 33,554,431, matched a bit-exact integer pattern. The
   standalone flush verifier now returns `FLUSH_OK` in a fresh process.

3. **[MEASURED] The prior rotation-control interpretation is invalid.** The
   16 MiB streaming control and the grid-128 pointer chase both used dynamic
   `blockDim.x`. Their CTAs aliased the block-0 address range. Consequently,
   the reported 16 MiB/launch bandwidth, full-32-MiB chase coverage, and
   high-MLP-vs-low-MLP explanation were not measurements of the stated byte
   sets. **[UNMEASURED]** The exact instruction-level trigger for Xid 13 is
   still not isolated; the same faulty dynamic-geometry construction also
   produced an Xid 109 timeout in an integer variant. This uncertainty does
   not block the constant-geometry measurement route.

4. **[MEASURED] A valid cache positive control now separates in both bracket
   orders.** Across six fresh target processes, the corrected 16 MiB read
   measured `cold-hot = +5.456..+5.632 us` in H/C/H and
   `+5.488..+5.712 us` in C/H/C. A matched hot-predecessor control measured
   the predecessor-QMD timestamp path at `-0.288 us` relative to an explicit
   start timestamp, so the approximately 5.6 us separation is not a timestamp
   handoff artifact.

5. **[MEASURED] All six exact retained R cubins are cache-sensitive.** N=16
   per arm, first four discarded, one fresh process per row, clocks locked,
   reverse H/C/H and C/H/C. Each cubin SHA in the artifact matches the file
   launched. Values required by the cooperative kernels are now passed to
   `fill_kernargs`; the prior probes omitted them.

   | row | count | P us | R=P-C us | hot us H/C/H | cold us H/C/H | delta H/C/H | delta C/H/C |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | q_coop_4096 | 17 | 8.416 | 3.107 | 5.984 | 12.512 | +6.528 | +6.128 |
   | q_g3_4096 | 19 | 8.704 | 0.953 | 8.496 | 15.136 | +6.640 | +6.544 |
   | o_epi_4096 | 36 | 9.184 | 1.486 | 8.672 | 15.552 | +6.880 | +6.688 |
   | flash_score | 36 | 6.272 | 2.614 | 4.816 | 7.360 | +2.544 | +2.624 |
   | kv_coop_1024 | 26 | 3.712 | 1.402 | 2.976 | 5.872 | +2.896 | +2.816 |
   | kv_g3_1024 | 28 | 4.768 | 0.970 | 4.320 | 6.576 | +2.256 | +2.496 |

6. **[INFERRED] Cache state is again a viable mechanism for R, but is not yet
   its measured partition.** Every row's fully-cold sensitivity is larger
   than its per-call `R`, and every production `P` lies between the corrected
   hot and fully-cold endpoints. The weighted fully-cold sensitivity is
   710.856 us/token, versus 282.14 us/token of R for these rows. The 710.856 us
   is therefore a stress ceiling, not recoverable wall and not a booked
   explanation of R. **[UNMEASURED]** The actual cache state established by
   each installed production predecessor remains the missing observable.

7. **[MEASURED—source] Two additional probe-construction defects could make
   the old fault path unstable.** The old cold arm reused one `flush_args`
   allocation for every in-flight flush, and the decisive probe repeated the
   same `chase_state`/`stream_state` QMD. A QMD resides in that mutable kernarg
   allocation. The corrected measurement path uses a unique args/QMD state
   for every dispatch, one queue/submission per sample, and never mutates
   `q.active_qmd`. **[INFERRED]** QMD aliasing can explain intermittent
   instability, but it is not needed to explain the deterministic block-0
   readback.

## Corrected disposition

- **[MEASURED]** Native-HCQ cache sensitivity: `PASS`.
- **[MEASURED]** Full 128 MiB eviction write and positive control: `PASS`.
- **[UNMEASURED]** Production cache share of `R`: still open.
- **[UNMEASURED]** C0-C3 predecessor-conditioned partition: not yet run.
- **[UNMEASURED]** K/V two-queue producer-to-join recovery: not yet run.
- **[UNMEASURED]** 240 tok/s buildability: unchanged; no wall recovery is
  booked from this probe.

The previous statement that cache state was unavailable because timestamp/QMD
chain stability required a production-runtime patch is refuted. The remaining
work is a live-buffer measurement problem, not a native-HCQ timestamp blocker.

## Exact next sequence

1. **[MEASUREMENT REQUIREMENT]** During one real decode process, use an
   analysis-only hook to retain the selected layer's live program objects,
   argument states, buffers, values, grid/block tuples, and cubin hashes. Run
   the experiments before the process exits; retained virtual addresses from
   an old capture are not reusable in a fresh process.
2. **[MEASUREMENT REQUIREMENT]** Clone the selected buffers so replay cannot
   mutate model state. Validate each replayed target's output checksum against
   its captured production output.
3. **[MEASUREMENT REQUIREMENT]** Measure nested target intervals, one
   submission per sample, with unique QMDs and no private queue-state writes:
   C0 isolated hot target; C1 hot target after a matched target predecessor;
   C2 target after its exact immediate production predecessor; C3 target after
   the full local production prefix; P installed command interval.
4. **[MEASUREMENT REQUIREMENT]** Partition in one clock domain:
   `P-C0 = (C1-C0) + (C2-C1) + (C3-C2) + (P-C3)`. Require zero residual within
   timestamp resolution before naming cache, predecessor interference, or
   admission as the production mechanism.
5. **[MEASUREMENT REQUIREMENT]** Build the K/V microgate from the same live
   cloned layer. Compare serialized Q-then-K/V against two-queue Q || K/V over
   the complete provider-to-join span. Use queue signals for the join and no
   per-kernel timestamps in the primary span gate. Check output hashes before
   interpreting timing.
6. **[PROMOTION GATE]** Only a token-SHA-matched reverse wall bracket can book
   recovery or change the 240 verdict.

## Evidence

- Raw evidence and manifest:
  `docs/task_workflow/evidence/nv-r-hcq-harness-adjudication-20260823/`
- Full-write proof: `flush-check.txt`
- Dynamic block-dimension proof: `blockdim-readback.log`
- Timestamp start-path control: `start-path-control.log`
- Reverse-bracket rows: `q_coop_4096.json`, `q_g3_4096.json`,
  `o_epi_4096.json`, `flash_score.json`, `kv_coop_1024.json`,
  `kv_g3_1024.json`
- Hash manifest: `sha256.txt`

Only measurement scripts under `extra/llm_research/decode/` were changed. No
production model, renderer, scheduler, graph, or runtime file was changed.
