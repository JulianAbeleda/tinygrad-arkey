# NV compact-Q8 to packed-Q4 IMMA combined-chain integration

## Verdict

PASS for the isolated production-shaped research gate. The exact compact Q8
producer feeds the specialized 170-owner packed-Q4 main and deterministic
fixup without an adapter or copy. The complete chain is correct on the real
Qwen gate tensor and is about 11.1% faster than corrected FP16.

This qualifies a fail-closed research binding, not general routing.

## Interface and lifecycle

The producer writes `int8 q[M,K]`, `float32 scale[M,K/32]`, and `float32
raw_sum[M,K/32]`. The sum retains FP32 reduction, FP16 round, FP32 widen
semantics. The same buffers feed the packed main: zero conversion bytes and
zero adapter launches.

Main and fixup are distinct one-symbol cubins. Main uses grid 170, block 256,
and 58,880 bytes requested shared memory (57,856-byte layout plus the runtime
1 KiB reservation). It writes final tiles or 340 bounded FP32 partial slots.
Fixup uses grid 384, block 256 and writes contiguous FP32 `[512,12288]`.

## Correctness

Fixture: Qwen3-8B-Q4_K_M, `blk.0.ffn_gate.weight`, seed `20260617`, tokens
`(i*7)%1000`, shape `M=512,N=12288,K=4096`.

The full oracle is the scalar-qualified faithful static packed kernel over
every tile. All 6,291,456 outputs were covered with zero sentinels; both arms
are finite. Persistent main+fixup versus static:

- max absolute error `1.6689300537109375e-6`;
- mean absolute error `1.1045216297134175e-8`;
- direct-tile max `0`, split/fixup-tile max `1.6689300537109375e-6`;
- `np.allclose(rtol=2e-5, atol=2e-3)`: PASS.

An independent scalar packed oracle checks tile 0 exactly: max and mean error
zero, with output `[0,0]=0.23542046546936035` in both. The earlier large
mismatch is retracted: that reference covered one 256-column slab and left
6,160,384 outputs unwritten.

## R9 timing and pp512 translation

Hot R9 chain minimum is `506.528 us`, median `506.752 us`. Producer minimum
is `3.584 us`; main minimum is `491.456 us`, with fixup roughly `11.5 us`.
Versus corrected FP16 at `569.7 us`, recovery is `63.172 us` or `11.09%`
per projection. Gate plus up brackets about `126 us` isolated recovery before
producer sharing, scheduling, or downstream work.

## Artifacts

- harness: `extra/llm_research/prefill/nv_q4_imma_combined_chain_gate.py`;
- R9: `docs/task_workflow/evidence/nv-q4-imma-combined-chain-20260828/full-real-r9-pass.json`;
- producer evidence: `docs/task_workflow/evidence/nv-q8-compact-producer-20260828/`.

No general production routing was changed.

## Actual-model overlay checkpoint

A default-off `NV_Q4_IMMA_PP512=1` research overlay admits only Qwen3-8B,
Q4_K gate/up, exact pp512 geometry on NV. Layer-0 real gate/up versus corrected
FP16 are finite and pass `rtol=0.02, atol=0.5`; max errors are `0.04203` and
`0.04630`, respectively.

The first whole-model candidate run is finite and preserves the selected token
(`198`). Candidate and FP16 last-token logits have ranges
`[-4.62983,14.37952]` and `[-4.62205,14.36699]`; the first 16 sampled logits
differ by at most about `0.0281`. Their SHA values differ, as expected for the
Q8 activation route.

The synchronous bridge is not admissible: a candidate cold whole forward took
`219,454 ms` versus `5,789 ms` for FP16. This localizes the remaining wall to
raw-cubin graph integration: synchronizing and realizing inside the block
breaks the nested `@function` and outer TinyJit capture. The isolated chain
remains fast; a capturable/replayable native CALL substrate is required before
whole-model R7/R9 timing has meaning.

The finalized PROGRAM adapter subsequently removed the synchronous bridge and
captured exactly 72 producer, 72 main, and 72 fixup calls. With the block
`@function` bypassed only in the explicit research arm, full last-token logits
pass versus FP16 (`max_abs=0.0486650`, `mean_abs=0.0112925`, same token 198).
Synchronized warm whole-model wall is nevertheless `94.306 ms` minimum versus
`83.793 ms` for FP16: `+10.513 ms` (`+12.55%`), so promotion remains NO_GO.

Restoring the matched per-block `@function(precompile=True)` retains all 216
native calls and improves the candidate to `91.862 ms`, but corrupts the token
from 198 to 112236. This isolates the current substrate wall to native PROGRAM
output/implicit-buffer binding through function callification. The research
arm therefore keeps the correct block-function bypass pending that fix; the
flag remains default off.

The precise computed-input wall is now known: a native projection inside
`@function` is exact when its Q8 source is an explicit function parameter, but
can consume the wrong buffer when that source is computed inside the same
function. Splitting at the FFN-norm boundary restores exact dependencies and
token 198, but costs a 72-call copy totaling about `5.003 ms`.

Matched warm device profiles are `/tmp/q4cap-profile.jsonl` and
`/tmp/fp16-profile.jsonl`. The candidate's 72 mains total `42.277 ms`, Q8
producers `0.244 ms`, and fixups `1.425 ms` (`43.946 ms` lifecycle). The 72
FP16 gate/up bodies total `41.075 ms`. Thus the 72-workspace, roughly 3.6 GiB
captured arm also loses `2.871 ms` in projection lifecycle; main rises from
about 489 us isolated to 587 us in-model. One shared workspace was tested but
did not establish WAW/RAW serialization and watchdog-hung, so it was fully
reverted. Admission remains closed.

## Final v4 lifecycle result

The `both8` inner-loop unroll won a captured 72-main proxy using all 72 distinct
real gate/up packed-weight buffers and distinct per-call scratch. The proxy
improved from `39.24045 ms` (`545.006 us/main`) on v3 to `37.03472 ms`
(`514.371 us/main`) on provider v4, a `2.20572 ms` (`5.62%`) recovery. The
proxy is `extra/llm_research/prefill/nv_q4_imma_72main_proxy.py`.

Before installation, v4 passed the complete real gate-weight oracle over all
6,291,456 outputs: max/mean absolute error `1.6689300537109375e-6` /
`1.1045216297134175e-8`, direct-tile max zero, split-tile max
`1.6689300537109375e-6`, exact scalar tile 0, finite/full sentinel coverage,
and bit-exact read-only input/weight checks. Its isolated R9 chain minimum and
median are `470.112 us` and `470.976 us`; main minimum is `455.040 us`.
Evidence is `full-real-both8-r9.json` and
`full-real-both8-readonly-pass.json` in the combined-chain evidence directory.

The correctness-preserving partial redirect keeps the exact fp16 normed
activation zero-copy into the native Q8 producer while materializing residual
`h`. A fully redirected precompiled FFN tail eliminated that copy but added
143 graph calls and `1.558 ms` GPU busy time; matched R7 regressed from
`95.3 ms` to `96.9 ms`, so the tail experiment was reverted. Dependency-carried
shared scratch also remains reverted because the scheduler cannot safely
represent repeated writes to one BUFFER identity.

With provider v4, synchronized whole-model R9 improves from `95.3 ms` to
`93.2 ms` (`5495 tok/s`). The profile retains exactly 72 calls each of Q8,
main, and fixup; summed durations are `0.229536 ms`, `39.842176 ms`, and
`1.863392 ms`. Main recovers `2.441472 ms` from matched v3. Total profiled GPU
busy is `96.335648 ms`. The corrected FP16 comparator remains `83.793 ms`,
leaving `+9.407 ms` (`+11.23%`). The research flag remains default-off and
promotion is **NO_GO**.

The next measured lever is graph-safe scratch/address-footprint reduction. The
72 distinct workspaces preserve correctness but retain roughly 3.6 GiB and
raise the in-model main above its isolated wall. This needs scheduler-level
ordered repeated-write ownership (or a non-aliasing scratch scheme), not raw
buffer aliasing. Kernel-only unrolling is now substantially exhausted.
