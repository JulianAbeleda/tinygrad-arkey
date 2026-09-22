# NV decode P4 coarse two-queue cut search

Date: 2026-08-05
Status: **CPU complete; no new GPU arm authorized**

## Question

After the dependency-correct Q-support cut forecast a 171.486-us saving but
measured 9.199--10.474 us slower, are there coarser K, V/K, flash/support, or
contiguous-region cuts that still justify another native-NV wall experiment?

## Authority and exact occurrence notation

The input is the current redirect-on d512 capture
`/tmp/nv_p4_redirect_on_dag_20260805.json`: 875 nodes, 4,080 dependency edges,
147 cross-group edges, and ordered-name SHA-256
`49838b8ab2e7118d0c384fb93d2b4c3085b3732f1fe8d5abc69d51d232a6b413`.
It contains exactly 36 occurrences of
`flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128`, at global program
IDs `F_b = 16 + 24*b`, for `b = 0..35`.  A row's offset set `O` means exactly
the selected global IDs `{F_b + o | b in 0..35, o in O}`.  This is an exact,
not class-based, selector.  The graph-local index is that global ID minus its
split-group base (`0, 32, 96, 224, 480` respectively); this gives the exact
local occurrence at construction time.

The stable local structural signature, which is the identity guard for this
notation, is:

| offset from `F` | normalized program identity |
| ---: | --- |
| -11 | `q4k_g3_lanemap_gemv_4096_4096` |
| -10 | `q4k_g3_lanemap_gemv_1024_4096` |
| -9 | `q6k_gen_partial_1024_4096_4` |
| -8 | `E_4_2_8_16_4` |
| -7 | `E_2_8_16_4` |
| -6 | `r_2_8_4_4_16` |
| -5 | `r_8_16_8` |
| -4 | `E_2_8_16_4_4` |
| -3 | `E_8_2_16_4` |
| -2 | `E_16_32_4_2` |
| -1 | `r_8_8_16_2_4` |
| 0 | `flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128` |
| +1 | `flash_fused_gmax_combine_32_128` |

Thus, for example, the exact K-support candidate is `O={-8,-6,-4}`:
108 selected calls, with the three identities shown above at every one of the
36 pinned flash occurrences.  It is not an inference from an `E_` prefix.

## Enumeration method

`extra/llm_research/decode/nv_dependency_closed_cut.py`'s execution-order
simulator was used with the existing measured 0.363-us effective-wait cost.
It preserves all captured edges, one serial timeline per queue, and HCQ's
monotonically cached opposite-queue signal.  I enumerated:

- all 255 subsets of the eight non-GEMV pre-flash support positions
  `{-8,-7,-6,-5,-4,-3,-2,-1}`;
- every contiguous repeated interval with endpoints in `[-24,+4]` around each
  of the 36 pinned flash occurrences; and
- the semantic K, V/K, QKV, flash/support, and FFN-region groupings below.

The table reports only *no-contention* schedule arithmetic.  `raw edges` is a
useful boundary-pressure diagnostic, not a count of emitted waits; `waits` is
the actual cached-signal event count used for the tax.

| rank / exact `O` | selected calls | raw edges | gross overlap us | waits / tax us | static net us | conservative disposition |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1, Q support `{-7,-5,-3,-1}` | 144 | 683 | 184.992 | 73 / 26.499 | 171.486 | **measured NO-GO**: -9.199 to -10.474 us wall |
| 2, K support `{-8,-6,-4}` | 108 | 573 | 160.320 | 73 / 26.499 | 147.252 | reject: lower forecast at identical boundary count and same light-support mechanism as rank 1 |
| 3, K + join `{-8,-6,-4,-2}` | 144 | 754 | 222.144 | 109 / 39.567 | 208.713 | reject: 49% more effective handoffs than the failed Q cut |
| 4, V chain `{-9,-1}` | 72 | 326 | 391.040 | 144 / 52.272 | 353.519 | reject: includes Q6 MMQ; native heavy-GEMM probe was -0.1% overlap and has twice the failed cut's waits |
| 5, K+V `{-10,-9,-8,-6,-4,-2,-1}` | 252 | 1,256 | 447.360 | 180 / 65.340 | 397.937 | reject: Q4/Q6 MMQ plus 2.47x the failed cut's waits |
| 6, QKV linears `{-11,-10,-9}` | 108 | 1,008 | 43.360 | 108 / 39.204 | 17.224 | reject: below 50-us gate; all work is MMQ |
| 7, full pre-flash interval `[-11,-1]` | 396 | 1,709 | 0.000 | 72 / 26.136 | -26.136 | reject: no static headroom |
| 8, flash + support `[-11,+1]` | 468 | 1,802 | 0.000 | 72 / 26.136 | -26.136 | reject: no static headroom; adds flash resource pressure |
| 9, post-flash/FFN bridge `[0,+8]` | 324 | 1,758 | 0.000 | 72 / 26.136 | -26.136 | reject: no static headroom |
| 10, FFN interval `[+3,+15]` | 468 | 1,514 | 41.728 | 107 / 38.841 | 15.592 | reject: below gate and includes MMQ |

The support-only exhaustive sweep has 214 subsets above the old 50-us static
gate, but only **11** have no more than the failed Q cut's 73 effective waits.
The failed Q cut is the largest of those (171.486 us); the next best is
`{-6,-4,-2}` at 157.439 us and 73 waits.  Every static result above it uses
109--217 waits.  Therefore there is no unmeasured support-only candidate with
both less synchronization pressure and more modeled headroom than the
already-measured negative control.

## Conservative resource accounting

The CPU simulator has no byte/SM contention model, so it cannot convert its
apparent 353--398-us V/K savings into a wall prediction.  The missing term is
not speculative here:

1. On this same native construction, independent light work co-schedules
   7--10%, while two 1024 GEMMs measured -0.1% overlap.  Moving `-11:-9`, V,
   K+V, a flash region, or an FFN region deliberately places Q4/Q6 MMQ or
   flash work beside the primary decode MMQ chain; its defensible overlap
   credit is therefore zero.
2. The light-only Q support cut is the closest possible positive control:
   144 support calls, 73 effective waits, bitwise-exact full logits, and a
   171.486-us static net.  Its reverse A/B/A wall result was a 9.199--10.474-us
   regression.  This consumes the entire static credit for any equal-or-worse
   low-boundary support cut.  Treating a lower K forecast as GPU-eligible
   would be selecting on an already-falsified model.
3. The apparent V/K wins are especially non-conservative: their 326--1,256
   cross edges are only compressed to 144--180 waits by the simulator, while
   their dominant byte traffic is the same quantized GEMV traffic that the
   native heavy-pair probe showed does not co-schedule.

This is deliberately a stronger requirement than the historical `>=50 us`
static gate: after one same-topology wall falsification, a new arm must have a
positive **evidence-based lower bound**, not merely a larger no-contention
upper bound.  None does.

## Verdict

**P4 coarse two-queue repartitioning is closed for this redirect-on d512
topology.**  No policy, test, or GPU job was added.  The existing default-off
two-queue substrate and failed Q policy remain useful correctness/construction
artifacts, but a K-only, V/K, flash/support, contiguous-region, or FFN variant
cannot claim a credible >=50-us wall prospect under the measured native
contention and synchronization evidence.

Reopen only with a different primitive that changes one of the two failed
premises: (a) a construction that demonstrably co-schedules MMQ/flash work, or
(b) a fused region that removes the cross-queue boundaries rather than moving
the same byte traffic across them.  Re-running any selector in this table is
not authorized by this record.
