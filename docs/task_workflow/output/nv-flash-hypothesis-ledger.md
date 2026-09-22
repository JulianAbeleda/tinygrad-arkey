# Flash hypothesis ledger

## Rule of evidence

No mechanism is called a lever from a cross-runtime difference alone.

- **Observed**: a matched counter/timing difference exists.
- **Causal primitive**: changing only the mechanism moves the intended counter
  and improves the isolated hot and cold body.
- **Graph converted**: the 36-layer production score/island service falls by
  the primitive amount without losing the saving to overlap.
- **Booked**: semantics pass and a reverse token bracket beats both controls.
- **Closed**: the intended mechanism was actually changed and failed its gate.
- **Open**: the observation exists, but no clean causal construction has
  changed only that mechanism yet.

The recoverable pools are not additive across hypotheses:

| measured pool | size |
|---|---:|
| hot score-body debt | 23.904--26.208 us/token |
| excess hot-to-production conversion | 33.500--35.804 us/token |
| combine debt | 13.215 us/token |
| complete Flash debt | 72.923 us/token |

## Facts that theories must explain

At matched S6 horizon, tinygrad and llama move approximately the same DRAM and
L2 bytes. Tinygrad executes fewer total instructions and uses fewer registers,
but requests more global sectors, expands shared loads into many more data-pipe
wavefronts, and has more long-scoreboard stalls.

| matched S6 fact | tinygrad | llama |
|---|---:|---:|
| DRAM reads | 3.183 MB | 3.163 MB |
| L2 traffic | 13.178 MB | 13.174 MB |
| dynamic instructions | 801,600 | 850,944 |
| registers/thread | 56 | 162 |
| achieved occupancy | 9.14% | 9.47% |
| all global-load instructions | 36,864 | 34,560 |
| all global-load sectors | 491,520 | 417,792 |
| all sectors/global request | 13.33 | 15.11 |
| K/V global-load sectors | 393,216 | 393,216 |
| excessive K/V sectors | 0 | 0 |
| Q global-load instructions | 12,288 | 3,072 |
| Q global-load sectors | 98,304 | 24,576 |
| shared-load instructions | 13,824 | 19,968 |
| shared-load wavefronts | 101,383 | 19,968 |
| long-scoreboard stalls | 65.84% | 58.98% |

Production state is cold in both runtimes. Llama reads about 3.16 MB from DRAM
at roughly a 75.5% L2 read-hit rate. The question is therefore not whether
llama avoids cold K/V, but how it services the same cold state faster.

## Complete theory matrix

| ID | exact theory | evidence | status | maximum pool it could explain | decisive test |
|---|---|---|---|---:|---|
| H1 | Tinygrad's S8 physical horizon performs avoidable empty work versus llama S6 | S6 removes about 25% of bytes/instructions; automatic dual capture now survives the S6→S8 crossing | **booked** | installed crossing recovery 40.224 us/token | Keep S6/S8 selection inseparable from one-time capture of both ping-pong pairs |
| H2 | Tinygrad reads more compulsory DRAM/L2 bytes even at equal horizon | matched S6 DRAM/L2 traffic is equal | **rejected** | 0 | Reopen only if a new matched counter contradicts the retained result |
| H3 | Global K/V requests are less productive/coalesced | PC decomposition shows K/V are identical and ideal; vector Q changes the intended requests | **rejected for K/V; Q translation closed** | 0 for tested vector-Q construction | Matched explicit-geometry A/C/A regresses 69.752 us/token despite exact tokens |
| H4 | K→score→V dependency order exposes too little memory-level parallelism | Removing llama's `minBlocks=1` contract changes 158→96 registers, expands its K/V 16-load spans from 32/40 to 279/266 instructions, and slows cold NCU 6.144→7.808 us at equal sectors. Adding the contract to tinygrad changes 56→96 registers, compacts K from 226→29 instructions, and improves matched S6 cold service 6.400→5.888 us. Rotating the first graph cap 32→33 removes the conversion seam | **causal and booked as paired compiler/capture policy** | installed A/C/A midpoint recovers 15.954 us/token, +0.953 tok/s in-bracket | Keep launch bounds and cap 33 inseparable and capture-scoped. Forced PTX V remains closed; other graph topologies and MoE require independent admission |
| H5 | Final PV shared-memory layout causes bank replay | 7.33 versus 1.00 shared wavefronts/instruction, but PC sampling assigns zero long-scoreboard samples to either final shared tail | **causal counter; rejected as primary cold-service explanation** | a bounded hot/tail subpool only | Static/vectorized conflict-free exchange with unchanged shared-load and total instruction counts; then hot/cold A/B |
| H6 | A simple transpose fixes H5 | transpose halves wavefronts but raises shared-load instructions 13,824→50,688 and regresses about 14% | **closed** | 0 for tested spelling | Do not retry dynamic scalar indexing; preserve compile-time/vector expansion |
| H7 | Fast transcendental/divide math is the main gap | removes 5.65% instructions but leaves bytes; graph recovers about 5 us/token | **real, small, wall-resolution/quality blocked** | 4.98--5.50 us | Quality contract plus lower-noise endpoint measurement |
| H8 | Total instruction count is the main equal-horizon gap | tinygrad S6 executes fewer instructions than llama | **rejected as a scalar explanation** | 0 | Instruction ordering/dependencies remain H4; total count alone is closed |
| H9 | Low occupancy or excessive registers limits tinygrad | occupancy is matched; H4 proves the opposite register failure mode: tinygrad under-spends registers needed for load ILP | **rejected as occupancy/excess-register theory; absorbed into H4** | 0 independently | Do not reduce registers merely to raise theoretical occupancy; judge register headroom through H4 service time and spills |
| H10 | Llama wins because it preserves hot K/V | direct llama production NCU is cold and matches the full-prefix replay | **rejected** | 0 | None |
| H11 | Producer topology/cache displacement creates tinygrad's extra conversion penalty | full history plus Q/K/V fork/join reproduces production latency; reheat removes it | **causal diagnostic, no useful construction** | 33.5--35.8 us | Protect K/V or change line admission without slowing producer overlap; require both target DRAM/time and token wall to fall |
| H12 | Blanket evict-first streaming of dense weights protects K/V | Q/K/V-only recovers ~6 us; whole-dense policy regresses ~24 us | **closed at blanket granularity** | 0 for tested policy | Reopen only with line/reuse-aware admission, not the same blanket rule |
| H13 | Serial prefetch/reheat is a useful solution | reheat proves residency cause but repeats the same reads and has no admissible token accounting | **diagnostic only** | 0 | A prefetch must overlap otherwise idle service and beat token wall |
| H14 | Separate K and V allocations improve address service | exact candidate was slower | **closed** | 0 | None without a different access grammar |
| H15 | Gating invalid upper-horizon K/V loads fixes cold service | faster in some hot cases, slower cold; no-go | **closed** | 0 | Horizon removal is H1, not another gate spelling |
| H16 | Read-only/constant load spelling is llama's advantage | all 32 vector sites changed to CONSTANT; resources same; cold candidate failed | **closed** | 0 | None |
| H17 | Base-address color or combined-cache stride is the main gap | colors move timings but no portable winner; separate allocation loses | **closed at tested layouts** | 0 | Reopen only with a deterministic mapping from address bits to service |
| H18 | Compiler loop unrolling alone exposes llama-like parallelism | pragma produced byte-identical machine code | **test was non-causal; theory remains under H4** | not independently claimable | An explicit schedule must change SASS/load dependency distance, not just source text |
| H19 | One-head/four-warp vector topology and wide loads are missing | already installed; prior promotion recovered 81.468 us/token | **booked/closed by promotion** | already realized | No further claim from renaming the same topology |
| H20 | More CTA partitions improve equal-horizon service | llama S8 remains as cold-insensitive as S6; tinygrad S7 regresses and S8 adds empty work | **closed for tested geometries** | 0 | Reopen only with a new occupancy/wave calculation and matched horizon |
| H21 | PDL or producer-ready launch-ahead explains llama Flash | PDL-off llama remains fast; tinygrad legal launch-ahead changes ~1.5 us and fails wall | **closed** | 0 | None for the same boundary |
| H22 | Score→combine or combine→O scheduling gaps are the loss | 35/36 score→combine pairs are same-group with zero gap; one score ends a graph batch and combine begins the next. Rotating the initial cap 32→33 preserves the H4 saving and passes the installed reverse wall | **ordinary gaps rejected; boundary conversion sink solved for admitted graph** | accounted under H4, not additive | Closed by the paired capture policy for this graph; reopen independently for other token topologies |
| H23 | Combine ownership recovers combine debt | width-only is closed; register-broadcast weights pass and are booked on installed S8, but fail installed S6 composition | **booked S8; S6 conversion-closed** | S8 installed recovery 9.396 us/token | Do not infer S6 from explicit geometry; distinct S6/S8 capture is required |
| H24 | Fusing score and combine removes partial traffic/boundary | tested single-stage mapping was resource/underfill negative | **closed for tested mapping, algorithmically open** | at most 13.215 us plus partial traffic | New construction must retain score occupancy and remove global partial ABI before production work |
| H25 | Six versus eight partials explains all combine debt | S6 helps fixed windows, but selector fails; llama ABI also differs | **partly H1, not independent** | bounded by 13.215 us | Equal-S6 combine/ABI A/B with identical score and token validity |
| H26 | fp16 versus fp32 output or split metadata is the main issue | structural difference observed; no isolated equal-cardinality ABI A/B | **open, low priority** | combine pool only | Matched six-part ABI microgate holding combine arithmetic/order fixed |
| H27 | Math-pipe saturation limits the cold body | math-pipe throttle is below 1%; fast math converts weakly cold | **rejected** | 0 beyond H7 | None |
| H28 | More predecessor history creates a deeper hidden state | two-history replay is no worse; exact fork/join closes production state | **rejected** | 0 | None |
| H29 | Cross-runtime timing instrumentation creates the apparent gap | hot/production deltas reproduce within each runtime and token/device ledgers close | **rejected as full explanation** | 0 | Do not compare raw NCU duration across runtimes; use counters and native authorities |
| H30 | Normalized grouped-Q ownership is the missing wide-Flash lever | QG1/S6, QG2/S12 and QG4/S24 hold 192 CTAs/768 warps/equal head-token work | **topology-only closed** | 0 for current non-sharing spelling | QG2 is exact but slower; QG4 is ~0.34% faster but non-exact. Reopen only with actual cross-head K/V sharing |
| H31 | Actual cross-head K/V sharing converts GQA reuse | QG2 stages one common K/V tile for two query heads and halves global-loading warps at fixed 192 CTAs/768 warps; result is bit-exact | **closed for shared-memory staging construction** | 0 | Device score service regresses 3.520→4.224 us/layer; barriers and shared traffic cost more than duplicate cache requests |
| H32 | Reuse-class FFN-down admission protects K/V without slowing the producer | ql/qh payload uses an evict-first alias while scale/d metadata stays ordinary; exact entry penalty falls +0.608→-0.016 us/layer | **mechanism passes; token construction closed** | diagnostic 22.5 us/token, zero booked | Reverse wall regresses 69.877 us/token / 4.151 tok/s. Reopen only with per-access cache semantics that preserve the original ABI and production service |

## Surviving candidate levers

There are now two proven body levers and three unbooked mechanisms:

1. **Vector float32 Q loads (H3 decomposition)** — causal request change but
   token-wall closed at +69.752 us/token.
2. **H4: compiler register headroom and dependency order** — causal at matched
   S6 in both directions and now booked through the paired cap-33 capture
   construction. The installed midpoint recovers 15.954 us/token.
3. **H5: conflict-free static PV exchange** — bank behavior is causal, but the
   first implementation changed instruction grammar and failed; PC attribution
   shows this is not the primary cold long-scoreboard source.
4. **H11: useful K/V protection/admission** — residency cause proven; blanket
   policy closed; line/reuse-aware construction absent.
5. **H24/H26: different combine boundary/ABI** — limited to roughly 13 us.

Fast math H7 is a separate roughly 5-us body/graph win with numerical and wall
resolution requirements. It cannot be the main Flash lever.

## Test order

The next tests should maximize information before implementation:

1. **Completed:** PC attribution assigns the long-scoreboard debt to global
   load consumers, not the final shared tail.
2. **Completed:** request decomposition rejects K/V coalescing as the aggregate
   difference and proves a small vector-Q primitive.
3. **Completed negative:** ordinary and forced 2-/4-column batching does not
   reproduce llama's register/dependency schedule and regresses.
4. Build a static/vectorized PV-exchange microgate with identical arithmetic,
   shared-load instruction count, and output. This directly adjudicates H5.
5. **Completed positive:** two-argument launch bounds reproduce llama's K
   cadence and cross its matched score service; llama ablation reverses both
   schedule and timing. Forced V preload is completed negative.
6. **Completed negative control:** launch bounds alone lose 16.978 us; profile
   accounting locates the sink at graph pacing.
7. **Completed positive conversion:** cap-33 reverse bracket and final installed
   reverse bracket both pass; promote only the paired capture contract.
8. If H4 does not close the cold conversion pool, test one line/reuse-aware
   producer admission rule under the exact fork/join history for H11.
9. Only a passing primitive advances to S6 and S8 production profiles, then
   numerical qualification and reverse token wall.

H4 and the active-horizon selector have passed causal and production
conversion gates.  The paired compiler/capture policy and automatic S6/S8
selection are booked.  Forced V scheduling, vector Q, topology-only grouped
ownership, and ordinary PDL are closed.  Real cross-head K/V sharing and a
line/reuse-aware producer policy remain new constructions, not unmeasured
settings of the current emitter.
