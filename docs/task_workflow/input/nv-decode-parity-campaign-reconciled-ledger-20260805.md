# NV decode parity campaign — reconciled implementation ledger

Date: 2026-08-05
Target: Qwen3-8B-Q4_K_M d512, RTX 5090 / driver 595.84, native `DEV=NV`
Method: reconciliation of qualified campaign records plus a final composed same-session GPU bracket

## Verdict

The fixed authority remains **5.612310 ms/token native NV** versus
**3.966140 ms/token llama.cpp**, an authority gap of **1646.170000 us/token**.
Six recoveries are now admitted, with one important interaction: the
ping-pong result is measured *on top of* the generic callify redirect, so it
must not be combined with its larger redirect-off isolated result. The M4
residual_add fold is booked as a fresh same-session section-6 delta, never
composed with the rejected Attention-O/FFN-down row.

```text
P1 JIT descriptor + reusable input-shadow                 66.6620940 us/token
P2 generic callify precompiled-output redirect            75.0307500
P5 ping-pong, incremental in redirect-on composition      91.6365625
Q4 cooperative/shared-Q8 g12, incremental composed        24.6764063
Q4 precision-budget max17 subset beyond g12, incremental  12.4620469
M4 residual_add fold, same-session section-6 delta        32.6100000
= strict booked total                                   303.077859625 us/token
fixed authority gap                                    1646.1700000 us/token
- strict booked total                                   303.077859625
= strict accounted counterfactual remainder            1343.092140375 us/token
```

The arithmetic above remains a **counterfactual causal ledger** tied to the
fixed authority; it must not be used to synthesize a current native token time
or parity ratio.  A fresh composed same-session measurement now exists
separately: steady llama A/C midpoint `4.0056768 ms/token` versus native
P1+P2+P5+Q4-g12 `5.3242440 ms/token`, or **0.75235x** llama throughput with a
**1.3185672 ms/token** gap.  That direct bracket, not subtraction from
`5.612310`, is the current absolute authority.

The older 69.1655-us diagnostic predispatch result is superseded by P1. Its
65.536-us and 28.372-us components overlap and are never added.

## Accepted recovery and interaction contract

| Item | Qualified evidence | Booking rule | Booked recovery |
| --- | --- | --- | ---: |
| P1 descriptor identity cache + reusable private input shadow | Reverse native A/B with exact full logits/tokens; OFF-A 5.607174000, ON 5.548225875, OFF-B 5.622601938 ms/token | Independent accepted change, default-on with rollback | **66.662094 us** |
| P2 generic precompiled-output/callify redirect | Exact full-logit SHA-256 `71c0a2...ae0f0`; OFF median 5.551528500, ON median 5.476497750 ms/token; 946 -> 875 programs | Independent accepted change, remains closed-default pending normal promotion | **75.030750 us** |
| P5 two-capture sampler-feedback ping-pong | Redirect-on exact tokens/full logits; control midpoint 5.466153875 and ping-pong midpoint 5.3745173125 ms/token | Incremental **only in the redirect-on composition**; retain its 91.6365625-us midpoint, do not add redirect-off ~104-us result | **91.6365625 us** |
| Q4 cooperative/shared-Q8 g12 | On top of P1/P2/P5: exact 160-token streams; d512 semantic relative L2 `8.36963e-4`; settled control midpoint 5.347317250 and candidate 5.322640844 ms/token | Incremental composed g12 result only; do not add g1/g4 or the isolated primitive result | **24.67640625 us** |
| Q4 precision-budget max17 subset (blocks 14--18) | On top of g12: exact block identities and semantics; g12 / subset / g12 settled bracket | Incremental beyond the booked g12 row only; do not add the 28.864734-us cumulative row | **12.462046875 us** |
| M4 residual_add fold (o-proj q4k GEMV epilogue, `decode-q4k-epilogue-resadd-route-policy.json`) | Section-6 gate rerun PASS 2026-08-08 (`m4-resadd-s4-gate-rerun-record-20260807.md`): same-session open vs closed d512 183.032 vs 181.946 tok/s and d2048 172.506 vs 170.982; census 912/36/36/1/36 at the gate oracle; pins 3/3 both arms; pg3 legacy sha `27857cb8ca03` unmoved | Fresh same-session section-6 delta only; do not compose with the rejected Attention-O/FFN-down row or the probe-2 ceiling (+100.1 us) | **32.61 us** |

The P5 implementation removes the 4-byte pre-graph alias-firewall copy by
alternating two captures with distinct fixed return buffers. It leaves the
generic alias firewall intact and retains the host `sampled.item()` API path.
At redirect-on it is 89.079--94.194 us/token faster in its reverse bracket.
The 91.6365625-us booking is the stated midpoint; it is not an assertion that
P2 and P5 have independent effects. In fact, P5's redirect-off benefit is
about 103--104 us, showing a roughly 10--15-us interaction with P2.
At d2048 the same redirect-on route retained exact token hashes and recovered
83.706 us/token against its A1/A2 control midpoint. That is a depth
non-regression row, not additional d512 credit. At d4096, a deterministic
resident-zero-KV gate avoided prompt-setup contamination and retained bitwise
full logits, identical token hashes, and an admitted zero-shadow capture
contract; its A1/B/A2 midpoint recovery was **91.858 us/token** (85.927--97.790
us versus the adjacent controls). This is likewise validation, not new d512
credit.

## Physical equation (location, not recovery)

```text
(1108.082 support attribution + 302.788 quant-core attribution
 - 8.111 llama internal gaps - 1.143 profile/device reconciliation)
+ 239.804933 outside-device delta + 4.749067 outer reconciliation
= 1646.170000 us/token
```

The 662.128-us fusion/dataflow/body term, 445.954-us hidden-overlap term, and
family Shapley rows assign elapsed-time ownership. They are not separately
recoverable savings. The reconciliation bridges are not optimization targets.

## Closed, rejected, and precisely blocked routes

| Work item | Current evidence | Disposition / recovery |
| --- | --- | --- |
| Ordinary `REDUCE_OUTPUT` RMSNorm wrapper | The exact chain is now traced: `REDUCE_OUTPUT -> RUNTIME_SCRATCH semantic -> reshape -> fp16 cast -> contiguous -> opaque CALL` (typed-fp16 omits the cast). The late STORE selector cannot see it; the exact typed-CALL producer attempt still produced 0 reducers / 875 programs. | **NO-GO for both exact constructions**, **0** credit. Do not broaden the predicate. |
| Real-topology shared-Q8 attention | Qwen topology is 18 Q4/Q4/Q6 plus 18 Q4/Q4/Q4. Provider/consumer ABI and progressive semantics work through g4, but real g4 reverse wall is **+18.4667 us/token** with unchanged 947 programs/partitions. | **NO-GO for current composition**, **0** credit. Synthetic group microgates are direction-only. |
| FFN-down shared-Q8 successor | A distinct 32-row cooperative W1/W3-to-Q8 producer was built; ABI, resources, 3 -> 2 topology and independent correctness pass. Included wall is 78.523325 -> 229.715160 us. | **WALL NO-GO, +151.191835 us**, **0** credit. |
| Flash single-stage d512 | Bitwise exact six-case kernel, but included reverse microgate is 64.011625 -> 146.516715 us, **+82.505090 us**. | **NO-GO**, **0** credit; resource/underfill construction is wrong. |
| Dependency-coherent two-queue cut | Exact 144/144 selection, exact logits, but reverse wall is **+9.2 to +10.5 us/token**. | **NO-GO**, **0** credit. |
| Coarse overlap cuts | CPU census includes no defensible survivor: heavy native GEMMs measured about -0.1% overlap, while candidate cuts add costly waits and compete for DRAM. | **Closed for this d512 redirect-on construction**, **0** credit. |
| Prior six-name native multi-queue support split | Correctness passes but 24--55 us slower. | **NO-GO**, **0** credit. |
| Q4 vector-carrier spelling | PTX/lane analysis: vector spelling compiles identically to current scalar FP32 recurrence. | **Closed spelling variant**, **0** credit; a real route must change representation, lane ownership, or schedule. |
| Q4 four-warp/shared-Q8 progression | Distinct runtime-loop mapping passes isolated cost and real-token g1/g4/g8/g12 semantics. Settled g12 wall is **-24.676 us/token**; g18 exact tokens/top-10 but relative L2 `1.27144e-3 > 1e-3`. | **g12 bounded PASS / g18 SEMANTIC STOP**. g12 is booked once above; no g35/default flip. |
| Q4 precision-budget blocks 19--35 | All 17 singleton additions were measured; 13 pass semantics. The nearest four-tail boundary `{23,24,27,33}` and top-ranked triple `{23,24,33}` fail fresh real-model relative L2 (`1.05305e-3` and `1.06455e-3`). Best passing singleton block 25 regresses settled incremental wall by **+12.917031 us/token**. | **TAIL EXPANSION NO-GO**, **0** new credit. The existing 17-block lease remains the maximum booked subset; additive final-logit ranking is direction-only and closed for promotion. |
| Q4 FFN subset | Exact production-shape Q8+DP4A wins the isolated included-cost gate; 16/18 singleton semantic arms pass. The first passing production singleton, layer 8, regresses settled wall by **+6.204734 us/token**. | **WALL NO-GO**, **0** credit; do not advance the predicted pair or combine it with attention-Q4 bookings. |
| Exact Q4 native four-warp / factorized substrate | Pinned live d512 llama is `mmvq.cu::mul_mat_vec_q<Q4_K,1,...>`: Q8_1 + DP4A / four warps, not MMQ/MMA. Exact four-warp is +2.868498 us; factorized follow-up is -0.207540 us (noise-scale). | **NO-GO / wall-neutral**, **0** credit; a reopen needs a third physical representation. |
| Q6 exact warp32 and integer-MMA premise | Exact warp32 is +56.363037 us; the observed live d512 path is Q8+DP4A, not an integer-MMA causal path. | **NO-GO / substrate unavailable**, **0** credit. |
| Q6 post-barrier stage | Faithful stage is +0.18535 us versus the flat Q8+DP4A control. | **NO-GO**, **0** credit; does not displace the flat primitive. |
| Q6 direct shared-Q8 consumer | g1/g4/g8/g12 semantic rows pass; exact g12 settled wall delta is **+7.00909375 us/token**. | **WALL NO-GO**, **0** credit. |
| Native concurrency construction | Two native GPFIFOs, exact dependencies, and 9.7056% light-kernel overlap pass construction. The prior decode support split remains wall-negative because queue/wait economics dominate. | **CONSTRUCTION PASS / TOKEN-SCHEDULE ECONOMICS NO-GO**, **0** credit; do not call this a generic decode-overlap recovery. |
| P2a RMSNorm boundary-free producer | Fresh production census admits zero routes (875 programs); the late selector cannot reach the real wrapper. | **NO-GO**, **0** credit. |
| P2b owned invocation-input support | Generic rule removes 35 copies (874 -> 841), but leaves 35 new per-fused-block copies and misses the <=804 topology gate. | **TOPOLOGY NO-GO**, **0** credit; second lifetime class remains unproven. |
| Scale-only RMSNorm -> Q4 gate/up | Isolated exact-scale consumer saves 6.42915 us, but block-0 full logits pass and settled model A/B/A is **+8.3694375 us/token**. | **WALL NO-GO**, **0** credit; no lease expansion. |
| Packed greedy argmax | Included primitive 71.874 -> 142.647 us. | **NO-GO**, **0** credit. |
| Attention-O custom epilogue and FFN-down residual composition | Three post-callify censuses reproduce 876 programs, 71 `E_86a2` copies (70 new), 35 fused-O calls and the correct token. FFN-down route recomputes activation. | **Topology NO-GO**, **0** credit; exact copy ownership remains unproven because the gate already failed. |
| M4 residual_add fold (o-proj epilogue variant, `decode-q4k-epilogue-resadd-route-policy.json`) | **PROMOTED 2026-08-08** after the S4 gate rerun PASS; booked +32.61 us/token in the accepted table above (see `m4-resadd-s4-gate-rerun-record-20260807.md`). Prior blocker history: the 08-06 production fold crashed the flash-decode schedule at `rangeify.cleanup_dead_axes`; that blocker was cleared by the census-fix stack (`a7944410e`, `123ea1b4c`, `c855309fc`). The combined Attention-O/FFN-down composition above stays NO-GO at 0 credit. | **LANDED**, **32.61 us** booked; the residual-slot copy elision also proves exact copy ownership for this half (copy class 1 = control only). |
| KV-store fusion / gate-up adapters / existing route swaps | KV chain already effectively one store per layer; native gate/up already fused; tested adapters regress or are wall-neutral. | **Closed exact constructions**, **0** credit. |
| CUDA substitutions | Q6 attention and Q4 FFN-down substitutions are useful causal direction only. | Cross-backend evidence, **not native recovery**. |

## Ranked next steps

Ranks use remaining credible impact, information value, and whether the next
test can rule in/out a construction without another broad route search.

| Rank | Work | Why now | Next decisive gate |
| ---: | --- | --- | --- |
| 1 | A distinct exact native Q4/Q6 DP4A substrate | Live d512 causality is MMVQ/DP4A, not MMQ/MMA. Exact four-warp, factorized Q4, exact Q6 warp32, post-barrier, attention-tail and FFN-subset variants are now bounded. | A third physical representation with an independent oracle, PTX/resources, and material included-cost win before one real family. |
| 2 | Native independently scheduled RM/HCQ work | Native construction now passes light-kernel overlap, but current decode queue/wait economics are negative. | A wait-adjusted decode forecast with a positive margin before another token schedule arm. |
| 3 | Residual sampler/vocab/RoPE/KV tails | Small remaining ownership; P1/P5 already consume descriptor and feedback parts. | A distinct topology/body mechanism with explicit P1/P5 interaction exclusion before GPU time. |

## Claims that remain unsupported

- Do not state a fresh native absolute time, parity ratio, or “current gap” by
  subtracting booked rows from fixed authority; use the final same-session
  composed bracket instead.
- Do not add P5 redirect-off timing to P2. The redirect-on P5 midpoint is the
  only composed P5 debit in this ledger.
- Do not add fusion/body and overlap ownership, CUDA substitutions, profiler
  node sums, Shapley allocations, simulated critical paths, or synthetic
  shared-Q8 microgates as recovery.
- Do not call the live d512 path MMQ/MMA: its pinned llama implementation is
  MMVQ/Q8_1+DP4A. Do not extrapolate the native light-kernel concurrency PASS
  into a decode scheduling recovery while waits remain wall-negative.
- Do not reopen RMSNorm by weakening the parent-chain predicate. The exact
  break is now located and both bounded construction attempts are closed.

## Next accounting checkpoint

The final composed same-session row is now measured: steady llama A/C midpoint
`4.0056768 ms/token` versus native P1+P2+P5+Q4-g12
`5.3242440 ms/token`, or **0.75235x** llama throughput with a fresh
**1.3185672 ms/token** gap. Using llama-bench's cold-inclusive reported average
instead gives 0.76444x and a 1.254189-ms gap; both are recorded in
`nv-decode-final-composed-same-session-record-20260805.md`.

The strict fixed-authority booked total is **303.077859625 us/token** and its
counterfactual remainder is **1343.092140375 us/token** after the M4
residual_add fold booking (+32.61 us/token, 2026-08-08). Those fixed-authority
numbers remain causal accounting; the same-session row is the current absolute
measurement. No current construction is parity-qualified.
