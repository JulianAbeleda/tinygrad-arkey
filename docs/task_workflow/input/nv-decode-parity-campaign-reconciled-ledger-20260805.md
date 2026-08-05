# NV decode parity campaign — reconciled implementation ledger

Date: 2026-08-05
Target: Qwen3-8B-Q4_K_M d512, RTX 5090 / driver 595.84, native `DEV=NV`
Method: reconciliation of qualified campaign records plus a final composed same-session GPU bracket

## Verdict

The fixed authority remains **5.612310 ms/token native NV** versus
**3.966140 ms/token llama.cpp**, an authority gap of **1646.170000 us/token**.
Four recoveries are now admitted, with one important interaction: the
ping-pong result is measured *on top of* the generic callify redirect, so it
must not be combined with its larger redirect-off isolated result.

```text
fixed authority gap                                  1646.1700000 us/token
- P1 JIT descriptor + reusable input-shadow              66.6620940
- P2 generic callify precompiled-output redirect          75.0307500
- P5 ping-pong, incremental in redirect-on composition    91.6365625
- Q4 cooperative/shared-Q8 g12, incremental composed      24.6764063
= strict accounted counterfactual remainder             1388.1641872 us/token
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
| Q6 flat four-warp Q8+DP4A partial | Included native Gate 1 is 66.74259 -> 66.92764 us, **+0.18505 us**. | **NO-GO for exact construction**, **0** credit. |
| Packed greedy argmax | Included primitive 71.874 -> 142.647 us. | **NO-GO**, **0** credit. |
| Attention-O custom epilogue and FFN-down residual composition | Three post-callify censuses reproduce 876 programs, 71 `E_86a2` copies (70 new), 35 fused-O calls and the correct token. FFN-down route recomputes activation. | **Topology NO-GO**, **0** credit; exact copy ownership remains unproven because the gate already failed. |
| KV-store fusion / gate-up adapters / existing route swaps | KV chain already effectively one store per layer; native gate/up already fused; tested adapters regress or are wall-neutral. | **Closed exact constructions**, **0** credit. |
| CUDA substitutions | Q6 attention and Q4 FFN-down substitutions are useful causal direction only. | Cross-backend evidence, **not native recovery**. |

## Ranked next steps

Ranks use remaining credible impact, information value, and whether the next
test can rule in/out a construction without another broad route search.

| Rank | Work | Why now | Next decisive gate |
| ---: | --- | --- | --- |
| 1 | A distinct exact native Q4/Q6 GEMV substrate | Q6 flat-four-warp, Q4 vector spelling, and Q8-cooperative expansion are now bounded. The remaining parity-scale quant direction must avoid g12's weak marginal scaling and g18's approximation stop while changing physical ownership/instruction mapping. | Identical-shape included-cost primitive using the production representation, independent oracle, final PTX/resources, and a material win before one real family. |
| 2 | Native independently scheduled RM/HCQ work | CUDA proves the device can co-schedule, but native extra-channel construction remains blocked and every dependency-coherent cut on the current single queue lost. This is still the only route to llama's hidden support tail if the RM construction becomes expressible. | Exact accepted RM sequence, then two independent 2--5 us kernels with >=5% interval saving before token scheduling. |
| 3 | Reopen graph overlap only after a new coarse region exists | No current d512 cut survives heavy-MMQ DRAM contention plus wait cost. | CPU DAG forecast excluding competing MMQs and predicting >=50 us net after measured edge costs. |
| 4 | Residual sampler/vocab/RoPE/KV tails | Small remaining ownership; P1/P5 already consume the descriptor and feedback parts. | A distinct topology/body mechanism with explicit P1/P5 interaction exclusion before GPU time. |

## Claims that remain unsupported

- Do not state a fresh native absolute time, parity ratio, or “current gap” by
  subtracting booked rows from fixed authority; use the final same-session
  composed bracket instead.
- Do not add P5 redirect-off timing to P2. The redirect-on P5 midpoint is the
  only composed P5 debit in this ledger.
- Do not add fusion/body and overlap ownership, CUDA substitutions, profiler
  node sums, Shapley allocations, simulated critical paths, or synthetic
  shared-Q8 microgates as recovery.
- Do not claim flash-body parity or generic native hardware overlap from the
  rejected current cuts.
- Do not reopen RMSNorm by weakening the parent-chain predicate. The exact
  break is now located and both bounded construction attempts are closed.

## Next accounting checkpoint

The final composed same-session row is now measured: steady llama A/C midpoint
`4.0056768 ms/token` versus native P1+P2+P5+Q4-g12
`5.3242440 ms/token`, or **0.75235x** llama throughput with a fresh
**1.3185672 ms/token** gap. Using llama-bench's cold-inclusive reported average
instead gives 0.76444x and a 1.254189-ms gap; both are recorded in
`nv-decode-final-composed-same-session-record-20260805.md`.

The strict fixed-authority booked total is **258.00581275 us/token** and its
counterfactual remainder is **1388.16418725 us/token**. Those fixed-authority
numbers remain causal accounting; the same-session row is the current absolute
measurement. No current construction is parity-qualified.
