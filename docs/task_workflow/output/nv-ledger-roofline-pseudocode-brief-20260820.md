# Decode ledger as roofline pseudocode: where the 717.505 us gap lives

Status: analysis only. No runtime code changed. The numbers are the locked
measurement result from
`docs/task_workflow/output/nv-weighted-inter-anchor-causal-gap-result-20260820.md`
and its ledger artifact
`docs/task_workflow/output/nv-weighted-inter-anchor-ledger-20260820.json`.

Audience note: this brief is written to be pasted into another model. It
separates the ledger arithmetic, the roofline, and the llama mechanism so the
three cannot blur into "tinygrad is slower because of kernels".

## 0. One-paragraph answer

The 717.505 us/token wall gap is exposed timeline, not kernel speed. Tinygrad
carries less kernel residence-time mass than llama (4738.496 vs 5020.797 us),
and its
mandatory GEMV anchor bodies are already slightly faster (2943.750 vs 2998.617
us). The loss is that tinygrad hides almost nothing (5.996 us of overlap)
while llama hides 1128.020 us. The decisive area is the per-layer S1 interval
between the Q GEMV anchor and the O GEMV anchor: tinygrad exposes 1152.250 us
there against llama's 517.916 us, a 634.334 us contribution. That is the
answer. Everything below proves why that area, and not the GEMV core, is where
the gap is.

## 1. Ledger definitions as pseudocode

Given a token's traced kernels and their dependency DAG:

```text
K = [(start_i, end_i, bytes_i, flops_i, deps_i) for i in kernels]

mass_i      = end_i - start_i
node_sum    = sum(mass_i)
union       = measure(union of all [start_i, end_i) intervals)
overlap     = node_sum - union
span        = max(end_i) - min(start_i)
wall        = union + host_launch_residual
critical_path = longest dependency path using mass_i as edge/node weight
```

Two roofline layers are both required:

```text
per-kernel time_i  = max(bytes_i / BW, flops_i / R, launch_latency_i)
timeline efficiency = mandatory_byte_floor / wall
```

Decode is at roughly 3.3 FLOP/byte, so the per-kernel roofline is
memory-bound for the GEMVs. The support kernels (norm, rope, elementwise,
reduce) are the opposite: their bytes and FLOPs are small, so
`launch_latency_i` dominates and each one spends most of its cost occupying a
serial position in the timeline. That distinction is the whole story.

## 2. The locked ledger

| quantity | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| unprofiled wall | 4723.214 | 4005.709 | +717.505 |
| device union (profiled) | 4732.500 | 3892.777 | +839.723 |
| host / launch residual | -9.286 | +112.932 | -122.218 |
| node sum (kernel residence-time mass) | 4738.496 | 5020.797 | -282.301 |
| overlap mass (node sum - union) | 5.996 | 1128.020 | -1122.024 |
| logical critical path | 4249.216 | 4443.435 | -194.219 |
| device span | 4842.250 | 3901.205 | +941.045 |

The negative tinygrad host residual is a profiler-tax artifact, not a measured
host win: its `PROFILE=1` device timestamps absorb tax, so that row is a lower
bound.

The identities close exactly:

```text
wall gap = device union gap + host gap
717.505  = 839.723 - 122.218

device union gap = delta_node_sum - delta_overlap
delta_node_sum   = tinygrad - llama = -282.301
delta_overlap    = tinygrad - llama = -1122.024
839.723         = (-282.301) - (-1122.024)

node_sum - union = overlap mass
tinygrad: 4738.496 - 4732.500 =   5.996
llama:    5020.797 - 3892.777 = 1128.020
```

In advantage form, llama hides 1122.024 us more and carries 282.301 us more
residence time, so the net device gap is 1122.024 - 282.301 = 839.723 us.

Terminology correction: `node_sum` is summed kernel residence-time mass, not
operations, bytes, or useful work. Under PDL a consumer interval can include
time spent waiting at a grid-dependency synchronization, so residence mass and
simultaneous useful execution are different quantities. The identity is
exact, but it does not say how much useful work ran concurrently.

## 3. Anchor versus support decomposition

The device gap decomposes into rows that sum exactly:

| row | tinygrad us | llama us | delta us |
| --- | ---: | ---: | ---: |
| Q/O/gate-up/down anchor union | 2943.750 | 2998.617 | -54.867 |
| per-layer support exposed | 1418.750 | 593.012 | +825.738 |
| vocab tail | 370.000 | 303.740 | +66.260 |
| interval overlap accounting residual | - | - | +2.592 |
| device union total | 4732.500 | 3892.777 | +839.723 |

The anchors are not the loss; tinygrad wins that row by 54.867 us. The exposed
support row carries 825.738 us of the 839.723 us device gap.

Per-layer support exposure by segment:

| segment | tinygrad us | llama us | delta us | tinygrad weighted | llama weighted |
| --- | ---: | ---: | ---: | ---: | ---: |
| S0 (down -> Q) | 181.000 | 31.299 | +149.701 | 179.744 | 219.326 |
| S1 (Q -> O) | 1152.250 | 517.916 | +634.334 | 558.912 | 664.924 |
| S2 (O -> gate/up) | 186.000 | 32.896 | +153.104 | 186.016 | 221.212 |
| S3 (gate/up -> down) | 0.000 | 15.937 | -15.937 | 0.000 | 29.856 |
| S4 (down -> next Q) | 170.250 | 30.884 | +139.366 | 169.120 | 218.111 |
| tail after vocab | 56.250 | 0.000 | +56.250 | - | - |

S1 alone is 634.334 of the 825.738 us exposed-support gap. S3 is the one place
tinygrad wins outright: its GLU is fused into the down epilogue.

Inside the S1 exposure windows:

| family | tinygrad in-window mass | llama in-window mass |
| --- | ---: | ---: |
| gemv_kv | 298.750 | 288.057 |
| quant_provider | 0.000 | 132.768 |
| rope | 0.000 | 127.998 |
| rmsnorm | 186.750 | 93.535 |
| kv store/get | 0.000 | 75.453 |
| flash_score | 266.750 | 74.559 |
| flash_combine | 105.750 | 51.680 |
| residual | 129.750 | 0.000 |
| reduce | 54.750 | 0.000 |
| total support mass | 1042.500 | 844.050 |
| window exposure | 1152.250 | 517.916 |

Llama overlaps its own S1 families by 326.134 us, so 844.050 us of work fits
in a 517.916 us window. It also hides another 764.057 us behind the MMQ
anchors. Tinygrad hides 0.000 us behind anchors in every family, and its S1
window also contains 109.75 us of dead device time, the token's entire
span-minus-union dead time.

Taxonomy note: the trace classifies tinygrad `E_*` elementwise programs as
`residual`, so tinygrad's Q rope appears inside that family rather than in a
separate `rope` family. The llama `rope` row is a distinct ggml role. Both
sides have the logical rope operation; the row names do not imply llama has
rope work that tinygrad lacks.

## 4. Where this is in tinygrad source

The area is the decode block chain in `FFNBlock.__call__` / `_run` and
`TransformerBlock._attention`:

- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:640) - `_run`,
  the per-layer decode body.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:755) -
  `_attention`, the Q -> O chain.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:762) - separate
  `attn_q`, `attn_k`, `attn_v` primitive calls.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:803) - separate
  Q/K RMSNorm calls after the projections.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:852) - Q rope.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:928) - the
  flash score/combine route.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:1004) - the O
  projection return.
- [model.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/model.py:557) - the FFN
  path whose fused gate/up and down epilogue explain S3 = 0.

Pseudocode of one tinygrad layer, with the measured anchors and segments:

```text
def tinygrad_layer(x):
    nx = rmsnorm(x)                 # S0: reduce_output_rmsnorm + casts
    q  = gemv(Wq, nx)               # ANCHOR Q, memory bound

    # S1 begins. Every row below occupies the wall between Q and O.
    q  = q_rmsnorm(q)               # r_/E_ norm kernels, hidden by nobody
    q  = rope(q)                    # elementwise, hidden by nobody
    k  = gemv(Wk, nx)               # off critical path but exposed here
    v  = gemv(Wv, nx)               # off critical path but exposed here
    kv = kv_store(k, v)             # store/cast chain, hidden by nobody
    s  = flash_score(q, kv)         # flash score
    a  = flash_combine(s, kv)       # flash combine
    o  = gemv(Wo, a, epilogue=residual_add)  # ANCHOR O, h = x + o folded

    # S2 begins: FFN input norm is still two separate programs.
    nh = rmsnorm(h)                 # r_16_256 + E_32_32_4 epilogue

    # S3 is zero: fused w1+w3 producer plus GLU, down residual folded.
    z  = fused_gate_up(nh)          # ANCHOR gate/up, SiLU in-kernel
    d  = gemv(Wd, z, epilogue=residual_add)  # ANCHOR down

    # S4 is next-layer preparation, not a residual add (already folded).
    q8 = quant_provider(d)          # rmsnorm_q8_1 provider, exposed
    return d
```

The exact first-layer device intervals from the canonical capture make the
S1 structure concrete (microseconds relative to token start):

```text
[20.000, 28.750] q4k_g3_lanemap_gemv_4096_4096        ANCHOR Q
[28.750, 32.500] reduce_output_rmsnorm_32_128         q norm
[32.500, 35.250] E_16_32_4_2                          q rope
[36.000, 41.000] q4k_g3_lanemap_gemv_1024_4096        K GEMV
[41.000, 45.500] q6k_v_four_warp_fp16_direct_1024_4096 V GEMV
[45.500, 48.000] reduce_output_rmsnorm_8_128          k norm
[48.000, 49.750] E_8_8_16_2                           store/cast
[53.750, 62.000] flash_block_tiled_xlane_score_...    flash score
[62.000, 65.000] flash_fused_gmax_combine_f16_...     flash combine
[65.000, 74.500] q4k_g3_lanemap_gemv_epi_resadd_...   ANCHOR O
```

Between Q end at 28.75 and O start at 65.00 there is no anchor running. Every
support kernel is serial, and there is even a 4 us hole at 49.75-53.75.

The names in the trace are concrete:

```text
anchors:  q4k_g3_lanemap_gemv_* / q6k_gen_* / w1w3 fused producer
support:  reduce_output_rmsnorm_*, r_*, E_*, flash_score,
          flash_combine, residual adds and casts
```

Each remaining support row is a separate program because the promoted folds
only cover the O residual, fused gate/up, and down residual. The model-level
gates that control the absorbed variants are visible in `_run` (residual and
norm admission, `model.py:648-674` and `model.py:680-736`) and in the
epilogue machinery in
[decode_kernels.py](/home/ubuntu/tinygrad-arkey/tinygrad/llm/decode_kernels.py:170).
The current head has fused w1+w3 and the down epilogue, which is exactly why
S3 is already zero; the corresponding absorption for the S0/S1/S2/S4 support
is what remains.

## 5. Why we know S1 is the issue

This is a chain of arithmetic, not a preference:

1. Tinygrad's anchor bodies are faster in aggregate: 2943.750 vs 2998.617 us.
   So "make the GEMV faster" cannot be the whole answer.
2. Tinygrad carries less residence-time mass: 4738.496 vs 5020.797 us. So
   "llama does fewer operations" is false at the node-sum level.
3. Tinygrad overlaps 5.996 us; llama overlaps 1128.020 us. The overlap gap of
   1122.024 us is larger than the entire 717.505 us wall gap.
4. Per-layer exposed support is 1418.750 vs 593.012 us, delta 825.738 us.
   Within that, S1 is 1152.250 vs 517.916 us, delta 634.334 us.
5. In S1, llama hides 764.057 us behind its MMQ anchors and overlaps 326.134
   us of support with itself. Tinygrad hides zero and exposes 1042.500 us of
   support mass plus 109.75 us dead time.
6. The critical-path check rules out "the support is off path, therefore it
   costs nothing". Tinygrad's span exceeds its critical path by 593.034 us,
   so the off-path support is filling serialized gaps instead of being hidden.
   Llama's span is 542.230 us below its logical critical path because PDL
   launch-completion lets consumers start before producers logically finish.
7. The queue and PDL experiments tested the non-fusion alternative and it did
   not transfer:

```text
2 compute queues vs 1:        +113.219 us (real, but 1/6 of the gap)
PDL, correct direction, 2q:    -11.641 us (slower, no recovery)
PDL, correct direction, 1q:     -8.201 us (slower, no recovery)
```

So the remaining mechanism is not "add more overlap"; it is to remove the
separate support programs and their byte round-trips by fusing epilogues into
the anchors. This is the existing legal ranking, not a new theory.

## 6. Roofline reading of the same evidence

Decode is memory bound, so use bandwidth accounting rather than FLOP
accounting:

```text
tinygrad: ~5.04 GB/token / 4723.214 us = ~1067 GB/s effective
llama:    ~4.70 GB/token / 4005.709 us = ~1173 GB/s effective
measured peak: 1700-1792 GB/s read on this RTX 5090
```

The 5.04 vs 4.70 GB difference is the extra fp32 intermediate round-trips from
the separate `E_*` and `r_*` plumbing. But byte count alone is not the whole
gap either: at the measured peak the 5.04 GB floor is about 2.96 ms, while the
observed union is 4.73 ms. Tinygrad is both moving more bytes and keeping the
DRAM idle during the support chain. The byte counts are accounting estimates,
not hardware DRAM counters.

The per-kernel roofline gives the secondary, in-kernel deficits:

```text
historical Q6_K partial (K/V): 0.20 TB/s vs llama 1.04 TB/s
current Q6_K V four-warp:      4.368 us vs llama 4.896 us (fixed, now faster)
historical Q6_K coop down:     0.82 TB/s vs llama 1.40 TB/s
current Q6_K down four-warp:   30.368 us vs llama 28.753 us (near parity)
Q4_K FFN down:                 20.944 us vs llama 19.232 us (still trails;
                                corrected floor; 11.776 us was attention-O)
```

The mechanism behind the fixed rows is geometry, not instruction choice.
`ncu` proved the one-warp Q4-down chain in
`docs/task_workflow/input/nv-ffn-down-gap-occupancy-proof-20260814.md`:
one warp/row reaches 38.8% occupancy and 54.5% DRAM, while four warps/row
reach 66.3% occupancy and 77.2% DRAM. The current head has promoted the
four-warp fp16 direct consumers for Q4/Q6 down and Q6 V, so the old one-warp
rows are historical. The current Q4-down row is the main remaining
inside-anchor deficit. In aggregate the anchor union is still
tinygrad-faster, so this inside-anchor work cannot explain the 717 us wall
gap at the current head; it is the second lever behind the exposed S1 chain.

There is a first-principles reason the overlap experiments were bounded.
The S1 anchors are memory bound and the support kernels are launch-latency
bound, so running two of them concurrently does not create bandwidth; it only
hides launch and tail latency. Fusion is different because it removes the
kernel boundary and the intermediate bytes altogether. That is why the second
queue recovered only 113 us and PDL was a net loss: the competing program
still paid its launch cost while sharing one DRAM budget.

## 7. How llama gets its number

Llama's route, in the same pseudocode vocabulary:

```text
def llama_layer(x):
    # One ggml graph per token. MMVQ is 128 threads/row (4 warps), Q8_1 + DP4A.
    q8 = quantize_q8_1(x)                          # separate, overlapped
    q  = MMVQ(Wq, q8)                              # ANCHOR Q
    qn = rms_norm(q)                               # separate, PDL-overlapped
    qr = rope(qn)                                  # separate, PDL-overlapped
    k  = MMVQ(Wk, q8); v = MMVQ(Wv, q8)            # off spine, overlapped
    s  = flash_attn_ext_vec(qr, kv)                # overlaps the MMQ shadow
    o  = MMVQ(Wo, q8(s), epilogue=residual_add)    # residual folded
    z  = MMVQ(Wg | Wu, q8(x), epilogue=swiglu)     # one fused 12288-row MMQ
    d  = MMVQ(Wd, q8(z), epilogue=residual_add)    # residual folded
    return d

    # Graph edges are PDL launch-completion edges: a small consumer is
    # released at the producer's last wave. The token span is therefore
    # shorter than the logical critical path.
```

Side by side for the decisive interval:

```text
tinygrad S1 (1152.250 us exposed):
[Q GEMV][q_norm][rope][K/V GEMVs][flash_score][flash_combine][O GEMV]

llama S1 (517.916 us exposed):
[MMQ_Q]
   [rms_norm][rope][quant/rope/flash work running in the MMQ shadow]
                                            [MMQ_O with residual folded]
```

The mechanism labels from the trace:

| llama mechanism | measured consequence |
| --- | --- |
| `FUSED_INTO` | GLU and residual live inside MMQ epilogues |
| `OVERLAPPED` | 764.057 us of support hidden behind MMQ anchors |
| `OVERLAPPED` | 326.134 us of S1 norm/rope/quant/flash overlaps itself |
| `PDL` | span 3901.205 us below logical critical path 4443.435 us |
| `OFF_PATH` | K/V work kept off the spine, same as tinygrad |

Llama's per-shape MMQ rows, against the matched tinygrad rows, make the point
that its advantage is structural rather than instruction-level:

| projection | llama us/node | tinygrad us/node | mechanism |
| --- | ---: | ---: | --- |
| Q 4096x4096 | 9.536 | 8.160 shared-Q8 / 9.63 legacy | tinygrad faster; llama overlaps norm/rope |
| K 1024x4096 | 3.328 | 3.840 shared-Q8 / 4.896 legacy | near parity; support overlapped |
| V Q6_K 1024x4096 | 4.896 | 4.368 four-warp | tinygrad faster |
| O 4096x4096 | 11.776 | 9.088 (with resadd) | tinygrad faster |
| fused gate/up | 37.856 | 37.952 | parity after MC3 |
| down Q4_K | 19.232 | 20.944 four-warp | remaining deficit (corrected floor) |
| down Q6_K | 28.753 | 30.368 four-warp | near parity |
| vocab Q6_K | 303.618 | 313.632 | near parity |

The tinygrad column is the canonical current-head capture median; the llama
column is the matched same-session audit row. The anchor aggregate in section
3 remains the locked current-head number. The point of the table is that
llama's per-node MMQ speed is not where the 717 us comes from. It comes from
those same bodies absorbing support work and from the graph's PDL edges
compressing the device timeline.

## 8. What the ledger implies next

Ranked by recoverable end-to-end wall, from the same result:

| rank | change | legal ceiling | status |
| ---: | --- | ---: | --- |
| 1 | residual elementwise add/mul into GEMV epilogue | 148.736 us | simulated |
| 2 | output-reduction folding in-kernel | 203.680 us | simulated |
| 3 | fold post-vocab norm/quant/reduce | 313.632 us | measured transfer too small |
| 4 | flash score/combine exposure | 267.328 / 105.056 us | no known legal change |
| 5 | quant_provider exposure | 30.976 us | small |
| 6 | K/V GEMV | 0.000 us | off path in both |

The already measured copy-free fp16 RMSNorm returned +12.506 to +17.141 us and
did not clear the 50 us promotion gate. The ceiling numbers require critical
path recomputation; they are not additive.

## 9. Questions to hand to a high-effort model

These are the open edges the ledger leaves for design work:

1. How do you make the S1 norm/rope/flash chain stop serializing? Llama keeps
   norm/rope/quant as separate kernels and overlaps them with PDL; tinygrad's
   native PDL did not transfer, so folding a bounded epilogue into the Q4K
   primitive (via the existing `Q4KGEMVEpilogue` machinery) is one candidate
   but not the only mechanism.
2. How do you remove the `E_*` residual and `r_*` output-reduction programs
   while preserving byte-identical logits? Those are rows 1-2 above.
3. Given every S1 program is memory bound, what is the theoretical upper bound
   on launch-overlap recovery at 1.7-1.79 TB/s? The measured 113 us from a
   second queue is the empirical hint that this bound is low.
4. Why does native PDL on tinygrad cost time while llama's PDL edges compress
   its span? Candidate answers are launch cost, QMD latch placement, or graph
   granularity, but the current evidence says the economics did not transfer.

The one conclusion to keep fixed: the gap is 634.334 us of exposed S1 support
plus the other support segments, not a missing kernel mechanism.
