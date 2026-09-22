# NV GEMV-core correction - the row is NOT at parity, and the closures are premature (2026-08-13)

Date: 2026-08-13
Branch: `nvidia-bringup-20260731` (HEAD `1591e406c`)
Status: **correction.** Supersedes the parity conclusion in
`nv-gemv-body-parity-audit-20260813.md`. That audit trusted the 08-12
attribution's folded-quantize framing and missed the source audit's core
deficit. This re-derives from the pinned semantic join
(`nv-decode-llama-live-gemv-route-audit-20260805.md`) and the raw CUPTI trace.

## 0. The mistake, stated plainly

The 08-12 attribution reports "GEMV bodies alone at parity (+3.1 us)" by
summing llama's separate `quantize_q8_1` (482 us) into its GEMV total. That is
true only because tinygrad has already folded the quantizer into its GEMV core,
so the *totals* match. It does not mean the GEMV core is at parity. The 08-05
source audit - the careful semantic join to the pinned trace - measures the
core separately:

| side | GEMV core (MMVQ/q4k/q6k only) | activation quant | total |
| --- | ---: | ---: | ---: |
| llama | 3579.8 us | 552.1 us | 4131.9 us |
| tinygrad native core | 3882.6 us | 0 (folded) | 3882.6 us |

So tinygrad's folded total is actually -249 us ahead, but its **core is
+302.8 us slower**. The +302.8 us is the real lever: it is the price tinygrad
pays for using a slower core (scalar Q4_K direct) while llama uses DP4A.
Recovering it without re-adding the 552 us quantize is the whole game, and the
"closed" verdicts below were written before that recovery path was tested.

## 1. The corrected per-shape core deficit (08-05 source audit, pinned)

| population | llama MMVQ us | native core us | native - llama | ratio |
| --- | ---: | ---: | ---: | ---: |
| Q6 attention V (18) | 89.4 | 307.3 | **+217.9** | 3.44x |
| Q4 FFN down (18) | 346.2 | 443.2 | **+97.0** | 1.28x |
| Q6 FFN down (18) | 520.8 | 601.9 | **+81.1** | 1.16x |
| Q4 attention K (36) | 117.4 | 152.4 | +35.0 | 1.30x |
| Q6 vocab (1) | 303.6 | 314.4 | +10.8 | 1.04x |
| Q4 attention V (18) | 75.8 | 75.2 | -0.7 | 0.99x |
| Q4 gate/up (36) | 1364.0 | 1360.2 | -3.8 | 1.00x |
| Q4 attention Q (36) | 342.9 | 316.2 | -26.7 | 0.92x |
| Q4 attention O (36) | 418.5 | 311.8 | -106.7 | 0.74x |
| **sum** | | | **~+302.8** | |

The three real deficits are Q6 attention V (+218), Q4 FFN down (+97), and
Q6 FFN down (+81). Everything else is at or ahead of llama. This is not
parity; it is ~303 us of recoverable core mass, ~+12 tok/s.

## 2. The Q4 FFN-down "2.27x" and its NO-GO are built on a copy-paste error

The 08-12 quant audit copied the attention-O value into the FFN-down row:

- `nv-decode-llama-live-gemv-route-audit-20260805.md:50` (the source) says
  FFN down Q4 = **346.209 us / 18 = 19.23 us/node**.
- `nv-quant-gemv-llama-audit-20260812.md:38` (the copy) lists the same total
  346.209 but a us/node of **11.776** (attention-O's value).

Raw CUPTI confirms the source: the fused-Q4 N=4096 kernel group is bimodal,
~10.4 us (attention O, K=4096) vs ~19.5 us (FFN down, K=12288), 2:1 by node
count. So the true down-Q4 floor is 19.2-19.5 us, not 11.776 us.

Consequence: the load-pattern sweep
(`nv-q4kd-load-pattern-measurement-record-20260812.md`) computed its standalone
floor as `11.776 / 1.26-1.7 = 6.9-9.3 us`, then closed NO-GO because the best
row was 11.43 us. With the correct floor the standalone equivalent is
`19.23 / 1.26-1.7 = 11.3-15.3 us` (and 13.3 us at this shape's own 1.42-1.45x
offset), so the **11.43 us quad-u128-smem row actually clears the floor**. The
NO-GO is invalid; that row should be re-censused in-loop (the MC2 lesson:
standalone winner must survive the census).

## 3. The DP4A path has an untested successor, not a dead end

`nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md` shows the
DP4A consumer is a device win (-6.37 us/layer, -28.9%) but the naive
provider+consumer adds a graph node and loses the wall (+6.2 us on layer 8).
The same record ends with a **dormant producer-owned successor**: its provider
reads the W1/W3 fp32 output and emits Q8_1 directly, keeping the topology at
3 nodes instead of 4, and it "has not taken a GPU gate." That is the llama
substrate done right - fold the activation quant into an existing producer so
it costs no extra node - and it was never measured.

## 4. What this changes

- The GEMV row is **not closed and not at parity**: ~303 us of core mass is
  open, dominated by Q6 attention V (+218 us), Q4 FFN down (+97 us), and
  Q6 FFN down (+81 us).
- The Q4-down load-pattern NO-GO is **invalid** (wrong floor); a winning row
  exists and needs an in-loop census.
- The DP4A substrate (llama's actual algorithm) has a **producer-owned Q8_1
  successor that was designed but never GPU-gated**.
- Full recovery needs llama's DP4A substrate on the three slow shapes with the
  activation quant folded into an existing producer, so the 552 us quantize is
  not re-added. That is the real substrate-to-parity work, not the fusion rows.

## Evidence

- `nv-decode-llama-live-gemv-route-audit-20260805.md` (pinned semantic join, +302.8 us)
- `nv-quant-gemv-llama-audit-20260812.md` (the copy-paste error at line 38/86)
- `nv-q4kd-load-pattern-measurement-record-20260812.md` (NO-GO against wrong floor)
- `nv-q4k-ffn-down-mmvq-included-cost-and-one-layer-record-20260805.md` (DP4A device win + dormant successor)
- raw traces: `/tmp/llama_tg10_node_20260812.sqlite` (bimodal 10.4 vs 19.5 us)
