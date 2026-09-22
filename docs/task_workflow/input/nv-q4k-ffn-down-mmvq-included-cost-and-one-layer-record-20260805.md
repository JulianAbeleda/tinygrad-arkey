# Native NV Q4_K FFN-down MMVQ included-cost and one-layer record

Date: 2026-08-05. Target: RTX 5090, native `DEV=NV`, sm_120, driver
595.84. Tinygrad base commit: `a1a51c349d1b8c55a6373631913ea7845e99cc8d`.
All GPU work used `/tmp/gpu-bench.lock`. This is a default-off qualification
record, not a promotion record.

## Verdict

The production-shape primitive is a **material included-cost device win**, and
16 of 18 one-layer semantic arms pass, but the first passing singleton is a
**production-wall NO-GO**. Layer 8 regresses by `+6.204734 us/token` in a
settled reverse bracket. Therefore nothing is promoted, no recovery is booked,
and the predicted two-layer subset is not advanced into the already-tight
accepted-attention precision composition.

## Exact boundary and construction

Both arms begin with the same already-materialized fp16 FFN activation and end
with one fp32 4096-element down-projection output. W1/W3 production is outside
both arms and is unchanged.

- Control: installed `q4k_g3_lanemap_gemv_4096_12288`.
- Candidate: `q8_1_llama_provider_12288` followed by
  `q4k_q8_mmvq_direct_4096_12288`.
- Provider launch: 48 CTAs x 256 threads, eight Q8_1 groups per CTA.
- Consumer launch: 4096 CTAs x 128 threads, four warps per output row,
  12 Q4_K blocks per warp, Q8_1 plus DP4A, direct fp32 output.
- Consumer staging: warps 1--3 publish one partial per lane (384 B shared), one
  CTA barrier, warp 0 loads the three partials and performs the five-shuffle
  final reduction. Generated SASS contains two static `IDP.4A` instructions in
  the runtime loop body, one `STS`, three `LDS`, one `BAR.SYNC`, five `SHFL`,
  and one final `STG`.
- The fixed production route retains a real
  `for (int Lidx2 = 0; Lidx2 < 12; Lidx2++)` loop. It has no replay-time bound
  symbol and introduced no model-JIT input or adapter.

The provider is byte-exact against the pinned live llama CUDA Q8 cubin for the
fixed production activation. This matters: llama CUDA spells quantization as
`roundf(x/d)`. The llama CPU reference instead uses reciprocal multiplication
and differs at one live fp16 tie; the CPU reference is diagnostic, not the
provider authority.

Independent scalar packed Q4 x Q8 algebra passed on rows 0, 1, 2047, and 4095.
Worst selected-row relative error was `6.501e-6`.

## Device-event reconciliation

Eleven completion-timed launches per kernel:

| arm / kernel | median us |
| --- | ---: |
| installed Q4 fp16 consumer | 22.016 |
| candidate live-exact Q8 provider | 1.696 |
| candidate four-warp Q4/Q8 consumer | 13.952 |
| candidate included sum | **15.648** |

Included delta: **-6.368 us per Q4 layer, -28.92%**. This reconciles with the
live semantic ledger's approximately 24.6 us installed per-call attribution;
unlike the host replay microgate, it is directly device completion-timed.

The reverse A/B/A host replay microgate also passed materially:
`124.201 -> 68.588 us`, `-55.613 us` (`-44.78%`). That host value is not
multiplied by 18 or booked as token wall.

## One-layer production full-logit gate

Block 4 was leased in an otherwise normal composed d512 model run. Across its
two ping-pong captures, the complete program-histogram delta was exactly:

- installed Q4 FFN-down: `-2`;
- Q8 provider: `+2`;
- four-warp consumer: `+2`;
- every other program: unchanged.

Thus the topology gate passed with no adapter or unrelated program delta.
Eight retained full-logit rows produced:

| contract item | result | gate |
| --- | ---: | --- |
| finite | true | pass |
| generated tokens | exact | pass |
| argmax | exact | pass |
| top-10 set / order | exact / exact | pass |
| relative L2 | **0.0020060** | **fail**, maximum 0.001 |
| max absolute | **0.0799179** | historical 0.01 also fails |
| `2*max_abs / min top1 margin` | 0.13103 | pass, maximum 1.0 |

Verdict: **FAIL_CLOSED_ONE_LAYER_NUMERICS**. The all-18 wall A/B/A was not run,
because the required correctness gate precedes family timing.

## Precision-budget sweep

The fresh-control singleton sweep covered the exact Q4 indices
`(4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29)`. Every arm passed the
exact production topology delta. Sixteen of eighteen arms also passed the full
semantic contract; only block 4 (`0.0020060`) and block 5 (`0.0011151`) failed
relative L2.

The exhaustive signed-delta search over all `2^18` subsets found only two
additively predicted semantic passes:

| predicted subset | relative L2 | status |
| --- | ---: | --- |
| block 16 | `0.000550215` | singleton measured PASS |
| blocks 8,16 | `0.000837233` | prediction only |

These predictions are directional, not qualification. A real blocks-8,16
semantic arm was intentionally not run because the required cheaper singleton
wall gate failed first.

## Settled one-layer production wall

Layer 8 was chosen as a semantically passing member of the predicted pair. The
direct bracket ran with the accepted P1 descriptor/input-shadow behavior,
`CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT=1` (P2), and composed ping-pong (P5).
It used six warmup decode calls followed by three uninterrupted 32-token
windows per fresh arm.

| arm | median ms/token |
| --- | ---: |
| control A | `5.361277781` |
| layer-8 candidate | `5.364128781` |
| control C | `5.354570313` |
| control midpoint | `5.357924047` |

All three 96-token stream hashes are exact and no sample was rejected. The
candidate is **`+6.204734 us/token` slower** (`-0.11567%` speedup), hence
`NO_GO_WALL`.

The topology explains why isolated device timing was not sufficient. Across
each capture, replacing the installed consumer changes `947 -> 948` logical
graph members, `730 -> 732` generic calls, and `217 -> 216` semantic calls:
one installed kernel becomes a provider plus consumer. The isolated prediction
was `-6.368 us`; production measured `+6.205 us`, leaving a `12.573 us`
interaction/replay residual. This record does not falsely assign that residual
to one mechanism without a device/host trace. It does establish that the extra
node and production interaction erase the kernel-body win.

The final accepted-attention three-way harness is implemented but dormant. It
would compare unleased baseline, accepted shared-Q8 blocks 1--12 and 14--18,
and accepted attention plus FFN candidate, requiring the final candidate to
remain within global relative L2 `<=1e-3`. The earlier singleton wall stop
correctly prevents spending GPU time or claiming a composition result.

## Exact profile decomposition and dormant producer-owned successor

The control PROFILE capture contains 875 program nodes and the layer-8
candidate 876. Their flattened longest-common subsequence has 874 exact-name
matches. The unmatched local region is exactly:

- control installed consumer: `25.504 us`;
- candidate provider plus consumer: `1.408 + 22.624 = 24.032 us`;
- local candidate delta: `-1.472 us`.

The 874 matched nodes drift by `+9.792 us` (`5157.472 -> 5167.264`), so total
node sum moves by `+8.320 us`. The full PROFILE equation is:

`node +8.320 + span/node rounding -0.070 + inter-group gap -259.000 = device window -250.750 us`;
adding outside-device `-34.392 us` gives PROFILE sync-wall `-285.142 us`.

That timestamp-heavy direction contradicts the settled wall bracket and is
therefore composition evidence only, not wall authority. The settled equation
remains isolated `-6.368 us` to wall `+6.204734 us`, an inversion of
`12.572734 us`. Using the actual PROFILE local delta instead leaves
`+7.676734 us` unexplained at settled wall. Neither number is labeled a fixed
launch tax: there is no copy/other histogram delta, the exact local one-to-two
region remains faster, and the matched-node drift is material.

The smallest dormant successor owns the former silu*mul-to-fp16 boundary: its
provider reads the W1/W3 fp32 output, performs the former fp16 rounding
internally, and emits Q8_1 directly. Isolated fresh CPU schedules prove exact
topology preservation:

- control: W1/W3 fused producer, fp16 materialization, installed down consumer;
- candidate: the same W1/W3 producer, fp32-to-fp16 Q8 provider, direct Q4/Q8
  consumer;
- exact count: `3 -> 3`, hence predicted whole-token `875 -> 875`.

The two arms must be scheduled from fresh graphs; scheduling one shared graph
materializes buffer identities and invalidates a second-arm census. The
executable hermetic test enforces isolated construction. This successor is
still default-off and has not taken a GPU gate.

## Artifacts and code

- `/tmp/q4k_ffn_down_mmvq_microgate_20260805_v2.json` SHA-256
  `54023bdc561685392b52ced25713b4b039998d40ff0414245aeabb07ed208b0f`.
- `/tmp/q4k_ffn_down_mmvq_one_layer_qualify_20260805.json` SHA-256
  `3fe8aaa30775f457d978e1ee17e0a598ef89d6e544f1fb28c459a388a69690be`.
- `/tmp/q4k_ffn_down_mmvq_subset_sweep_20260805.json` SHA-256
  `bca9dd01731d8bf6d438fdafe2e4664af951cfca67fa6750640baa97e03c0c2a`.
- `/tmp/q4k_ffn_down_mmvq_layer8_wall_20260805.json` SHA-256
  `10f42f9b11ad82ac5ad5e3810e42f2c918540546180dc3979cbaea4bddea6355`.
- `tinygrad/llm/q4k_ffn_down_mmvq.py`: closed-default per-block admission,
  provider, consumer, and exact-shape/target guards.
- `tinygrad/llm/decode_routes.py`: absent-admission fast fallback; the normal
  installed route is unchanged.
- `extra/llm_research/decode/q4k_ffn_down_mmvq_microgate.py`: live-cubin,
  independent algebra, device-event, and host replay gate.
- `extra/llm_research/decode/q4k_ffn_down_mmvq_qualification.py`: full-logit,
  topology, timing, resumable singleton sweep, and additive subset search.
- `test/unit/test_q4k_ffn_down_mmvq.py`: 13 passed (three unrelated pytest
  configuration warnings).

No commit or push was made. Default behavior remains closed.
