# NV production gap re-pin at HEAD d284eb9d6 (2026-08-14)

Date: 2026-08-14. Target: RTX 5090, native `DEV=NV`, sm_120. Branch
`nvidia-bringup-20260731`, HEAD `d284eb9d6`. Fresh DEBUG=2 prime-token census
(`route_kernel_census.py --depth 512`), classified with the population ledger,
compared against the pinned llama node ledger. Measurement-only; nothing promoted.

## Headline

| side | tok/s | kernels/token | kernel-sum |
| --- | ---: | ---: | ---: |
| tinygrad (this HEAD, prime token) | **193.1** | 596 | ~5413 us |
| llama (pinned nsys ledger, 08-12) | 245.5 | 762 | 4774 us (span 3835 us) |
| gap | -52.4 | -166 | +639 us |

The production graph has already shrunk from 582 kernels (08-12) to 596 today,
but the token cost moved only ~+0.9 tok/s in the same window: the remaining
cost is kernel work, not kernel count.

## Per-class delta (current production vs llama node-sum)

Class boundary caveat: llama's `rms_norm` row and tinygrad's fused
`reduce_output_rmsnorm_*` bodies do not share a clean line. The rows below use
the population-ledger split and sum to the capture total.

| class | llama us | tinygrad us | delta us | status |
| --- | ---: | ---: | ---: | --- |
| GEMV + folded quant (anchor) | 3721.2 | 3809.0 | **+87.8** | 3 shapes NO-GO |
| vocab GEMV | 303.6 | 321.3 | +17.7 | near parity |
| flash score | 113.9 | 241.7 | **+127.8** | structural, 2.05x |
| flash combine | 120.5 | 120.7 | +0.2 | parity |
| norms + reduce-output + residual plumbing | 307.6 | 616.2 | **+308.6** | see below |
| rope + kv | 201.0 | 158.0 | **-43.0** | ahead |
| residual + materialized reduces | 4.8 | 75.8 | +71.0 | partial |
| vocab aux | 0.0 | 71.1 | +71.1 | tail |
| total | 4774.4 | ~5413 | +639 | |

The norm/plumbing `+308.6` row is the sum of the promoted `reduce_output_rmsnorm_*`
bodies (378 us, 91 launches), the still-unfused input norm reduce `r_16_256`
(150.5 us, 37), and the `E_32_32_4` epilogues (87.5 us, 38, mixed norm/residual).
The FFN-norm half of that was already gated NO-GO on the wall (correctness-clean,
removes 199 kernels, but flat wall), so it is not a free fold.

## Where the three anchor rows sit

The `+87.8 us` anchor deficit is the known three-shape tail, all measured NO-GO
against llama floors on this branch:

| shape | tinygrad us/node | llama us/node | ratio | record |
| --- | ---: | ---: | ---: | --- |
| Q4 FFN-down 4096x12288 | 26.75 | ~11.8-19.2 | 1.4-2.3x | load-pattern sweep NO-GO; DP4A resadd recovers ~25.6 us/token of the row |
| Q6 FFN-down 4096x12288 | 35.14 | 28.75 | 1.22x | MC2 coop NO-GO |
| Q6 attention V 1024x4096 | 17.91 | 4.90 | 3.65x | MC2 partial NO-GO |

## Verdict

The re-pin confirms there is no single buildable "substrate to 240" left on this
branch. The dominant mass is the quant GEMV core (~75% of kernel-sum), and every
remaining deficit inside it is a measured NO-GO at a local optimum. The support
mass that llama overlaps is already fused (rope/kv, flash combine, q/k reduce-output
are at parity or ahead); the FFN-norm fold was built and gated NO-GO on the wall.
What remains is a fragmented tail: flash score structure (+128 us, 2.05x),
unfused input norms (+150 us), vocab aux (+71 us), and launch hiding (~33 us,
exhausted). Each is worth roughly +1-3 tok/s and none collapses to parity.

## Evidence

- tinygrad prime-token census: `/tmp/census_nv_20260814.json`
  (`route_kernel_census.py --depth 512`, DEV=NV, HEAD d284eb9d6)
- llama pinned node ledger: `docs/task_workflow/evidence/nv-llama-d512-node-ledger-20260812.json`
- prior attribution: `docs/task_workflow/input/nv-decode-gap-attribution-same-session-20260812.md`
