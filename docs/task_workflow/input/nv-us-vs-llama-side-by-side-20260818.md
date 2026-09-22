# NV decode: us vs llama, side by side (2026-08-18)

Date: 2026-08-18
Branch: `nvidia-bringup-20260731`
Status: **clean same-session comparison. Read-only. One table per question.**
Authority: the exact same-session wall account
(`nv-240-exact-wall-account-20260817.md`, residual 0.0), which measured both
sides in one session (Qwen3-8B-Q4_K_M, d512, RTX 5090): llama `ac4cddeb0`
(nsys CUPTI, 762 nodes/replay), tinygrad HEAD `07e9b2abe` (NV HCQ node
ledger, 594 nodes/token). Fresh HEAD position (205.99 tok/s census,
`daf591ad3`) is noted where it matters; the comparison itself is the 08-17
same-session pair.

## 1. The headline (same session, both unprofiled)

| side | tok/s | ms/token |
| --- | ---: | ---: |
| llama | 246.37 | 4.0589 |
| tinygrad | 208.84 | 4.7883 |
| **gap** | | **+729.4 us** |

## 2. The wall equation (exact, residual 0.0)

`wall = GPU busy + host gap`; `GPU busy = node_sum - overlap`.

| term | tinygrad | llama | delta (tg - llama) |
| --- | ---: | ---: | ---: |
| wall | 4788.3 | 4058.9 | **+729.4** |
| GPU busy (kernel union) | 4519.3 | 3890.5 | **+628.8** |
| host gap | 269.0 | 168.3 | **+100.6** |
| node_sum (all kernels) | 4519.3 | 5015.7 | **-496.3 (we win)** |
| overlap mass (node_sum - union) | 0.0 | 1125.1 | **-1125.1 (they win)** |

Check: `628.8 + 100.6 = 729.4`; `-496.3 - (-1125.1) = +628.8`. Exact.

## 3. Class by class (sums exactly to the node-sum delta)

Common taxonomy (llama classes folded to our roles: llama `mmq +
quantize_q8_1` = our `gemv` because we fold quant in-kernel; llama `rope +
kv_set_rows` = our `rope_kv`; llama `get_rows` = `residual_cast`).

| class | tinygrad us | llama us | delta | who wins |
| --- | ---: | ---: | --- | --- |
| gemv (incl. folded quant) | 3477.4 | 4138.0 | -660.6 | **us** |
| reduce_output | 312.1 | 0.0 | +312.1 | llama (absorbs in-kernel) |
| norms | 206.2 | 307.7 | -101.4 | **us** |
| flash_score | 213.1 | 173.6 | +39.4 | llama (structural) |
| flash_combine | 99.6 | 189.7 | -90.1 | **us** |
| rope_kv | 100.5 | 201.9 | -101.4 | **us** |
| vocab_aux | 59.5 | 0.0 | +59.5 | llama (host-side top-1) |
| other (small plumbing) | 49.1 | 1.9 | +47.2 | llama |
| residual_cast | 1.9 | 2.9 | -0.9 | parity |
| **node sum** | **4519.3** | **5015.7** | **-496.3** | **us** |

We win 5 of 9 classes and the total. We are behind on exactly four rows:
`reduce_output` (+312.1), `vocab_aux` (+59.5), `flash_score` (+39.4),
`other` (+47.2) = 458.2 us of node mass.

## 4. Where the fall-off is (testing the "starting point" intuition)

The intuition "the starting point should be the same, so the fall-off must be
before the fusion/hiding decision" is testable, and the data says the opposite:

1. **The anchor (the "starting point") is where we are fastest, not where we
   lose.** The GEMV chain with folded quant is 3477.4 us vs llama's 4138.0 -
   we are **-660.6 us ahead** on the anchor. The fall-off is not in the anchor.
2. **The fall-off is that our fusion is incomplete, and that incompleteness is
   exactly llama's absorbed work.** llama never pays `reduce_output` (it does
   the norm sumsq + normalize inside one `rms_norm_f32` kernel) and never pays
   `vocab_aux` (it does top-1 on the host from the logits buffer). We still
   emit those as separate GPU kernels: +312.1 and +59.5 us of node mass that
   llama simply does not have. That is 371.6 us of the behind-rows.
3. **The scheduler hides 1125 us of llama's work and 0 us of ours.** This is
   the single largest term: it is the +628.8 us of busy delta. llama's CUDA
   graph overlaps 1125.1 us (quantize/norm/rope/flash pipelining) behind its
   mmq anchor; our decode runs one serial NV compute queue, zero overlap
   (verified at the raw-kernel level: 0 overlapping pairs across 34 steady
   tokens, minimum gap 0.000 us).
4. **Host gap is +100.6 us.**

So: the anchor is equal-or-better on our side; the loss is (a) work llama
absorbs that we still emit (reduce_output + vocab_aux), (b) the absence of any
overlap on our side, and (c) a larger host gap. The "fall-off" is after the
fusion decision, not before it: fusion is where we win; the residuals we did
not (or could not) fuse are what we lose, and they lose harder because they sit
fully exposed with no shadow to hide them in.

## 5. The serialization counterfactual (why overlap is worth real tok/s)

If llama's kernels were serialized per token (union forced to node_sum, host
gap unchanged), its wall would grow 4058.9 -> 5183.99 us, dropping
246.38 -> **192.90 tok/s**. The 1125 us of overlap is worth ~53.5 tok/s to
llama on its own body. tinygrad is exactly serial, so our wall is our node_sum
plus host gap with no discount.

## 6. What the fresh HEAD census adds

Fresh census at `daf591ad3` (this session): **205.99 tok/s**, node_sum 4999.6
us, 596 kernels/token, 5 graph groups. node_sum > wall (~4855 us) implies
~145 us of slack may now exist after the S4 PDL exec-path landing; this is
implied by the census, not yet re-verified through the exact wall-account
identity. The same-session 08-17 pair remains the authoritative llama
comparison until a fresh llama run is done.

## 7. The bottom line

We are not slower because llama's kernels are faster. We are slower because
llama (a) absorbs work we still emit as kernels (reduce_output, vocab_aux),
(b) hides 1125 us behind its anchor while we hide 0, and (c) has a 100.6 us
smaller host gap. On raw kernel work we are 496 us ahead; on schedule we are
1125 us behind. Fusion has already won the work ledger; the gap is entirely
schedule + the two absorbed rows.
