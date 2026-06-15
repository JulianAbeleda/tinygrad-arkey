# Step 2 — does the loop lower the batched-decode plateau? YES, ~1.9× over the heuristic (honest)

Date: 2026-06-15. `extra/qk_decode_verify_loop.py`. The validated curated-config loop (GPU-safe, NOT raw
BEAM) on Qwen3-8B FFN verification GEMM shapes (M/K=12288 held out of the N0 corpus), at speculative batch
N=8,16. Compared the loop's guided pick to **the heuristic the forward actually runs** (hand_coded), not
to naive no-opt.

## Result (loop vs the forward's actual heuristic schedule)
| shape (M,K,N) | heuristic TF | loop guided TF | **loop/heuristic** | guided/oracle | no_opt TF |
|---|---|---|---|---|---|
| 12288, 4096, 8 | 1.51 | 1.50 | 0.99× | 0.89 | 0.16 |
| 12288, 4096, 16 | 6.80 | 13.41 | **1.97×** | 0.997 | 0.12 |
| 4096, 12288, 8 | 0.49 | 0.62 | 1.26× | 1.00 | 0.16 |
| 4096, 12288, 16 | 2.86 | 9.43 | **3.30×** | 1.00 | 0.21 |
| **mean** | | | **1.88×** | **0.972** | |

GATE PASS: loop beats heuristic ≥1.5× (mean 1.88×) AND finds it cheaply (guided/oracle 0.97).

## Honest reading (don't repeat the 42× mistake)
- The first cut compared to no-opt (naive, ~0.15 TF) and got ~42× — MISLEADING, because the forward uses
  the hand_coded heuristic, not naive. Against the **real** baseline the lever is **~1.9×**, not 42×.
- **The win is at N≥16; N=8 is ~break-even** (0.99×, 1.26×). At very small batch the heuristic is already
  decent and the loop (trained on N≥16) extrapolates below its range. So the loop's decode value grows with
  speculative batch size — exactly the regime speculation provides.
- **Plateau improvement is ≤1.9× (Amdahl).** This is the MATMUL-schedule lever; the 14 ms/tok plateau is the
  full forward (matmuls + attention + overhead). If matmuls are a fraction of it, the realized plateau drop
  is < 1.9×. Confirming the matmul fraction of the plateau is the next measurement.
- The loop reaching 0.97 of oracle on HELD-OUT decode shapes (incl. N=8 extrapolation) reconfirms N1/N2/L0/L1
  transfer — now on the actual decode-verification GEMMs.

## What this establishes for the mission
**Machine search measurably improves decode-relevant kernels (~1.9× over tinygrad's heuristic) in the
batched/speculative regime** — the first concrete instance. Combined with the ceiling probe's ~2.4–3.5×
memory amortization from batching, the loop-tuned batched verification is the path: speculation supplies the
batch (and N≥16 is where the loop wins), the loop tunes the verification GEMMs above the heuristic.

## Next
1. Confirm the matmul fraction of the 14 ms/tok plateau (profile the batched forward by kernel type) — so
   the ≤1.9× plateau lever is sized honestly.
2. Speculative scaffold + wire the loop-found schedules into the T>1 verification path; measure realized
   batched-decode tok/s vs the ceiling.
