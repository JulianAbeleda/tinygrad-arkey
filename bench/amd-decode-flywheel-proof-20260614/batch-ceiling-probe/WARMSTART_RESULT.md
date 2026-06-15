# Step 3 — warm-start injection: BLOCKED by symbolic-batch JIT (and that's the plateau's root cause)

Date: 2026-06-15. `extra/qk_decode_warmstart.py` + a gated hook in `apply_opts` (postrange.py,
`_WARMSTART_OPTS`, default-off, correctness-safe). Goal: force the loop's Step-2 schedule onto the decode
forward's FFN matmuls without BEAM, and measure the plateau drop.

## What happened
The loop's guided-best for both FFN verify shapes is **TC(-1,2,1) + UNROLL:0:4** (13.51 / 9.43 TF) — i.e.
the win is **tensor cores**, which the heuristic doesn't use here.

Forcing those onto the real forward (T=16, the actual JIT'd decode path):
| run | ms/tok | warmstart_stats |
|---|---|---|
| baseline (heuristic) | 14.23 | match 0 |
| warmstart, strict match (all-int) | 14.25 | **match 0** (no fire) |
| warmstart, match concrete dims | 14.19 | **match 4, apply 0, error 4** |

The hook MATCHES the FFN matmuls (the shape-keyed injection mechanism works), but **the loop's TC schedule
errors (`KernelOptError`) on every one** → safe fallback to heuristic → no change.

## Root cause (and why it's the plateau itself)
The forward JITs ONE matmul kernel for a SYMBOLIC batch dim (T bound to a `toks` variable, N=1..32), not
concrete N=16. **Tensor cores need concrete, aligned dims**, so a TC schedule cannot apply to a
symbolic-N kernel — it `KernelOptError`s. This is not just why the warm-start fails; it is **why the
verification matmuls run at ~2% of peak in the first place**: the heuristic also can't put TC on the
symbolic-batch kernel, so the decode-verification matmuls run WITHOUT tensor cores. The plateau is a
no-tensor-cores plateau, caused by the symbolic-batch JIT.

## The honest end-state of the realization thread
- **Lever proven** (Step 2): loop beats heuristic ~1.9× on the verification GEMMs — via TC — standalone,
  concrete shape.
- **Realization blocked** through every standard path: JITBEAM intractable (timed out tuning ~730 kernels),
  raw BEAM hangs gfx1100 (S1), and now warm-start injection — the schedule matches but TC won't apply to the
  forward's symbolic-batch matmul.
- **Newly located root cause**: the decode-verification matmuls don't use tensor cores because the batch
  dim is a symbolic JIT variable. That is the 2%-of-peak plateau.

## What would actually realize it (next, concrete)
Make the verification forward JIT a CONCRETE batch size (specialize the prefill graph at N=K_spec, e.g. 16),
so the matmul dims are concrete → TC applies → the loop's schedule (or even the heuristic's TC path) kicks
in. This is a JIT-specialization change (compile-per-batch-size), not a scheduling trick. Then the warm-start
hook (which already matches and is correctness-safe) would apply the loop's TC schedule and the ~1.9×
(likely more, since the baseline has no TC) becomes realizable.

This is the same recurring shape as the whole program: the kernel-level win is real, and e2e realization is
gated by a tinygrad framework structure — here, symbolic-batch JIT blocking tensor cores on decode
verification. The hook is committed (default-off, no change to normal decode); the next lever is batch
specialization.
