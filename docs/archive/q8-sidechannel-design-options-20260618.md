# q8 side-channel — design options (Phase 2) 2026-06-18

From the producer audit: the only ≤4.8µs path is folding q8 into the producer's own data pass. Options:

## A. Custom fused RMSNorm+q8 kernel (the real side-channel)
A hand-written custom kernel replacing `nn.RMSNorm` for the FFN-norm: one pass — reduce mean(x²) → produce
normalized*weight (write fp) → per-32 max inline → quant+pack (write qpacked + scales).
- **expected cost:** norm already = 2 kernels / 19.4µs; a 1-kernel fused norm+q8 could be ~12-20µs total, so the
  q8 *effective* cost (fused − norm-only) could be **~0-5µs** (the fusion may even absorb the norm's 2→1 kernel).
  Plausibly ≤4.8µs.
- **code changes:** new custom kernel; swap `ffn_norm` to emit (fp, qpacked, scales); thread q8 to gate/up;
  must handle **decode (T=1) AND prefill (T>1)**, the residual fp path, and an fp fallback. Multi-output +
  two-granularity reduction (mean over 4096 + max over 32) in one kernel — non-trivial plumbing
  (multi-store/multi-output has repeatedly fought custom_kernel).
- **quality:** q8-lossy (rel 0.006) → dNLL gate.
- **complexity:** HIGH (replaces a hot, shared op).
- **≤4.8µs?** plausibly yes (unproven; the probe was not built — see verdict).
- **preserves fp fallback?** yes (emits fp too).

## B. Specialized post-RMSNorm fused pack kernel (1 kernel, not folded into norm)
One custom kernel after the norm: per-block max + quant + pack.
- **cost:** ~12µs (measured fused-quant-pack floor; redundant max) → **> 4.8µs**.
- **changes:** modest (a new pack kernel, no norm surgery).
- **≤4.8µs?** NO. Fails break-even. (This is the lifecycle-probe result.)

## C. Graph side-cache (pure Tensor ops, explicit side tensor)
Compute `xq8 = pack(ffn_norm(h))` once as a side tensor, feed gate/up.
- **cost:** the pure-graph pack does **not fuse** → 4 kernels / 29.7µs (proven). TinyJit already commons it
  across gate/up; manual `.realize()` doesn't help.
- **≤4.8µs?** NO. This is the already-refuted current state (0.96× coop).

## D. Model-transform (SmoothQuant/Atom-style)
Rescale activations/weights offline so q8 is cheaper/needs less correction.
- **cost:** large separate arc (calibration, weight rewrite, GGUF format, quality). Not a side-channel.
- **≤4.8µs?** changes the premise; out of scope here.

## Summary
Only **A** can hit the cost target, and only as a hand-written fused custom norm kernel (a pure-graph side-channel
B/C cannot fuse). A is feasible-in-principle but a deep, hot-path, multi-output, dual-shape kernel build for a
lossy ~+3-4% decode gain. See `q8-sidechannel-ffn-verdict-20260618.md`.
