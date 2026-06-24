# Small-Ops / Activation Fusion — Scope (2026-06-23)

## Verdict: `SMALL_OPS_FUSION_SCOPE_READY` (fallback lane; lower confidence than runtime-KV)

Runtime-KV (the biggest lever) is core-runtime-blocked. The next **bounded** lane is fusing the unfused small-op /
activation kernels that the post-default audit found tinygrad emits where llama fuses. **Caveat up front**: these
are heavily **overlapped** (tinygrad GPU-busy 13.7ms ≫ wall 11.7ms), so the WALL transfer of any fusion is
**uncertain** — this scope must prove transfer on ONE fusion before any broad work.

## Mission
Reduce the residual decode gap to llama (~12–15%) by fusing small-op/activation kernel groups, **starting from
rendered-source evidence** (not stale bucket labels), and only if a single fusion clears ≥1–2% W==D.

## Corrected bucket map (post-default audit, rendered kernels, ctx1024 GPU-busy)
- **FFN activation (silu/gate)**: ~1.5ms — tinygrad emits a separate activation kernel; llama fuses silu into the
  GEMV epilogue (+ q8 quant). Gap vs llama ~+1.0ms.
- **genuine norm/rope/residual + small reduces**: ~2.3ms (`r_1024_16_4_2_32`, `r_16_256`, `r_2_8_128…` KV-proj
  reduce, rope/norm). llama fuses (rmsnorm 613 + rope 365 + residual 84 µs). Gap ~+1.2ms.
- NOT in scope: KV materialization (E_49152, runtime-KV/core-blocked), attention (parity), weight-GEMV (parity).

## Candidate fusions (by rendered-source fingerprint, verify first)
1. **silu(gate) * up → FFN-GEMV epilogue** (the activation kernel). Most llama-like; clearest.
2. **RMSNorm + residual-add** fusion.
3. **RoPE into the q/k projection epilogue** (if rendered as a separate kernel).
4. KV-proj small reduces (`r_2_8_128…`) — verify whether these are genuine or mislabeled before touching.

## Lifecycle classification
`ISA_CODEGEN_GAP` — tinygrad's scheduler emits many small unfused kernels; the fix is fusion/codegen (learn from
llama's epilogue fusion), not a new hand-kernel.

## First bounded gate (do this before anything broad)
1. Render the exact kernels for ONE candidate (silu/gate) — confirm it is a separate kernel group, not overlapped
   into the GEMV already.
2. Fuse it (tinygrad-native, e.g. express the activation in the GEMV's output expression so the scheduler fuses).
3. Token correctness: byte-identical to default for ≥64 tokens, multi-prompt.
4. **W==D ≥1–2%** at ctx1024/4096 (wall, `.item()` in window, repeated). If the fusion removes a kernel but does
   NOT move wall (overlap), classify `SMALL_OPS_FUSION_NO_WD_TRANSFER` and stop.

## Stop rules
- No broad codegen rewrite before the single-fusion W==D gate.
- No stale bucket labels — rendered-source evidence only.
- No local-only success — W==D wall is authority.
- If the first fusion shows no wall transfer (overlap), declare small-ops a non-lever and the 8B bounded space
  exhausted.

## Boundaries
No attention/GEMV/runtime-KV work; no default flip; no 14B/32B; no new hand-kernels.
