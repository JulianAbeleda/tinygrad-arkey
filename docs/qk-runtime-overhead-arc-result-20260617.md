# Arc 4 — host/runtime overhead & low-sync decode: RESULT (2026-06-17)

Goal: find/remove the highest-value host/runtime overhead in decode (faster normal decode, or spec decode with
one sync/pass). **Result: the runtime-overhead premise is REFUTED for normal decode — it is GPU-bound (host
~0%). Spec decode is still ~0.24× and the cause is a deeper two-model jit-alternation overhead, not per-token
sync.** Nothing shipped; defaults untouched.

## 1. Normal decode wall/GPU/host split (Phase 0, `bench/qk-decode-runtime-overhead/`)

Clean method (no DEBUG=2 unbatch inflation): W = real decode feeding the device output token back (`out=step(out)`),
`.item()` per token; D = same but NO per-token `.item()` (one final sync); host_sync = W − D.

| ctx | W (real) | D (no per-token sync) | host_sync | tok/s |
|---:|---:|---:|---:|---:|
| 128 | 20.35 ms | 20.41 ms | **0.0 ms (0%)** | 49.1 |
| 512 | 27.08 | 27.17 | **0% ** | 36.9 |
| 1024 (flash) | 29.62 | 29.74 | **0%** | 33.8 |
| 4096 (flash) | 51.92 | 52.09 | **0%** | 19.3 |

**W == D everywhere → the per-token `.item()` readback and host dispatch add NOTHING. Normal decode is
GPU-bound.** A first harness version showed ~38% "host" — that was entirely per-step `Tensor([[tokid]])` CREATION
(a measurement artifact); feeding the device output token back (as `model.generate` does) eliminates it. This
**refutes the banked "~55% host overhead"** (same contamination class) and confirms `model.generate` ≈ 48–54
tok/s is GPU-bound. **6 programs/token here** (the eager-step count; the jit graph-batches them — host launch is
already amortized).

## 2. Sync/readback sources
The only per-token sync is the `.item()` readback — and it costs ~0 (it waits for GPU work you pay anyway; the
jit graph batches the ~hundreds of kernels into one submission). There is **no removable host sync** in normal
decode.

## 3. Spec decode sync breakdown (Phases 1+3)
The gated spec prototype (`extra/qk_spec_decode_generate.py`, 0.6B draft) is **greedy-EXACT** but ~0.24× (4×
slower) — and the cause was isolated by elimination:
- per-step `Tensor([[cur]])` creation → fixed (feed device tensor): no recovery (0.15→0.19×).
- in-loop prefill counted in the timer → fixed (time decode-only): baseline corrected to 54 tok/s, spec still 0.24×.
- `.item()` sync → Phase 0 proved it's free; isolated draft decode (both models loaded) = **233 tok/s** (≈ the
  273 standalone) → coexistence is fine.
- **Remaining cause: the spec-loop structure — alternating the draft's rollout jit and the target's verify jit
  each pass (4 draft + 1 verify).** The draft forwards run ~10× slower *in that alternating loop* than isolated
  (179 ms/pass vs the ~37 ms the GPU model predicts). Switching between two large captured graphs per pass is a
  deeper tinygrad jit/dispatch interaction — NOT the simple per-token sync this arc targeted.

## 4. Design matrix (Phase 2, abbreviated)
Options A (remove .item) / B (batched draft, no host round-trip) / F (device token buffer) all target the
per-token `.item()` sync — which **Phase 0 proved is already ~free**, so they cannot help normal decode and do
not address the spec slowness (which is jit-alternation, not .item). Option C (on-device accept) and D
(dual-graph) reduce host reads but don't fix the two-graph-alternation dispatch overhead. **No small option
targets the measured cause.**

## 5–6. Prototype attempted & before/after
Smallest option tried: device-token-feed in the draft loop (kills per-step creation) + decode-only timing. Spec
0.15→0.24×, still 4× below the gate. **Refuted** (≥1.2× not reached). Not shipped; defaults unchanged. (Greedy
exactness was confirmed True before the timing refactor offset the comparison.)

## 7. Is runtime overhead the primary 8B blocker? **NO.**
Normal decode is **GPU-bound** (host ~0%, W==D, jit graph-batches). The "structural runtime wall" premise is
refuted: the 8B gap vs llama is GPU-work (kernel efficiency + program granularity that's GPU-side), not host
overhead. Spec decode *could* help algorithmically (acceptance 2.84/pass, exact), but its in-tinygrad execution
hits a two-model jit-alternation overhead that's a deeper dispatch investigation, not a bounded local fix.

## 8. Exact next step
Every bounded 8B decode lever is now exhausted or refuted: sub4 (quality), big-copy (artifact), small-op fusion
(<3%), GEMV final-mile / Q4K_FUSE (−18%), ring2 (HBM), flash-decode (shipped, long-ctx), spec decode (exact but
runtime-bound), and now **runtime overhead (normal decode is GPU-bound — not the lever)**. The 8B gap is
GPU-kernel-structural with no remaining bounded local fix. **Recommendation: move to 14B** (GPU-dominated, where
the competitive GEMV bandwidth matters most and per-token fixed overheads are a smaller fraction) — or a
dedicated codegen/kernel arc (high risk) if staying on 8B. Spec decode would need a two-model low-overhead
dispatch path (deep runtime work) to realize its algorithmic win.
