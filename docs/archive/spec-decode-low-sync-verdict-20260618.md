# Low-sync speculative decode — VERDICT (Phases 3-8): CORRECT + low-sync PROVEN; production speed needs efficient-loop integration 2026-06-18

The arc's hard technical parts are DONE and correct. The production-speed gate is **unresolved by the standalone
harness** (it is host-overhead-bound), but the algorithm, low-sync structure, reusable graph, and greedy-exactness
are proven. Default decode untouched; nothing routed.

## What's proven
- **Phase 3 — device-token feedback:** a custom TinyJit unrolling K draft steps with argmax fed on-device (no
  `.item()`) is byte-exact to host-stepped draft greedy. ✓
- **Phase 4 — reusable proposal graph:** K distinct symbolic start_pos vars (sp_i=base+i, rebound/pass) → byte-
  exact across L=8/12/6, ONE sync, NO recompile (flat 20-35ms across rebinds 7→300). ✓
  (`extra/qk_spec_decode_lowsync_probe.py`)
- **Phases 5-7 — integrated loop (`extra/qk_spec_decode_lowsync.py`):** draft propose graph + target verify graph
  (T=K+1, one pass) + host accept + KV self-correction (both caches re-process from the corrected position;
  draft full-accept hole closed by a (K+1)-th cache-only forward). **GREEDY BYTE-IDENTICAL on every prompt.** ✓
  **2 syncs/pass** (proposal realize + verify realize) vs naive K+1. accept ~2.1-2.8/pass (consistent w/ the gate).

## What's NOT resolved — production speed
The standalone harness measures **baseline 9.4 tok/s and spec 12.5 tok/s (1.33×, greedy-exact)** — but this is a
**host-overhead-bound regime**: the baseline stays 9.4 even after forcing MCLK 96→1249MHz, so the GPU is not the
bottleneck; per-token Tensor creation + `.item()` + python loop dominate. The banked production decode (~55-68
tok/s) comes from the cli's host-efficient loop (GPU-bound). So:
- The **1.33×** reflects spec doing fewer host-bound passes/token — real *in this regime*, but not the production
  number.
- In the GPU-bound production regime the verdict depends on spec's per-pass host overhead (2 syncs + accept)
  vs the target-passes it saves (~2.1/pass). Estimate (GPU-bound, host-light): ~1.3-1.4× (≈92 tok/s if a pass ≈
  verify 1 target-pass + draft 4×0.6B ≈ 22ms for ~2.1 tokens). Estimate (host-heavy): could erode to <1×.
- **The clock-ramp confound bit hard:** all absolute numbers were ~1/10th until MCLK was forced (was stuck at
  96MHz idle). Even then the harness is host-bound. (Memory `amd-decode-measurement-confounds`.)

## Verdict: A-pending (correct + low-sync proven; confirm production speed before ship)
- **Greedy byte-identical: YES.** Stable. Low-sync (2/pass) achieved. Reusable graph, no recompile. KV protocol
  correct (zero/partial/full accept all greedy-exact across prompts).
- **Speed gate: 1.33× in the host-bound harness** (≥1.2× met there), but the **production gate (vs the GPU-bound
  cli decode) is unproven** — the standalone harness can't measure it.
- **Recommendation:** the remaining work is wiring spec into the cli/`model.generate` host-efficient loop (Phase 8
  proper) and measuring with `--warmup` under sustained load (so MCLK ramps), behind `SPEC_DECODE=1` (default
  off). Only flip the default if that confirms ≥1.2× greedy-exact at full clock. Do NOT ship on the host-bound
  1.33× alone.

## Remaining bottlenecks
1. Per-pass host overhead (2 syncs + tolist + python accept) — fine in GPU-bound, fatal in host-bound. Move accept
   on-device + reduce per-pass Tensor creation (Phase 9).
2. Production-loop integration (the harness ≠ the cli's efficient loop).
3. GPU clock ramp must be ensured for any benchmark (force perf=high or sustained load).

## Files
`[test]` `extra/qk_spec_decode_lowsync.py` (integrated loop), `extra/qk_spec_decode_lowsync_probe.py` (proposal
graph); `[docs]` arc + phase4 + this verdict; `bench/qk-spec-decode-low-sync/baseline.json`. No kernel/model/
default changes.
