# Prefill: the Tensile "86% of llama" SETTLED — it was a nosync artifact - 2026-06-20

Verdict: `TENSILE_86_WAS_NOSYNC`

Run:

```bash
DEV=AMD PYTHONPATH=. python3 extra/qk_prefill_tensile_settle.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
```

Baseline WMMA, the dependency-free graph route, and the external Tensile `.co` route, measured on ONE synced
axis (arbiter: K forwards / one `dev.synchronize()` / total/K), each in a fresh subprocess, plus the nosync
number to show which one the historical 86% was.

| path | synced ms/512 | synced tok/s | **% llama (synced)** | nosync tok/s | "% llama (nosync)" |
|---|---:|---:|---:|---:|---:|
| baseline WMMA | 414.2 | 1236 | **40.9%** | 24717 | 818% |
| Tensile `.co` | 276.0 | 1855 | **61.4%** | 25102 | 831% |
| **graph route (ours)** | 256.3 | 1998 | **66.2%** | 25715 | 851% |
| llama pp512 | 170 | 3020 | 100% | — | — |

## Conclusion

1. **The Tensile 86% was a NOSYNC artifact.** Synced, the Tensile route is **61% of llama**, not 86%. The
   historical "~86–87% llama" came from an unsynchronized A/B harness (best-of-N on `realize()` catches the
   fastest host dispatch). The nosync column here is self-evidently impossible (818–851% of llama ≈ ~400
   TFLOPS), confirming nosync measures host dispatch, not GPU compute.
2. **Nothing reaches 86% synced** — not Tensile, not us. The entire "80–95% of llama" prefill family (the
   PREFILL_V2 "93%", the Tensile "86%") was the same nosync mirage.
3. **The dependency-free graph route (66%) is the best prefill path** — it beats both the WMMA baseline (41%)
   and the external Tensile `.co` (61%) on a synced axis. So the Tensile `.co` dependency is **not needed**:
   our kernel wins in-model too, consistent with it beating Tensile in isolation (74 vs 71 TFLOPS gold-standard).

(The graph 256 ms is the stable Gate-1 figure across 3 sessions; Tensile 276 ms is single-shot — the ~8%
graph-over-Tensile margin is within plausible noise, but Tensile is unambiguously ~61%, nowhere near 86%, and
not ahead of our route.)

## What this leaves

The honest, verified prefill ladder is **41% (baseline) → 61% (Tensile, dependency) → 66% (our promoted
dependency-free route)**. To actually approach llama (100%) the remaining levers are real GPU-compute
reductions — int8-quantized GEMM (llama's bandwidth trick) and tighter attention/in-model integration — not a
measurement fix and not the Tensile `.co`.
