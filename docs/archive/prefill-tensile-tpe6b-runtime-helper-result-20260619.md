# TPE-6b RESULT — runtime helper: matmul batching isn't the lever; full-forward TinyJit capture is (PASS, ~1.74× projected)

Executed TPE-6b (`prefill-tensile-tpe6b-runtime-helper-scope-20260619.md`): build the single-dispatch batch primitive
and locate the real source of the TPE-6 host overhead. **Verdict: PASS** — the graph-integrated FFN block projects to
**~1.74× the PREFILL_V2 plateau**, clearing the 1.20× gate, **but the lever is full-forward TinyJit graph capture, not
matmul batching.** This sharpens (and upgrades) the TPE-6 REDIRECT. Probe: `extra/qk_tensile_block_graph.py`; artifact:
`bench/qk-tensile-extraction/block_graph.json`. No model route, no defaults, decode untouched.

## What RH-1 found — matmul batching is NOT the lever [M]
A single AMDComputeQueue with all 3 Tensile execs (one submit/wait) vs 3 separate `wait=True` submits:

| | ms | vs device-sum |
|---|---:|---:|
| 3-matmul device-time sum (HCQ signals) | 2.18 | 1.00× |
| 3 matmuls, per-kernel `wait=True` (naive) | 2.79 | 1.28× |
| 3 matmuls, **one submit/wait (batched)** | 2.56 | 1.17× |

**Batching saved only 0.23 ms.** The per-kernel sync was *not* the TPE-6 lever — the matmul dispatch already carries
near-zero host overhead.

## What the TPE-6 6.2 ms overhead actually was [M]
The TPE-6 naive block wall (8.69 ms) − GPU matmul (2.53 ms) = 6.2 ms. RH-2 isolates it: the per-op **tinygrad
`.realize()` of the transpose and SiLU·mul** cost ~1.0 ms and ~1.45 ms respectively — but their *GPU* work is trivial
(bandwidth-bound: a 12.6 MB transpose ≈ 26 µs, the SiLU·mul ≈ 40 µs at ~960 GB/s). So **~95% of each is tinygrad
per-realize host scheduling**, not GPU time. The overhead was Python/realize dispatch on tiny elementwise ops, which
**only a single forward graph (TinyJit) eliminates** — exactly what the real model uses and the standalone probe
cannot replicate (tinygrad's UOp-`Ops.PROGRAM` elementwise kernels can't be cheaply re-enqueued into a manual queue).

## RH-2 — graph-integrated block projection [I]
Single forward dispatch (no per-realize host cost): matmul device time + bandwidth-bound elementwise.

| | ms |
|---|---:|
| routed matmul (3 Tensile kernels, device sum) | 2.18 |
| + bandwidth-bound elementwise (2 transpose + SiLU·mul) | +0.09 |
| **projected routed block** | **2.27** |
| PREFILL_V2-plateau (40 TFLOPS) matmul + same elementwise | 3.96 |
| **projected block speedup** | **1.74×** (matmul-only 1.77×) |

The elementwise is shared between both sides and negligible at GPU bandwidth, so the block speedup is matmul-dominated
and lands ~1.74× — higher than the TPE-6 figure (1.53×) because the device-sum matmul time (2.18 ms) is cleaner than
TPE-6's per-kernel-synced 2.53 ms.

## Gates → PASS
- RH-1 (matmul batches cheap: batched ≤ 1.3× device-sum) ✓ (1.17×).
- RH-2 (projected single-dispatch block ≥ 1.20×) ✓ (1.74×).
- Caveat: RH-2 is a projection (bandwidth-bound elementwise assumption); the actual end-to-end proof requires in-model
  TinyJit capture (RH-3 / TPE-7), since the standalone probe can't capture tinygrad's elementwise into one dispatch.

## Verdict + next step
**TPE-6b PASS → the runtime helper to build is an in-model TinyJit-captured node, not a standalone batch queue.**
The decisive learning: matmul single-submit batching is a red herring (0.23 ms); the entire reachable win comes from
running the whole FFN block inside the model's existing single forward graph, where the Tensile launch joins the
tinygrad transpose/SiLU·mul ops with no per-realize host scheduling. Concretely for **RH-3 / TPE-7**: make
`NamedAMDProgram`'s launch a node the PREFILL_V2 forward's TinyJit/HCQGraph captures (a custom op or a JIT-cache
ExecItem wrapping the precompiled kernel), behind a research flag, on PREFILL_V2 fp16 weights; then measure warm
pp512 + dNLL. Expected ~1.74× on the FFN matmul bucket → consistent with the TPE-5 weighted ~1.40× full pp (~95% of
llama). Still no model default; external-artifact policy remains a separate decision. KILL only if TinyJit capture of
the precompiled kernel proves to carry irreducible per-call overhead.

## Files
`extra/qk_tensile_block_graph.py`, `bench/qk-tensile-extraction/block_graph.json`, this doc, scope
`prefill-tensile-tpe6b-runtime-helper-scope-20260619.md`. Reuses `qk_tensile_hcq_launch.py` + `kernarg_all.jsonl`.
No kernel/model/default changes; no runtime files modified.
