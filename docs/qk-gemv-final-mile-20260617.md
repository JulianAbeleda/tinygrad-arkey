# GEMV final-mile audit (Arc 3) — REFUTED: no local GEMV fix improves decode (2026-06-17)

Arc 3 hypothesis: in-model GEMVs may not hit the standalone ~76% HBM, and a local policy/storage/layout fix per
role could win +5–15% decode. **Verdict: REFUTED.** The GEMV path is competitive; per-role in-model bandwidth
can't be cleanly isolated (measurement confounds); and the one obvious local lever — horizontal GEMV fusion
(`Q4K_FUSE`) — is broken on prefill and **18% SLOWER on decode**.

## Phase 0/1 — per-role efficiency (measurement-confounded)

In-model per-role GEMV bandwidth is **not cleanly measurable**: the batched JIT graph hides per-kernel timing,
and DEBUG=2 unbatches kernels → per-kernel tm inflated (eager step gives 9–33% "peak" — implausible for
HBM-bound GEMVs; lm_head "9.5%" reading 510 MB is an artifact). Standalone re-benches are also confounded
(wall = host-dominated ~5–8 Q4-GB/s; device timing needs DEBUG=2 which inflates). With the actual decode policy
(serial, `LOCAL:0:64`, 1 kernel, iters=200, device-timed): ffn_gate **357 Q4-GB/s** vs attn_q **169** — the
*smaller* GEMVs show ~2× lower effective bandwidth (fixed per-kernel overhead dominates small batch-1 reads).
Both are below the banked standalone 76%, but the gap is measurement overhead (short single-token reads), not a
confirmed primitive deficiency. `bench/qk-gemv-role-efficiency/`.

## Phase 4/5 — the local fix (Q4K_FUSE) is REFUTED

The relative signal (small GEMVs overhead-bound) pointed to the obvious fix: **horizontal fusion** — `Q4K_FUSE`
fuses q/k/v→attn_qkv and gate/up→ffn_gateup (one GEMV over concatenated weight rows), already implemented.
Tested (`extra/qk_fuse_decode_probe.py`, decode-only):

| config | programs/token | decode tok/s |
|---|---:|---:|
| baseline | 780 | 55.42 |
| Q4K_FUSE=1 | 744 | **45.20 (−18%)** |

So fusion **reduces kernel count but makes decode 18% slower** — the fused, wider-concatenated GEMV is *less*
efficient at batch-1 (the per-kernel overhead wasn't the bottleneck; the fused shape tiles/occupies worse). It
also **crashes on a T>32 prefill** (the fused linear has no `.weight` fallback: `None.transpose()`). Both ways,
not a win.

## Decision (Arc 3 kill conditions)

- "no role-level fix gives ≥5% e2e" → met (the only available local fix is **−18%**).
- "in-model GEMVs already match standalone efficiency" → the GEMV primitive is competitive (banked 76%
  standalone); the in-model per-role gap can't be cleanly established, and the fusion lever refutes the
  overhead hypothesis.

**REFUTED. No local GEMV change improves decode.** The GEMV final-mile is exhausted. Q4K_FUSE left default-off
(broken + slower); not worth fixing the prefill fallback given the −18% decode result.

## Where this leaves the 8B short-decode arcs

GEMV final-mile (Arc 3) joins the exhausted/refuted list. The residual short-8B gap (54–64 vs llama ~80–100) is
confirmed **structural** (program granularity + competitive-not-faster GEMV + attention reduce shape), with no
remaining bounded local lever. Of the five arcs, the GEMV (most bounded) is now refuted; the rest are
attention-reduce-codegen (high risk), decode-block-fusion (very high, compiler-arch), host/runtime-graph
(medium-high), and speculative decoding (algorithmic, different axis — the only one with large upside left).
Recommendation: stop chasing kernel-level 8B decode; if pursuing, **speculative decoding** is the next real
upside, else move to 14B.
