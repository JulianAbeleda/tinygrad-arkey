# MMVQ fused cooperative-row — VERDICT: B/D (ceiling ~53-54%, fails gate) — 2026-06-18

Per `qk-mmvq-fused-coop-row-arc-20260618.md`.

## 1. Chosen design
Design A (manual LDS in-kernel reduction) — after `OptOps.GROUP` (the optimizer's automatic cooperative reduce)
was found BROKEN on custom kernels (err 0.95).

## 2. Correctness
GROUP variant: WRONG (err 0.95 — drops work). Manual-LDS variant: built up to the cross-lane single-output write,
which hit a custom_kernel `UNROLL-on-STORE` plumbing limit; not completed. The serial (no-coop) reduce is correct
(err 2e-6) but 3% peak.

## 3. Source proof
GROUP emits an LDS reduce but mis-scheduled (wrong). The work-ceiling was measured directly instead (partial
kernel alone, no stage-2).

## 4. Microkernel speed (% HBM peak)

| variant | % peak |
|---|---|
| base fp | 40 |
| fp coop (+ stage-2) | 48 |
| **fp coop partial ALONE** | **53** |
| `_sdot4` | 49 |
| opaque asm | 52 |
| **fused quadrant ceiling (measured)** | **~53-54** |
| llama / READRAW | 70 |

## 5. Full-linear estimate
Fused best case ~53-54%. Gate needs: ≥1.3× base (40→53 = 1.33× ✓), **≥1.15× fp coop (= 55.2%) ✗**, **≥1.05×
opaque (= 54.6%) ✗**. Fails by ~1-2 points.

## 6. vs 52% opaque and 70% llama
The fused quadrant edges opaque (52→~53-54%) by recovering the stage-2, but the **partial-alone work-ceiling
(53%) is essentially the opaque level** — the dequant ALU wall. The 53→70% gap is unchanged (coalesced-AND-
register-tight scheduling tinygrad can't thread).

## 7. Model route earned? NO
Fails the full-linear gate (≥1.15× coop / ≥1.05× opaque). Do not route.

## Verdict: B/D

**B** (in-kernel reduction's benefit ≈ the stage-2 it removes; gets to the 53% work-ceiling, no further) **+ D**
(the residual 53→70% is the coalesced-AND-register-tight backend-scheduling wall, confirmed again). The missing
quadrant was the right thing to test — and it **did** reveal that the coalesced dequant *work* alone is 53%
(stage-2 was a real 10% the 48% number hid) — but recovering the stage-2 only reaches ~53-54%, short of the gate
and far from llama 70%.

## What this leaves
The four MMVQ quadrants are now all tested:
- register-tight/uncoalesced (base fp): 40%
- coalesced + partials + stage-2 (fp coop): 48%
- coalesced + native dot4: 49%
- coalesced + in-kernel reduce + one write (this arc): ceiling ~53-54%

All cap at ≤~53-54%; opaque hand-asm 52%; llama 70%. The **53% work-ceiling** is the coalesced Q4_K dequant ALU
limit under tinygrad's custom_kernel codegen; closing 53→70% needs the coalesced-AND-register-tight inner loop
that requires register-allocator/scheduler work (not a kernel-structure transform). The MMVQ research space's
*structural* quadrants are exhausted; remaining is true backend/codegen-internals work.

## Recommendation
Next steps within the current target (per standing preference, not pivoting): either (a) accept the 8B MMVQ
ceiling and bank the durable wins (`_sdot4` helper, the quadrant map, the 53% work-ceiling finding), or (b) a
deep register-allocator/scheduler investment (very high risk, the only path to 53→70%). The structural-kernel
research is complete; (b) is framework-internals, not kernel research.

## Files / commits
`[docs]` arc + this verdict; `[test]` `bench/qk-mmvq-fused-coop-row/baseline.json` (stage-2 decomposition +
quadrant ceiling). No `[codegen]` (GROUP broken; manual-LDS not completed — transient probes), no `[nn]`, no
defaults.
