# Audit — which deferrals leaned on the (now-refuted) "BEAM hangs gfx1100" premise, and are any now reachable?

The Route-B spike refuted the long-standing "BEAM hangs gfx1100" wall (BEAM=2/4/8 all complete on matmul + prefill
shapes) but found BEAM **underperforms** (14–17 TFLOPS prefill matmul, below the default 17.7 and the warmstart 48,
vs Tensile 66). This audits every deferral that cited BEAM-hang, against both findings.

## Is BEAM the replacement for Tensile? — NO [M]
Measured prefill matmul (512×12288←4096 fp16), warm, fair:
| schedule | TFLOPS |
|---|---:|
| BEAM=2 / 4 / 8 | 14.7 / 17.1 / 14.3 |
| default heuristic (BEAM=0) | 17.7 |
| tinygrad warmstart (PREFILL_V2) | ~48 |
| **Tensile (rocBLAS)** | **66** |
BEAM doesn't reach even the warmstart, never mind Tensile — and it picks *worse* kernels than the default (test-size
mis-ranking + TC not applied effectively; `clear_l2` + higher width didn't help). **BEAM is not the Tensile
replacement and not the dependency-free prefill lever.**

## Deferrals that cited "BEAM hangs" — corrected status
| deferral | what it claimed | corrected status |
|---|---|---|
| **prefill LDS-tiling** (`prefill-wmma-lds-tiling`) | "the lever; BEAM would find it but hangs" | **premise FALSE + lever INDEPENDENTLY REFUTED** (PWLT-A2: hand-LDS = 1.02× default, IC-served). No win either way. |
| **decode/spec batched-K LDS-tiled GEMM** (`8b-decode-remaining-gap`) | "BEAM finds the GROUP/LOCAL-into-LDS opt but hangs" | premise FALSE; spec-verify closed (distributed T-scaling); LDS-tiling IC-served (PWLT-A2). No win. |
| **prefill software-pipeline** (`prefill-codegen-pipeline-redo`) | "project-level / BEAM-hang class" | premise FALSE + INDEPENDENTLY REFUTED (CG-R1: pipeline expressible but IC-served, no gain). No win. |
| **decode hand-coded opts** (`amd-decode-fix-plan`) | "targeted stand-in for BEAM (which hangs)" | premise FALSE, but the hand-coded opts were the RIGHT choice anyway — BEAM underperforms them (14–17 < shipped). No regret. |
| **dp4a search-reachability** (`amd-decode-dp4a-vocabulary`) | "add a search action so BEAM can choose it" | premise FALSE, but dp4a/int-dot whole-linear was refuted on the q8-pack wall regardless; BEAM wouldn't rescue it. |
| **loop cost-model → live BEAM** (`amd-decode-final-report`/`loop-live-plan`) | "wire the cost model into BEAM warm-start" | premise FALSE; but the cost model is out-of-distribution on native-BEAM partial schedules, and BEAM itself underperforms — not a win. |

## The honest conclusion
**The "BEAM hangs gfx1100" premise was FALSE in ~6 places — a real project-wide correction** (it was never
root-caused; it was an overgeneralized assumption). **But NO deferral becomes a new win**, because in every case the
lever was *either*:
- **independently refuted** on its own merits (LDS-tiling and software-pipeline are Infinity-Cache-served on gfx1100,
  proven by hand-kernel probes PWLT-A2 / CG-R1, not by BEAM), *or*
- **not rescued by BEAM** anyway (BEAM is too weak — 14–17 TFLOPS, below the warmstart — so even a reachable BEAM
  wouldn't have found the schedule).

So the deferrals' **conclusions all stand**; only their stated *reason* ("BEAM hangs") was wrong. The corrected
reason is "the lever is IC-served / refuted" or "BEAM is too weak to find it."

## The one genuinely-new lead (not a win yet)
BEAM **under-ranks** good schedules (it chose a 14.3 kernel as "best", below the 17.7 default) — a real BEAM *quality*
bug on gfx1100 (suspected `allow_test_size` mis-ranking + ineffective TC search). Fixing BEAM's matmul search could
make it a general tuning lever — but:
- even a *correct* BEAM would target the warmstart level (~48), not Tensile (66) — the 48→66 gap is the
  copy-vectorization (Route A, hand-coded), which BEAM's action space (UPCAST/TC/etc.) *might* reach only if the copy
  axis is searchable AND the ranking is fixed;
- so "fix BEAM quality" is a speculative, separate arc — it does not change the current verdict.

## Verdict
- **BEAM is not the Tensile replacement** (14–17 vs 66) and not the dependency-free prefill lever.
- **The BEAM-hang wall is refuted** — a useful correction to the project's mental model; future search-based work
  should not assume BEAM hangs (it doesn't on these shapes) but should expect it to *underperform* the warmstart.
- **No deferred lever is newly unlocked** — every one is independently refuted or BEAM-unreachable.
- The dependency-free prefill lever remains **Route A (hand copy-vectorization, project-level)**; the only validated
  speedup remains the **external Tensile route (1.41× llama, dependency)**; rest state **PREFILL_V2 (~80% llama)**.

## Provenance
Route-B spike: `bench/qk-codegen-wmma/inmodel_matmul.json` (CG_W3_routeB_beam_spike); refuted-lever probes:
`prefill-wmma-lds-tiling-result` (PWLT-A2), `prefill-codegen-pipeline-redo-result` (CG-R1). No code/default changes
(audit + framing).
