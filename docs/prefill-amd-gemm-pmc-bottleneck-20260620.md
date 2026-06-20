# Prefill AMD GEMM — PMC Bottleneck Classification

Date: 2026-06-20

## Purpose

The occupancy diagnostic showed BK32 is contention-limited (interior optimum at wg2) but couldn't name the
contended resource. This **names it with hardware counters**. tinygrad's native PMC (`PMC=1`) *does*
instrument our hand-asm `run_linear`/HCQ dispatches (confirmed — rocprofv3 can't, but the native path can), so
we captured per-kernel counters for the same BK32 kernel at wg4 vs wg2, plus BK16 and the LLVM authority.

Probe: `extra/qk_amd_gemm_pmc_probe.py` → `bench/.../amd_gemm_pmc_result.json`.
Run: `DEV=AMD PMC=1 PROFILE=1 PYTHONPATH=. python3 extra/qk_amd_gemm_pmc_probe.py`.

## Counters (normalized by GRBM_GUI_ACTIVE = wall cycles; reproduced 3×)

| config (TFLOPS) | cycles(M) | L2 hit% | VALU/act | busy/act | LDSact/act | **bankcf/act** |
|---|---:|---:|---:|---:|---:|---:|
| bk32 **wg4** (~49) | 3.4 | **56.0** | 2.5 | 31.8 | 31.3 | 22.1 |
| bk32 **wg2** (~57.7) | 2.4 | **63.4** | 3.6 | 37.0 | 44.6 | **31.5** |
| bk16 (~42) | 5.3 | 67.6 | 1.6 | 37.9 | 10.6 | **4.7** |
| authority_llvm (global-direct) | 3.6 | 57.7 | 9.1 | 36.8 | **0.0** | **0.0** |

PMC counters are perturbing and instance-summed (per the measured-map discipline), so the trustworthy signals
are **ratios and wg4-vs-wg2 / vs-BK16 deltas**, not absolute rates.

## Finding 1 — the high-occupancy (wg4) penalty is L2 cache contention

wg4 vs wg2 (same kernel, same work): wg4 takes **1.43× the cycles**, with **L2 hit 56% vs 64% (−7.4 pts)** and
lower per-cycle compute (`busy/act` 0.86×, `VALU/act` 0.70×). Four workgroups/CU **thrash L2**; at wg2 (two
workgroups) the working set stays hotter and every per-cycle rate rises. This *names* the occupancy curve's
interior optimum: high occupancy → L2 contention, not a lack of waves to hide latency. (It also refutes the
earlier audit's latency-via-occupancy guess, now with a counter.)

## Finding 2 — what bounds wg2 (the residual to Tensile) is LDS bank conflicts

The LDS family carries **huge LDS bank conflicts**: BK32 wg2 = **31.5/cycle**, **6.7× BK16's 4.7**, vs the
global-direct authority's **0** (it uses no LDS). The LDS unit is near-saturated (`LDSact/act` 44.6). So the
LDS-staged family's throughput is **LDS-bank-conflict bound** — the WMMA-fragment reads collide on LDS banks,
and deeper BK (more LDS reads) multiplies the conflicts (4.7 → 31.5 from BK16 → BK32).

This is exactly the resource **Tensile's selected kernel pads against**: `LdsPadB=8` (bank-conflict padding,
`LBSPPB128`) + `PLR1` (prefetch the next fragment so the read isn't on the critical path). Our naive `PAD8`
sweep config *regressed* (family sweep: 31 vs 55) because it padded the wrong way — it changed LDS size /
occupancy without matching the 16×16 WMMA fragment access stride, so it didn't remove the conflicts. The fix
is a **bank-conflict-free LDS layout matched to the fragment reads**, not arbitrary padding.

## Synthesis — why we're stuck, named end-to-end

| layer | measured | named bottleneck |
|---|---|---|
| occupancy | interior optimum wg2 (57.7) | L2 contention at wg4 (hit 56 vs 64) |
| LDS family ceiling | bankcf 31.5/cyc (6.7× BK16, authority 0) | **LDS bank conflicts** on WMMA-fragment reads |
| Tensile's edge | `LdsPadB` + `PLR1` | bank-conflict-free LDS layout + read prefetch |

So the dependency-free frontier (~58 at wg2, beating the LLVM authority ~53) is **bounded by LDS bank
conflicts**, and the ~58→66 gap to Tensile is a **concrete, named lever**: a bank-conflict-free LDS layout for
the 16×16 fragment reads (Tensile's `LdsPadB` done correctly), optionally with read prefetch (`PLR`). It is
**not** "more occupancy" (refuted) and **not** the vague scheduling wall — it is a specific LDS-layout problem
we can see in the counters.

## Honesty

- Single prefill shape; PMC perturbs timing and sums across instances — the **deltas** (L2 −7.4 pts at wg4;
  bankcf 6.7× BK16) are the robust claims, reproduced 3×; absolute rates are not.
- `busy/act` for wg2 vs the authority is ~parity and run-noisy, so the wg2 story rests on the bank-conflict
  delta (robust), not on a SIMD-idle argument.
- The authority is global-direct (LDS=0); it bounds differently (VALU/act 9.1 = address-arith heavy), which is
  why it's only ~53 despite high SIMD utilization.

## Verdict & next

`CONFIRMED_CONTENTION_L2_PLUS_LDS_BANKCONFLICT_BOUND`. The next step is finally a **specific, justified kernel
change**, not a sweep:

1. **Design a bank-conflict-free LDS layout** for the 16×16 WMMA fragment reads (the Tensile `LdsPadB` idea,
   but padded to match the actual `ds_load_b128` stride so conflicts actually drop). Re-measure bankcf/cycle
   under PMC and TFLOPS under the interleaved gate — target: cut bankcf toward BK16's ~5 while keeping BK32's
   compute density.
2. Keep wg2 occupancy (the L2 sweet spot) while doing it.
3. If conflicts drop and throughput rises toward ~66, the dependency-free path reaches Tensile; if not, the
   residual is the read-prefetch (`PLR`) scheduling wall and the vendored `.co` stays the only ~66 route.
