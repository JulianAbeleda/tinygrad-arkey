# Prefill AMD GEMM — Tensile-Source Gap Audit (banks the BK32 frontier)

Date: 2026-06-20

## Purpose

Bank the dependency-free BK32 ~55-TFLOPS frontier and, using AMD's **cloned Tensile source** as ground truth,
answer the one question that matters: **why ~55, not ~66 (Tensile)?** No GPU, no build, no timing — this is a
source + kernel-name audit that classifies the gap so we stop guessing.

Source: `/home/ubuntu/rocm-libraries-tensile-sparse/shared/tensile/Tensile/` (confirmed present).
Probe: `extra/qk_amd_tensile_source_gap_audit.py` → `bench/.../amd_tensile_source_gap_audit_result.json`.

## The framing that breaks the problem open

The selected Tensile ffn_gate/up kernel is **DepthU = 16** and reaches **~66 TFLOPS**. Our numbers at the
matching tile (128×128, MI16×16×16):

| kernel | depth | sched | TFLOPS |
|---|---|---|---:|
| Tensile selected | DepthU16 | SIA1+PLR1+PGR1+SLW1+WGM8 | **~66** |
| ours `build_gemm_lds2` BK16 | DepthU16-equiv | **SIA0** (none) | ~42 |
| ours BK32 (frontier) | 2× depth | SIA0 | ~55 |

**So depth is NOT Tensile's lever.** At the *same* DepthU=16, Tensile gets 66 and our SIA0 kernel gets 42 — a
**1.57× gap from scheduling alone**. Our BK32 (deeper K) is *our* compensation, recovering 1.31× (42→55); a
**1.20× residual to Tensile remains**. The questions below are answered from the Tensile source.

## Q1 — Is K-block depth Tensile's lever? **No.**

Tensile wins at fixed DepthU=16. Depth (BK32) is a lever *we* used to partially close a gap Tensile doesn't
open with depth. The earlier "BK-depth exhausted at ~55" result is therefore the right read: depth is our
knob, capped by VGPRs, and it cannot reach Tensile because it attacks the wrong axis.

## Q2 — What IS Tensile's lever? Instruction scheduling + one-iteration-ahead prefetch.

Decoded from the kernel name, cited from `Common.py`:

| feature | token | Tensile-source meaning | our `build_gemm_lds2` BK32 |
|---|---|---|---|
| **ScheduleIterAlg** | `SIA1` | "0 = minimal/no scheduling: Global Read, then local reads, then local writes, then MACs"; ≥1 **interleaves** those four classes within the iteration | **SIA0-equivalent** — coarse blocks: all `global_load` → `ds_store` → `s_barrier` → all `ds_load` → `s_waitcnt` → all 16 `v_wmma`. *This is Tensile's null baseline.* |
| **PrefetchLocalRead** | `PLR1` | prefetch next-iter LDS reads while the current MAC runs (`iter0: plr[1] MAC_r[0]`) — hides `ds_load` latency behind WMMA | **PLR0** — all `ds_load`, then wait, then WMMA; next iter's reads not overlapped |
| **PrefetchGlobalRead** | `PGR1` | double-buffer global→vgpr→lds; prefetch next K-tile's global loads while computing current | **partial/refuted** — register DBUF was ~neutral; full barrier serializes load→compute |
| **ScheduleLocalWrite** | `SLW1` | schedule `ds_store` *into* the local-read iterations, not one block | **SLW0** — `ds_store` is a coarse block behind a full barrier |
| **WorkGroupMapping** | `WGM8` | remap wg ids (`wgSerial = wg0 + (wg1 % WGM)·nwg0`) so concurrent wgs hit L2 best — raises effective bandwidth | **none** — plain `gx=s[2]`, `gy=s[3]` |
| LocalReadVectorWidth | `LRVW16` | `ds_load_b128` wide fragments | **matched** (we emit `ds_load_b128`) |
| MI atom / macro tile | `MI16x16x16` / `MT128x128x16` | WMMA atom + 128×128 tile | **matched** |

The gap is **not** the WMMA atom, **not** LDS-vs-global, **not** depth, **not** vector width — all matched.
It is concentrated in **SIA1 + PLR1 + PGR1 + SLW1** (instruction scheduling + prefetch latency-hiding) plus
**WGM8** (L2 bandwidth locality).

## Q3 — What is our kernel, exactly? Tensile's `SIA0` null baseline.

The Tensile source literally describes `ScheduleIterAlg=0` as "Global Read, followed by local reads, followed
by local writes, followed by MACs." That is, line-for-line, the structure our `build_gemm_lds2` loop emits.
We are at Tensile's *zero-scheduling* baseline; BK32 deep-K brute-forces more MACs between barriers to
amortize the un-hidden latency, which is why it helps but plateaus at ~55.

## Q4 — Is the lever expressible dependency-free? It is the standing codegen wall.

PLR1/PGR1/SIA1 are **instruction-level loop scheduling** — issue iteration *k+1*'s loads interleaved among
iteration *k*'s WMMAs so memory latency hides behind compute. Our hand-asm emits coarse phases on the
`assemble_linear` path with no instruction scheduler; **DBUF (register double-buffer) was measured ~neutral
precisely because, without SIA-style interleaving, the full barrier still serializes** load→compute. Closing
this needs a real per-iteration instruction scheduler (the software-pipelined-K-loop capability), **not**
another tile/depth/wave knob. This is the same wall named in the SW-pipeline charter — now *precisely
attributed* to Tensile's SIA/PLR/PGR/SLW via their source.

## Banked conclusion

- **Frontier banked**: dependency-free hand-asm prefill = **~55 TFLOPS (BK32, 128×128), reaching the LLVM
  authority (~53), ~1.9× global-direct, correct (2.1e-4)** — the closest dependency-free result on record.
- **Gap attributed (from Tensile source)**: the residual ~55→66 (and the underlying ~42→66 at fixed depth) is
  **Tensile's SIA1 instruction scheduling + PLR1/PGR1 prefetch latency-hiding + SLW1 + WGM8 L2 locality** —
  three lever classes: `instruction_scheduling`, `prefetch_latency_hiding`, `memory_locality`.
- **Practical ceiling**: the SIA0 phase-blocked family tops out ~55 (~85% of Tensile). ~66 requires either the
  instruction-scheduler codegen capability (the wall) or the vendored Tensile `.co` (declined).

## Next (not another sweep)

1. **PMC/occupancy confirmation** of the BK32 kernel — this audit *predicts* it is LDS-read/global-load
   latency-bound (un-hidden because SIA0); a per-kernel PMC (LDS-wait / VALU-vs-WMMA-issue / occupancy) would
   confirm the mechanism the source points to. Cheap, decisive, no new kernel.
2. **Only then**, if pursued: a minimal PLR-style local-read prefetch on the hand-asm path (one new schedule
   capability), measured under the same interleaved gate. Bounded, no BEAM.
3. The depth/tile/wave sweep space is **closed** — depth was the wrong axis; the audit says so from source.
