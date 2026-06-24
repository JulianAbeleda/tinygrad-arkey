# Prefill ASM Instruction Scheduler — Inc 3 Scope + Result (2026-06-23)

## Verdict: `ASM_SCHED_WAITCNT_RELOCATION_WINS` (config-dependent) — the first NON-NEUTRAL lever
Inc 2 showed pure instruction *reordering* is perf-neutral. Inc 3 changes the instruction *set* — **waitcnt
relocation** — and gets a **real, reproducible speedup**: DBUF1 **~+6%**, the PLRA route config **~+2%**, plain ~+1.7%
on clean clock-pinned isolated timing. It is **config-dependent** (kv_halved *regresses* ~−4%), so it needs per-config
gating and whole-prefill confirmation before any promotion. This is the first lever in the asm-scheduler line that
moves the needle.

## Complete Inc 3 scope (as planned, with outcomes)
| step | plan | outcome |
|---|---|---|
| branch-offset fixup | recompute branch simm16 after instruction insertion (non-mutating) | **DONE** — `capture_branch_targets`/`fix_branches` (shallow-copy, no shared-Inst mutation) |
| compute-block relocation | strip full-drain `lgkm(0)`, interleave per-WMMA minimal waits | **DONE** — `relocate_lgkm_waits` |
| correctness gates | register DAG + wait model + branch boundaries + verify_wait_correct | **DONE** — byte-correct + verify across configs/sizes |
| measurement | clean clock-pinned isolated A/B | **DONE** — config-dependent win (DBUF1 +6%, PLRA +2%, kv −4%) |

## The lever
Each compute block is `[N ds_loads][lgkm(0) full drain][M wmmas]` — every WMMA waits for ALL fragment loads. Relocation:
- remove the full drain;
- issue WMMAs in **frag-ready order** (by max producer issue-score);
- before each, insert the **minimal `lgkmcnt`** for just that WMMA's fragments (the Inc-1 wait model: for a fragment
  load at issue-position `s` among `P` producers, `lgkmcnt ≤ P−1−s`); dedup consecutive equal counts.

This overlaps WMMA compute with the tail of LDS-load latency. Because it INSERTS waits, branch offsets are recomputed
by `fix_branches` (target located by Inst identity; branch replaced with a shallow copy so the un-relocated stream is
never corrupted — `S2 NON_MUTATING`).

## Correctness (`extra/qk_asm_scheduler_inc3_test.py` — S1/S2 PASS, the gate)
- **S1**: byte-correct (rmse ≤ 3e-4) + `verify_wait_correct` across `plain / DBUF1 / PLRA_route / kv_halved`, and across
  K sizes NBLK 16/32/128 (proves branch offsets are right at scale — no MMU fault).
- **S2**: relocating a fresh build leaves a separately-built identity stream byte-identical (non-mutating).
- The Inc-1 `verify_wait_correct` gate earned its keep here: an early double-emit bug produced a register-legal-looking
  stream that the gate flagged (`DS_LOAD consumes vN before its load drained`), catching a real WAW-across-substeps
  hazard before it shipped.

## Measurement (S3, informational — clean, clock-pinned, isolated, copies excluded; 512×4096×4096)
| config | identity | reloc | Δ |
|---|---|---|---|
| **DBUF1** | ~58 TFLOPS | ~62 TFLOPS | **+5.7 … +6.3% (reproduced 3×)** |
| PLRA (route default for big GEMMs) | ~62.6 TFLOPS | ~63.9 TFLOPS | +2.1% |
| plain | ~62.5 TFLOPS | ~63.6 TFLOPS | +1.7% (near noise) |
| kv_halved (route default for small-N) | ~63.8 TFLOPS | ~61.3 TFLOPS | **−4.0% (REGRESSION)** |

**Why config-dependent:** the win comes from overlapping WMMA compute with residual LDS-load latency. DBUF1 (2× LDS →
lower occupancy → less latency-hiding from other waves) has the most exposed latency to recover (+6%). High-occupancy
small-N `kv_halved` already hides the latency across waves, so the extra waits are pure overhead (−4%).

## Honest standing
- **The asm-scheduler line is no longer all-neutral.** Pure reorder (Inc 2) was neutral; waitcnt *relocation* (Inc 3) is
  a real lever — but instruction-set-changing, and config-dependent.
- **Practical relevance to the route:** the production route uses PLRA for the big projection GEMMs (+2.1%) and
  kv_halved for the small kv_proj (−4.0%). A net win therefore requires applying relocation **only to the PLRA roles**
  (excluding kv) — a per-config/per-role gate. Even then the dominant-role gain is modest (~+2%), consistent with the
  overall ≤~2–3% schedulable-upside estimate (part of the Tensile gap is a `beta` work confound).
- **Promotion gate not crossed:** isolated timing is a SIGNAL, not authority. Net benefit must be confirmed on
  clock-pinned synced **whole-prefill** (with kv excluded) before any default flip. Given the modest+mixed signal, that
  wiring is scoped but not done; the capability is shipped default-off.

## Files
New: `extra/qk_asm_scheduler_inc3_test.py`, this doc. Modified (additive): `extra/qk_asm_scheduler.py`
(`capture_branch_targets`, `fix_branches`, `relocate_lgkm_waits`). +1 ledger. No `tinygrad/` source, no production
path, no default flip, no whole-prefill speed claim.

## Follow-up A — the kv_halved regression IS occupancy (causal breakdown)
Varying *only* the LDS allocation of the **same** kv_halved kernel (which changes workgroups-per-CU, not the logic):

| occupancy (LDS) | relocation Δ |
|---|---|
| ~4 WG/CU (natural ~15 KB) | +0.08% |
| ~2 WG/CU (32 KB) | −3.03% |
| ~1 WG/CU (64 KB) | **+4.26%** |

So the lever decomposes into two primitives: **benefit = LDS-latency overlap ∝ 1/occupancy** and **cost = extra-waitcnt
overhead (≈ constant, small)**. Net = benefit − cost flips sign with occupancy: low-occupancy (DBUF1, kv@1WG/CU) is
benefit-dominated (+4–6%); high-occupancy small-N kv (natural) is the zero-benefit, noise-dominated regime (−3%…+0%,
not a robust −4%). This is exactly why kv roles (high occupancy) must be excluded and only the lower-occupancy PLRA
roles (waves_n=2) gated in.

## Follow-up B — wired into the route, but the isolated win does NOT transfer to whole-prefill
Wired `relocate_lgkm_waits` into `extra/qk_prefill_graph_gemm_route.py` behind additive `PREFILL_GEMM_RELOC` (default
off), gated to `waves_n==2` (non-kv PLRA roles). Clock-pinned synced **whole-prefill** (`qk_prefill_whole_synced.py`):

| ctx | baseline #1 | reloc | baseline #2 |
|---|---|---|---|
| @512 | 3732 | 3713 | 3705 |
| @1024 | 3645 | 3629 | 3626 |
| @2048 | 3408 | 3391 | 3388 |
| @4096 | 3001 | 2989 | 2989 |

The two baselines differ by ~0.7% (run-to-run noise), which is **larger** than the reloc-vs-baseline gap. **Net
whole-prefill effect = none (within noise).** The isolated +2% on the projection GEMMs is consumed by in-model
integration — the GEMM is only a fraction of prefill (attention, norms, kv-proj, activations), the real per-role shapes
sit at higher occupancy than the isolated 4096³ probe, and the harness noise floor (~0.7%) swamps any sub-percent
benefit. This is the project's standing lesson ([[inference-perf-measured-map]]): isolated kernel wins don't transfer.

**Decision:** `PREFILL_GEMM_RELOC` ships **default-off** (no net whole-prefill benefit, like `PREFILL_GEMM_8WAVE`). The
capability is real and verified; it just doesn't move the in-model needle.

## Arc close (Inc 0–3)
The asm-scheduler line is **complete and conclusive**: register DAG (faithful) → wait model (hand waits already
minimal) → cross-motion sound + pure reorder neutral → waitcnt relocation is the one isolated-non-neutral lever (+2–6%,
config-dependent) but **does not transfer to whole-prefill**. The prefill→Tensile residual is **not recoverable by any
instruction-scheduling transform** of the current kernel. Closing it requires either vendored Tensile (full parity, opaque
dep) or a structurally different emit (deeper DepthU / cross-iteration pipelining), neither of which is instruction
scheduling. Capability + three correctness gates shipped default-off; no production change.
