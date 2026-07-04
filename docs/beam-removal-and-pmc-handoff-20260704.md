# Handoff: gfx11 PMC unlock, 8B-vs-14B verdict, inherited-beam removal — 2026-07-04

Exhaustive resume point. Spans two repos: `tinygrad-arkey` (this repo) and `BoltBeam`
(`/home/ubuntu/BoltBeam`). Read top-to-bottom; the **IMMEDIATE BLOCKER** is first.

Related memory notes (`~/.claude/projects/-home-ubuntu/memory/`):
`resume-after-amdgpu-dkms-reboot`, `tinygrad-native-pmc-misses-gemm-graph-kernels`,
`bubblebeam-not-inherited-beam`, `killing-tinygrad-amd-wedges-mes-ring`.

---

## 0. TL;DR — where we are

1. **DONE + shipped:** amdgpu-dkms 6.16.13 reboot unlocked the gfx11 PMC counters. BoltBeam's
   profiler now emits measured occupancy/VALU/L2 per kernel. Two BoltBeam fixes committed+pushed to
   `main` (`da91934`, `effda84`).
2. **DONE (analysis):** 8B-vs-14B question answered — 14B is *proportional more work*, not a
   14B-specific efficiency loss. The shared Q4_K prefill GEMM is latency/occupancy/codegen bound at
   ~45% occupancy / ~1% of fp16 peak. Config sweep proved occupancy is NOT tunable via workgroup/parts.
3. **DONE + committed (`9bdb249a1`), NOT yet GPU-validated:** removed tinygrad's inherited timing beam
   so BubbleBeam/FutureSight is the sole path. 6 files + `search.py` deleted. Verified via import +
   `DEV=PYTHON`. On-GPU validation still owed (do a small-model prefill after GPU recovery).
4. **IMMEDIATE BLOCKER:** the AMD GPU is wedged (MES ring full, D-state ttm threads) from
   kill-timeouts on tinygrad AMD runs. Needs GPU reset or reboot before any `DEV=AMD` work.

---

## 1. IMMEDIATE BLOCKER — GPU wedge, recover first

`dmesg` spams `amdgpu 0000:08:00.0: amdgpu: MES ring buffer is full`; ~41 `kworker/u65:*+ttm`
threads stuck in uninterruptible D-state. Every `DEV=AMD` realize hangs. `DEV=PYTHON` still works
(that's how we proved it's the GPU, not the beam-removal code).

**Cause:** hard-killing a `DEV=AMD` tinygrad process mid-kernel (a `timeout` SIGTERM on a long prefill
and a `pkill` on a hung smoke test) jams the RDNA3 MicroEngine Scheduler ring. See memory
`killing-tinygrad-amd-wedges-mes-ring`. **Lesson: never `timeout`/`pkill` a live DEV=AMD run — bound
the WORK (smaller model/context), or `run_in_background` and wait.**

**Recovery (pick one):**
- Reboot — cleanest. amdgpu-dkms reloads on boot; PMC counters stay unlocked (DKMS driver persists).
- GPU reset without reboot: `sudo sh -c 'echo 1 > /sys/kernel/debug/dri/*/amdgpu_gpu_recover'`
  (passwordless sudo works here). May fail with a jammed MES and need a reboot anyway.

**After recovery, re-verify the GPU + counters (from `resume-after-amdgpu-dkms-reboot`):**
1. `cat /sys/module/amdgpu/version` ≈ `6.16.13`; `/dev/kfd` + `/dev/dri/renderD128` present.
2. Counter probe (should show ~6/8 nonzero, incl GRBM_GUI_ACTIVE / SQ_INSTS_VALU / GL2C_*):
   `cd /home/ubuntu/tinygrad-arkey && DEV=AMD PROFILE=1 PMC=1 PMC_GRAPH=1 .venv/bin/python extra/qk/pmc_graph_microbench.py`
   (LDS counters read 0 — expected for matmul, not a bug.)

---

## 2. Inherited-beam removal (tinygrad-arkey) — uncommitted, needs GPU validation then commit

**Goal (user):** "we use bubblebeam not beam — regular beam should be eliminated … decouple both and
use futuresight and bubblebeam as the canonical path." No hand-written kernels.

**Taxonomy** (`docs/bubblebeam-futuresight-terminology-20260625.md`): BubbleBeam = this fork's search;
FutureSight = its static pre-timing selector (emits `opts_to_apply`); *inherited beam* = tinygrad's
timing `beam_search`. FutureSight applies its COALESCE via `opts_to_apply` → `apply_opt`
(`postrange.py`), never through the timing beam — so they were already decoupled at the apply layer.

**Branch:** `gfx11-pmc-profiler-enable`. Beam removal is COMMITTED as `9bdb249a1` (pushed to origin),
on top of the PMC profiler-enable commit `8ba989337`.

**Exact edits made:**
- `tinygrad/codegen/opt/search.py` — **DELETED** (whole file; it was only inherited-beam machinery:
  `actions`, `get_kernel_actions`, `_time_program`, `beam_search`, `beam_pool`). Its only caller was
  postrange; nothing else in the repo imported it.
- `tinygrad/codegen/opt/postrange.py` — removed the `elif beam >= 1:` branch in `apply_opts` and the
  `beam` param (now `def apply_opts(ast, ren)`); removed now-unused `Context` import. `bufs_from_ast`
  KEPT (also used by `realize.py:91`).
- `tinygrad/codegen/__init__.py` — `apply_opts(sink, ren)` (dropped `beam=ast.arg.beam`); updated two
  stale comments.
- `tinygrad/uop/ops.py` — removed `KernelInfo.beam: int = 0` field.
- `tinygrad/engine/realize.py` — removed `pm_beam` PatternMatcher; `compile_linear(linear, validate=)`
  (dropped `beam` param + the `pm_beam` rewrite); fixed `time_call` caller (`beam=0` gone,
  `Context(BEAM=0…)`→`Context(…)`); removed orphaned `BEAM` import.
- `tinygrad/engine/jit.py` — `compile_linear(linear)` (dropped `beam=getenv("JITBEAM", BEAM.value)`);
  removed the `Context(BEAM=0 if IGNORE_JIT_FIRST_BEAM …)` guard; removed orphaned `BEAM`/`Context`
  imports.
- `tinygrad/helpers.py` — removed the `BEAM` ContextVar from line 238.

**Diff:** `6 files changed, 14 insertions(+), 29 deletions(-)` + `search.py` −199 lines.

**Verified so far (NON-GPU):** `import tinygrad` OK; `KernelInfo` has no `beam` attr; `DEV=PYTHON`
matmul realizes correctly (`8.0`). No repo code (tinygrad or extra/ or test/) references the removed
symbols (grepped: `beam_search`, `BEAM` ContextVar, `KernelInfo.beam`, `opt.search`).

**REMAINING (post-reboot GPU validation of the already-committed `9bdb249a1`):**
1. GPU recovery (section 1) — reboot in progress / done.
2. On-GPU validation: run a real prefill and confirm FutureSight still produces the same kernels and
   correct output with beam gone:
   `cd /home/ubuntu/BoltBeam && python3 -m boltbeam.cli collect-hw-trace --provider tinygrad --model /home/ubuntu/models/Qwen3-0.6B-Q8_0.gguf --workload prefill --context 128 --gpu-health fail --out /tmp/beam_removal_check.json`
   (use a SMALL model — 0.6B — to keep it fast; do NOT kill it mid-run). Confirm rc=0 and kernels present.
   Also sanity: `DEV=AMD .venv/bin/python extra/qk/pmc_graph_microbench.py` still runs.
   If validation fails, the beam removal is `9bdb249a1` on `gfx11-pmc-profiler-enable` (revert/fix there).

---

## 3. PMC unlock + BoltBeam profiler (DONE, shipped)

- amdgpu-dkms 6.16.13 (ROCm 7.2.4) rebooted → gfx11 PMC counters unlocked. Microbench:
  **6/8 counters nonzero, eager+graph** (SQ_BUSY_CYCLES, SQ_INSTS_VALU, SQ_INSTS_SALU,
  GRBM_GUI_ACTIVE, GL2C_HIT, GL2C_MISS; SQC_LDS_* = 0, expected for matmul).
- BoltBeam `main` commits (pushed): `da91934` (counter-aware causal classifier — was hardcoding
  "gfx11 PMC gap" even when counters present) and `effda84` (baseline verdict distinguishes
  efficiency gain from loss; 433 tests green).
- `collect-hw-trace` on real Qwen3-14B prefill (ctx 512): 35/36 rows carry measured counters.

---

## 4. 8B-vs-14B verdict (DONE) + occupancy finding

Collected 8B baseline, ran `profiler-report --baseline`. Hot `ffn_gate_up` counters near-identical
8B vs 14B (occ 45.8/45.5%, valu 37.2/37.1%, l2 97.9/96.8%). Per-role work-x vs time-x:

| role        | work × | time × | eff × | verdict |
|-------------|--------|--------|-------|---------|
| ffn_gate_up | 1.97   | 1.55   | 1.27  | more_work_efficiency_gain |
| ffn_down    | 1.97   | 1.67   | 1.18  | more_work |
| attn_qo     | 1.74   | 1.39   | 1.25  | more_work_efficiency_gain |
| attn_kv     | 1.39   | 1.09   | 1.28  | more_work_efficiency_gain |

**Verdict: `proportional_more_work`.** Every role's time grows slower than its FLOPs → 14B is
marginally MORE efficient per FLOP; no efficiency loss from scaling. Real target = the SHARED Q4_K
prefill GEMM inefficiency (~45% occupancy, ~37% VALU, L2-resident → latency/occupancy/codegen bound;
fixing it helps both models).

**Occupancy sweep (PMC-measured, env-tunable knob added then REVERTED — route_policy.py is clean):**
occupancy is INVARIANT to config — LOCAL 64/128/256 → occ 45.5%; parts=2 → 46.0% (noise). The ~45%
ceiling is baked into the q4k direct-out PRIMITIVE's fixed UOp schedule (register footprint of the
dequant→MAC inner loop), NOT a tiling choice. These are `custom_kernel` primitives; moving occupancy
needs primitive schedule/codegen work (BubbleBeam/FutureSight territory), not a config pick and not a
hand-written kernel. Under "no hand-written kernels," the config surface is EXHAUSTED for this shape.

---

## 5. Open work / next steps

- **P5 (8B-vs-14B): DONE** — verdict above.
- **Occupancy on the hot GEMM:** the only lever left is the primitive's schedule/codegen via
  BubbleBeam/FutureSight. Alternative axis: VALU is only 37% and data's in L2, so the dequant
  instruction mix (int-dot / v_dot paths, cf. decode work in `extra/qk/`) may matter more than
  occupancy. Both are primitive-level — needs a decision on scope.
- **P3 (BoltBeam vendor importers):** ncu importer exists (`boltbeam/profiler/importers/ncu.py`,
  tested) but is NOT CLI-wired; no rocprof-compute importer yet. Registry already declares
  `rocprof_compute` provider mappings. This is unblocked, GPU-independent work.
- **PMC_GRAPH** stays opt-in (never a collector default) per the MVP doc.

---

## 6. File & artifact locations

- Beam-removal diff: `git -C /home/ubuntu/tinygrad-arkey diff` (+ `search.py` deletion staged).
- PMC microbench: `extra/qk/pmc_graph_microbench.py`. Profiler-enable ioctl:
  `tinygrad/runtime/ops_amd.py` `require_profile_mode()`.
- Route policy (q4k per-role parts/LOCAL): `tinygrad/llm/route_policy.py` `q4k_policy()`.
  Hot GEMM primitive: `tinygrad/llm/qk_primitives.py`, `tinygrad/llm/prefill_routes.py`.
- BoltBeam profiler: `boltbeam/profiler/report.py` (classifier `_classify_kernel`, `_baseline_compare`).
- **Persistent artifacts (survive reboot): `/home/ubuntu/pmc-handoff-artifacts/`** — the 14B/8B
  hw_traces, reports (`qwen3_14b_report.md`, `qwen3_14b_vs_8b_report.md`), and the 5-config sweep
  JSONs + `sweep_gateup.sh`. (The originals in the `/tmp/...scratchpad/pmc/` dir are EPHEMERAL.)
- Models: `/home/ubuntu/models/Qwen3-{0.6B-Q8_0,8B-Q4_K_M,14B-Q4_K_M}.gguf`.

## 7. Repo state snapshot (2026-07-04)

- `tinygrad-arkey`: branch `gfx11-pmc-profiler-enable`, HEAD `9bdb249a1` (beam removal, pushed to
  origin) on top of `8ba989337` (PMC profiler-enable). `route_policy.py` clean. Pending: on-GPU
  validation only.
- `BoltBeam`: branch `main`, HEAD `effda84`, clean, pushed to origin. Feature branch
  `profiler-report-pmc-diagnostics` merged + deleted.
