# FRONTIER SCOPE — beyond Route A: the 4 residual high-EV levers (all non-kernel or deep-internals)

Paste as a fresh-session opener. Self-contained. Working dir `/home/ubuntu/tinygrad-arkey`, gfx1100 / RX 7900 XTX.

## Meta-finding (why these 4, and not another kernel)
Route A (A1 correct WMMA asm → A2 global-direct pipeline 24–32 TFLOPS → A3 LDS multi-wave refuted) plus the prior
decode/prefill/flash/fusion arcs have **exhausted the dependency-free hand-kernel space**. Nearly every "refuted"
conclusion bottoms out at one root cause: **gfx1100 is Infinity-Cache/L2-served for these workloads**, which kills
the classical-GPU playbook (LDS staging, locality primitives, software pipelining, prefetch) because the cache
already does their job. Decode rests at 66–69% llama (GPU-bound, residual = per-thread codegen), prefill at A2 ~32
dependency-free / PREFILL_V2 ~80% llama. The "BEAM hangs gfx1100" wall was also **refuted** (BEAM completes but
*under-ranks* — picks 14–17 < 17.7 default; a quality bug), and that audit found **no deferred kernel lever is
newly unlocked** (`beam-hang-premise-audit-20260619.md`).

So the residual high-EV work is **not another kernel** — it's the 4 things we've been routing around. Sequencing
rationale: **#1 (rocprof) is the cheap decisive gate** — it confirms or reopens the IC-served premise that the
entire rest-state depends on, and it tells #4 *where* the codegen gap is. Then #2/#3 are engineering wins (proven
perf, blocked on plumbing). #4 is deepest/last, informed by #1.

Order to execute: **1 → 3 → 2 → 4** (measure-first; ship the highest-confidence win; unblock the ceiling; then the
deep grind). Each is independently scoped below; pick by EV/appetite.

---

## #1 — rocprof IC-served ground-truth (DO FIRST: cheap, decisive, gates #4)
**Goal.** Replace inference with measurement. Directly answer: (a) are the decode GEMV + prefill matmul really
IC/L2-served (→ locality levers stay dead, high-confidence rest)? (b) what is the *actual* bottleneck — occupancy,
VALU, or memory stall? (c) vs llama's same kernel, *where* is the 66→100% decode gap (scheduling? occupancy?
regalloc?) — turning #4 from a guess into a target.

**Instruments (installed, confirmed):** `/opt/rocm/bin/rocprofv3` (counters: `GL2C_HIT`/`GL2C_REQ` → L2 hit-rate,
`SQ_WAVES` → occupancy, `GRBM_GUI_ACTIVE`/`GRBM_COUNT` → GPU busy, `SQ_INSTS_VALU`/`SQ_BUSY_CYCLES` → VALU%),
`rocprof-compute` (omniperf — full panel: L2/MALL hit, wavefront occupancy, VALU/MFMA busy, LDS bank conflicts,
mem-unit busy/stall). MALL = the Infinity Cache (last-level); check if a MALL hit counter is exposed
(`rocprofv3 --list-avail | grep -i mall`).

**⚠ FEASIBILITY SPIKE FIRST (the one real risk):** tinygrad's AMD backend (`tinygrad/runtime/ops_amd.py`) talks
**directly to KFD/HSA, bypassing the ROCm HIP runtime** that rocprof normally hooks. rocprofv3 may not capture
tinygrad kernels. Spike: `rocprofv3 --kernel-trace --stats -- <python decode harness>` and check for kernel rows.
If empty, fallbacks: (a) rocprofv3 counter-collection via the KFD path / PC-sampling if supported; (b) tinygrad may
have its own SQTT/perf hooks — grep `ops_amd.py`/`tinygrad/runtime/support` for `sqtt`, `perfcounter`, `pm4`; (c)
get the *llama* and *PyTorch-rocBLAS* counters (they use HIP, rocprof works) as the reference, and infer tinygrad's
via the relative kernel timings already measured. **Bound the spike to ~1 session; if HCQ is untraceable, pivot to
the reference-kernel approach.**

**Targets.** Decode GEMV (run `extra/qk_decode_runtime_overhead.py`, W==D), prefill matmul (PREFILL_V2 via
`extra/qk_prefill_v2_measure.py`; A2 via `GEMM=1 USEPIPE=1`), and **llama.cpp decode/prefill under rocprof** for the
apples comparison.

**Decision table (the output):**
- L2/MALL hit-rate on weight reads **>~90%** → IC-served CONFIRMED → locality levers stay closed; rest the kernel work with high confidence.
- hit-rate **low** → IC-served REFUTED → reopen LDS/prefetch/pipeline (a whole class) — major.
- decode occupancy low + VALU low → **latency-bound** (→ #4 = occupancy/scheduling); VALU high → ALU-bound (codegen near ceiling).
- tinygrad-kernel VALU%/occupancy **vs llama's same kernel** → pinpoints the 66→100% gap concretely.

**Gate / kill.** Deliver a counter table + a one-line verdict per question. No code change. ~1–2 sessions. This is
the highest-information-per-token move in the whole project.

---

## #3 — Ship spec-decode (highest-confidence unshipped win; self-contained; DECODE)
**Goal.** Take spec-decode from "A-pending" to shipped: ~1.3–1.4× decode, greedy-byte-identical, behind
`SPEC_DECODE=1` (default off), flipped only if it confirms ≥1.2× at full clock.

**What's PROVEN** (`spec-decode-low-sync-verdict-20260618.md`, `extra/qk_spec_decode_lowsync.py`): device-token
draft feedback (no `.item()`); reusable K-symbolic-start_pos proposal graph (ONE sync, no recompile); integrated
loop = draft propose + target verify (T=K+1, one pass) + host accept + KV self-correction; **greedy byte-identical
on every prompt**; 2 syncs/pass; accept ~2.1–2.8/pass. Draft = Qwen3-0.6B.

**The blocker (not research — measurement+integration).** The standalone harness measures 9.4→12.5 tok/s (1.33×)
but is **host-overhead-bound** (baseline stuck at 9.4 even at forced MCLK); the production decode (~55–68 tok/s) is
GPU-bound via the cli's host-efficient loop. So the production gate is **unmeasured**.

**Plan (Phase 8 proper).** (1) Wire the draft-propose + target-verify graphs + KV self-correction protocol into the
**cli/`model.generate` warm loop** (the GPU-bound one) — `tinygrad/llm/model.py` `generate`. (2) Two TinyJit graphs
(draft 0.6B unrolled K, target verify T=K+1) alternating in the warm loop; carry var_vals for the symbolic
start_pos. (3) Measure with `--warmup` under sustained load so MCLK ramps to full (the clock-ramp confound bit hard
— see `amd-decode-measurement-confounds`). (4) Sweep K (draft length) for best accept×speed.

**Gate / kill.** Flip default ONLY if ≥1.2× greedy-exact at full clock vs the production GPU-bound decode. The
verdict doc estimates 1.3–1.4× GPU-bound but warns it could erode to <1× if per-pass host overhead (2 syncs +
accept) outweighs the ~2.1 tokens saved — so **measure, don't assume**. Risk: draft VRAM (0.6B fp16 + 8B) on 24GB
(fits); two-jit alternation overhead in the warm loop.

**Expected.** The single largest dependency-free decode lever left; most self-contained; ~1.3–1.4× if it holds.

---

## #2 — Unblock external Tensile/rocBLAS (the only path to the hardware ceiling; PREFILL)
**Goal.** Convert the **measured 66–77 TFLOPS = 1.41× llama** external route from "blocked" to usable. This is the
*only* thing that reaches the gfx1100 ceiling; A1–A3 tried to hand-build it and capped at ~32.

**The blocker (packaging, not perf).** rocBLAS 7.2.4 vs the runtime's HIP 5.7 **toolchain split** — the rocBLAS host
library won't load against the mismatched HIP. Extensive machinery already exists:
`extra/qk_tensile_inmodel.py` (route_pf16/install), `qk_tensile_hcq_launch.py`, `qk_tensile_selection.py`,
`qk_tensile_kernarg_capture.cpp`, `qk_tensile_disasm.py`, `qk_tensile_runtime.py`, `qk_tensile_rebindable_node.py`.

**The promising route (b): bypass the rocBLAS *host library* entirely.** We don't need librocblas.so — we need the
Tensile **GPU code object (HSACO/.co)** + its kernarg ABI + launch params, then launch via tinygrad's **HCQ**
directly (the machinery exists: `qk_tensile_hcq_launch.py` + `kernarg_capture.cpp` + `disasm.py`). The HSACO is just
gfx1100 machine code — version-agnostic to the host HIP. So: (1) obtain/extract a gfx1100 Tensile HSACO for the
prefill ffn shapes (from the rocBLAS install's Tensile library, or hipBLASLt); (2) capture the kernarg layout +
grid/workgroup/LDS (kernarg_capture.cpp already does this); (3) launch via HCQ, flag-gated. Alt route (a):
build/obtain a version-matched rocBLAS/hipBLASLt; alt (c): hipBLASLt instead of rocBLAS.

**Gate.** warm pp512 ≥ PREFILL_V2 warmstart by the measured margin, **dNLL ≤ 0.01** (`extra/qk_prefill_v2_nll_eval.py`),
**decode W==D untouched**. Ship behind a flag.

**Policy call for the user (flag this, don't decide it):** this introduces a **bundled vendored HSACO blob** — a
single version-pinned binary, lighter than a runtime rocBLAS dependency, but not "pure dependency-free." If
"dependency-free" is a hard requirement, #2 is out (and the ceiling stays at A2/PREFILL_V2). If "self-contained
bundled blob" is acceptable, #2 is the highest-ceiling win. **Ask before building.**

**Risk.** HSACO is gfx1100/shape-specific and version-fragile; kernarg ABI must match exactly (a mismatch =
silent-wrong or hang). Selection across prefill shapes (`qk_tensile_selection.py`, `shape_matrix.py`).

---

## #4 — tinygrad AMD codegen quality (deepest; DO LAST, gated by #1; the only dependency-free decode lever)
**Goal.** Close the decode 66→100% residual (and help prefill toward warmstart ~48) via better **per-thread
codegen** on the GEMV/GEMM kernels. The gap is internals, not algorithm.

**What's known.** Decode GEMV at 57–76% of peak; gap to llama is "per-thread codegen" (`qk-runtime-overhead-arc`).
Handwritten HIP 65% vs tinygrad 57% = a measured **+8% codegen lever on int-dot** (fp is ALU-ceilinged, no lever —
`handwritten-mmvq-codegen-lever`). `_sdot4`→native `__builtin_amdgcn_sudot4` already shipped. BEAM completes but
**under-ranks** (chose 14.3 over 17.7 default) — a real **BEAM quality bug** (suspected `allow_test_size`
mis-ranking + ineffective TC search; `beam-hang-premise-audit-20260619.md`).

**Two concrete sub-levers (pick by #1's verdict):**
- **(4a) Fix BEAM's matmul ranking/TC-search** so search becomes a usable general tuning lever toward warmstart ~48.
  Entry: reproduce the under-ranking (`bench/qk-codegen-wmma/inmodel_matmul.json` CG_W3_routeB_beam_spike), audit
  the cost-model/`allow_test_size` ranking in `tinygrad/engine/search.py`, check why TC opts aren't applied/ranked.
- **(4b) Targeted renderer improvements** in `tinygrad/renderer/amd/`: VOPD dual-issue packing on the GEMV inner
  loop, tighter `s_waitcnt` scheduling, dp4a/sudot4 selection coverage. Validate against the +8% int-dot ceiling.

**Gate / kill.** #1 must first show decode is **latency/scheduling-bound** (not already at the ALU ceiling) — else
there's no room and #4 is dead on arrival. Bounded upside: ~+8% int-dot GEMV; BEAM-fix → warmstart-class prefill
(~48, still < Tensile 66). Highest effort, lowest certainty, most general (helps all kernels). **Only fund if #1
says there's headroom AND #2/#3 are done or rejected.**

---

## Decision tree
1. Run **#1** (spike rocprof on HCQ; if untraceable, use reference-kernel counters). → confirms IC-served (rest
   kernel work) or reopens it; tells #4 where the gap is.
2. Run **#3** (ship spec-decode) — independent of #1, highest-confidence decode win, self-contained.
3. **Ask the user the deps policy**, then **#2** if bundled-HSACO is acceptable — the only path to 66–77 / 1.41×.
4. **#4** only if #1 shows decode headroom and #2/#3 are settled — deepest, most general, bounded.

## Provenance
IC-served basis: PWLT-A2 (`prefill-wmma-lds-tiling-result`), CG-R1 (`prefill-codegen-pipeline-redo-result`), A3
(`route-a-a3-p2-p3-lds-refuted-20260619.md`). BEAM audit: `beam-hang-premise-audit-20260619.md`. Spec verdict:
`spec-decode-low-sync-verdict-20260618.md`. Tensile machinery: `extra/qk_tensile_*`. Decode/prefill harnesses:
`extra/qk_decode_runtime_overhead.py`, `extra/qk_prefill_v2_measure.py`. Route A: `route-a-a2-pipeline-result`,
`route-a-a3-p2-p3-lds-refuted` (both 20260619).
