# SCOPE — Prefill WMMA occupancy lever (gfx1100, 2026-06-19)

Follow-up to `prefill-boost-resolution-result-20260619.md`. That work proved tinygrad WMMA prefill is **bimodal per
process — ~1438 tok/s (~47% llama) OR ~2674 (~87% llama)** — latched at process init, stable within a process, and
**NOT clock/thermal/power-state**: the stuck runs sit at a HIGHER real clock (sclk 2333 vs 2315), are fully busy
(32/32), and draw absurdly low power (~55 W = an **under-occupied / stalled** kernel, not compute-saturated). No
clock/power lever (profile_peak, manual DPM, dense-matmul primer) forces the fast state.

**Goal: find what sets the per-process execution state and force the fast (~87% llama) state reliably and
dependency-free.** The ~55 W-at-full-clock-fully-busy signature points at **occupancy / wave-scheduling / SMU
power-grant** for the WMMA FFN matmul — NOT clock and NOT the matmul algorithm.

**Measure/diagnose-first: no kernel/route/default change until a lever is validated ≥5/5 fresh launches,
byte-identical greedy, real-clock-verified.** Real GFXCLK via `rocm-smi --showgpuclocks` from a SEPARATE process;
wall-clock time-base for window correlation. N≥5 fresh launches per claim.

## Key existing handles (from last turn)
- **A reliably-FAST harness:** `extra/qk_tensile_ab_measure.py` → WMMA(OFF) = ~2674, 6/6 runs.
- **A reliably-STUCK harness:** `extra/qk_prefill_boost_probe.py` MODE=wmma → ~1438, 15/15 runs.
- These two differ only in setup → the split is bisectable (P0/P1). The WMMA jit, model, chunk, warmstart, and
  clock are identical between them.

## Phase plan

### P0 — Confirm determinism + lock the two reference harnesses
Re-confirm under one controlled back-to-back sweep (N≥5 each, interleaved order to kill thermal drift): is
`ab_measure`-WMMA reliably FAST and `boost_probe`-WMMA reliably STUCK? If yes → the trigger is in the code path and
is bisectable. If it turns out random (both harnesses sometimes flip) → skip to P2a (pure hardware-state path), since
no code bisection exists. Artifact: `bench/qk-prefill-boost/p0_determinism.txt`.

### P1 — Kernel-identity gate (the decisive fork)
Capture, in a FAST process and a STUCK process, for the WMMA FFN matmul kernel(s):
- launch geometry: global dims, local/workgroup dims (`DEBUG>=2` per-kernel line, or `Program`/`ProgramSpec`).
- resource usage / occupancy: VGPR, SGPR, LDS bytes, waves/SIMD (AMD code-object metadata from the compiled program;
  tinygrad AMD renderer/compiler exposes the ELF; `llvm-readobj`/`.kd` descriptor or tinygrad's program metadata).
- the compiled binary itself (hash the code object).
**Decisive fork:**
- **Same binary + same launch dims (fast == stuck)** → it is a pure GPU/driver execution state (the GPU runs the
  *identical* kernel faster or slower per process). Go **P2a**.
- **Different binary or different launch dims** → tinygrad codegen/schedule is non-deterministic per process. Go
  **P2b** (find + pin the divergent decision).
Artifact: `bench/qk-prefill-boost/p1_kernel_identity.json` (fast vs stuck geometry/occupancy/hash).

### P2a — Hardware/driver execution-state path (if kernel identical)
The kernel is the same; the GPU executes it under-occupied (~55 W) in stuck processes. Investigate and try to force
full occupancy / power grant, dependency-free:
- **CU mask / active-CU count:** is the stuck process running on fewer CUs? Inspect via gpu_metrics binary interface
  (`/sys/class/drm/card0/device/gpu_metrics`) or rocm-smi; test forcing all CUs.
- **SMU power grant:** the ~55 W ceiling at full clock suggests a power/voltage state. Probe `gpu_metrics`
  (current_socket_power, average_gfx_activity, voltage) in fast vs stuck; test power-profile writes
  (`pp_power_profile_mode`) — COMPUTE profile (not the perf-level lanes already refuted).
- **KFD/HSA init env:** does an env/queue-priority/`HSA_*` setting at process start determine the state? Test
  candidate env vars (queue priority, `HSA_ENABLE_SDMA`, `GPU_MAX_HW_QUEUES`, etc.) across fresh launches.
Gate: a dependency-free setting that yields fast (≥2400 tok/s) ≥5/5 fresh, with `gpu_metrics` showing higher
activity/power. Artifact: `bench/qk-prefill-boost/p2a_hw_state.json`.

### P2b — Codegen/schedule path (if kernel differs)
Bisect `ab_measure` → `boost_probe` setup until WMMA flips fast↔stuck; the flipping step is the trigger. Likely
suspects from last turn (none individually confirmed): building/running the Tensile jit, the order of block-flag
setting, a pre-realized large tensor, the warmstart-opts application timing, JIT batch-ramp state. Once isolated,
identify WHY it changes the compiled WMMA kernel (different `_pf16_warmstart` opts applied? a different TC schedule?
a stale cache key?) and pin the fast schedule deterministically. Artifact:
`bench/qk-prefill-boost/p2b_bisect.txt` + the identified divergent decision.

### P3 — Build + validate the lever
Apply the minimal dependency-free change (occupancy-forcing setting from P2a, or schedule pin from P2b). Validate:
- ≥5/5 fresh cold launches reach the fast band (≥2400 tok/s ≈ ~80%+ llama), real-clock-verified.
- byte-identical greedy output vs the current default (rel_err 0 on logits / same sampled token).
- realistic generate-path single-prefill also lands fast (P2-style), not just the tight loop.
- no decode regression (decode path untouched), no default change unless the lever is safe + always-on.
Artifact: `bench/qk-prefill-boost/p3_validation.txt`.

## Gates (hard)
- Diagnose before changing: P1 kernel-identity gate decides the path; do NOT build a lever before it.
- A lever "works" only at ≥5/5 fresh launches in the fast band, real-clock-verified, byte-identical greedy.
- Power/occupancy claims must cite `gpu_metrics` (current_socket_power / gfx_activity), not just tok/s.
- No default/route change until P3 passes; if the lever is an env/runtime knob, ship it gated first.

## Deliverables
- `docs/prefill-occupancy-lever-result-20260619.md`.
- probe extensions in `extra/qk_prefill_boost_probe.py` (kernel-geometry dump, gpu_metrics reader, lever knobs) or a
  new `extra/qk_prefill_occupancy_probe.py`.
- `bench/qk-prefill-boost/{p0_determinism,p1_kernel_identity,p2a_hw_state|p2b_bisect,p3_validation}.*`.
- README pointer. NO default/route change unless P3 validates a safe always-on lever.

## Definition of done
- EITHER a **dependency-free lever** (occupancy-forcing runtime setting or a pinned schedule) that reliably locks
  WMMA prefill into the fast ~87%-llama state — a real shippable prefill win — validated ≥5/5 fresh + byte-identical;
- OR proof the fast state is an **uncontrollable GPU/driver lottery** (kernel identical, no setting forces it) → the
  honest dependency-free prefill ceiling is the stuck ~47% llama, and the ~87% is opportunistic only. Either outcome
  settles whether prefill has a real dependency-free lever beyond the shipped concrete-KV 1.24×.
