# >>> CORRECTION (2026-07-24, post-review) — THIS DOC'S ORIGINAL DIAGNOSIS BELOW IS WRONG <<<
#
# A py-spy + dmesg review found the REAL root cause. Read this, not the "environment
# degradation" theory below (kept for the record; its suspects — 18GB cache, IPC/semaphore
# leaks, KFD handle leaks — are RED HERRINGS, not in the stall path).
#
# ROOT CAUSE — a reproducible CODE bug, triggered BY a clean GPU (not degradation):
#   Clean GPU w/ ~25GB free -> FULL_RESIDENT_OVERLAY admitted -> prefill_v2=True ->
#   from_gguf (model.py:1313) -> _build_packed_wmma_warmstart (model.py:952) -> gate_combo
#   -> _run_gate (packed_wmma_prefill_candidates.py:171) -> run_canary
#   (packed_wmma_correctness_canary.py:152) -> run_isolated (process_isolated.py:97,
#   multiprocessing spawn). The spawned canary compiles+runs a packed-WMMA GEMM that
#   HW-FAULTS the GPU (RuntimeError: HW fault ... memory_lost=1) -> full driver reset
#   (dmesg: "amdgpu: GPU reset(81) succeeded", "device wedged, but recovered through reset").
#   THE DRIVER-RESET WINDOW IS THE ">240s HANG"; auto-recovery is why VRAM reclaims + rocm-smi
#   looks healthy after. Secondary deadlock in flat/unguarded scripts: spawn re-execs the
#   whole script as __main__; the child re-runs load_model, dies on the admission
#   "budget 0.0GB, KV admits 0" (parent holds ~19GB overlay) BEFORE draining the spawn pipe
#   -> parent deadlocks in Process.start()/pipe_write, BEFORE run_isolated's timeout loop
#   (process_isolated.py:97 vs 99-104) so the hard timeout never fires.
#
# PROVEN MITIGATION (immediate): TINYGRAD_PREFILL_PACKED_WMMA=0 -> repro completes ~15s,
#   correct token, no spawn/fault/hang. (8B fast path is graph-GEMM, unaffected; disables
#   only the packed-WMMA prefill route, which is 14B's fast path.)
# REAL FIX (Codex): the packed-WMMA canary GEMM HW-faults (OOB/bad-tile) -> debug the
#   kernel/geometry (_mutate_payload/PACKED_WMMA_GEOM/packed_wmma_correctness_canary.py).
#   Robustness: (a) don't run the warmstart canary after the parent realized ~19GB overlay;
#   (b) make run_isolated's hard timeout cover Process.start(), not just queue.get;
#   (c) add if __name__=="__main__" guards to flat entry/repro scripts.
# >>> END CORRECTION — original (incorrect) diagnosis follows <<<

# BoltBeam / GPU prefill-run HANG — diagnosis handoff (2026-07-24)

Handoff for Codex to diagnose + fix. Symptom: **BoltBeam prefill runs (and even a plain
model load + one forward) HANG indefinitely on a clean, idle GPU** after a session of
many runs. This blocks on-GPU verification (e.g. the fused-attention live trace). The
CODE is fine and committed; this is an **environment / runtime degradation** problem.
Goal: root-cause *why fresh runs wedge* and make prefill runs reliably terminate.

## Confirmed state at handoff
- **Git:** clean tree, `HEAD == origin/master == 928b8a013`. All feature work committed
  (fused-attention route, graph-GEMM regression revert `c7da22a61`, tracing fix
  `429e5c5ed`/`928b8a013`).
- **GPU hardware:** HEALTHY and IDLE right now — `rocm-smi`: 0.2 GB / 25.75 GB used,
  0% util, SCLK 0 MHz, 44 °C, no GPU-holding PIDs, no stray python/compile procs.
  → The driver/hardware is NOT wedged; the problem is process/compile-layer, not GPU state.
- **tinygrad compile cache:** `~/.cache/tinygrad` = **18 GB** (bloated by ~20 model runs
  this session). UNTESTED whether clearing it fixes the hang — prime first experiment.

## The symptom (precise)
A run that took ~2–3 min **early** in the session now **hangs > 240 s with no output**,
holding ~19 GB VRAM, with **stuck child python processes**. Confirmed for:
- `extra/llm_research/prefill/prefill_whole_synced.py:prefill_authority` (the heavy census path:
  FULL_RESIDENT_OVERLAY + graph-GEMM candidate compilation of ~254 kernels), AND
- a **minimal** `load_model_and_tokenizer(...) + one model.logits(512-token) forward`
  (no census, no timing bursts).

## Minimal reproduction (hangs)
```
bash gpu_wait_clear.sh 18 60 5   # guard: confirms >=18GB free first (it passes: "GPU CLEAR: 25GB free")
timeout 240 env PYTHONPATH=. DEV=AMD PARALLEL=0 \
  .venv/bin/python light_trace.py /home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf
# light_trace.py = load_model_and_tokenizer(path,1024) ; shared_attention_attribution(m) ;
#                  set block._prefill_v2/_use_flash/_is_prefill=True ; m.logits(1..512).argmax()
# RESULT: 240s timeout, NO output. Process tree shows parent python + 2 child python
#         (2952508/2952509) + a ~8-thread pool, all sleeping (SNl).
```
Notes: hangs **even with `PARALLEL=0`** (so it is not solely the parallel-compile pool),
**even guard-gated on a clean 25 GB-free GPU** (so not VRAM), **even single forward**
(so not the census/overlay heaviness alone).

## RULED OUT (with evidence)
1. **NOT a VRAM leak.** VRAM fully reclaims on process kill (0.2 GB after kills). The
   19 GB is *held-while-alive* by the stuck run, not leaked.
2. **NOT GPU/driver hardware wedge.** GPU is idle + healthy at handoff (0% util, 44 °C).
3. **NOT the fused-attention / tracing code.** Pure-python validated; the hang PREDATES
   the tracing commits (earlier runs hung too); the fused route was proven to fire
   cleanly (36×/8B, 40×/14B, correct tokens 198/90310) via the same dispatch point
   EARLIER the same session when the env was fresh.
4. **NOT the OOM/admission errors.** The earlier `RuntimeError: ... budget 0.0GB, KV
   admits 0` were the memory-adaptive admission (`tinygrad/llm/admission.py`) CORRECTLY
   fail-closing under VRAM contention (two big models fighting one 24 GB card). That is
   the admission working as designed, resolved by serializing runs + the VRAM guard. It
   is a SEPARATE symptom from the hang.

## Leading suspects (for Codex to investigate, ranked)
1. **The 18 GB `~/.cache/tinygrad` compile cache** — slow lookups and/or a poisoned entry
   stalling `compile_cached` (`tinygrad/device.py:compile_cached`). FIRST EXPERIMENT:
   `mv ~/.cache/tinygrad ~/.cache/tinygrad.bak` and re-run the minimal repro. If it now
   completes, the cache is the culprit (bounded-size / eviction / corruption-guard fix).
2. **Stuck compile / generation worker SUBPROCESSES.** The hung process spawns child
   pythons (`2952508`, `2952509`) + a thread pool. Even `PARALLEL=0` hangs, so suspect
   the **subprocess-isolation-for-generation** path (per structure/ overrides:
   `tinygrad/llm/generate.py` spawns a child per policy mode) and/or a compile worker that
   deadlocks on a leaked lock/semaphore. Check for leaked IPC: `ipcs -m`, `ipcs -s`,
   `ls /dev/shm`. A dozen SIGKILL'd runs this session likely leaked shm/semaphores that
   a fresh run then blocks on.
3. **Leaked HIP/KFD contexts across SIGKILLs.** ~12 processes were `kill -9`'d mid-run.
   ROCm reclaims VRAM but may leave stale `/dev/kfd` handles / queues that make a fresh
   HIP context creation or first compile stall. Test: does a **reboot** (or
   `sudo rmmod amdgpu && modprobe amdgpu`, if permitted) restore fast runs? If yes →
   driver-state accumulation; mitigation = periodic reset or avoiding SIGKILL (graceful
   teardown).

## The single most valuable diagnostic (do this first)
**`py-spy dump` a hung run to get the exact stall stack.** Launch the minimal repro
(no timeout), wait ~90 s until it's clearly hung, then:
```
pip install py-spy 2>/dev/null; py-spy dump --pid <hung_pid>          # parent
for c in $(pgrep -P <hung_pid>); do echo "--child $c--"; py-spy dump --pid $c; done
```
This reveals WHERE it blocks — `compile_hip`/`comgr` (cache/compiler), a
`multiprocessing`/`Queue.get` (worker deadlock), a HIP `hipModuleLoad`/synchronize
(driver), or model-load I/O. That one stack likely names the root cause outright.

## Secondary diagnostics
- `ipcs -m -s` + `ls -la /dev/shm` before/after killed runs (leaked shm/sems).
- `ls /dev/kfd` handles / `rocm-smi --showpids` for zombie GPU queues.
- Re-run with `CCACHE=0` (bypass compile cache) to isolate cache vs compiler-invocation.
- Re-run in a FRESH shell / after logout (isolate per-shell env leakage).
- Compare wall time of `compile_hip` on the FIRST kernel vs later (is compile itself
  slow, or is it a lock wait before compile?).

## Impact / what is NOT blocked
- All feature work is committed, pushed, and validated. Prefill fast path restored
  (8B ~3.6k / 14B ~1.9k tok/s), fused-attention route governed + firing (proven),
  tracing fix committed (pure-python validated).
- The ONLY thing pending is the **live GPU re-confirmation** of the new trace fields
  (`shared_attention.fusion_proven=True`, `custom_kernel_attention_trace.dispatches≈36`)
  in a real `prefill_authority` run — a ~2-minute guarded light-trace re-run **once the
  environment is healthy** (post cache-clear / fresh env / reboot).

## Mitigation already in place
- `gpu_wait_clear.sh <min_free_gb> [timeout] [poll]` — general (param-driven, device-read
  total) pre-run VRAM gate that WAITS for the GPU to clear before a run, so serialized
  big-model runs can't OOM each other. Currently a session tool; can be committed as a
  repo pre-flight if wanted. NOTE: it gates on VRAM, which does NOT prevent the compile
  hang (VRAM is clean when the hang occurs) — the hang fix is separate (this doc).
