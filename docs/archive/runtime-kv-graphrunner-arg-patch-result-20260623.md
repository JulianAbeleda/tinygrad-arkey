# Runtime-KV GraphRunner Arg-Patch — Result (2026-06-23)

## Verdict: **RUNTIME_KV_ARG_PATCH_VALUES_CORRECT_DATA_STALE**

Kernel-level instrumentation proves the **GraphRunner arg patching is correct** — `start_pos` advances per replay
and every append/tile call is patched. The baking is a **DATA** failure: the owned AMDGCN tile is fed **NaN K**
in-model. And the root cause is **not RUNTIME_KV at all** — the **owned tile route itself is broken for real
multi-step decode** (B4 with the fp32 cache bakes too; gqa works). The arg-patch / graph-staleness / device-buffer
direction is **refuted**. Route reverted (`model.py` clean, default decode byte-identical).

## 1. Verdict
`RUNTIME_KV_ARG_PATCH_VALUES_CORRECT_DATA_STALE` — args correct; the data fed to the owned tile is NaN.

## 2. What previous results refuted
Buffer-identity/rebase (`RUNTIME_KV_BUFFER_IDENTITY_DIFF_NOT_FOUND`): cache is not the cause. This task refutes
the remaining lane (arg patching / graph-replay scalar staleness / device-buffer position).

## 3. Instrumentation method
`extra/qk_runtime_kv_graphrunner_arg_probe.py` monkeypatches `HCQGraph` to record, per decode graph: `self.vars`,
`var_vals_replace` for every `kv_append`/`owned_flash` call, the per-call `ProgramInfo.vars`, and the `start_pos`
passed to each replay. Plus targeted hooks on `kv_append_node` (input k,v finiteness) and `amdgcn_flash_decode`
(tile Q,K finiteness), a standalone tile sweep, and the B4-vs-RUNTIME-vs-gqa comparison. **Token correctness** is
the authority throughout (positions-written was abandoned as unreliable).

## 4. Microbench vs full-model arg-patch comparison
- **Args are CORRECT in the full model.** `start_pos` ∈ every append-graph `.vars` = True; every `kv_append` call
  ∈ `var_vals_replace` (`replace=[(0,0)]`) = True; `start_pos` passed to replays advances **2049→2050→2051→2052**.
  (`runtimevars` is only `'core_id'`, so `start_pos` is never excluded from patching — jit.py:138/144.)
- The HCQ graph submits `var_vals` (with the updated `start_pos`) on each replay (hcq.py:275/290).
- **Disabling the graph does not help**: `JIT_BATCH_SIZE=1` (no graphing) bakes; **plain `m.forward` each step
  (no TinyJit at all) bakes.** So it is not graph-replay scalar staleness.

## 5. Append/tile start_pos analysis
Both `kv_append` and `owned_flash_tile_gqa` declare `prog_vars=['start_pos']` and are in `var_vals_replace`. The
scalar is patched and advances. **Scalar patching is not the bug.**

## 6. Per-layer / layer-divergence analysis
- A **misread is corrected**: the "first token correct" (13876/34208) was always the **prefill output**
  (`toks[0]`). **Every decode step is garbage from step 1.** The whole prior saga's "first correct, replays
  garbage" was "prefill output correct, all decode garbage."
- **First divergent layer = layer 0.** Tile-input hook: layer 0 receives **Q finite (absmax 8.7) but K = NaN**
  (`K_nonzero = 2,098,176 = exactly Hkv·2049·Hd`, the valid prefix). Layer 0's attention output is NaN → layer 1
  inputs NaN → cascade → logits → token `151936`.
- **Prefill cache K[0:2048] is CLEAN** (finite, all 36 blocks, absmax 222). The **decode append introduces the
  NaN** at the appended position (the per-step scan shows `pos2048` finite=False after the append) — despite the
  append's logical k input being **finite** (absmax 222), `src` being **contiguous** with the correct layout, and
  the **isolated microbench append being correct**.
- **The tile is correct standalone**: `amdgcn_flash_decode` is finite + matches numpy at K magnitude 0.5/5/50/200,
  n_valid 2049 (rel_rmse e-7..e-5). So the tile math is fine; the failure is the **in-model data fed to it**.

## 7. The real root cause — the owned tile route, not RUNTIME_KV
- **B4 owned tile with the fp32 canonical cache (`DECODE_ATTN_AMDGCN_TILE=1`, no RUNTIME_KV) ALSO bakes**
  (`[13876, 151936, …]`); **gqa baseline is correct** (`[13876, 38835, 34208, …]`). So the owned-tile route is
  broken for real multi-step decode independent of RUNTIME_KV/fp16/cache.
- **B4's specific bug**: the tile expects `__half` K but the canonical cache is `dtypes.float` (fp32) → it reads
  fp32 bytes as fp16 = garbage → NaN. **Masked** because the owned tile was only ever validated in the W==D
  harness with a **degenerate/uninitialized (zero) cache**, where real K (~222) was never exercised.
- **RUNTIME_KV's bug** (fp16 cache, correct dtype for the tile): the **decode append writes NaN** at the appended
  cache position in the full-model context — the remaining open thread (likely a full-model realization/aliasing
  of the in-place opaque write, or a k value the fp16 store turns to inf that `numpy()` of the lazy tensor doesn't
  surface).

## 8. Correctness result
**FAIL** (DATA). No fix reached the correctness gate.

## 9. W==D result
**Not reached** — correctness failed.

## 10. Candidate / default decision
**Not registered.** Additionally, this result **flags that the existing default-eligible owned-tile candidate
(`decode_attention_owned_amdgcn`, `DECODE_ATTN_AMDGCN_TILE`) is broken for real multi-step decode** and must be
fixed or de-listed before any promotion. Defaults unchanged here.

## 11. Remaining blockers
The owned-tile **integration** is the blocker, not KV-cache arg patching:
1. **B4 owned tile**: fp32-cache-read-as-fp16 → de-list or make it read/cast fp16; re-validate against gqa with a
   **real (non-zero) cache** under token correctness.
2. **RUNTIME_KV append**: why the opaque decode append writes NaN at the appended cache position in the full model
   despite finite logical k, contiguous src, and a passing microbench.

Both are owned-tile-route DATA correctness; the arg-patch / device-buffer-position lane is exhausted.

## 12. Artifacts and commands
- `extra/qk_runtime_kv_graphrunner_arg_probe.py` → `bench/qk-runtime-managed-kv-cache/graphrunner_arg_probe.json`.
- Probe: `DEV=AMD JIT=1 Q4K_GEMV_WARP=1 Q4K_GEMV_WARP_DOWN=1 RUNTIME_KV_CACHE=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_graphrunner_arg_probe.py`.
- Key checks (commands in the session): B4 bake `... DECODE_ATTN_AMDGCN_TILE=1 ...` chunked-prefill decode; gqa
  correct (no tile flag); no-graph `... JIT_BATCH_SIZE=1 ...`; standalone tile sweep.

## 13. Files changed
Source: **none shipped** — `RUNTIME_KV_CACHE` route re-applied for instrumentation then **reverted** (`model.py`
byte-clean, default decode byte-identical `[279, 1156, 22148, …]`). New: `extra/qk_runtime_kv_graphrunner_arg_probe.py`,
`docs/runtime-kv-graphrunner-arg-patch-result-20260623.md`, `bench/qk-runtime-managed-kv-cache/graphrunner_arg_probe.json`.

## 14. Working tree status
`model.py` clean (reverted). New probe + doc + bench artifact only. No default change; no 14B/32B; no paged KV;
no new attention tile; no RoPE kernel; no activation/norm/GEMV work; no broad GraphRunner redesign.
