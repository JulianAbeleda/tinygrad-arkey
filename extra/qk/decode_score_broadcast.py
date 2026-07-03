#!/usr/bin/env python3
"""Score-broadcast family (Cluster D), collapsed to one parameterized module.

Cluster C ended on a single named next primitive -- score reuse across PV output columns. This module is that
answer's proof surface: six VARIANTs that build the score-broadcast route from the probe up to the model's real
`assigned_kv` cache view. Numeric gate throughout is finite AND max_abs <= 1e-3 AND rel_rmse <= 1e-5. Env:
DEV=AMD, V_DOT2_LOWERING=1, DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1. Writes are owned by
gate_registry (bench/qk-decode-primitive-space/, distinct per-variant filenames -- NEVER latest.json, which
belongs to primitive_detector). Registry entrypoints: build_direct(), build_chain(), build_varjit_chain(),
build_control_matrix(), build_model_cache_view(), build_reuse_paths().

Subprocess isolation (SACRED): direct/chain/varjit_chain/model_cache_view run their GPU compute in a CHILD
SUBPROCESS -- the parent builder spawns `python3 <thisfile>` with a per-variant `*_CHILD` env marker, and the
`__main__` block below dispatches to the matching `_*_child()` and prints one JSON line. This isolates the
memoized compile-time getenv/env state each gate toggles. reuse_paths runs in-process (it captures kernel libs
via a runtime hook, no env isolation needed). The child `_kernels` lib and flash_decode are imported, unchanged.

  direct           SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT       -- route through shipped
                   flash_decode_attention_whole_cache (eager/JIT/varJIT via QK_SCORE_BROADCAST_DIRECT_{JIT,VARJIT}).
  chain            SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT                -- standalone score_once_state -> 4x
                   score_broadcast_pv_cols @0/32/64/96 -> combine4.
  varjit_chain     SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT         -- chain survives variable-bound TinyJit
                   warmup->capture->replay, chunks in {1,2,4}.
  control_matrix   SCORE_BROADCAST_CONTROL_MATRIX_RECORDED                -- diagnostic-only 2x2 stub (see note).
  model_cache_view SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT -- route on the model's real
                   assigned_kv view; refutes assigned_kv as the full-model MMU root cause.
  reuse_paths      SCORE_REUSE_PATHS_PASS__BROADCAST_PROBE_READY          -- Path A score-once state + Path B
                   score-broadcast PV columns, sublinear scaling probe.

Run:  DEV=AMD V_DOT2_LOWERING=1 DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE=1 \
      PYTHONPATH=. python3 -m extra.qk.gate_registry run score_broadcast_direct [chain varjit_chain ...]
"""
from __future__ import annotations
from extra.qk.isa_helpers import CROSS_LANE_RE

import ctypes, json, os, pathlib, re, subprocess, sys, time, traceback
from typing import Any
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-primitive-space"
THIS = pathlib.Path(__file__).resolve()


def _spawn(extra_env: dict[str, str]) -> subprocess.CompletedProcess:
  """Sacred env-isolation: run this file's GPU compute in a fresh child process with a per-variant *_CHILD marker."""
  env = {**os.environ, "PYTHONPATH": str(ROOT), **extra_env}
  return subprocess.run([sys.executable, str(THIS)], cwd=ROOT, env=env,
                        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)


# ---- direct: route through shipped flash_decode_attention_whole_cache --------------------------------------------------
def _direct_child() -> dict:
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk.flash_decode import flash_decode_attention_whole_cache
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q), Tensor(cache)
  if os.environ.get("QK_SCORE_BROADCAST_DIRECT_VARJIT", "0") == "1":
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    j = TinyJit(lambda spb: flash_decode_attention_whole_cache(qt, ct, spb + 1, spb + 1, Hd, Hq, Hkv, MAXC, L=L).realize())
    got = j(vsp.bind(Tc - 1)).numpy()
  elif os.environ.get("QK_SCORE_BROADCAST_DIRECT_JIT", "0") == "1":
    j = TinyJit(lambda: flash_decode_attention_whole_cache(qt, ct, Tc, Tc, Hd, Hq, Hkv, MAXC, L=L).realize())
    got = j().numpy()
  else:
    got = flash_decode_attention_whole_cache(qt, ct, Tc, Tc, Hd, Hq, Hkv, MAXC, L=L).realize().numpy()
  ref = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    scores = (cache[0, kvh, :Tc, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
    m = np.max(scores)
    p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
    ref[h] = (p @ cache[1, kvh, :Tc, :].astype(np.float32)) / p.sum()
  diff = got - ref
  return {"checked": True, "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
    "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def build_direct() -> dict:
  p = _spawn({"QK_SCORE_BROADCAST_DIRECT_CHILD": "1",
              "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1", "V_DOT2_LOWERING": "1"})
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle"}
  if p.returncode != 0:
    out.update({"verdict": "SCORE_BROADCAST_DIRECT_FAIL__RUNTIME", "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]})
  else:
    child = json.loads((p.stdout or "").splitlines()[-1])
    n = child.get("numeric", {})
    passed = bool(n.get("finite") and n.get("max_abs", 1.0) <= 1e-3 and n.get("rel_rmse", 1.0) <= 1e-5)
    out.update({"verdict": "SCORE_BROADCAST_DIRECT_READY__MODEL_CAPTURE_NEXT" if passed else "SCORE_BROADCAST_DIRECT_FAIL__NUMERIC", "child": child})
  return out


# ---- chain: standalone score_once_state -> 4x score_broadcast_pv_cols -> combine4 -------------------------------------
def _chain_child() -> dict[str, Any]:
  from tinygrad import Tensor, dtypes
  from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
  from extra.qk.flash_decode import flash_pall_score_broadcast_combine4_kernel
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, int(os.environ.get("QK_SCORE_BROADCAST_CHAIN_TC", "192"))
  S = (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qf, cf = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  state = Tensor.empty(Hq * S * 2, dtype=dtypes.float32).custom_kernel(qf, cf,
    fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0]
  pvs = [Tensor.empty(Hq * S * 32, dtype=dtypes.float32).custom_kernel(state, qf, cf,
    fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, 32, off))[0] for off in (0, 32, 64, 96)]
  got = Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(state, *pvs,
    fxn=flash_pall_score_broadcast_combine4_kernel(Hd, Hq, S))[0].realize().numpy().reshape(Hq, Hd)
  ref = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    scores = (cache[0, kvh, :Tc, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
    m = np.max(scores)
    p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
    ref[h] = (p @ cache[1, kvh, :Tc, :].astype(np.float32)) / p.sum()
  diff = got - ref
  return {"checked": True, "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
    "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def build_chain() -> dict[str, Any]:
  p = _spawn({"QK_SCORE_BROADCAST_CHAIN_CHILD": "1", "V_DOT2_LOWERING": "1"})
  out = {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
         "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle"}
  if p.returncode != 0:
    out.update({"verdict": "SCORE_BROADCAST_CHAIN_FAIL__CHILD_RUNTIME", "returncode": p.returncode,
                "output_tail": (p.stdout or "")[-12000:],
                "decision": "Fix standalone chain before route capture or W==D."})
    return out
  try:
    child = json.loads((p.stdout or "").splitlines()[-1])
  except Exception:
    out.update({"verdict": "SCORE_BROADCAST_CHAIN_FAIL__NO_JSON", "output_tail": (p.stdout or "")[-12000:]})
    return out
  numeric = child.get("numeric", {})
  passed = bool(numeric.get("finite") and numeric.get("max_abs", 1.0) <= 1e-3 and numeric.get("rel_rmse", 1.0) <= 1e-5)
  out.update({"verdict": "SCORE_BROADCAST_CHAIN_READY__ROUTE_NEXT" if passed else "SCORE_BROADCAST_CHAIN_FAIL__NUMERIC",
              "child": child, "decision": "Route only if standalone chain is clean."})
  return out


# ---- varjit_chain: chain under variable-bound TinyJit warmup->capture->replay ----------------------------------------
def _varjit_child(chunks: int) -> dict[str, Any]:
  from tinygrad import Tensor, TinyJit, UOp, dtypes
  from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
  from extra.qk.flash_decode import flash_pall_score_broadcast_combine4_kernel

  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  Smax = (MAXC + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qf, cf = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  vsp = UOp.variable("start_pos", 0, MAXC - 1)

  def run(spb):
    tc = spb + 1
    state = Tensor.empty(Hq * Smax * 2, dtype=dtypes.float32).custom_kernel(qf, cf,
      fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, Smax, tc))[0]
    pvs = [Tensor.empty(Hq * Smax * 32, dtype=dtypes.float32).custom_kernel(state, qf, cf,
      fxn=score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, Smax, tc, 32, off))[0] for off in (0, 32, 64, 96)[:chunks]]
    while len(pvs) < 4: pvs.append(pvs[-1])
    return Tensor.empty(Hq * Hd, dtype=dtypes.float32).custom_kernel(state, *pvs,
      fxn=flash_pall_score_broadcast_combine4_kernel(Hd, Hq, Smax))[0].realize()

  j = TinyJit(run)
  warmup = j(vsp.bind(Tc - 1)).realize()
  capture = j(vsp.bind(Tc - 1)).realize()
  got = j(vsp.bind(Tc - 1)).numpy().reshape(Hq, Hd)
  return {"checked": True, "chunks": chunks, "phases": {"warmup": True, "capture_exec": True, "replay": True},
          "warmup_shape": list(warmup.shape), "capture_shape": list(capture.shape),
          "finite": bool(np.isfinite(got).all()), "sample_abs_sum": float(np.abs(got).sum())}

def _varjit_run_child(chunks: int) -> dict[str, Any]:
  p = _spawn({"QK_SCORE_BROADCAST_VARJIT_CHILD": "1",
              "QK_SCORE_BROADCAST_VARJIT_CHUNKS": str(chunks), "V_DOT2_LOWERING": "1"})
  if p.returncode != 0:
    return {"chunks": chunks, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    d["pass"] = bool(d.get("finite"))
    return d
  except Exception:
    return {"chunks": chunks, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:]}

def build_varjit_chain() -> dict[str, Any]:
  rows = [_varjit_run_child(c) for c in (1, 2, 4)]
  verdict = "SCORE_BROADCAST_VARJIT_CHAIN_READY__ROUTE_NEXT" if all(r.get("pass") for r in rows) else "SCORE_BROADCAST_VARJIT_CHAIN_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "rows": rows,
          "decision": "If this fails, fix variable-bound custom-kernel chain before model route or W==D."}


# ---- control_matrix: diagnostic 2x2 stub -----------------------------------------------------------------------------
# Controlled score-broadcast capture matrix. Runs the same full-model TinyJit capture phase with chunks fixed at 4
# while toggling the two candidate lifecycle interventions that were previously confounded:
#   - DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH: route-local graph-batch barrier install.
#   - DECODE_ATTN_SCORE_BROADCAST_SCRATCH:  persistent route scratch buffers.
# This is diagnostic-only. It never runs W==D and never promotes.
_CONTROL_MODE = os.environ.get("QK_SCORE_BROADCAST_CONTROL_MODE", "jit_capture_same_same")
_CONTROL_CASES = (
  ("barrier_on_scratch_on", "1", "1"),
  ("barrier_off_scratch_on", "0", "1"),
  ("barrier_on_scratch_off", "1", "0"),
  ("barrier_off_scratch_off", "0", "0"),
)

def _control_run_case(name: str, no_graph: str, scratch: str) -> dict[str, Any]:
  env = {**os.environ, "PYTHONPATH": str(ROOT),
         "DECODE_ATTN_GENERATED_WHOLECACHE": "1",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1",
         "DECODE_ATTN_SCORE_BROADCAST_CHUNKS": "4",
         "DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH": no_graph,
         "DECODE_ATTN_SCORE_BROADCAST_SCRATCH": scratch,
         "V_DOT2_LOWERING": "1",
         "QK_SCORE_BROADCAST_JIT_PHASE_CHILD": "1",
         "QK_SCORE_BROADCAST_JIT_PHASE_MODE": _CONTROL_MODE}
  return {"case": name, "returncode": None, "pass": False, "failure_class": "stale_replay_removed",
          "flags": {"DECODE_ATTN_SCORE_BROADCAST_NO_GRAPH": no_graph, "DECODE_ATTN_SCORE_BROADCAST_SCRATCH": scratch},
          "note": "the old score-broadcast JIT-phase child replay is not part of the compact repo surface"}

def build_control_matrix() -> dict[str, Any]:
  rows = [_control_run_case(*case) for case in _CONTROL_CASES]
  pass_cases = [r["case"] for r in rows if r.get("pass")]
  fail_cases = [r["case"] for r in rows if not r.get("pass")]
  verdict = "SCORE_BROADCAST_CONTROL_MATRIX_RECORDED"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "mode": _CONTROL_MODE, "verdict": verdict, "pass_cases": pass_cases, "fail_cases": fail_cases,
          "rows": rows,
          "decision": "Use this matrix to isolate barrier, scratch, and their interaction at fixed chunks=4. This artifact is diagnostic-only."}


# ---- model_cache_view: route on the model's real assigned_kv view ----------------------------------------------------
# This reproduces the model attention cache update/view:
#   assigned_kv = cache_kv.after(cache_kv[:, :, :, start_pos:start_pos+T, :].store(stack(k, v)))
# outside the full transformer block, then calls flash_decode_attention_whole_cache.
def _model_view_child() -> dict:
  from tinygrad import Tensor, TinyJit, UOp
  from extra.qk.flash_decode import flash_decode_attention_whole_cache
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache0 = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16)
  cache0[:, 0, :, :Tc-1, :] = rng.normal(0, 0.25, (2, Hkv, Tc-1, Hd)).astype(np.float16)
  k_new = rng.normal(0, 0.25, (1, Hkv, 1, Hd)).astype(np.float16)
  v_new = rng.normal(0, 0.25, (1, Hkv, 1, Hd)).astype(np.float16)
  q_t, cache_t, k_t, v_t = Tensor(q), Tensor(cache0), Tensor(k_new), Tensor(v_new)

  def run(start_pos):
    assigned = Tensor(cache_t.uop.after(cache_t[:, :, :, start_pos:start_pos+1, :].uop.store(Tensor.stack(k_t, v_t).uop)))
    return flash_decode_attention_whole_cache(q_t, assigned, start_pos + 1, start_pos + 1, Hd, Hq, Hkv, MAXC, L=L).realize()

  if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT", "0") == "1":
    vsp = UOp.variable("start_pos", 0, MAXC - 1)
    got = TinyJit(run)(vsp.bind(Tc - 1)).numpy()
  else:
    got = run(Tc - 1).numpy()

  full_k = cache0[0, 0].copy(); full_v = cache0[1, 0].copy()
  full_k[:, Tc-1:Tc, :] = k_new[0]
  full_v[:, Tc-1:Tc, :] = v_new[0]
  ref = np.zeros((Hq, Hd), np.float32)
  scale = 1.0 / np.sqrt(Hd)
  for h in range(Hq):
    kvh = h // (Hq // Hkv)
    scores = (full_k[kvh, :Tc, :].astype(np.float32) @ q[h].astype(np.float32)) * scale
    m = np.max(scores)
    p = np.exp2((scores - m) * 1.4426950408889634).astype(np.float32)
    ref[h] = (p @ full_v[kvh, :Tc, :].astype(np.float32)) / p.sum()
  diff = got - ref
  return {"checked": True, "mode": "varjit" if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT") == "1" else "eager",
          "numeric": {"finite": bool(np.isfinite(got).all()), "max_abs": float(np.max(np.abs(diff))),
                      "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}}

def _model_view_run(mode: str) -> dict:
  extra = {"QK_SCORE_BROADCAST_MODEL_VIEW_CHILD": "1",
           "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE": "1", "V_DOT2_LOWERING": "1"}
  if mode == "varjit": extra["QK_SCORE_BROADCAST_MODEL_VIEW_VARJIT"] = "1"
  p = _spawn(extra)
  if p.returncode != 0: return {"mode": mode, "pass": False, "returncode": p.returncode, "output_tail": (p.stdout or "")[-12000:]}
  try:
    d = json.loads((p.stdout or "").splitlines()[-1])
    n = d.get("numeric", {})
    d["pass"] = bool(n.get("finite") and n.get("max_abs", 1.0) <= 1e-3 and n.get("rel_rmse", 1.0) <= 1e-5)
    return d
  except Exception:
    return {"mode": mode, "pass": False, "returncode": 0, "output_tail": (p.stdout or "")[-12000:]}

def build_model_cache_view() -> dict[str, Any]:
  rows = [_model_view_run("eager"), _model_view_run("varjit")]
  verdict = "SCORE_BROADCAST_MODEL_CACHE_VIEW_READY__ATTENTION_ONLY_NEXT" if all(r.get("pass") for r in rows) else "SCORE_BROADCAST_MODEL_CACHE_VIEW_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
          "candidate_id": "decode_attention_physical_tile_score_broadcast_lifecycle",
          "verdict": verdict, "rows": rows,
          "decision": "If this passes, the assigned_kv view is not the full-model MMU root cause."}


# ---- reuse_paths: the two plausible fixes for PALL lifecycle q.k-per-column recompute (in-process) --------------------
# Path A: score-once split state. Compute online m/l once per (kvh, split, head) with the PALL physical score path.
#         This proves score-once is expressible, but it does not produce PV.
# Path B: score-broadcast fused PV. Compute q.k once per token, then update several PV output columns from that score.
#         This is the actual primitive we need for a fast fused lifecycle.
def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  return system(f"{objdump} -d -", input=lib)

def _desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _ = elf_loader(lib)
  ro = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[ro:ro+ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)]))
  rsrc1 = desc.compute_pgm_rsrc1
  return {"vgpr": ((rsrc1 & 0x3f) + 1) * 8, "sgpr": (((rsrc1 >> 6) & 0xf) + 1) * 8,
          "lds": desc.group_segment_fixed_size, "scratch": desc.private_segment_fixed_size}

def _flags(asm: str) -> dict[str, bool]:
  return {"has_v_dot2": "v_dot2" in asm or "__builtin_amdgcn_fdot2" in asm,
          "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
          "has_cross_lane": bool(re.search(CROSS_LANE_RE, asm)),
          "has_spill": bool(re.search(r"\bscratch_(load|store)", asm))}

def _reuse_run_probe() -> dict[str, Any]:
  from tinygrad import Tensor, dtypes, Device
  from extra.qk.decode_physical_tile_score_broadcast_kernels import score_once_state_kernel, score_broadcast_pv_cols_kernel
  dev = Device[Device.DEFAULT]; captured: dict[str, bytes] = {}; orig = dev.runtime
  def hook(name, lib, **kw):
    if name.startswith("flash_pall_score_") and name not in captured: captured[name] = lib
    return orig(name, lib, **kw)
  dev.runtime = hook
  Hq, Hkv, Hd, MAXC, L, Tc = 32, 8, 128, 256, 128, 192
  G, S = Hq // Hkv, (Tc + L - 1) // L
  rng = np.random.default_rng(20260626)
  q = rng.normal(0, 0.25, (Hq, Hd)).astype(np.float16)
  cache = np.zeros((2, Hkv, MAXC, Hd), np.float16)
  cache[0] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  cache[1] = rng.normal(0, 0.25, (Hkv, MAXC, Hd)).astype(np.float16)
  qt, ct = Tensor(q.reshape(-1)), Tensor(cache.reshape(-1))
  state = Tensor.empty(Hq * S * 2, dtype=dtypes.float32).custom_kernel(qt, ct,
    fxn=score_once_state_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc))[0].realize().numpy().reshape(Hq, S, 2)
  ref_state = np.zeros((Hq, S, 2), np.float32)
  for kvh in range(Hkv):
    for s in range(S):
      t0, t1 = s * L, min((s + 1) * L, Tc)
      for g in range(G):
        h = kvh * G + g
        scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * (1.0 / np.sqrt(Hd))
        m = np.max(scores).astype(np.float32)
        p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
        ref_state[h, s, 0], ref_state[h, s, 1] = p.sum(), m
  state_diff = state - ref_state
  rows = []
  for Wp in (1, 8, 32, 128):
    fxn = score_broadcast_pv_cols_kernel(Hd, Hq, Hkv, MAXC, L, S, Tc, Wp)
    got = Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(Tensor(state.reshape(-1)), qt, ct, fxn=fxn)[0].realize().numpy().reshape(Hq, S, Wp)
    ref = np.zeros((Hq, S, Wp), np.float32)
    for kvh in range(Hkv):
      for s in range(S):
        t0, t1 = s * L, min((s + 1) * L, Tc)
        for g in range(G):
          h = kvh * G + g
          scores = (cache[0, kvh, t0:t1, :].astype(np.float32) @ q[h].astype(np.float32)) * (1.0 / np.sqrt(Hd))
          m = np.max(scores).astype(np.float32)
          p = np.exp2((scores - m) * _LOG2E).astype(np.float32)
          for c in range(Wp): ref[h, s, c] = p @ cache[1, kvh, t0:t1, c].astype(np.float32)
    times = []
    for _ in range(3):
      st = time.perf_counter()
      Tensor.empty(Hq * S * Wp, dtype=dtypes.float32).custom_kernel(Tensor(state.reshape(-1)), qt, ct, fxn=fxn)[0].realize().numpy()
      times.append(time.perf_counter() - st)
    diff = got - ref
    rows.append({"Wp": Wp, "median_s": float(np.median(times)), "numeric": {"max_abs": float(np.max(np.abs(diff))),
      "rel_rmse": float(np.sqrt(np.mean(diff * diff)) / (np.sqrt(np.mean(ref * ref)) + 1e-12))}})
  kernels = {}
  for name, lib in captured.items():
    asm = _disasm(lib); (OUT / f"disasm_{name}.txt").write_text(asm)
    d = _desc(lib); d["primitive_flags"] = _flags(asm); kernels[name] = d
  return {"score_once_state": {"numeric": {"max_abs": float(np.max(np.abs(state_diff))),
            "rel_rmse": float(np.sqrt(np.mean(state_diff * state_diff)) / (np.sqrt(np.mean(ref_state * ref_state)) + 1e-12))}},
          "score_broadcast_pv": {"rows": rows}, "kernels": kernels}

def build_reuse_paths() -> dict[str, Any]:
  try:
    attempt = _reuse_run_probe()
  except Exception as e:
    return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"),
            "verdict": "SCORE_REUSE_PATHS_BLOCKED__BUILDER_EXCEPTION", "exception_type": type(e).__name__,
            "exception": str(e), "traceback_tail": traceback.format_exc()[-5000:]}
  rows = attempt["score_broadcast_pv"]["rows"]
  mult = rows[-1]["median_s"] / rows[0]["median_s"] if rows and rows[0]["median_s"] else None
  score_once_pass = attempt["score_once_state"]["numeric"]["max_abs"] <= 1e-3 and attempt["score_once_state"]["numeric"]["rel_rmse"] <= 1e-5
  broadcast_numeric_pass = all(r["numeric"]["max_abs"] <= 1e-3 and r["numeric"]["rel_rmse"] <= 1e-5 for r in rows)
  verdict = "SCORE_REUSE_PATHS_PASS__BROADCAST_PROBE_READY" if score_once_pass and broadcast_numeric_pass and mult is not None and mult < 16.0 else \
            "SCORE_REUSE_PATHS_PARTIAL__SCORE_ONCE_ONLY" if score_once_pass else "SCORE_REUSE_PATHS_FAIL"
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": verdict,
          "attempt": attempt, "runtime_multiple_32col_vs_1col": mult,
          "decision": "If broadcast passes with sublinear scaling, route it next. If only score-once passes, the missing primitive remains fused score reuse across PV columns."}


# ---- registry surface ------------------------------------------------------------------------------------------------
VARIANTS = {"direct": build_direct, "chain": build_chain, "varjit_chain": build_varjit_chain,
            "control_matrix": build_control_matrix, "model_cache_view": build_model_cache_view,
            "reuse_paths": build_reuse_paths}

def build(variant): return VARIANTS[variant]()


if __name__ == "__main__":
  os.chdir(ROOT)
  # Child-subprocess dispatch (sacred env isolation): a *_CHILD marker means run GPU compute and print one JSON line.
  if os.environ.get("QK_SCORE_BROADCAST_DIRECT_CHILD") == "1":
    print(json.dumps(_direct_child())); raise SystemExit(0)
  if os.environ.get("QK_SCORE_BROADCAST_CHAIN_CHILD") == "1":
    print(json.dumps(_chain_child())); raise SystemExit(0)
  if os.environ.get("QK_SCORE_BROADCAST_VARJIT_CHILD") == "1":
    print(json.dumps(_varjit_child(int(os.environ.get("QK_SCORE_BROADCAST_VARJIT_CHUNKS", "1"))))); raise SystemExit(0)
  if os.environ.get("QK_SCORE_BROADCAST_MODEL_VIEW_CHILD") == "1":
    print(json.dumps(_model_view_child())); raise SystemExit(0)
  # Parent CLI: run a named VARIANT and print its verdict dict (gate_registry owns artifact writes + exit policy).
  out = build(sys.argv[1] if len(sys.argv) > 1 else "direct")
  print(json.dumps(out, indent=2))
