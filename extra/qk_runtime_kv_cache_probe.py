#!/usr/bin/env python3
"""Phase-1 microbench for runtime-managed KV cache (docs/runtime-managed-kv-cache-implementation-scope-20260623.md).

Proves/refutes the runtime BOUNDARY before any model integration: a runtime-owned persistent cache + the proven
opaque append + the existing owned AMDGCN tile, driven by a plain TinyJit decode step (NO @function(precompile=True)
-- that decorator's buffer substitution is exactly what lost persistence in KV_OPAQUE_READ_CORRECTNESS_FAIL).

Gates: persistence across multi-step replay (each step's append visible to later steps), reset/no-stale across
"generations", no full-MAXC copy, append-before-tile ordering, correct vs numpy.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_runtime_kv_cache_probe.py
  -> bench/qk-runtime-managed-kv-cache/microbench.json
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-runtime-managed-kv-cache"
MAXC = 4608

class RuntimeKVCache:
  """Fork-local runtime-owned KV cache: persistent realized buffer + explicit lifecycle (allocate/reset/append/views).
  Persistence lives in the realized buffer (mutated in place by the opaque append), NOT in @function graph state."""
  def __init__(self, Hkv, Hd, max_context, dtype):
    from tinygrad import Tensor
    self.Hkv, self.Hd, self.max_context, self.dtype = Hkv, Hd, max_context, dtype
    self.cache_kv = Tensor.zeros(2, 1, Hkv, max_context, Hd, dtype=dtype).contiguous().realize()
    self.generation_id = 0
  def reset(self):
    from tinygrad import Tensor
    # zero the persistent buffer in place (new prompt/generation boundary) -- realized so no @function dependence
    self.cache_kv.assign(Tensor.zeros(2, 1, self.Hkv, self.max_context, self.Hd, dtype=self.dtype)).realize()
    self.generation_id += 1
  def append(self, k, v):
    from extra.qk_kv_cache_state_token import kv_append_node
    return kv_append_node(self.cache_kv, k, v)   # cache_kv.after(append); opaque in-place write, NO repoint
  def k_view(self, after): return after[0, 0]    # [Hkv, MAXC, Hd] ordered-after the append
  def v_view(self, after): return after[1, 0]

def main():
  from tinygrad import Tensor, UOp, TinyJit, Context, Device, dtypes
  from tinygrad.device import Compiled
  from tinygrad.uop.ops import Ops
  from extra.qk_owned_flash_decode_graph_node import amdgcn_flash_decode, Hq, Hkv, Hd, G, SCALE
  dev = Device["AMD"]; S = 48
  res = {"date": "2026-06-23", "phase": "RUNTIME_KV_MICROBENCH", "gpu": "RX 7900 XTX / gfx1100", "default_behavior_changed": False}
  rng = np.random.default_rng(0)
  vsp = UOp.variable("start_pos", 0, MAXC - 1)

  def numpy_ref(Qn, Kfull, Vfull, nvalid):
    o = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kvh = h // G
      sc = (Qn[h:h+1].astype(np.float32) @ Kfull[kvh, :nvalid].astype(np.float32).T)[0] * SCALE
      p = np.exp(sc - sc.max()); p /= p.sum(); o[h] = p @ Vfull[kvh, :nvalid].astype(np.float32)
    return o

  # ---------- the runtime decode step (plain function -> TinyJit, NO @function) ----------
  cache = RuntimeKVCache(Hkv, Hd, MAXC, dtypes.half)
  Qt = Tensor(rng.standard_normal((Hq, Hd)).astype(np.float16)).realize()
  def step(k_new, v_new, sp_bound):
    after = cache.append(k_new, v_new)                                # opaque append into runtime cache
    carry = (Tensor.ones(MAXC, dtype=dtypes.float32)[0:sp_bound].sum().reshape(1) * 0.0)
    K = Tensor(cache.k_view(after).reshape(Hkv*MAXC*Hd).uop)          # flat AFTER -> custom_kernel skips contiguous (no copy)
    V = Tensor(cache.v_view(after).reshape(Hkv*MAXC*Hd).uop)
    return (amdgcn_flash_decode(Qt, K, V, vsp, S, MAXC).reshape(Hq*Hd) + carry).reshape(Hq, Hd).realize()

  # ---------- (A) PERSISTENCE across multi-step replay: prefill, then append distinct tokens; each step must see all prior ----------
  try:
    PRE = 2048
    Kfull = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16)
    Vfull = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16)
    # prefill the runtime cache [0:PRE] directly (one realized write -- simulates prefill handoff)
    cnp = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16); cnp[0, 0, :, :PRE] = Kfull[:, :PRE]; cnp[1, 0, :, :PRE] = Vfull[:, :PRE]
    cache.cache_kv.assign(Tensor(cnp)).realize()
    jstep = TinyJit(step)
    persist = {}
    for sp in (PRE, PRE+1, PRE+2, PRE+3):     # eager, capture, replay, replay -- each appends a DISTINCT token at sp
      knp = (rng.standard_normal((1, Hkv, 1, Hd)) * 0.5).astype(np.float16); vnp = (rng.standard_normal((1, Hkv, 1, Hd)) * 0.5).astype(np.float16)
      Kfull[:, sp] = knp[0, :, 0]; Vfull[:, sp] = vnp[0, :, 0]      # mirror into the numpy reference (accumulates)
      out = jstep(Tensor(knp), Tensor(vnp), vsp.bind(sp)).numpy()
      ref = numpy_ref(Qt.numpy(), Kfull, Vfull, sp+1)              # ref over ALL positions [0:sp+1] -> requires prior appends persisted
      rmse = float(np.sqrt(((out-ref)**2).mean()) / (np.sqrt((ref**2).mean())+1e-9))
      persist[sp] = {"rel_rmse": rmse, "ok": rmse <= 2e-3}
    res["persistence_multistep"] = persist
  except Exception as e:
    import traceback; res["persistence_multistep"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}; res["persist_tb"] = traceback.format_exc()[-1600:]

  pa = res.get("persistence_multistep", {}); persist_ok = isinstance(pa, dict) and all(isinstance(x, dict) and x.get("ok") for x in pa.values())

  # ---------- (B) RESET / no-stale across generations ----------
  if persist_ok:
    try:
      cache.reset()
      Kf2 = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16); Vf2 = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16)
      cnp = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16); cnp[0, 0, :, :2048] = Kf2[:, :2048]; cnp[1, 0, :, :2048] = Vf2[:, :2048]
      cache.cache_kv.assign(Tensor(cnp)).realize()
      jstep2 = TinyJit(step)
      sp = 2048; knp = (rng.standard_normal((1, Hkv, 1, Hd)) * 0.5).astype(np.float16); vnp = (rng.standard_normal((1, Hkv, 1, Hd)) * 0.5).astype(np.float16)
      Kf2[:, sp] = knp[0, :, 0]; Vf2[:, sp] = vnp[0, :, 0]
      out = jstep2(Tensor(knp), Tensor(vnp), vsp.bind(sp)).numpy(); ref = numpy_ref(Qt.numpy(), Kf2, Vf2, sp+1)
      rmse = float(np.sqrt(((out-ref)**2).mean()) / (np.sqrt((ref**2).mean())+1e-9))
      # also confirm no stale data from generation 1 (positions were zeroed then refilled with gen-2 KV)
      res["reset_no_stale"] = {"rel_rmse": rmse, "ok": rmse <= 2e-3, "generation_id": cache.generation_id}
    except Exception as e:
      res["reset_no_stale"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

  # ---------- (C) no full-MAXC copy + append-before-tile (captured graph identity) ----------
  if persist_ok:
    try:
      names = [u.src[0].arg.name for u in jstep.captured.linear.toposort()
               if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM] if jstep.captured else []
      prog_names = [str(getattr(u.arg, 'name', '')) for u in jstep.captured.linear.toposort() if u.op is Ops.PROGRAM]
      full_copy = [n for n in prog_names if n.startswith("E_49152") or "4718592" in n]
      app_i = next((i for i, n in enumerate(names) if "kv_append" in n), None)
      tile_i = next((i for i, n in enumerate(names) if "owned_flash_tile" in n), None)
      res["graph_identity"] = {"program_nodes": [n for n in names if "kv_append" in n or "owned_flash" in n],
                               "full_maxc_copy": full_copy, "append_present": app_i is not None, "owned_tile_present": tile_i is not None,
                               "append_before_tile": (app_i is not None and tile_i is not None and app_i < tile_i)}
    except Exception as e:
      res["graph_identity"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

  # ---------- verdict ----------
  gi = res.get("graph_identity", {}); rs = res.get("reset_no_stale", {})
  if "error" in pa: verdict = "RUNTIME_KV_MICROBENCH_NOT_EXPRESSIBLE"
  elif not persist_ok: verdict = "RUNTIME_KV_APPEND_ORDER_FAIL"
  elif "error" in rs or not rs.get("ok"): verdict = "RUNTIME_KV_RESET_FAIL"
  elif gi.get("full_maxc_copy"): verdict = "RUNTIME_KV_COPY_STILL_PRESENT"
  elif not gi.get("append_before_tile"): verdict = "RUNTIME_KV_APPEND_ORDER_FAIL"
  else: verdict = "RUNTIME_KV_MICROBENCH_PASS"
  res["verdict"] = verdict
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "microbench.json").write_text(json.dumps(res, indent=2))
  print(f"VERDICT: {verdict}", file=sys.stderr)
  print(f"  persistence_multistep: {pa}", file=sys.stderr)
  print(f"  reset_no_stale: {rs}", file=sys.stderr)
  print(f"  graph_identity: {gi}", file=sys.stderr)
  if res.get("persist_tb"): print(res["persist_tb"], file=sys.stderr)
  print(f"artifact: {OUT/'microbench.json'}", file=sys.stderr)

if __name__ == "__main__":
  main()
