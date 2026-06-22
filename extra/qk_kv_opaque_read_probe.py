#!/usr/bin/env python3
"""Phase-1 probe for the Runtime-KV Opaque-Read route (docs/runtime-kv-opaque-read-followon-scope-20260623.md).

Isolates the core mechanism OUTSIDE the model: an opaque KV append (extra/qk_kv_cache_state_token.py) writes the
current token's K/V into the persistent cache IN PLACE, and the EXISTING owned AMDGCN tile
(extra/qk_owned_flash_decode_graph_node.py) reads that persistent cache directly -- both opaque custom_kernels, so
the same-graph read-after-write hazard that killed the `gqa_coop_vec` functional-reduce pairing should NOT fire, and
there is no `assigned_kv = cache.after(full store)` full-MAXC copy.

Gates: (1) correctness vs numpy flash-decode; (2) no full-MAXC copy kernel; (3) graph order append-before-tile;
(4) TinyJit capture/replay with changing start_pos.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_opaque_read_probe.py
  -> bench/qk-kv-opaque-read/probe.json
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-kv-opaque-read"
MAXC = 4608

def main():
  from tinygrad import Tensor, UOp, TinyJit, Context, Device, dtypes
  from tinygrad.device import Compiled
  from tinygrad.uop.ops import Ops
  from extra.qk_kv_cache_state_token import kv_append_node
  from extra.qk_owned_flash_decode_graph_node import amdgcn_flash_decode, Hq, Hkv, Hd, G, SCALE
  dev = Device["AMD"]; S = 48
  res = {"date": "2026-06-23", "phase": "KV_OPAQUE_READ_PROBE", "gpu": "RX 7900 XTX / gfx1100", "default_behavior_changed": False}
  rng = np.random.default_rng(0)
  Qn = rng.standard_normal((Hq, Hd)).astype(np.float16)
  Kn = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16)   # the "prior" cached KV (full)
  Vn = (rng.standard_normal((Hkv, MAXC, Hd)) * 0.5).astype(np.float16)
  Qt = Tensor(Qn).realize()
  def ref(nvalid):
    o = np.zeros((Hq, Hd), np.float32)
    for h in range(Hq):
      kvh = h // G
      sc = (Qn[h:h+1].astype(np.float32) @ Kn[kvh, :nvalid].astype(np.float32).T)[0] * SCALE
      p = np.exp(sc - sc.max()); p /= p.sum(); o[h] = p @ Vn[kvh, :nvalid].astype(np.float32)
    return o

  # ---------- (1) correctness (eager, multiple start_pos): append the token at sp (== its already-cached value, so
  # idempotent), owned tile reads [0:sp+1] of the PERSISTENT cache. exact vs numpy. ----------
  vsp = UOp.variable("start_pos", 0, MAXC - 1)
  corr = {}
  try:
    for sp in (511, 1023, 2047, 4095):
      cache_np = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16)
      cache_np[0, 0] = Kn; cache_np[1, 0] = Vn        # pre-fill the persistent cache with the prior KV
      cache = Tensor(cache_np).contiguous().realize()
      k_new = Tensor(Kn[:, sp].reshape(1, Hkv, 1, Hd).copy())   # the token's K/V == its cached value (idempotent append)
      v_new = Tensor(Vn[:, sp].reshape(1, Hkv, 1, Hd).copy())
      kv_append_node(cache, k_new, v_new)              # opaque in-place write at start_pos (repoints cache.uop)
      carry = (Tensor.ones(MAXC, dtype=dtypes.float32)[0:vsp.bind(sp)].sum().reshape(1) * 0.0)
      out = (amdgcn_flash_decode(Qt, cache[0, 0], cache[1, 0], vsp, S, MAXC).reshape(Hq * Hd) + carry).reshape(Hq, Hd).numpy()
      r = ref(sp + 1); rmse = float(np.sqrt(((out - r) ** 2).mean()) / (np.sqrt((r ** 2).mean()) + 1e-9))
      corr[sp] = {"rel_rmse": rmse, "ok": rmse <= 1e-3}
    res["correctness"] = corr
  except Exception as e:
    import traceback; res["correctness"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}; res["correctness_tb"] = traceback.format_exc()[-1800:]

  corr_ok = isinstance(res.get("correctness"), dict) and all(isinstance(v, dict) and v.get("ok") for v in res["correctness"].values())

  # ---------- (2)+(3) no full-MAXC copy + graph order (profile one forward) ----------
  if corr_ok:
    try:
      cache_np = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16); cache_np[0, 0] = Kn; cache_np[1, 0] = Vn
      cache = Tensor(cache_np).contiguous().realize()
      k_new = Tensor(Kn[:, 2047].reshape(1, Hkv, 1, Hd).copy()); v_new = Tensor(Vn[:, 2047].reshape(1, Hkv, 1, Hd).copy())
      with Context(PROFILE=1):
        kv_append_node(cache, k_new, v_new)
        carry = (Tensor.ones(MAXC, dtype=dtypes.float32)[0:vsp.bind(2047)].sum().reshape(1) * 0.0)
        base = len(Compiled.profile_events)
        (amdgcn_flash_decode(Qt, cache[0, 0], cache[1, 0], vsp, S, MAXC).reshape(Hq * Hd) + carry).realize()
        dev.synchronize(); dev._at_profile_finalize()
        names = []
        for e in Compiled.profile_events[base:]:
          if type(e).__name__ != "ProfileGraphEvent": continue
          for ent in e.ents: names.append(str(ent.name))
      full_copy = [n for n in names if n.startswith("E_49152") or (n.startswith("E_") and "4718592" in n)]
      halfish = [n for n in names if n.startswith("E_") and n not in full_copy]
      res["graph_profile"] = {"kernels": names, "append_present": any("kv_append" in n for n in names),
                              "owned_tile_present": any("owned_flash" in n for n in names),
                              "full_maxc_copy": full_copy, "other_E_copies": halfish}
    except Exception as e:
      res["graph_profile"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

  # ---------- (4) TinyJit capture/replay with changing start_pos ----------
  if corr_ok:
    try:
      cache_np = np.zeros((2, 1, Hkv, MAXC, Hd), np.float16); cache_np[0, 0] = Kn; cache_np[1, 0] = Vn
      cache = Tensor(cache_np).contiguous().realize()
      k_new = Tensor(Kn[:, 2047].reshape(1, Hkv, 1, Hd).copy()); v_new = Tensor(Vn[:, 2047].reshape(1, Hkv, 1, Hd).copy())
      def run(sp_bound):
        kv_append_node(cache, k_new, v_new)
        carry = (Tensor.ones(MAXC, dtype=dtypes.float32)[0:sp_bound].sum().reshape(1) * 0.0)
        return (amdgcn_flash_decode(Qt, cache[0, 0], cache[1, 0], vsp, S, MAXC).reshape(Hq * Hd) + carry).realize()
      jf = TinyJit(run)
      replay = {}
      for n in (2047, 2047, 1023, 3071):    # eager, capture, replay(diff), replay(diff)
        o = jf(vsp.bind(n)); replay[n] = {"shape": list(o.shape), "finite": bool(np.isfinite(o.numpy()).all())}
      captured = [u.src[0].arg.name for u in jf.captured.linear.toposort()
                  if u.op is Ops.CALL and len(u.src) and u.src[0].op is Ops.PROGRAM] if jf.captured else []
      res["jit_replay"] = {"per_start_pos": replay, "all_finite": all(v["finite"] for v in replay.values()),
                           "captured_program_nodes": [n for n in captured if "owned_flash" in n or "kv_append" in n],
                           "append_before_tile": ([n for n in captured if "kv_append" in n] and [n for n in captured if "owned_flash" in n]) and
                                                  (captured.index(next(n for n in captured if "kv_append" in n)) <
                                                   captured.index(next(n for n in captured if "owned_flash" in n)))}
    except Exception as e:
      import traceback; res["jit_replay"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-1200:]}

  # ---------- verdict ----------
  gp = res.get("graph_profile", {}); jr = res.get("jit_replay", {})
  if not corr_ok:
    verdict = "KV_OPAQUE_READ_CORRECTNESS_FAIL" if "error" not in res.get("correctness", {"error":1} if not corr_ok else {}) else "KV_OPAQUE_READ_HAZARD_PERSISTS"
    if isinstance(res.get("correctness"), dict) and "error" in res["correctness"]: verdict = "KV_OPAQUE_READ_HAZARD_PERSISTS"
  elif gp.get("error") or jr.get("error"): verdict = "KV_OPAQUE_READ_HAZARD_PERSISTS"
  elif gp.get("full_maxc_copy"): verdict = "KV_OPAQUE_READ_COPY_STILL_PRESENT"
  elif not jr.get("all_finite"): verdict = "KV_OPAQUE_READ_JIT_REPLAY_FAIL"
  elif not jr.get("append_before_tile"): verdict = "KV_OPAQUE_READ_ORDERING_FAIL"
  else: verdict = "KV_OPAQUE_READ_PROBE_PASS"
  res["verdict"] = verdict
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "probe.json").write_text(json.dumps(res, indent=2))
  print(f"VERDICT: {verdict}", file=sys.stderr)
  print(f"  correctness: {res.get('correctness')}", file=sys.stderr)
  print(f"  graph_profile: full_maxc_copy={gp.get('full_maxc_copy')} other_E={gp.get('other_E_copies')} append={gp.get('append_present')} tile={gp.get('owned_tile_present')}", file=sys.stderr)
  print(f"  jit_replay: all_finite={jr.get('all_finite')} append_before_tile={jr.get('append_before_tile')} nodes={jr.get('captured_program_nodes')}", file=sys.stderr)
  if res.get("correctness_tb"): print(res["correctness_tb"], file=sys.stderr)
  print(f"artifact: {OUT/'probe.json'}", file=sys.stderr)

if __name__ == "__main__":
  main()
