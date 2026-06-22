#!/usr/bin/env python3
"""Design A microprobe (docs/kv-cache-stateful-jit-capability-scope-20260622.md, Phase 1).

Proves/refutes the smallest stateful-KV-append capability OUTSIDE the full model: an OPAQUE custom_kernel that
writes the current token's K/V into a persistent cache buffer slice IN PLACE at a symbolic `start_pos`, returns
`cache.after(kernel)` (same buffer, ordered, NO full-buffer copy), and survives TinyJit capture/replay with a
changing `start_pos` (start_pos carried as a ProgramInfo runtime scalar var, never baked into a captured index).

Gates: (1) byte-correct append + prefix read; (2) no full-MAXC copy kernel; (3) JIT replay with changing start_pos.

  run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_kv_append_microprobe.py
  -> bench/qk-kv-cache-stateful-jit/design_a_microprobe.json
"""
from __future__ import annotations
import json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-kv-cache-stateful-jit"
KVH, HD = 8, 128                       # Qwen3-8B GQA kv-heads, head_dim
NELEM = 2 * KVH * HD                   # one (K,V) append at T=1, B=1 = 2048 fp16

def append_kernel(cache_f, src_f, start_pos_var, MAXC):
  """opaque RDNA3 kernel: 1024 threads, each stores one b32 (2 fp16) from src[tid] into cache at the start_pos slice.
  layout cache[2,1,KVH,MAXC,HD]; dest_fp16(tid) = kv*(KVH*MAXC*HD) + h*(MAXC*HD) + start_pos*HD + 2*d2,
  kv=tid>>9, h=(tid>>6)&7, d2=tid&63 (b32 covers fp16 pair 2*d2,2*d2+1; HD contiguous in src&dest)."""
  from tinygrad.uop.ops import UOp, Ops, KernelInfo
  from tinygrad.renderer import Estimates
  from tinygrad.runtime.autogen.amd.rdna3.ins import (s_load_b128, s_load_b32, s_waitcnt_lgkmcnt, s_mov_b32,
    s_lshl_b32, v_lshrrev_b32_e32, v_and_b32_e32, v_mul_lo_u32, v_lshlrev_b32_e32, v_add_nc_u32_e32,
    global_load_b32, s_waitcnt_vmcnt, global_store_b32, s_endpgm)
  from tinygrad.renderer.amd.dsl import s, v, NULL
  KVHMAXCHD, MAXCHD = KVH * MAXC * HD, MAXC * HD
  threads = UOp.special(1024, "lidx0")
  insts = [
    s_load_b128(s[4:7], s[0:1]),                 # s4:5=cache(out) ptr, s6:7=src(in) ptr
    s_load_b32(s[8], s[0:1], offset=0x10),       # s8 = start_pos
    s_waitcnt_lgkmcnt(sdst=NULL, simm16=0),
    s_mov_b32(s[10], KVHMAXCHD), s_mov_b32(s[11], MAXCHD),
    v_lshrrev_b32_e32(v[1], 9, v[0]),            # kv = tid>>9
    v_lshrrev_b32_e32(v[2], 6, v[0]), v_and_b32_e32(v[2], 7, v[2]),   # h = (tid>>6)&7
    v_and_b32_e32(v[3], 63, v[0]),               # d2 = tid&63
    v_mul_lo_u32(v[4], v[1], s[10]),             # kv*KVHMAXCHD
    v_mul_lo_u32(v[5], v[2], s[11]), v_add_nc_u32_e32(v[4], v[4], v[5]),   # + h*MAXCHD
    s_lshl_b32(s[9], s[8], 7), v_add_nc_u32_e32(v[4], s[9], v[4]),         # + start_pos*HD
    v_lshlrev_b32_e32(v[6], 1, v[3]), v_add_nc_u32_e32(v[4], v[4], v[6]),  # + 2*d2 -> dest_fp16
    v_lshlrev_b32_e32(v[4], 1, v[4]),            # dest_byte = dest_fp16*2
    v_lshlrev_b32_e32(v[7], 2, v[0]),            # src_byte = tid*4
    global_load_b32(v[8], v[7], saddr=s[6:7]),
    s_waitcnt_vmcnt(sdst=NULL, simm16=0),
    global_store_b32(addr=v[4], data=v[8], saddr=s[4:5]),
    s_endpgm(),
  ]
  sink = UOp.sink(cache_f.base, src_f.base, start_pos_var, threads,
                  arg=KernelInfo("kv_append", estimates=Estimates(ops=NELEM, mem=NELEM * 4)))
  return UOp(Ops.PROGRAM, src=(sink, UOp(Ops.DEVICE, arg="AMD"), UOp(Ops.LINEAR, src=tuple(UOp(Ops.INS, arg=x) for x in insts))))

def main():
  from tinygrad import Tensor, UOp, Context, Device, dtypes
  from tinygrad.engine.realize import run_linear
  from tinygrad.device import Compiled
  import functools
  dev = Device["AMD"]
  res = {"date": "2026-06-22", "phase": "DESIGN_A_MICROPROBE", "gpu": "RX 7900 XTX / gfx1100",
         "default_behavior_changed": False}
  MAXC = 16
  rng = np.random.default_rng(0)
  srcs = [rng.standard_normal(NELEM).astype(np.float16) for _ in range(4)]
  spvar = UOp.variable("start_pos", 0, MAXC - 1)   # UNBOUND; value supplied per-run via var_vals (test_variable pattern)

  def build(cache, src):   # returns the cache.after(append_kernel) tensor (opaque write; start_pos is a runtime var)
    return Tensor.custom_kernel(cache.flatten(), src.flatten(),
                                fxn=functools.partial(append_kernel, start_pos_var=spvar, MAXC=MAXC))[0]

  # ---------- correctness at CONCRETE start_pos (schedule once per cache, run with var_vals) ----------
  conc = {}
  try:
    for sp in (0, 3, 9):
      cache = Tensor.zeros(2, 1, KVH, MAXC, HD, dtype=dtypes.half).contiguous().realize()
      lin = build(cache, Tensor(srcs[0])).schedule_linear()
      run_linear(lin, var_vals={"start_pos": sp})
      got = cache.numpy().reshape(2, KVH, MAXC, HD); want = srcs[0].reshape(2, KVH, HD)
      others = np.delete(np.arange(MAXC), sp)
      conc[sp] = {"slice_correct": bool(np.array_equal(got[:, :, sp, :], want)), "rest_zero": bool(np.all(got[:, :, others, :] == 0))}
    res["concrete_correctness"] = conc
  except Exception as e:
    import traceback; res["concrete_correctness"] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}; res["traceback"] = traceback.format_exc()[-1800:]

  conc_pass = isinstance(res.get("concrete_correctness"), dict) and res["concrete_correctness"].get(3, {}).get("slice_correct")
  # ---------- copy-free check: profile a run; no full-MAXC copy kernel, append node present ----------
  if conc_pass:
    try:
      cache = Tensor.zeros(2, 1, KVH, MAXC, HD, dtype=dtypes.half).contiguous().realize()
      lin = build(cache, Tensor(srcs[1])).schedule_linear()
      run_linear(lin, var_vals={"start_pos": 5}); dev.synchronize(); dev._at_profile_finalize()
      with Context(PROFILE=1):
        b = len(Compiled.profile_events); run_linear(lin, var_vals={"start_pos": 6}); dev.synchronize(); dev._at_profile_finalize()
        names = [str(ent.name) for e in Compiled.profile_events[b:] if type(e).__name__ == "ProfileGraphEvent" for ent in e.ents]
      res["copy_free"] = {"kernels": names, "append_present": any("kv_append" in n for n in names),
                          "full_buffer_copy_present": any(n.startswith("E_") for n in names),
                          "note": "any E_* elementwise kernel here would be a buffer copy/materialization; the opaque append should be the ONLY kernel"}
    except Exception as e:
      res["copy_free"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

  # ---------- capture/replay with CHANGING start_pos: ONE compiled linear, run with different var_vals ----------
  if conc_pass:
    try:
      cache = Tensor.zeros(2, 1, KVH, MAXC, HD, dtype=dtypes.half).contiguous().realize()
      lin = build(cache, Tensor(srcs[3])).schedule_linear()   # compiled ONCE
      replay = {}
      for sp in [2, 7, 11, 4]:
        run_linear(lin, var_vals={"start_pos": sp})           # replay with changing start_pos (no recompile)
        got = cache.numpy().reshape(2, KVH, MAXC, HD)
        replay[sp] = bool(np.array_equal(got[:, :, sp, :], srcs[3].reshape(2, KVH, HD)))
      res["jit_replay"] = {"per_start_pos_correct": replay, "all_correct": all(replay.values()), "method": "single compiled linear, changing var_vals"}
    except Exception as e:
      import traceback; res["jit_replay"] = {"error": f"{type(e).__name__}: {str(e)[:300]}", "tb": traceback.format_exc()[-1200:]}

  # ---------- verdict ----------
  cc = res.get("concrete_correctness", {})
  conc_ok = isinstance(cc, dict) and all(isinstance(cc.get(sp), dict) and cc[sp].get("slice_correct") and cc[sp].get("rest_zero") for sp in (0, 3, 9))
  jit = res.get("jit_replay", {})
  if "error" in cc: verdict = "DESIGN_A_NOT_EXPRESSIBLE"
  elif not conc_ok: verdict = "DESIGN_A_ALIAS_FAIL"
  elif "error" in jit: verdict = "DESIGN_A_JIT_REPLAY_FAIL"
  elif jit.get("all_correct") and not res.get("copy_free", {}).get("full_buffer_copy_present", True): verdict = "DESIGN_A_MICROPROBE_PASS"
  elif not jit.get("all_correct"): verdict = "DESIGN_A_JIT_REPLAY_FAIL"
  else: verdict = "DESIGN_A_ORDERING_FAIL"
  res["verdict"] = verdict
  OUT.mkdir(parents=True, exist_ok=True); (OUT / "design_a_microprobe.json").write_text(json.dumps(res, indent=2))
  print(f"VERDICT: {verdict}", file=sys.stderr)
  print(f"  concrete: {cc}", file=sys.stderr)
  print(f"  copy_free: {res.get('copy_free')}", file=sys.stderr)
  print(f"  jit_replay: {res.get('jit_replay')}", file=sys.stderr)
  if res.get("traceback"): print(res["traceback"], file=sys.stderr)
  print(f"artifact: {OUT/'design_a_microprobe.json'}", file=sys.stderr)

if __name__ == "__main__":
  main()
