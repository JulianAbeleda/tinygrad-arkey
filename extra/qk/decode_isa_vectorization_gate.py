#!/usr/bin/env python3
"""Authoritative ISA vectorization gate for generated decode attention tiles.

Correctness-only gates can pass while the generated tile still emits scalar cache
loads. This gate is the authority for the vectorization claim: capture the target
generated decode tile, disassemble it, count wide-load markers including RDNA3
`global_load_b*` naming, and PASS only when an accepted wide load is present.

Run:
  DEV=AMD JIT=1 REG_STORE_DEVEC=1 PYTHONPATH=. python3 extra/qk/decode_isa_vectorization_gate.py
"""
from __future__ import annotations

import json, os, pathlib, re, subprocess, sys, time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-decode-isa-vectorization"
MAXC = 4608
CTX = int(os.environ.get("QK_ISA_VEC_CTX", "1024"))
TARGET = "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128" if os.environ.get("DECODE_ATTN_BLOCK_TILE", "0") != "0" else \
  "flash_fused_xlane_score_pv_tile_whole_cache_32_128"

_ZERO = ("DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
         "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
         "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
         "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
         "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
         "DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE", "DECODE_ATTN_PHYSICAL_TILE_PALL_LIFECYCLE",
         "DECODE_ATTN_PHYSICAL_TILE_SCORE_BROADCAST_LIFECYCLE")


def _markers(asm: str) -> dict[str, int]:
  def c(pat: str) -> int: return len(re.findall(pat, asm))
  return {
    "global_load_dwordx4": c(r"\bglobal_load_dwordx4\b"),
    "global_load_dwordx2": c(r"\bglobal_load_dwordx2\b"),
    "global_load_dword": c(r"\bglobal_load_dword\b"),
    "global_load_d16": c(r"\bglobal_load_(u?short|d16)"),
    "global_load_b128": c(r"\bglobal_load_b128\b"),
    "global_load_b96": c(r"\bglobal_load_b96\b"),
    "global_load_b64": c(r"\bglobal_load_b64\b"),
    "global_load_b32": c(r"\bglobal_load_b32\b"),
    "ds_read": c(r"\bds_(read|load)"),
    "ds_write": c(r"\bds_(write|store)"),
    "s_barrier": c(r"\bs_barrier\b"),
    "s_waitcnt": c(r"\bs_waitcnt\b"),
    "v_dot2": c(r"\bv_dot2"),
    "cross_lane": c(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)"),
    "scratch": c(r"\bscratch_(load|store)"),
    "v_fma": c(r"\bv_fma"),
    "v_exp": c(r"\bv_exp"),
    "v_mul": c(r"\bv_mul"),
    "v_add": c(r"\bv_add"),
  }


def _route_env() -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT)}
  for k in _ZERO: env[k] = "0"
  env["DECODE_ATTN_GENERATED_WHOLECACHE"] = "1"
  env["DECODE_ATTN_FUSED_XLANE_SCORE_PV_TILE"] = "1"
  env["V_DOT2_LOWERING"] = "1"
  return env


def _run_json(cmd: list[str], env: dict[str, str]) -> dict[str, Any]:
  p = subprocess.run(cmd, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  for line in reversed((p.stdout or "").splitlines()):
    try:
      out = json.loads(line)
      out["_returncode"] = p.returncode
      return out
    except Exception:
      pass
  txt = p.stdout or ""
  try:
    out = json.loads(txt[txt.index("{"):txt.rindex("}")+1])
    out["_returncode"] = p.returncode
    return out
  except Exception:
    pass
  return {"failed": True, "_returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}


def _capture() -> dict[str, Any]:
  from tinygrad import Context, Device, GlobalCounters, Tensor, TinyJit, UOp
  from extra.llm.generate import load_model_and_tokenizer
  from extra.qk.harness_contract import DEFAULT_MODEL
  from extra.qk.isa_helpers import _disasm, _hist, _parse_desc

  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig = dev.runtime

  def hook(name, lib, **kw):
    if name.startswith(TARGET) and name not in captured: captured[name] = lib
    return orig(name, lib, **kw)

  dev.runtime = hook
  m, _tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  q4k = getattr(m, "_q4k_linears", None)
  for lin in (q4k.linears if q4k else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False
  v = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  with Context(DEBUG=0):
    for _ in range(8): step(tk, v.bind(CTX), temp).realize().item()
  GlobalCounters.reset()

  res: dict[str, Any] = {"captured": list(captured.keys())}
  for name, lib in captured.items():
    asm = _disasm(lib)
    try: desc = _parse_desc(lib)
    except Exception as e: desc = {"error": str(e)}
    markers = _markers(asm)
    (OUT / f"disasm_{name}.txt").write_text(asm)
    res["tile"] = {"name": name, "resources": desc, "hist": _hist(asm), "markers": markers, "asm_lines": len(asm.splitlines())}
    break
  return res


def _run_capture_child() -> dict[str, Any]:
  env = _route_env()
  env["QK_ISA_VEC_CHILD"] = "1"
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=env,
                     text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-8000:]}


def build() -> dict[str, Any]:
  env = _route_env()
  microgate = "extra/qk/decode_attention_block_tile_microgate.py" if os.environ.get("DECODE_ATTN_BLOCK_TILE", "0") != "0" else \
    "extra/qk/decode_attention_fused_xlane_score_pv_microgate.py"
  micro = _run_json([sys.executable, str(ROOT / microgate)], env)
  route = {"checked": False, "verdict": "ROUTE_GATE_REMOVED_STALE_REPLAY",
           "note": "the old fused-xlane route-gate replay script is not part of the compact repo surface"}
  cap = _run_capture_child()
  timestamp = time.strftime("%Y%m%d-%H%M%S")
  base = {"date": "2026-06-26", "timestamp": timestamp, "authority": "vectorization",
          "ctx": CTX, "target": TARGET, "numeric_correctness": micro, "route_cleanliness": route, "capture": cap}
  tile = cap.get("tile")
  if not tile:
    return {**base, "verdict": "ISA_VEC_AUTHORITATIVE_FAIL__CAPTURE", "pass": False}
  markers = tile.get("markers", {})
  wide = sum(markers.get(k, 0) for k in ("global_load_d16", "global_load_dwordx4", "global_load_b128", "global_load_b64"))
  scalar = sum(markers.get(k, 0) for k in ("global_load_dword", "global_load_b32"))
  lds = (tile.get("resources") or {}).get("lds")
  verdict = "ISA_VEC_AUTHORITATIVE_PASS" if wide > 0 else "ISA_VEC_AUTHORITATIVE_FAIL__SCALAR_LOADS"
  return {**base, "verdict": verdict, "pass": verdict == "ISA_VEC_AUTHORITATIVE_PASS",
          "wide_load_count": wide, "scalar_load_count": scalar, "lds_bytes": lds,
          "markers": markers, "tile": tile,
          "decision": ("Wide-load ISA is present; next authority is block tiling/LDS and W==D transfer."
                       if wide > 0 else
                       "No accepted wide-load ISA marker; do not claim coalesced/vectorized generated cache loads.")}


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  if os.environ.get("QK_ISA_VEC_CHILD") == "1":
    print(json.dumps(_capture()))
    return 0
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"isa-vectorization-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if out["verdict"] == "ISA_VEC_AUTHORITATIVE_PASS" else 1


if __name__ == "__main__":
  raise SystemExit(main())
