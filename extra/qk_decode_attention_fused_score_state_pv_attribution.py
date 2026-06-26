#!/usr/bin/env python3
"""Attribution audit for the generated fused score+state+PV decode-attention route.

This is diagnostic, not a promotion benchmark. It answers why a route that is
standalone-numeric-clean and route-clean loses W==D by combining:
  - DEBUG=2 per-program proxy timing for baseline vs fused route,
  - compiled AMD kernel resource/ISA histograms captured from tinygrad runtime,
  - static work-shape/economics for the fused generated tile,
  - the canonical W==D artifact summary.
"""
from __future__ import annotations

import contextlib, ctypes, io, json, os, pathlib, re, subprocess, sys, time
from collections import Counter
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-attention-fused-score-state-pv-attribution"
MAXC = 4608
CTX = int(os.environ.get("QK_ATTR_CTX", "1024"))
TARGET_PREFIXES = (
  "flash_fused_score_state_pv_tile_whole_cache_32_128",
  "flash_state_gmax_32_128",
  "flash_state_combine_32_128",
  "owned_flash_tile_gqa_whole",
  "owned_flash_combine",
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")
DEBUG_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem")
DEBUG_TM = re.compile(r"\s(\d+\.\d+)ms")

CLEAR_FLAGS = (
  "DECODE_ATTN_GENERATED_SKELETON", "DECODE_ATTN_GENERATED_WHOLECACHE", "DECODE_ATTN_SCORE_VDOT2",
  "DECODE_ATTN_SCORE_XLANE", "DECODE_ATTN_TILE_PLACEHOLDER", "DECODE_ATTN_TILE_SCORE_MAX",
  "DECODE_ATTN_TILE_PROB", "DECODE_ATTN_TILE_PARTIAL_PV", "DECODE_ATTN_TILE_PROB_PARTIAL_PV",
  "DECODE_ATTN_ONLINE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE", "DECODE_ATTN_ONLINE_STATE_PV_TILE_XLANE",
  "DECODE_ATTN_ONLINE_STATE_SPLIT_XLANE", "DECODE_ATTN_FUSED_PV_TILE", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE",
  "V_DOT2_LOWERING", "WARP_REDUCE_LOWERING",
)
ARMS = {
  "baseline": {},
  "fused_score_state_pv_tile": {"DECODE_ATTN_GENERATED_WHOLECACHE": "1", "DECODE_ATTN_FUSED_SCORE_STATE_PV_TILE": "1"},
}


def _env(arm: str) -> dict[str, str]:
  env = {**os.environ, "PYTHONPATH": str(ROOT), "QK_FUSED_SSPV_ATTR_CHILD": "1", "QK_FUSED_SSPV_ATTR_ARM": arm}
  for k in CLEAR_FLAGS: env[k] = "0"
  env.update(ARMS[arm])
  return env


def _program_names(captured) -> list[str]:
  from tinygrad.uop.ops import Ops
  if captured is None: return []
  return [str(getattr(u.arg, "name", "")) for u in captured.linear.toposort() if u.op is Ops.PROGRAM]


def _parse_debug(lines: list[str]) -> dict[str, Any]:
  rows = []
  for line in lines:
    m = DEBUG_LINE.search(line)
    if not m: continue
    name = m.group(1).strip()
    tm = DEBUG_TM.search(line)
    rows.append({"program": name, "debug_ms": float(tm.group(1)) if tm else None, "line": line[:500]})
  by_name: dict[str, float] = {}
  for r in rows:
    if r["debug_ms"] is not None: by_name[r["program"]] = by_name.get(r["program"], 0.0) + r["debug_ms"]
  return {"rows": rows, "program_ms": dict(sorted(by_name.items(), key=lambda kv: -kv[1])), "program_counts": Counter(r["program"] for r in rows).most_common()}


def _parse_desc(lib: bytes) -> dict[str, Any]:
  from tinygrad.runtime.support.elf import elf_loader
  from tinygrad.runtime.autogen import amdgpu_kd
  image, sections, _relocs = elf_loader(lib)
  rodata_entry = next((sh.header.sh_addr for sh in sections if sh.name == ".rodata"), -1)
  desc_sz = ctypes.sizeof(amdgpu_kd.llvm_amdhsa_kernel_descriptor_t)
  desc = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytes(image[rodata_entry:rodata_entry+desc_sz]))
  rsrc1 = desc.compute_pgm_rsrc1
  gran_vgpr = rsrc1 & 0x3f
  gran_sgpr = (rsrc1 >> 6) & 0xf
  return {"vgpr": (gran_vgpr + 1) * 8, "sgpr": (gran_sgpr + 1) * 8, "lds": desc.group_segment_fixed_size,
          "scratch": desc.private_segment_fixed_size, "kernarg": desc.kernarg_size, "rsrc1": hex(rsrc1)}


def _disasm(lib: bytes) -> str:
  from tinygrad.helpers import system
  objdump = "/opt/rocm/llvm/bin/llvm-objdump"
  if not pathlib.Path(objdump).exists(): objdump = "llvm-objdump"
  try: return system(f"{objdump} -d -", input=lib)
  except Exception as e: return f"DISASM_ERROR: {e}"


def _hist(asm: str) -> dict[str, int]:
  h = {"total": 0, "valu": 0, "s_inst": 0, "vmem_load": 0, "vmem_store": 0, "ds": 0, "exp": 0,
       "fma_dot": 0, "cross_lane": 0, "scratch": 0, "branch": 0}
  for line in asm.splitlines():
    m = re.search(r"\b([sv]_[a-z0-9_]+|global_[a-z0-9_]+|buffer_[a-z0-9_]+|ds_[a-z0-9_]+|scratch_[a-z0-9_]+)\b", line)
    if not m: continue
    op = m.group(1); h["total"] += 1
    if op.startswith("v_"): h["valu"] += 1
    if op.startswith("s_"): h["s_inst"] += 1
    if op.startswith("global_load") or op.startswith("buffer_load"): h["vmem_load"] += 1
    if op.startswith("global_store") or op.startswith("buffer_store"): h["vmem_store"] += 1
    if op.startswith("ds_"): h["ds"] += 1
    if "exp" in op and op.startswith("v_"): h["exp"] += 1
    if "fma" in op or "dot" in op or "mac" in op: h["fma_dot"] += 1
    if op.startswith(("ds_bpermute", "ds_permute", "ds_swizzle")) or op.startswith("v_permlane"): h["cross_lane"] += 1
    if op.startswith("scratch_"): h["scratch"] += 1
    if op.startswith("s_branch") or op.startswith("s_cbranch"): h["branch"] += 1
  return h


def _kernel_resources(captured: dict[str, bytes]) -> dict[str, Any]:
  out = {}
  OUT.mkdir(parents=True, exist_ok=True)
  for name, lib in captured.items():
    if not any(name.startswith(p) for p in TARGET_PREFIXES): continue
    try: desc = _parse_desc(lib)
    except Exception as e: desc = {"error": str(e)}
    asm = _disasm(lib)
    desc["hist"] = _hist(asm)
    desc["primitive_flags"] = {
      "has_v_dot2": "v_dot2" in asm,
      "has_lds": bool(re.search(r"\bds_(load|store|read|write)", asm)),
      "has_cross_lane": bool(re.search(r"\b(ds_bpermute|ds_permute|ds_swizzle|v_permlane)", asm)),
      "has_vector_global_load": "global_load" in asm or "buffer_load" in asm,
      "has_spill": bool(re.search(r"\bscratch_(load|store)", asm)),
    }
    out[name] = desc
    (OUT / f"disasm_{name}.txt").write_text(asm)
  return out


def _child(arm: str) -> dict[str, Any]:
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm_generate import load_model_and_tokenizer
  from extra.qk_harness_contract import DEFAULT_MODEL

  dev = Device[Device.DEFAULT]
  captured: dict[str, bytes] = {}
  orig_runtime = dev.runtime
  def runtime_hook(name, lib, **kw):
    if any(name.startswith(p) for p in TARGET_PREFIXES) and name not in captured: captured[name] = lib
    return orig_runtime(name, lib, **kw)
  dev.runtime = runtime_hook

  m, _tok = load_model_and_tokenizer(os.environ.get("QK_MODEL", DEFAULT_MODEL), MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []): lin.decode_enabled = True
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False

  v = UOp.variable("start_pos", 0, MAXC - 1)
  temp = Tensor([0.0])
  step = TinyJit(m.forward)
  tk = Tensor([[100]], dtype="int32").contiguous()
  for _ in range(8):
    o = step(tk, v.bind(CTX), temp).realize(); o.item()
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2):
    GlobalCounters.reset(); step(tk, v.bind(CTX + 1), temp).realize(); gpu_ms = GlobalCounters.time_sum_s * 1e3
  lines = [ANSI.sub("", l) for l in buf.getvalue().splitlines() if "***" in l]
  names = _program_names(step.captured)
  resources = _kernel_resources(captured)
  return {
    "arm": arm,
    "ctx": CTX,
    "env": {k: os.environ.get(k, "") for k in os.environ if k.startswith("DECODE_ATTN")},
    "debug2_unbatched_gpu_ms": round(gpu_ms, 3),
    "captured_program_counts": Counter(names).most_common(100),
    "target_programs_in_graph": [n for n in names if any(n.startswith(p) for p in TARGET_PREFIXES)],
    "generated_attention_programs": [n for n in names if n.startswith("flash_")],
    "owned_attention_programs": [n for n in names if n.startswith("owned_flash_")],
    "debug": _parse_debug(lines),
    "kernel_resources": resources,
  }


def _run_child(arm: str) -> dict[str, Any]:
  p = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve())], cwd=ROOT, env=_env(arm), text=True,
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if p.returncode != 0: return {"arm": arm, "failed": True, "returncode": p.returncode, "output_tail": (p.stdout or "")[-10000:]}
  for line in reversed((p.stdout or "").splitlines()):
    try: return json.loads(line)
    except Exception: pass
  return {"arm": arm, "failed": True, "returncode": 0, "error": "no json", "output_tail": (p.stdout or "")[-10000:]}


def _latest_wd() -> dict[str, Any]:
  runs = sorted((ROOT / "bench/qk-decode-eval/runs").glob("*-decode_attention_fused_score_state_pv_tile.json"))
  if not runs: return {"available": False}
  p = runs[-1]
  d = json.loads(p.read_text())
  return {"available": True, "path": str(p.relative_to(ROOT)), "verdict": d.get("verdict"), "wd": d.get("wd", {})}


def _static_work_shape() -> dict[str, Any]:
  Hq, Hkv, Hd, L = 32, 8, 128, 128
  S = (CTX + L - 1) // L
  W = Hd + 2
  return {
    "ctx": CTX,
    "Hq": Hq, "Hkv": Hkv, "Hd": Hd, "G": Hq // Hkv, "L": L, "S": S, "W": W,
    "fused_tile_workgroups": Hkv * S,
    "fused_tile_local_columns": W,
    "qk_dot_reductions_per_workgroup": (Hq // Hkv) * W,
    "qk_dot_muladds_per_workgroup": (Hq // Hkv) * W * L * Hd,
    "qk_redundancy_note": "The generated fused tile computes q.k inside the local output-column axis, so each PV/l/m column repeats the Hd dot unless codegen/commoning/cross-lane sharing removes it.",
    "baseline_owned_workgroups_approx": Hkv * int(os.environ.get("DECODE_ATTN_AMDGCN_S", "48")),
  }


def _classify(arms: dict[str, Any], wd: dict[str, Any], work: dict[str, Any]) -> dict[str, Any]:
  fused = arms.get("fused_score_state_pv_tile", {})
  base = arms.get("baseline", {})
  resources = fused.get("kernel_resources", {})
  tile = next((v for k, v in resources.items() if k.startswith("flash_fused_score_state_pv_tile")), {})
  flags = tile.get("primitive_flags", {})
  debug_ms = fused.get("debug", {}).get("program_ms", {})
  tile_ms = sum(v for k, v in debug_ms.items() if k.startswith("flash_fused_score_state_pv_tile"))
  state_ms = sum(v for k, v in debug_ms.items() if k.startswith("flash_state_"))
  baseline_owned_ms = sum(v for k, v in base.get("debug", {}).get("program_ms", {}).items() if k.startswith("owned_flash_"))
  blockers = []
  if not flags.get("has_v_dot2"): blockers.append("generated fused tile has no v_dot2/native packed dot")
  if not flags.get("has_lds"): blockers.append("generated fused tile has no LDS/tile-staged K/V reuse")
  if not flags.get("has_cross_lane"): blockers.append("generated fused tile has no cross-lane q.k sharing/reduction primitive")
  if work.get("qk_dot_reductions_per_workgroup", 0) > work.get("G", 1): blockers.append("q.k dot appears repeated across local output columns")
  if tile_ms and state_ms and tile_ms > state_ms * 2: bottleneck = "fused_tile"
  elif state_ms and state_ms >= tile_ms: bottleneck = "state_gmax_combine"
  else: bottleneck = "unknown_or_mixed"
  return {
    "verdict": "FUSED_SCORE_STATE_PV_ATTRIBUTED__GENERATED_TILE_PHYSICALLY_BAD" if blockers else "FUSED_SCORE_STATE_PV_ATTRIBUTION_INCONCLUSIVE",
    "bottleneck_proxy": bottleneck,
    "debug_ms": {"fused_tile": round(tile_ms, 4), "state_kernels": round(state_ms, 4), "baseline_owned_attention": round(baseline_owned_ms, 4)},
    "blockers": blockers,
    "interpretation": "The route is structurally pure but lacks the physical primitives that make decode attention fast; W==D loss is expected unless codegen exposes LDS/vectorized/cross-lane/v_dot2 or avoids repeated q.k across local columns.",
    "wd_path": wd.get("path"),
  }


def build() -> dict[str, Any]:
  arms = {a: _run_child(a) for a in ARMS}
  wd = _latest_wd()
  work = _static_work_shape()
  if any(v.get("failed") for v in arms.values()):
    diagnosis = {"verdict": "FUSED_SCORE_STATE_PV_ATTRIBUTION_FAIL__CHILD", "failed_arms": [k for k, v in arms.items() if v.get("failed")]}
  else:
    diagnosis = _classify(arms, wd, work)
  return {"date": "2026-06-26", "timestamp": time.strftime("%Y%m%d-%H%M%S"), "verdict": diagnosis["verdict"],
          "arms": arms, "wd_summary": wd, "work_shape": work, "diagnosis": diagnosis,
          "decision": "Do not promote. If continuing pure-search decode, expose/search the missing physical primitives rather than retuning this route."}


def main() -> int:
  os.chdir(ROOT)
  if os.environ.get("QK_FUSED_SSPV_ATTR_CHILD") == "1":
    print(json.dumps(_child(os.environ.get("QK_FUSED_SSPV_ATTR_ARM", "baseline"))))
    return 0
  OUT.mkdir(parents=True, exist_ok=True)
  out = build()
  (OUT / "latest.json").write_text(json.dumps(out, indent=2) + "\n")
  (OUT / f"decode-attention-fused-score-state-pv-attribution-{out['timestamp']}.json").write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps(out, indent=2))
  return 0 if not out["verdict"].endswith("__CHILD") else 1


if __name__ == "__main__":
  raise SystemExit(main())
