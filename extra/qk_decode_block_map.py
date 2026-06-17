#!/usr/bin/env python3
"""Decode-block primitive map — Phase 0 census (post-hoisted-flash HEAD).

Refreshes the per-layer / per-region decode program + GPU-time picture AFTER the shipped flash-decode
`hoisted` default, at ctx 512/1024/4096. Design/audit only: NO code changes beyond instrumentation, NO
default changes.

Method (carried-forward discipline):
  - per-kernel anatomy: ONE eager decode step with a BOUND symbolic start_pos (so the flash path fires;
    concrete start_pos would take SDPA), DEBUG=2, warm. GPU time is the eager DEBUG=2 `tm` = RELATIVE proxy
    (eager unbatches -> inflates absolute; ratios/shares are the signal). programs/token = eager kernel count.
  - tok/s: the W==D warm device-token-feed method (real decode wall vs dispatch ceiling), the robust path.
  - region/role classification reuses the GEMV output-dim role map from qk_decode_layer_census (DRY).

Run: DEV=AMD JIT=1 PYTHONPATH=. .venv/bin/python extra/qk_decode_block_map.py
"""
from __future__ import annotations

import io, json, os, pathlib, re, statistics, subprocess, sys, time, contextlib
from collections import defaultdict, OrderedDict

from extra.qk_decode_layer_census import GEMV_ROLE, _GEMV  # reuse the GEMV role map (single source of truth)

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us")
CKPTS = [512, 1024, 4096]; MAXC = 4608; NLAYERS = 36; NMEAS = 30
Hq, G = 32, 4  # n_heads, n_heads//n_kv_heads

def region(name:str, kv:int) -> str:
  n = name.lower(); g = _GEMV.search(n)
  if g: return GEMV_ROLE.get((int(g.group(1)), int(g.group(2))), "gemv_other[GEMV]")
  nums = [int(x) for x in re.findall(r"\d+", name)]
  if n.startswith("flash_prob"): return "attn_flash_prob"
  if n.startswith("flash_partial"): return "attn_flash_partial"
  if n.startswith("flash_max") or n.startswith("flash_gmax"): return "attn_flash_max"
  if n.startswith("flash_den") or n.startswith("flash_combine"): return "attn_flash_reduce"
  if n.startswith("flash"): return "attn_flash_other"
  if n.startswith("r_32_4") or n.startswith("r_32_8"): return "attn_qk_scores"
  if n.startswith("copy") and " 4 b" in n: return "input_upload(sync)"
  if n.startswith("copy"): return "kv_write/copy"
  if n.startswith("r_") and 16 in nums and 256 in nums: return "rmsnorm"
  if kv in nums or 1024 in nums: return "attn_other"
  if n.startswith("r_"): return "reduce(other)"
  if n.startswith("e_") or n.startswith("e "): return "elementwise(rope/residual/cast)"
  return "other"

def provenance() -> dict:
  def run(c):
    try: return subprocess.run(c, shell=True, text=True, capture_output=True, timeout=30).stdout
    except Exception as e: return f"<err {e}>"
  mk = run("rocminfo 2>/dev/null | grep 'Marketing Name' | grep -i radeon | head -1").strip()
  vram = run("rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Total Memory'").strip()
  model_line = run("rocm-smi --showproductname 2>/dev/null | grep -i 'Card model'").strip()
  from tinygrad import Device
  return {"rocminfo_marketing": mk, "rocm_smi_vram": vram, "rocm_smi_card_model": model_line,
          "tinygrad_device": str(Device.DEFAULT), "arch": getattr(Device[Device.DEFAULT], "arch", None),
          "vram_gib": round(25753026560 / 1024**3, 2),
          "note": "RX 7900 XTX (rocminfo marketing + 24GB VRAM); rocm-smi Card-model 'GRE' is a misread"}

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm_generate import load_model_and_tokenizer
  dev = Device[Device.DEFAULT]
  prov = provenance()
  m, tok = load_model_and_tokenizer(model, MAXC, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 800)
  ids = (ids * (1 + MAXC // max(1, len(ids))))[:MAXC]
  v_sp = UOp.variable("start_pos", 0, MAXC - 1); temp = Tensor([0.0])
  for b in m.blk: b._use_flash, b._prefill_v2 = True, False

  rows = []
  for ck in CKPTS:
    kv = ck + 1; tokid = int(ids[ck])
    # --- tok/s via W==D (TinyJit) ---
    step = TinyJit(m.forward)
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()
    out, W = Tensor([[tokid]], dtype="int32").contiguous(), []
    for i in range(NMEAS):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); _ = int(out.item())
      W.append(time.perf_counter() - t0)
    out = Tensor([[tokid]], dtype="int32").contiguous(); dev.synchronize(); t0 = time.perf_counter()
    for i in range(NMEAS): out = step(out, v_sp.bind(ck + i), temp)
    dev.synchronize(); D = (time.perf_counter() - t0) / NMEAS
    w_ms = statistics.median(W) * 1e3

    # --- per-kernel anatomy: ONE eager step, bound start_pos (flash path), DEBUG=2 ---
    for i in range(3): m.forward(out, v_sp.bind(ck + i), temp).realize()  # warm eager
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); m.forward(out, v_sp.bind(ck + 9), temp).realize()
    kernels = []
    for l in buf.getvalue().splitlines():
      mt = _LINE.search(_ANSI.sub("", l))
      if mt: kernels.append((mt.group(1).strip(), float(mt.group(2))))
    total = sum(t for _, t in kernels) or 1.0
    # cluster by name -> per-layer (count>=18) vs tail
    clusters = defaultdict(lambda: [0, 0.0])
    for nm, t in kernels: clusters[nm][0] += 1; clusters[nm][1] += t
    layer_progs = sum(c for c, _ in clusters.values() if c >= 18)
    tail_progs = sum(c for c, _ in clusters.values() if c < 18)
    reg = defaultdict(lambda: [0, 0.0])
    for nm, t in kernels: r = region(nm, kv); reg[r][0] += 1; reg[r][1] += t
    region_tab = {r: {"kernels": c, "per_layer": round(c / NLAYERS, 2), "us": round(u, 1),
                      "pct_gpu_proxy": round(100 * u / total, 1)}
                  for r, (c, u) in sorted(reg.items(), key=lambda kv2: -kv2[1][1])}
    rows.append({"ctx": ck, "tok_s_W": round(1000 / w_ms, 1), "tok_s_D_ceiling": round(1 / D, 1),
                 "wall_ms": round(w_ms, 3), "programs_per_token": len(kernels),
                 "programs_per_layer_est": round(layer_progs / NLAYERS, 1), "layer_programs": layer_progs,
                 "tail_programs": tail_progs, "total_us_proxy": round(total, 1), "regions": region_tab})
    print(f"ctx {ck:5}: {1000/w_ms:5.1f} tok/s | {len(kernels)} progs/tok ({layer_progs/NLAYERS:.1f}/layer + "
          f"{tail_progs} tail)", file=sys.stderr)
    for r, d in list(region_tab.items())[:8]:
      print(f"    {d['pct_gpu_proxy']:5.1f}%  {d['per_layer']:5}/layer  {r}", file=sys.stderr)

  out_obj = {"model_id": pathlib.Path(model).stem, "hardware": prov, "ckpts": CKPTS, "nmeas": NMEAS,
             "method": "eager bound-start_pos DEBUG=2 (relative GPU proxy) for anatomy; W==D for tok/s; "
                       "programs_per_token = eager kernel count; GEMV roles from qk_decode_layer_census",
             "rows": rows}
  art = pathlib.Path("bench/qk-decode-block-map/result.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps(out_obj, indent=2)); print(f"\nartifact: {art}", file=sys.__stderr__)
  print("@@DONE@@", file=sys.__stderr__)

if __name__ == "__main__":
  main()
