#!/usr/bin/env python3
"""Arc 1 Phase 1: SDPA vs flash-decode short-context anatomy. For ctx 512/1024/4096, with FLASH_DECODE forced
(env 0 or 1), measure: clean decode tok/s (device-token feed, no per-step Tensor creation -- the Arc-4 method),
total kernels/token, attention kernels/layer + GPU%, and argmax (greedy sanity). Run twice (FLASH_DECODE=0 then
=1) and diff. Answers: why is flash-decode only ~1.05x at ctx512? which attention kernels disappear/appear?
does the cost move to GEMV/other? No code change.

Run: DEV=AMD JIT=1 FLASH_DECODE=0 PYTHONPATH=. .venv/bin/python extra/qk_attention_sdpa_vs_flash.py
     DEV=AMD JIT=1 FLASH_DECODE=1 PYTHONPATH=. .venv/bin/python extra/qk_attention_sdpa_vs_flash.py
"""
from __future__ import annotations
import io, json, os, pathlib, re, statistics, sys, time, contextlib

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(.+?)\s+arg\s+\d+\s+mem\s+[\d.]+\s+GB\s+tm\s+([\d.]+)us")
_GEMV = re.compile(r"q[46]k_gemv"); HD, NLAYERS = 128, 36
def _nums(s): return [int(x) for x in re.findall(r"\d+", s)]

def main():
  model = os.environ.get("QK_MODEL", "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  mode = str(os.environ.get("FLASH_DECODE", "0"))
  from tinygrad import Tensor, UOp, TinyJit, Context, GlobalCounters, Device
  from extra.llm_generate import load_model_and_tokenizer
  import tinygrad.llm.model as M
  dev = Device[Device.DEFAULT]
  m, tok = load_model_and_tokenizer(model, 4608, seed=20260617)
  for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode("the quick brown fox jumps. " * 900)
  ids = (ids * (1 + 4608 // max(1, len(ids))))[:4608]
  v_sp = UOp.variable("start_pos", 0, 4607); temp = Tensor([0.0])

  rows = []
  for ck in (128, 256, 384, 512, 768):
    use_flash = M.should_use_flash_decode(v_sp.bind(ck), 1, False)
    for b in m.blk: b._use_flash, b._prefill_v2 = use_flash, False
    step = TinyJit(m.forward); tokid = int(ids[ck])
    out = Tensor([[tokid]], dtype="int32").contiguous()
    for i in range(8): out = step(out, v_sp.bind(ck + i), temp).realize()
    argmax = int(out.item())
    W = []
    for i in range(60):
      t0 = time.perf_counter(); out = step(out, v_sp.bind(ck + i), temp); int(out.item()); W.append(time.perf_counter() - t0)
    # kernel capture
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), Context(DEBUG=2):
      GlobalCounters.reset(); step(out, v_sp.bind(ck + 40), temp).realize()
    ks = [(mt.group(1).strip(), float(mt.group(2))) for mt in (_LINE.search(_ANSI.sub("", l)) for l in buf.getvalue().splitlines()) if mt]
    kv = ck + 1
    attn = [(nm, us) for nm, us in ks if not _GEMV.search(nm.lower()) and (kv in _nums(nm) or HD in _nums(nm))]
    tot = sum(u for _, u in ks) or 1.0
    w_ms = statistics.median(W) * 1e3
    rows.append({"ctx": ck, "flash": bool(use_flash), "tok_s": round(1000 / w_ms, 1), "wall_ms": round(w_ms, 3),
                 "total_kernels": len(ks), "attention_kernels": len(attn), "attention_per_layer": round(len(attn) / NLAYERS, 2),
                 "attention_pct": round(100 * sum(u for _, u in attn) / tot, 1), "argmax": argmax})
    print(f"FLASH={mode} ctx{ck:5} flash={use_flash!s:5}: {rows[-1]['tok_s']:6.1f} tok/s | {len(ks)} kernels | "
          f"attn {rows[-1]['attention_per_layer']}/layer {rows[-1]['attention_pct']}% | argmax {argmax}", file=sys.__stderr__)

  art = pathlib.Path(f"bench/qk-8b-attention-sdpa-vs-flash/flash_{mode}.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps({"flash_decode_env": mode, "rows": rows}, indent=2))
  print(f"artifact: {art}", file=sys.__stderr__); print("@@DONE@@")

if __name__ == "__main__":
  main()
