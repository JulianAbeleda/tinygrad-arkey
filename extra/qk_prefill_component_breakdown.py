#!/usr/bin/env python3
"""Arc A Phase PWR-1 — prefill component target selection (diagnostic, no routes).

Decompose the warm PREFILL_V2 forward (concrete 512 chunk, fp16 + realized weights + warmstart-TC) into components
to find which one can move pp throughput. Per-kernel shares are captured from the FIRST __call__ under DEBUG=2
(TinyJit's record pass runs+prints the kernels with real tm); the warm pp tok/s from qk_prefill_v2_measure is the
authoritative total. Decision gate (scope PWR-1): a component / shared-primitive family must be >=30% of warm
prefill AND a 2x component win must imply >=1.2x full prefill, else don't build the weight kernel.

NOTE: PREFILL_V2 pre-realizes fp16 weights at load, so the forward has NO in-forward Q4_K dequant -- it is fp16
TC matmuls (FFN + attn QKVO) + SDPA attention (512x512) + norm/rope/swiglu. The quantized-weight-reuse primitive
only matters if (a) a matmul component dominates AND (b) the fp16-materialize path is not already capturing it
(it OOMs on 14B/32B -> VRAM-frugal quant reuse is the alternative).

  DEV=AMD PREFILL_V2=1 PYTHONPATH=. .venv/bin/python extra/qk_prefill_component_breakdown.py
"""
from __future__ import annotations
import os, io, contextlib, re, json, pathlib
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LINE = re.compile(r"\*\*\*\s+\S+\s+\d+\s+(\S+).*?tm\s+([0-9.]+)us")

def classify(name:str, N:int) -> str:
  n = _ANSI.sub("", name)
  dims = re.findall(r"\d+", n)
  # attention SDPA over the N=512 chunk (scores [N,N], softmax, P@V): kernels carrying N (or N+pad) as a dim
  if any(str(N + d) in dims for d in range(-2, 3)): return "attention"
  # TC / WMMA matmuls (fp16 FFN + attn QKVO) -- tinygrad names TC kernels with WMMA or the matmul r_ over 4096/12288
  if "WMMA" in n or "wmma" in n: return "matmul_tc"
  if n.startswith("r_") and any(d in dims for d in ("4096", "12288", "11008", "14336")): return "matmul_tc"
  if n.startswith("E_"): return "elementwise_norm"
  if n.startswith("r_"): return "reduce_other"
  return "unknown"

def main():
  if not os.environ.get("PREFILL_V2"):
    print("run with PREFILL_V2=1"); raise SystemExit(2)
  from tinygrad import Tensor, UOp, Context
  from tinygrad.llm.model import Transformer, PREFILL_UBATCH
  Tensor.manual_seed(0)
  model, _ = Transformer.from_gguf(pathlib.Path("/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf").expanduser(), 2048)
  N = PREFILL_UBATCH
  vsp = UOp.variable("start_pos", 0, model.max_context - 1)
  temp = Tensor([0.0])
  t = Tensor([5, 6, 7, 8, 9, 10] * 200 + [0] * (model.max_context - 1200), dtype="int32").reshape(1, model.max_context)
  chunk = t[:, vsp.bind(0):vsp.bind(0) + N]
  # first __call__ under DEBUG=2 = TinyJit record pass -> emits per-kernel tm (real GPU time; compile overhead ignored)
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf), Context(DEBUG=2): model(chunk, vsp.bind(0), temp).realize()
  comp = {}
  for ln in buf.getvalue().splitlines():
    mm = _LINE.search(ln)
    if mm: comp[classify(mm.group(1), N)] = comp.get(classify(mm.group(1), N), 0.0) + float(mm.group(2))
  tot = sum(comp.values())
  print(f"\n=== PREFILL_V2 forward component shares (N={N}, first-call DEBUG2 tm; directional) ===")
  for c, us in sorted(comp.items(), key=lambda x: -x[1]):
    print(f"  {c:18}: {us/1000:8.2f}ms  {100*us/tot:5.1f}%")
  # top kernels for sanity
  top = {}
  for ln in buf.getvalue().splitlines():
    mm = _LINE.search(ln)
    if mm:
      nm = _ANSI.sub("", mm.group(1)); top.setdefault(nm, [0, 0.0]); top[nm][0] += 1; top[nm][1] += float(mm.group(2))
  print("  --- top kernels ---")
  for nm, (c, us) in sorted(top.items(), key=lambda x: -x[1][1])[:10]:
    print(f"  {us/1000:8.2f}ms {100*us/tot:5.1f}% x{c:3d}  {nm[:54]}")
  art = pathlib.Path("bench/qk-prefill-weight-reuse-20260618/pwr1-components.json"); art.parent.mkdir(parents=True, exist_ok=True)
  art.write_text(json.dumps({"N": N, "component_us": comp, "note": "first-call DEBUG2 shares directional; warm pp tok/s in pwr0 authoritative"}, indent=2))
  print(f"\nartifact: {art}")

if __name__ == "__main__":
  main()
