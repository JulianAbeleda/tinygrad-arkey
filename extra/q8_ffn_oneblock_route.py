#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, statistics, time

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "1")
os.environ.setdefault("QK_PRIMITIVE_STORAGE", "shared")

from tinygrad import Tensor, dtypes
from tinygrad.device import Buffer, Device
from extra.llm_generate import load_model_and_tokenizer
from extra.q8_ffn_hcq_artifact import NORM_SOURCE, MMVQ_SOURCE
from extra.qk_layout import q8_1_dequantize, q8_1_quantize
from extra.qk_nll_eval import CALIB_TEXT

def realized_buf(t:Tensor):
  t.realize()
  buf = t.uop.base.realized or t.uop.buffer
  buf.ensure_allocated()
  return buf._buf

def empty_realized(shape, dtype=dtypes.float32, device="AMD") -> Tensor:
  return Tensor.empty(*shape, dtype=dtype, device=device).contiguous().realize()

def median_ms(samples:list[float]) -> float:
  return statistics.median(samples) * 1000.0

def q4_words(linear, device:str) -> Tensor:
  words = linear.q4k_storage.words.to(device)
  if linear.q4k_storage.mode == "q4_ondemand": words = words.contiguous()
  return words.realize()

def tensor_stats(t:Tensor) -> dict:
  import numpy as np
  a = t.numpy().astype("float32", copy=False)
  finite = np.isfinite(a)
  return {
    "shape": list(a.shape),
    "finite": int(finite.sum()),
    "size": int(a.size),
    "min": float(np.nanmin(a)) if a.size else 0.0,
    "max": float(np.nanmax(a)) if a.size else 0.0,
    "mean": float(np.nanmean(a)) if a.size else 0.0,
  }

def diff_stats(a:Tensor, b:Tensor) -> dict:
  import numpy as np
  av, bv = a.numpy().astype("float32", copy=False), b.numpy().astype("float32", copy=False)
  d = np.abs(av - bv)
  finite = np.isfinite(d)
  return {
    "shape_a": list(av.shape),
    "shape_b": list(bv.shape),
    "finite": int(finite.sum()),
    "size": int(d.size),
    "max_abs": float(np.nanmax(d)) if d.size else 0.0,
    "mean_abs": float(np.nanmean(d)) if d.size else 0.0,
  }

def q8_route_ffn(block, h:Tensor, norm_prg, mmvq_prg, *, warmups:int, iters:int) -> tuple[Tensor, dict, dict]:
  device = h.device
  assert h.shape == (1, 1, block.config.dim), h.shape
  norm_w = block.ffn_norm.weight.cast(dtypes.float32).to(device).contiguous().realize()
  h_vec = h.reshape(block.config.dim).contiguous().realize()
  norm_out = empty_realized((block.config.dim,), dtypes.float32, device)
  q8buf = Buffer(device, (block.config.dim // 32) * 36, dtypes.uint8).ensure_allocated()

  gate_t = empty_realized((block.config.hidden_dim,), dtypes.float32, device)
  up_t = empty_realized((block.config.hidden_dim,), dtypes.float32, device)
  gate_words = q4_words(block.ffn_gate, device)
  up_words = q4_words(block.ffn_up, device)

  prod_ms, gate_ms, up_ms = [], [], []
  for i in range(warmups + iters):
    p = norm_prg(realized_buf(norm_out), q8buf._buf, realized_buf(h_vec), realized_buf(norm_w),
                 global_size=(1,1,1), local_size=(256,1,1), wait=True)
    g = mmvq_prg(realized_buf(gate_t), realized_buf(gate_words), q8buf._buf,
                 global_size=(block.config.hidden_dim,1,1), local_size=(32,4,1), wait=True)
    u = mmvq_prg(realized_buf(up_t), realized_buf(up_words), q8buf._buf,
                 global_size=(block.config.hidden_dim,1,1), local_size=(32,4,1), wait=True)
    if i >= warmups:
      prod_ms.append(float(p)); gate_ms.append(float(g)); up_ms.append(float(u))

  gate = gate_t.reshape(1, 1, block.config.hidden_dim)
  up = up_t.reshape(1, 1, block.config.hidden_dim)
  out = block.ffn_down(gate.silu().contiguous() * up).realize()
  stats = {"norm_out": tensor_stats(norm_out), "gate": tensor_stats(gate_t), "up": tensor_stats(up_t), "out": tensor_stats(out)}
  timing = {
    "producer_ms": median_ms(prod_ms),
    "gate_ms": median_ms(gate_ms),
    "up_ms": median_ms(up_ms),
    "gate_up_lifecycle_ms": median_ms(prod_ms) + median_ms(gate_ms) + median_ms(up_ms),
    "samples": {
      "producer_ms": [round(x * 1000.0, 6) for x in prod_ms],
      "gate_ms": [round(x * 1000.0, 6) for x in gate_ms],
      "up_ms": [round(x * 1000.0, 6) for x in up_ms],
    },
  }
  return out, timing, stats

def q8_proxy_ffn(block, h:Tensor) -> tuple[Tensor, Tensor]:
  x = block.ffn_norm(h).realize()
  q, scales = q8_1_quantize(x.reshape(-1, block.config.dim).cast(dtypes.float32))
  xq = q8_1_dequantize(q, scales).reshape(*x.shape)
  out = block.ffn_down(block.ffn_gate(xq).silu().contiguous() * block.ffn_up(xq)).realize()
  return out, x

def fp_ffn(block, h:Tensor) -> Tensor:
  return block._feed_forward(block.ffn_norm(h)).realize()

def main() -> None:
  ap = argparse.ArgumentParser(description="A2 one-block eager q8 handwritten FFN route")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--block", type=int, default=0)
  ap.add_argument("--seed", type=int, default=20260616)
  ap.add_argument("--warmups", type=int, default=6)
  ap.add_argument("--iters", type=int, default=20)
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/oneblock_route.json"))
  args = ap.parse_args()

  dev = Device["AMD"]
  t0 = time.perf_counter()
  norm_prg = dev.runtime("q8_rmsnorm_side", dev.compiler.compile(NORM_SOURCE))
  mmvq_prg = dev.runtime("q8_mmvq", dev.compiler.compile(MMVQ_SOURCE))
  compile_s = time.perf_counter() - t0

  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True
  block = model.blk[args.block]
  if hasattr(block, "ffn_gate_exps"): raise RuntimeError("A2 q8 handwritten route is scoped to dense FFN blocks")
  for name in ("ffn_gate", "ffn_up"):
    lin = getattr(block, name)
    if lin.__class__.__name__ != "Q4KPrimitiveLinear":
      raise RuntimeError(f"{name} is {lin.__class__.__name__}, expected Q4KPrimitiveLinear")

  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)
  h = (x + block._attention(block.attn_norm(x), 0)).contiguous().realize()

  route_out, timing, route_stats = q8_route_ffn(block, h, norm_prg, mmvq_prg, warmups=args.warmups, iters=args.iters)
  proxy_out, norm_ref = q8_proxy_ffn(block, h)
  fp_out = fp_ffn(block, h)

  route_proxy = diff_stats(route_out, proxy_out)
  proxy_fp = diff_stats(proxy_out, fp_out)
  result = {
    "date": "2026-06-19",
    "phase": "A2",
    "model": args.model,
    "block": args.block,
    "input": "token0 post-attention hidden state",
    "compile_s": compile_s,
    "timing": timing,
    "route_stats": route_stats,
    "correctness": {
      "route_vs_q8_proxy_max_abs": route_proxy["max_abs"],
      "route_vs_q8_proxy_mean_abs": route_proxy["mean_abs"],
      "route_vs_q8_proxy_finite": route_proxy["finite"],
      "route_vs_q8_proxy_size": route_proxy["size"],
      "q8_proxy_vs_fp_max_abs": proxy_fp["max_abs"],
      "q8_proxy_vs_fp_mean_abs": proxy_fp["mean_abs"],
      "norm_ref_max_abs": float(norm_ref.abs().max().item()),
    },
    "gates": {
      "route_vs_proxy_max_abs_lte_2e_2": route_proxy["finite"] == route_proxy["size"] and route_proxy["max_abs"] <= 2e-2,
      "hcq_eager_lifecycle_lte_modeled_us": timing["gate_up_lifecycle_ms"] * 1000.0 <= 107.64 * 1.20,
      "no_hip_runtime_in_process": "libamdhip64.so" not in pathlib.Path("/proc/self/maps").read_text(errors="ignore"),
      "default_changed": False,
    },
    "redirect": {
      "reason": "COMGR-compiled HCQ artifact is correct but too slow; hipcc-quality object is not currently loadable by AMDProgram",
      "hipcc_loader_probe": "HIPCCCompiler(gfx1100) compiled MMVQ_SOURCE, AMDProgram rejected object with unknown AMD reloc 10",
    },
  }
  result["verdict"] = "PASS" if all(result["gates"].values()) else "FAIL"
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))
  if result["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
