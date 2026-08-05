#!/usr/bin/env python3
"""B3 per-class numerics probe: flash decode, q4k gemv, q6k gemv, rmsnorm, residual add.

Runs the production kernel emitters standalone on the selected route (DEV=NV or
DEV=CUDA with CUDA_GRAPH_STREAMS=1) with fixed synthetic inputs and compares the
device output against an fp32 CPU reference on host-sampled elements, reporting
max abs err, max rel err, sampled FNV-1a 64-bit hash over the expected/actual
double bit patterns (E3/B1 convention), and a PASS/FAIL against a documented
per-class tolerance. Identical inputs on every route, so the per-class rows are
directly comparable between NV and CUDA.

Run: PYTHONPATH=. DEV=NV .venv/bin/python extra/llm_research/decode/route_class_numerics.py --out /tmp/numerics_nv.json
     PYTHONPATH=. DEV=CUDA CUDA_GRAPH_STREAMS=1 .venv/bin/python extra/llm_research/decode/route_class_numerics.py --out /tmp/numerics_cuda.json
"""
from __future__ import annotations

import argparse, json, struct

import numpy as np

from tinygrad import Tensor, dtypes

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"
SAMPLE_STRIDE = 4096


def fnv1a64_init() -> int: return 0xcbf29ce484222325
def fnv1a64_update(h:int, data:bytes) -> int:
  for b in data:
    h ^= b
    h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
  return h


def fnv1a64_double(h:int, v:float) -> int:
  return fnv1a64_update(h, struct.pack("<d", float(v)))


def sampled_stats(expected:np.ndarray, actual:np.ndarray, max_samples:int=SAMPLE_STRIDE):
  flat_e, flat_a = expected.reshape(-1), actual.reshape(-1)
  n = flat_e.size
  stride = max(1, n // max_samples)
  idx = np.arange(0, n, stride)[:max_samples]
  e, a = flat_e[idx], flat_a[idx]
  err = np.abs(a - e)
  denom = np.maximum(np.abs(e), 1e-6)
  h = fnv1a64_init()
  for ev, av in zip(e.tolist(), a.tolist()):
    h = fnv1a64_double(h, ev)
    h = fnv1a64_double(h, av)
  return {"samples": int(idx.size), "max_abs_err": float(err.max()), "max_rel_err": float((err / denom).max()),
          "fnv1a64": f"{h:016x}"}


def check_class(name:str, got:np.ndarray, ref:np.ndarray, tol_rel:float, tol_abs:float):
  stats = sampled_stats(ref, got)
  stats["class"] = name
  stats["tol_rel"] = tol_rel
  stats["tol_abs"] = tol_abs
  stats["shape"] = list(got.shape)
  stats["pass"] = bool(np.allclose(got, ref, rtol=tol_rel, atol=tol_abs))
  return stats


# ---- flash decode attention (production 8B geometry, KV_BOTH + fused combine) ----
def probe_flash(dev:str) -> dict:
  HQ, HKV, HD, MAXC, S, TC = 32, 8, 128, 512, 48, 512
  seed = 20260804
  rng = np.random.default_rng(seed)
  q = rng.normal(0, .04, (1, HQ, 1, HD)).astype(np.float16)
  k = rng.normal(0, .04, (1, HKV, TC, HD)).astype(np.float16)
  v = rng.normal(0, .04, (1, HKV, TC, HD)).astype(np.float16)
  cache_np = np.zeros((2, 1, HKV, MAXC, HD), dtype=np.float16)
  cache_np[0, 0, :, :TC, :] = k[0]
  cache_np[1, 0, :, :TC, :] = v[0]
  cache = Tensor(cache_np, device=dev)
  q_dev = Tensor(q, device=dev).reshape(HQ * HD)

  from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
  spec = describe_flash_decode_attention(HQ, HD, HKV, MAXC, S, staging="KV_BOTH", fused_combine=True)
  partial = execute_research_program(Tensor.empty(HQ * S * (HD + 2), dtype=dtypes.float32, device=dev),
    q_dev, cache, program=KernelProgram("exp.route_class_numerics", spec.tile.kernel_name,
    KernelProgramProvenance.RESEARCH_ONLY, spec.emit_tile(TC)))
  out = execute_research_program(Tensor.empty(HQ * HD, dtype=dtypes.float32, device=dev), partial,
    program=KernelProgram("exp.route_class_numerics", spec.combine.kernel_name,
    KernelProgramProvenance.RESEARCH_ONLY, spec.emit_combine())).reshape(HQ, HD)
  got = out.numpy().astype(np.float32)

  from extra.llm_research.attention_harness_common import reference_attention
  mask = Tensor.full((1, 1, 1, TC), 0.0, dtype=dtypes.float16, device="CPU").contiguous()
  ref_t = reference_attention(Tensor(q, device="CPU"), Tensor(k, device="CPU"), Tensor(v, device="CPU"),
                              mask, HQ, HKV)
  ref = ref_t.numpy().astype(np.float32).reshape(HQ, HD)
  return check_class("flash_decode_attention", got, ref, 0.03, 0.006)


# ---- q4k gemv (production shapes, packed random words) ----
def _make_q4k_words(rows:int, k:int, seed:int) -> tuple[np.ndarray, np.ndarray]:
  from extra.llm_research.layout import Q4_K_BLOCK_BYTES, Q4_K_BLOCK_ELEMS
  rng = np.random.default_rng(seed)
  nblocks = (rows * k) // Q4_K_BLOCK_ELEMS
  raw = rng.integers(0, 256, size=nblocks * Q4_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q4_K_BLOCK_BYTES)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  dmin = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  raw[:, 0:2] = d.view(np.uint8).reshape(nblocks, 2)
  raw[:, 2:4] = dmin.view(np.uint8).reshape(nblocks, 2)
  words = raw.reshape(-1).copy().view(np.uint32)
  return words, raw


def probe_q4k(dev:str, rows:int, k:int) -> dict:
  from tinygrad.llm.decode_kernels import q4k_g3_lanemap_gemv_kernel
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
  from extra.llm_research.layout import q4_k_reference
  words_np, raw = _make_q4k_words(rows, k, seed=rows + 4096)
  x = np.random.default_rng(1000 + rows).normal(0, 1.0, k).astype(np.float16)
  words = Tensor(words_np.copy(), device=dev).contiguous().realize()
  x_dev = Tensor(x, device=dev)
  out = execute_research_program(Tensor.empty(rows, dtype=dtypes.float32, device=dev), words, x_dev,
    program=KernelProgram("exp.route_class_numerics", f"q4k_g3_lanemap_gemv_{rows}_{k}",
    KernelProgramProvenance.RESEARCH_ONLY, q4k_g3_lanemap_gemv_kernel(rows, k)))
  got = out.numpy().astype(np.float32)
  w_ref = q4_k_reference(Tensor(raw.reshape(-1).copy()), rows * k).numpy().astype(np.float32).reshape(rows, k)
  ref = w_ref @ x.astype(np.float32)
  return check_class(f"q4k_gemv_{rows}_{k}", got, ref, 1e-2, 1e-2)


# ---- q6k gemv (coop, external_sum, production shapes) ----
def _make_q6k_halfs(rows:int, k:int, seed:int) -> np.ndarray:
  from extra.llm_research.layout import Q6_K_BLOCK_BYTES
  rng = np.random.default_rng(seed)
  nblocks = (rows * k) // 256
  raw = rng.integers(0, 256, size=nblocks * Q6_K_BLOCK_BYTES, dtype=np.uint8).reshape(nblocks, Q6_K_BLOCK_BYTES)
  d = (rng.standard_normal(nblocks).astype(np.float32) * 5e-4).astype(np.float16)
  raw[:, -2:] = d.view(np.uint8).reshape(nblocks, 2)
  return raw.reshape(-1).copy().view(np.uint16)


def probe_q6k(dev:str, rows:int, k:int, row_tile:int=4) -> dict:
  from tinygrad.llm.decode_kernels import q6k_spec_for_role, emit_q6k_gemv_kernel
  from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, execute_research_program
  from extra.llm_research.layout import q6_k_reference
  halfs_np = _make_q6k_halfs(rows, k, seed=rows * 7 + k)
  x = np.random.default_rng(2000 + rows).normal(0, 1.0, k).astype(np.float16)
  halfs = Tensor(halfs_np.copy(), device=dev).contiguous().realize()
  x_dev = Tensor(x, device=dev)
  spec = q6k_spec_for_role(rows, k, parts=1, row_tile=row_tile, use_coop=True, reduction="external_sum")
  partials = execute_research_program(Tensor.empty(rows, spec.partial_axis_extent, dtype=dtypes.float32, device=dev),
    halfs, x_dev, program=KernelProgram("exp.route_class_numerics", spec.kernel_name,
    KernelProgramProvenance.RESEARCH_ONLY, emit_q6k_gemv_kernel(spec)))
  got = partials.sum(axis=1).numpy().astype(np.float32)
  w_ref = q6_k_reference(Tensor(halfs_np.reshape(-1).copy().view(np.uint8), dtype=dtypes.uint8), rows * k).numpy().astype(np.float32).reshape(rows, k)
  ref = w_ref @ x.astype(np.float32)
  return check_class(f"q6k_gemv_{rows}_{k}_rt{row_tile}", got, ref, 1e-2, 1e-2)


# ---- rmsnorm (nn.RMSNorm math, fp16 x / fp32 w, the decode norm class) ----
def probe_rmsnorm(dev:str, dim:int=4096, eps:float=1e-6) -> dict:
  rng = np.random.default_rng(3000)
  x = rng.normal(0, 1.0, (1, 1, dim)).astype(np.float16)
  w = rng.normal(1.0, 0.01, dim).astype(np.float32)
  x_dev = Tensor(x, device=dev)
  w_dev = Tensor(w, device=dev)
  normed = x_dev.float().square().mean(-1, keepdim=True).add(eps).rsqrt()
  out = normed.cast(dtypes.float16).mul(w_dev)
  got = out.numpy().astype(np.float32).reshape(-1)
  xf = x.astype(np.float32)
  mean = (xf * xf).mean(-1, keepdims=True)
  ref = (xf / np.sqrt(mean + eps)).astype(np.float16).astype(np.float32) * w
  return check_class(f"rmsnorm_{dim}", got, ref, 1e-2, 1e-2)


# ---- residual add (elementwise/fusion class) ----
def probe_residual(dev:str, n:int=4096) -> dict:
  rng = np.random.default_rng(4000)
  a = rng.normal(0, 1.0, (1, 1, n)).astype(np.float16)
  b = rng.normal(0, 1.0, (1, 1, n)).astype(np.float16)
  got = Tensor(a, device=dev).add(Tensor(b, device=dev)).numpy().astype(np.float32).reshape(-1)
  ref = (a.astype(np.float32) + b.astype(np.float32))
  return check_class(f"residual_add_{n}", got, ref, 1e-2, 1e-2)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  from tinygrad import Device
  dev = Device.DEFAULT
  rows: dict[str, list[dict]] = {}
  rows["flash_decode_attention"] = [probe_flash(dev)]
  rows["q4k_gemv"] = [probe_q4k(dev, 4096, 4096), probe_q4k(dev, 4096, 12288), probe_q4k(dev, 12288, 4096)]
  rows["q6k_gemv"] = [probe_q6k(dev, 4096, 12288), probe_q6k(dev, 1024, 4096)]
  rows["norms"] = [probe_rmsnorm(dev, 4096), probe_rmsnorm(dev, 128)]
  rows["elementwise"] = [probe_residual(dev)]
  out = {"route": dev, "model": MODEL,
         "git_commit": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd="/home/ubuntu/tinygrad-arkey").decode().strip(),
         "classes": rows}
  print(json.dumps(out, indent=1))
  with open(args.out, "w") as f:
    json.dump(out, f, indent=1)


if __name__ == "__main__":
  main()
