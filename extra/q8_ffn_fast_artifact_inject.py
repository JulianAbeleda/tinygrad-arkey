#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os, pathlib, time

os.environ.setdefault("DEV", "AMD")
os.environ.setdefault("JIT", "1")
os.environ.setdefault("QK_PRIMITIVE_STORAGE", "shared")

import numpy as np

from tinygrad import Tensor, TinyJit, UOp, dtypes
from tinygrad.device import Device
from tinygrad.uop.ops import AxisType, KernelInfo
import tinygrad.engine.realize as R
import tinygrad.runtime.ops_amd as ops_amd
from extra.llm_generate import load_model_and_tokenizer
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, hip_norm_source
from extra.q8_ffn_oneblock_route import diff_stats, q4_words, q8_proxy_ffn
from extra.qk_nll_eval import CALIB_TEXT

DIM, HIDDEN, Q8_BYTES = 4096, 12288, (4096 // 32) * 36
PROD_THREADS = 1024

class Q8ArtifactRunner:
  def __init__(self, prg, global_size:tuple[int, int, int], local_size:tuple[int, int, int]):
    self.prg, self.q8_global, self.q8_local = prg, global_size, local_size
  def __getattr__(self, name): return getattr(self.prg, name)
  def __call__(self, *bufs, global_size=None, local_size=None, vals=(), wait=False, timeout=None):
    return self.prg(*bufs, global_size=self.q8_global, local_size=self.q8_local, vals=vals, wait=wait, timeout=timeout)

_orig_exec = ops_amd.AMDComputeQueue.exec
def _patched_exec(self, prg, args_state, global_size, local_size):
  if isinstance(prg, Q8ArtifactRunner): global_size, local_size = prg.q8_global, prg.q8_local
  return _orig_exec(self, prg, args_state, global_size, local_size)
ops_amd.AMDComputeQueue.exec = _patched_exec

def producer_stub(norm_out:UOp, q8:UOp, x:UOp, w:UOp) -> UOp:
  l0 = UOp.range(PROD_THREADS, 0, AxisType.LOCAL)
  norm, q8f, xf, wf = norm_out.flatten(), q8.flatten(), x.flatten(), w.flatten()
  # Placeholder body only establishes the graph contract. runtime_cache swaps execution to the precompiled artifact.
  st0 = norm[l0].store(xf[l0] + wf[l0] * 0.0)
  st1 = q8f[l0].store(q8f[l0])
  return UOp.group(st0, st1).end(l0).sink(arg=KernelInfo(name="q8_rmsnorm_side_inject", opts_to_apply=()))

def gateup_stub(gate:UOp, up:UOp, gate_words:UOp, up_words:UOp, q8:UOp) -> UOp:
  g0 = UOp.range(HIDDEN, 0, AxisType.GLOBAL)
  g1 = UOp.range(2, 1, AxisType.GLOBAL)
  l0 = UOp.range(32, 2, AxisType.LOCAL)
  l1 = UOp.range(4, 3, AxisType.LOCAL)
  gf, uf = gate.flatten(), up.flatten()
  # Touch all inputs so ProgramInfo.globals orders buffers as gate, up, gate_words, up_words, q8.
  v = (gate_words.flatten()[0].cast(dtypes.float32) * 0.0) + (up_words.flatten()[0].cast(dtypes.float32) * 0.0) + \
      (q8.flatten()[0].cast(dtypes.float32) * 0.0) + (l0.cast(dtypes.float32) * 0.0) + (l1.cast(dtypes.float32) * 0.0)
  st0 = gf[g0].store(v)
  st1 = uf[g0].store(v + g1.cast(dtypes.float32) * 0.0)
  return UOp.group(st0, st1).end(l1, l0, g1, g0).sink(arg=KernelInfo(name="q8_mmvq_gateup_inject", opts_to_apply=()))

def install_runtime_swaps(producer_threads:int) -> dict:
  dev = Device["AMD"]
  prod_prg = Q8ArtifactRunner(dev.runtime("q8_rmsnorm_side_injected_artifact", compile_hipcc_linked(hip_norm_source(producer_threads), "gfx1100")),
                              (1,1,1), (producer_threads,1,1))
  gateup_prg = Q8ArtifactRunner(dev.runtime("q8_mmvq_gateup_injected_artifact", compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, "gfx1100")),
                                (HIDDEN,2,1), (32,4,1))
  keys: list[tuple[bytes, str, str, tuple[int, ...], tuple[int, ...]|None]] = []
  orig = R.get_runtime
  def hook(device, ast, cache=True):
    if ast.arg.name in {"q8_rmsnorm_side_inject", "q8_mmvq_gateup_inject"}:
      keys.append((ast.key, device, ast.arg.name, tuple(ast.arg.global_size), tuple(ast.arg.local_size or ())))
    return orig(device, ast, cache)
  R.get_runtime = hook
  try:
    x = Tensor.ones(DIM, dtype=dtypes.float32, device="AMD").contiguous().realize()
    w = Tensor.ones(DIM, dtype=dtypes.float32, device="AMD").contiguous().realize()
    norm = Tensor.empty(DIM, dtype=dtypes.float32, device="AMD").contiguous()
    q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device="AMD").contiguous()
    norm.custom_kernel(q8, x, w, fxn=producer_stub)[:2][0].realize()
    gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
    up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device="AMD").contiguous()
    gw = Tensor.empty(HIDDEN * DIM // 2, dtype=dtypes.uint8, device="AMD").contiguous().realize()
    uw = Tensor.empty(HIDDEN * DIM // 2, dtype=dtypes.uint8, device="AMD").contiguous().realize()
    gate.custom_kernel(up, gw, uw, q8.realize(), fxn=gateup_stub)[:2][0].realize()
  finally:
    R.get_runtime = orig
  by_name = {name: (key, device, gs, ls) for key, device, name, gs, ls in keys}
  if set(by_name) != {"q8_rmsnorm_side_inject", "q8_mmvq_gateup_inject"}:
    raise RuntimeError(f"did not capture both q8 injected program keys: {[(n, gs, ls) for _,_,n,gs,ls in keys]}")
  prod_key, prod_dev, prod_gs, prod_ls = by_name["q8_rmsnorm_side_inject"]
  gu_key, gu_dev, gu_gs, gu_ls = by_name["q8_mmvq_gateup_inject"]
  R.runtime_cache[(prod_key, prod_dev)] = prod_prg
  R.runtime_cache[(gu_key, gu_dev)] = gateup_prg
  return {
    "producer_program": {"placeholder_global_size": list(prod_gs), "placeholder_local_size": list(prod_ls),
                         "artifact_global_size": list(prod_prg.q8_global), "artifact_local_size": list(prod_prg.q8_local),
                         "kernarg_size": prod_prg.kernargs_segment_size},
    "gateup_program": {"placeholder_global_size": list(gu_gs), "placeholder_local_size": list(gu_ls),
                       "artifact_global_size": list(gateup_prg.q8_global), "artifact_local_size": list(gateup_prg.q8_local),
                       "kernarg_size": gateup_prg.kernargs_segment_size},
  }

def injected_ffn(block, h:Tensor) -> Tensor:
  device = h.device
  norm_w = block.ffn_norm.weight.cast(dtypes.float32).to(device).contiguous()
  h_vec = h.reshape(DIM).contiguous()
  norm = Tensor.empty(DIM, dtype=dtypes.float32, device=device).contiguous()
  q8 = Tensor.empty(Q8_BYTES, dtype=dtypes.uint8, device=device).contiguous()
  norm, q8, *_ = norm.custom_kernel(q8, h_vec, norm_w, fxn=producer_stub)
  gate_words, up_words = q4_words(block.ffn_gate, device), q4_words(block.ffn_up, device)
  gate = Tensor.empty(HIDDEN, dtype=dtypes.float32, device=device).contiguous()
  up = Tensor.empty(HIDDEN, dtype=dtypes.float32, device=device).contiguous()
  gate, up, *_ = gate.custom_kernel(up, gate_words, up_words, q8, fxn=gateup_stub)
  gate = gate.reshape(1, 1, HIDDEN)
  up = up.reshape(1, 1, HIDDEN)
  return block.ffn_down(gate.silu().contiguous() * up).realize()

def tensor_max_mean_abs(a:Tensor, b:Tensor) -> tuple[float, float]:
  av, bv = a.numpy().astype("float32", copy=False), b.numpy().astype("float32", copy=False)
  d = np.abs(av - bv)
  return float(np.nanmax(d)), float(np.nanmean(d))

def main() -> None:
  ap = argparse.ArgumentParser(description="A3 q8 fast artifact runtime_cache/TinyJit injection proof")
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--max-context", type=int, default=4096)
  ap.add_argument("--block", type=int, default=0)
  ap.add_argument("--seed", type=int, default=20260616)
  ap.add_argument("--execute", action="store_true", help="unsafe: execute the swapped q8 runtimes; previous attempt faulted on HCQ")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-handwritten-oracle/fast_artifact_inject.json"))
  args = ap.parse_args()

  t0 = time.perf_counter()
  install = install_runtime_swaps(PROD_THREADS)
  compile_s = time.perf_counter() - t0

  if not args.execute:
    res = {
      "date": "2026-06-19",
      "phase": "A3-injection",
      "route": "runtime_cache_swapped_PROGRAM_UOps_for_q8_producer_and_fused_gateup",
      "compile_s": compile_s,
      "install": install,
      "verdict": "BLOCKED_UNSAFE",
      "reason": "A swapped-runtime execution attempt reached HCQ but caused an AMD MMU fault during eager route output copyout. "
                "The direct HCQ artifact route is correct; the Tensor-visible injection contract is not yet safe.",
      "last_fault": {"kind": "AMD_MMU_fault", "address": "0x76205F713000", "stage": "eager injected route"},
      "next": "Do not rerun execution by default. First build a contract verifier for ProgramInfo.globals/outs/ins and "
              "kernarg buffer order, or inject explicit Ops.PROGRAM nodes instead of runtime-cache swapping placeholder kernels.",
      "execute_flag_required": "--execute",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    raise SystemExit(1)

  model, tok = load_model_and_tokenizer(args.model, args.max_context, seed=args.seed)
  for lin in getattr(model, "_q4k_linears", None).linears if getattr(model, "_q4k_linears", None) else []:
    lin.decode_enabled = True
  block = model.blk[args.block]
  ids = (tok.prefix() if hasattr(tok, "prefix") else []) + tok.encode(CALIB_TEXT)
  token = Tensor([[ids[0]]], dtype=dtypes.int32, device="AMD").contiguous()
  x = model.token_embd(token).float().realize()
  block._init_state(x)
  h = (x + block._attention(block.attn_norm(x), 0)).contiguous().realize()
  proxy, _ = q8_proxy_ffn(block, h)

  eager = injected_ffn(block, h)
  eager_diff = diff_stats(eager, proxy)

  @TinyJit
  def jroute(inp:Tensor):
    return injected_ffn(block, inp)

  jit_diffs = []
  for _ in range(4):
    out = jroute(h).realize()
    Device["AMD"].synchronize()
    mx, mean = tensor_max_mean_abs(out, proxy)
    jit_diffs.append({"max_abs": mx, "mean_abs": mean})

  maps = pathlib.Path("/proc/self/maps").read_text(errors="ignore")
  res = {
    "date": "2026-06-19",
    "phase": "A3-injection",
    "route": "runtime_cache_swapped_PROG_UOps_for_q8_producer_and_fused_gateup",
    "compile_s": compile_s,
    "install": install,
    "eager": {
      "max_abs": eager_diff["max_abs"],
      "mean_abs": eager_diff["mean_abs"],
      "correct": eager_diff["finite"] == eager_diff["size"] and eager_diff["max_abs"] <= 2e-2,
    },
    "tinyjit": {
      "diffs": jit_diffs,
      "replay_correct": all(x["max_abs"] <= 2e-2 for x in jit_diffs[2:]),
      "calls": 4,
      "replay_calls": [2, 3],
    },
    "no_hip_runtime_in_process": "libamdhip64.so" not in maps,
    "default_changed": False,
  }
  res["verdict"] = "PASS" if res["eager"]["correct"] and res["tinyjit"]["replay_correct"] and res["no_hip_runtime_in_process"] else "FAIL"
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(res, indent=2) + "\n")
  print(json.dumps(res, indent=2))
  if res["verdict"] != "PASS": raise SystemExit(1)

if __name__ == "__main__":
  main()
