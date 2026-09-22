#!/usr/bin/env python3
"""Exact one-K256 A/B for the wide Q6_K ping/pong phase pipeline."""
from __future__ import annotations

import argparse, json, pathlib, re, statistics
import numpy as np

from tinygrad import Device, Tensor
from tinygrad.runtime.ops_nv import NVProgram
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record, _sass
from extra.llm_research.prefill.nv_compiler_q6k_wide_double_buffer import (
  bound_compiler_q6k_wide_k256, transform_compiler_q6k_wide_double_buffer)

M, N, K = 512, 4096, 12288
GRID, BLOCK = (32, 4, 1), (32, 2, 4)


def _buf(t: Tensor): return t.uop.buffer.get_buf("NV")


def _stats(samples: list[float]) -> dict[str, object]:
  return {"samples_us": samples, "min_us": min(samples), "median_us": statistics.median(samples), "max_us": max(samples)}


def _program(source: str, stem: pathlib.Path) -> tuple[NVProgram, dict[str, object]]:
  name = re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(', source)
  if name is None: raise ValueError("wide Q6 symbol not found")
  binary = Device["NV"].compiler.compile(source)
  return NVProgram(Device["NV"], name.group(1), binary), _sass(binary, stem)


def _time(program: NVProgram, args: tuple[object, ...], rounds: int) -> list[float]:
  program(*args, global_size=GRID, local_size=BLOCK, wait=True)
  return [program(*args, global_size=GRID, local_size=BLOCK, wait=True)*1e6 for _ in range(rounds)]


def main() -> int:
  root = pathlib.Path(__file__).resolve().parents[3]
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--base-source", default=str(root/"docs/task_workflow/evidence/nv-q6-wide-live-publication-20260831/control-artifacts/wide_direct.cu"))
  ap.add_argument("--rounds", type=int, default=31)
  ap.add_argument("--out", required=True)
  ap.add_argument("--artifacts", required=True)
  args = ap.parse_args()
  if args.rounds < 9: raise ValueError("qualification requires R9 or greater")
  artifacts = pathlib.Path(args.artifacts); artifacts.mkdir(parents=True, exist_ok=True)

  base = pathlib.Path(args.base_source).read_text()
  control_source = bound_compiler_q6k_wide_k256(base)
  candidate_source = transform_compiler_q6k_wide_double_buffer(control_source)
  (artifacts/"control.cu").write_text(control_source)
  (artifacts/"candidate.cu").write_text(candidate_source)
  control_program, control_sass = _program(control_source, artifacts/"control")
  candidate_program, candidate_sass = _program(candidate_source, artifacts/"candidate")

  model = pathlib.Path(args.model); meta = read_metadata(model)
  info = next(i for i in meta.infos if i.name == "blk.0.ffn_down.weight")
  if info.typ != GGML_Q6_K: raise RuntimeError(f"illegal fixture {info}")
  weights = packed_u16_slice(model, meta, info, device="NV").contiguous().realize()
  records = Tensor(_record(M, K)[0], device="NV").contiguous().realize()
  control_out = Tensor.full((M, N), float("nan"), device="NV").contiguous().realize()
  candidate_out = Tensor.full((M, N), float("nan"), device="NV").contiguous().realize()
  control_args = (_buf(control_out), _buf(records), _buf(weights))
  candidate_args = (_buf(candidate_out), _buf(records), _buf(weights))
  control_samples = _time(control_program, control_args, args.rounds)
  candidate_samples = _time(candidate_program, candidate_args, args.rounds)

  expected, got = control_out.numpy(), candidate_out.numpy()
  diff = np.abs(got-expected)
  control_t, candidate_t = _stats(control_samples), _stats(candidate_samples)
  speedup = (float(control_t["median_us"])-float(candidate_t["median_us"]))/float(control_t["median_us"])
  exact = {"finite": bool(np.isfinite(got).all()), "max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
           "array_equal": bool(np.array_equal(got, expected)),
           "allclose_rtol2e5_atol2e3": bool(np.allclose(got, expected, rtol=2e-5, atol=2e-3))}
  gates = {"exact": exact["array_equal"], "finite": exact["finite"], "faster": speedup > 0.0,
           "no_local_load": candidate_sass["local_load"] == 0, "no_local_store": candidate_sass["local_store"] == 0,
           "zero_stack": candidate_sass["resources"]["stack_bytes"] == 0,
           "imma_preserved": candidate_sass["imma"] >= control_sass["imma"],
           "two_buffers": "buf1[40960]" in candidate_source,
           "one_pipeline_barrier": candidate_source.count("__syncthreads();") == 1}
  result = {"schema": "tinygrad.nv_q6_k256_double_buffer.v1", "shape": {"M":M,"N":N,"K_epoch":256},
            "fixture": {"model":str(model),"weight":info.name,"format":"Q6_K"}, "correctness":exact,
            "timing": {"control":control_t,"candidate":candidate_t,"median_speedup_fraction":speedup},
            "sass": {"control":control_sass,"candidate":candidate_sass}, "gates":gates,
            "passed": all(gates.values())}
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(result, indent=2)+"\n")
  print(json.dumps(result, sort_keys=True))
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
