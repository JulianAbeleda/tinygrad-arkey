#!/usr/bin/env python3
"""Isolated occupancy-safe Gate A qualification for pp512 Q4_K K projections.

This does not change model routing.  It exercises the ordinary compiler-owned
packed matmul contract at (M,N,K)=(512,1024,4096), compares every output with
the qualified static Q4_K/Q8_1 oracle, and measures the executed compiler
binary directly so Python scheduling time is not mistaken for kernel service.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess
from dataclasses import dataclass
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider,
  Q8ActivationRecordTransform, Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.nv_compiler_q4k_production_gate import (_Context, _activation_carrier,
  _weight_carrier, _buf)

M, N, K, TILE_K = 512, 1024, 4096, 64
GEOMETRIES = (
  ("underfilled_128x128", 128, 128, 2, 4, 256),
  ("underfilled_64x64_2warp", 64, 64, 1, 2, 64),
  ("underfilled_64x64_4warp", 64, 64, 2, 2, 128),
  ("safe_32x64_2warp", 32, 64, 1, 2, 64),
  ("safe_64x32_2warp", 64, 32, 2, 1, 64),
  ("safe_64x32_4warp", 64, 32, 2, 2, 128),
  ("safe_32x32_1warp", 32, 32, 1, 1, 32),
)


def _context(tm:int, tn:int, wm:int, wn:int, threads:int):
  wt, at = PackedWeightTransform("Q4_K", N, K), Q8ActivationRecordTransform(M, K)
  wp, ap = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q4KQ8GroupAccumulatorContract(wp, ap)
  stride = TILE_K + (TILE_K//16)*4
  geometry = KernelTileGeometry((tm, tn, TILE_K), (wm, wn), threads, 32,
    (KernelLDSWindow("A", 0, tm*stride, stride),
     KernelLDSWindow("B", tm*stride, (tm+tn)*stride, stride)))
  identity = hashlib.sha256(repr((geometry, wp.identity, ap.identity, accumulator.abi)).encode()).hexdigest()
  return wt, at, geometry, identity, _Context("boltbeam.full_kernel_candidate.v1", identity, geometry,
    wt, wp, at, ap, accumulator)


def _stats(xs:list[float]) -> dict[str, object]:
  return {"samples_us":xs, "first_us":xs[0], "min_us":min(xs), "median_us":statistics.median(xs), "max_us":max(xs)}


def _sass(binary:bytes, stem:pathlib.Path) -> tuple[str, dict[str, object]]:
  cubin, sass_path = stem.with_suffix(".cubin"), stem.with_suffix(".sass")
  cubin.write_bytes(binary)
  nvdisasm = pathlib.Path(__file__).resolve().parents[3]/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env = dict(os.environ, NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
  cp = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-resource-usage", "--dump-sass", str(cubin)],
                      capture_output=True, text=True, env=env)
  text = cp.stdout+cp.stderr
  sass_path.write_text(text)
  return text, {"cubin":str(cubin), "sass":str(sass_path), "cuobjdump_returncode":cp.returncode,
                "imma_16832_s8":text.count("IMMA.16832.S8.S8"), "bar_sync":text.count("BAR.SYNC"),
                "ldsm":text.count("LDSM"), "local_load":text.count("LDL"), "local_store":text.count("STL")}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--weight", default="blk.0.attn_k.weight")
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--out", required=True)
  ap.add_argument("--artifacts", required=True)
  args = ap.parse_args()
  if args.rounds < 9: raise ValueError("qualification requires R9 or greater")

  from extra.llm_research.layout import GGML_Q4_K, packed_u32_slice, read_metadata
  from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC, lexical_src
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

  model_path, artifacts = pathlib.Path(args.model), pathlib.Path(args.artifacts)
  artifacts.mkdir(parents=True, exist_ok=True)
  metadata = read_metadata(model_path)
  info = next(i for i in metadata.infos if i.name == args.weight)
  if info.typ != GGML_Q4_K or tuple(reversed(info.dims)) != (N, K): raise RuntimeError(f"illegal Q4 K/V fixture {info}")
  words = packed_u32_slice(model_path, metadata, info, device="NV").contiguous().realize()
  words_before = words.numpy().copy()

  q8_np = (((np.arange(M*K, dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M, K)
  groups = np.arange(M*(K//32), dtype=np.int64).reshape(M, K//32)
  scales_np = (2.0**((groups%7)-5)).astype(np.float32)
  sums_np = q8_np.reshape(M, K//32, 32).astype(np.int32).sum(2).astype(np.float32)
  record_np = np.frombuffer(q8_np.reshape(-1).tobytes()+scales_np.reshape(-1).tobytes()+sums_np.reshape(-1).tobytes(),
                            np.uint32).copy()
  record = Tensor(record_np, device="NV").contiguous().realize()
  q8 = Tensor(q8_np.reshape(-1), device="NV").contiguous().realize()
  scales = Tensor(scales_np.reshape(-1), device="NV").contiguous().realize()
  sums = Tensor(sums_np.reshape(-1), device="NV").contiguous().realize()

  # The static body is a correctness authority only.  Its 32-CTA grid is
  # explicitly inadmissible as a production-performance claim on 170 SMs.
  static_binary = NVRTCCompiler(Device["NV"].arch, ptx=False, cache_key="q4k_k_role_static_oracle").compile(lexical_src(SRC))
  static_program = NVProgram(Device["NV"], "q4k_imma_complete", static_binary, shared_mem=57856+1024)
  reference = Tensor.full((M*N,), float("nan"), dtype=dtypes.float32, device="NV").contiguous().realize()
  static_times = []
  for _ in range(args.rounds):
    static_times.append(static_program(_buf(reference), _buf(words), _buf(q8), _buf(scales), _buf(sums), vals=(M,N,K),
      global_size=(N//128,M//128,1), local_size=(256,1,1), wait=True)*1e6)
  ref = reference.numpy().reshape(M,N)
  if not np.isfinite(ref).all(): raise RuntimeError("static oracle left non-finite outputs")

  from tinygrad.codegen import to_program_cache
  arms = {}
  for name, tm, tn, wm, wn, threads in GEOMETRIES:
    wt, at, geometry, identity, context = _context(tm,tn,wm,wn,threads)
    key = warmstart_key({M,N}, K, wt.storage_dtype)
    to_program_cache.clear()
    with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)}, {key:context}):
      output = _activation_carrier(record,at).matmul(_weight_carrier(words,wt).transpose(), dtype=dtypes.int) \
        .cast(dtypes.float).contiguous()
      output.realize()
    programs = list(to_program_cache.values())
    if len(programs) != 1: raise RuntimeError(f"{name}: expected one program, got {len(programs)}")
    program, got = programs[0], output.numpy()
    pinfo = program.arg
    sources = [u.arg for u in program.src if u.op is Ops.SOURCE and isinstance(u.arg,str)]
    binaries = [u.arg for u in program.src if u.op is Ops.BINARY and isinstance(u.arg,bytes)]
    if len(sources) != 1 or len(binaries) != 1: raise RuntimeError(f"{name}: missing source or binary")
    source_path = artifacts/f"{name}.cu"; source_path.write_text(sources[0])
    sass, sass_rec = _sass(binaries[0], artifacts/name)

    # Re-launch the exact compiler binary into a NaN-poisoned buffer.  This is
    # both the sentinel test and the low-overhead event-timing authority.
    poison = Tensor.full((M,N), float("nan"), dtype=dtypes.float32, device="NV").contiguous().realize()
    native = NVProgram(Device["NV"], pinfo.name, binaries[0])
    direct_times = [native(_buf(poison), _buf(record), _buf(words), global_size=pinfo.global_size,
                           local_size=pinfo.local_size, wait=True)*1e6 for _ in range(args.rounds)]
    direct = poison.numpy()
    diff = np.abs(direct-ref)
    ctas = int(np.prod(pinfo.global_size))
    exact_contract = (getattr(pinfo.candidate_context,"canonical_identity",None) == identity and
                      pinfo.candidate_context.geometry.tile[2] == TILE_K)
    arm = {"geometry":{"tile":[tm,tn,TILE_K],"waves":[wm,wn],"threads":threads,
                              "global_size":list(pinfo.global_size),"local_size":list(pinfo.local_size),"ctas":ctas},
           "identity":identity, "compiler":{"programs":1,"ordinary_matmul":True,"expanded_global_weight_allocation":False,
             "candidate_identity_exact":exact_contract,"source":str(source_path),"signed_imma_ptx":
             "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in sources[0],"sass":sass_rec},
           "correctness":{"finite":bool(np.isfinite(direct).all()),"unwritten_sentinels":int(np.isnan(direct).sum()),
             "nonzero":int(np.count_nonzero(direct)),"max_abs":float(diff.max()),"mean_abs":float(diff.mean()),
             "allclose_rtol2e5_atol2e3":bool(np.allclose(direct,ref,rtol=2e-5,atol=2e-3)),
             "tensor_path_matches_direct":bool(np.array_equal(got,direct))},
           "timing":_stats(direct_times), "occupancy_safe":ctas >= 170,
           "production_claim_admissible":ctas >= 170}
    arm["passed"] = bool(arm["occupancy_safe"] and exact_contract and arm["compiler"]["signed_imma_ptx"] and
      sass_rec["imma_16832_s8"] > 0 and arm["correctness"]["finite"] and arm["correctness"]["unwritten_sentinels"] == 0 and
      arm["correctness"]["allclose_rtol2e5_atol2e3"] and arm["correctness"]["tensor_path_matches_direct"])
    arms[name] = arm

  readonly = {"words":bool(np.array_equal(words.numpy(),words_before)), "record":bool(np.array_equal(record.numpy(),record_np))}
  safe = [a for a in arms.values() if a["passed"]]
  winner = min(safe, key=lambda a:a["timing"]["min_us"]) if safe else None
  rec = {"schema":"tinygrad.nv_compiler_q4k_k_role_gate.v1","shape":{"M":M,"N":N,"K":K},
         "fixture":{"model":str(model_path),"weight":info.name,"format":"Q4_K"},
         "static_oracle":{"topology":"32 CTAs, correctness-only, performance claim rejected","timing":_stats(static_times)},
         "readonly":readonly,"arms":arms,
         "winner":next((name for name,a in arms.items() if a is winner),None),
         "passed":bool(winner is not None and all(readonly.values()))}
  out_path = pathlib.Path(args.out); out_path.parent.mkdir(parents=True,exist_ok=True); out_path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,sort_keys=True))
  if not rec["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
