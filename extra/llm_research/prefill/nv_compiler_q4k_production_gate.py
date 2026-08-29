#!/usr/bin/env python3
"""Real-shape Gate A for the compiler-owned Q4_K/Q8_1 IMMA path."""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, statistics, subprocess, time
from dataclasses import dataclass
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider, Q8ActivationRecordTransform,
  Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry

M, N, K = 512, 12288, 4096


@dataclass(frozen=True)
class _Context:
  schema_version: str
  canonical_identity: str
  geometry: KernelTileGeometry
  packed_weight: PackedWeightTransform
  packed_fragment_provider: Q4KInt8FragmentProvider
  packed_activation: Q8ActivationRecordTransform
  packed_activation_provider: Q8Int8FragmentProvider
  group_accumulator: Q4KQ8GroupAccumulatorContract
  pipeline: None = None


def _weight_carrier(words:Tensor, transform:PackedWeightTransform) -> Tensor:
  blocks, halfwords = transform.rows * transform.blocks_per_row, int(transform.block_bytes)//2
  return words.bitcast(dtypes.uint16).reshape(blocks, halfwords).pad(((0, 0), (0, 128-halfwords))) \
    .reshape(blocks, 128, 1).expand(blocks, 128, 2).reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows, transform.k//2) \
    .reshape(transform.rows, transform.k//2, 1).expand(transform.rows, transform.k//2, 2) \
    .reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _buf(t:Tensor): return t.uop.buffer.get_buf("NV")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds", type=int, default=7)
  ap.add_argument("--tile-k", type=int, default=32, choices=(32,64,128,256))
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  from extra.llm_research.layout import GGML_Q4_K, packed_u32_slice, read_metadata
  from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import SRC as MAIN_SRC, lexical_src
  from extra.llm_research.prefill.nv_q4_imma_provider import PARTIAL_SLOTS, compile_provider
  from tinygrad.runtime.ops_nv import NVProgram
  from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

  model_path = pathlib.Path(args.model)
  metadata = read_metadata(model_path)
  info = next(i for i in metadata.infos if i.name == "blk.0.ffn_gate.weight")
  if info.typ != GGML_Q4_K or tuple(reversed(info.dims)) != (N, K): raise RuntimeError(f"illegal gate fixture {info}")
  words = packed_u32_slice(model_path, metadata, info, device="NV").contiguous().realize()

  # Deterministic, non-degenerate legal Q8_1 packet. Metadata values are all
  # exactly representable in fp16, so compiler and v4 share an exact ABI input.
  q8_np = (((np.arange(M*K, dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M, K)
  group_ids = np.arange(M*(K//32), dtype=np.int64).reshape(M, K//32)
  scales_np = (2.0**((group_ids%7)-5)).astype(np.float32)
  sums_np = q8_np.reshape(M, K//32, 32).astype(np.int32).sum(2).astype(np.float32)
  record_np = np.frombuffer(q8_np.reshape(-1).tobytes()+scales_np.reshape(-1).tobytes()+sums_np.reshape(-1).tobytes(), np.uint32).copy()
  record = Tensor(record_np, device="NV").contiguous().realize()
  q8 = Tensor(q8_np.reshape(-1), device="NV").contiguous().realize()
  scales = Tensor(scales_np.reshape(-1), device="NV").contiguous().realize()
  sums = Tensor(sums_np.reshape(-1), device="NV").contiguous().realize()

  wt, at = PackedWeightTransform("Q4_K", N, K), Q8ActivationRecordTransform(M, K)
  wp, apv = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q4KQ8GroupAccumulatorContract(wp, apv)
  stride = ((args.tile_k + (args.tile_k//16)*4 + 15)//16)*16
  geometry = KernelTileGeometry((128, 128, args.tile_k), (2, 4), 256, 32,
    (KernelLDSWindow("A", 0, 128*stride, stride),
     KernelLDSWindow("B", 128*stride, 256*stride, stride)))
  identity = hashlib.sha256(repr((geometry, wp.identity, apv.identity, accumulator.abi)).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, wt, wp, at, apv, accumulator)
  key = warmstart_key({M, N}, K, wt.storage_dtype)

  @TinyJit
  def generated(record_arg:Tensor, words_arg:Tensor):
    return _activation_carrier(record_arg, at).matmul(_weight_carrier(words_arg, wt).transpose(), dtype=dtypes.int) \
      .cast(dtypes.float).contiguous().realize()

  from tinygrad.codegen import to_program_cache
  to_program_cache.clear()
  generated_times = []
  with warmstart_candidate_state({key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}):
    for iteration in range(args.rounds+3):
      Device["NV"].synchronize(); started = time.perf_counter_ns(); out = generated(record, words); Device["NV"].synchronize()
      if iteration >= 3: generated_times.append((time.perf_counter_ns()-started)/1e3)
    programs = list(to_program_cache.values())
  sources = [u.arg for p in programs for u in p.src if u.op is Ops.SOURCE and isinstance(u.arg, str)]

  # The proven static v4 body is the full-grid oracle and timing comparator.
  static_source = lexical_src(MAIN_SRC).replace("row*K+blk*256", "row*4096+blk*256").replace("K/256", "16") \
    .replace("row*N+col", "row*12288+col")
  static_program = NVProgram(Device["NV"], "q4k_imma_complete",
    NVRTCCompiler(Device["NV"].arch, ptx=False, cache_key="q4k_compiler_gate_static_v4").compile(static_source), shared_mem=57856+1024)
  reference = Tensor.full((M*N,), float("nan"), dtype=dtypes.float32, device="NV").contiguous().realize()
  v4_times = []
  for iteration in range(args.rounds+3):
    elapsed = static_program(_buf(reference), _buf(words), _buf(q8), _buf(scales), _buf(sums), vals=(M,N,K),
                             global_size=(96,4,1), local_size=(256,1,1), wait=True)*1e6
    if iteration >= 3: v4_times.append(elapsed)

  # Promotion threshold uses the actually-qualified v4 Stream-K main+fixup
  # chain, not the slower static full-grid oracle above.
  v4_provider = compile_provider(Device["NV"])
  v4_output = Tensor.empty(M*N, dtype=dtypes.float32, device="NV").realize()
  v4_partials = Tensor.empty(PARTIAL_SLOTS*128*128, dtype=dtypes.float32, device="NV").realize()
  v4_ids = Tensor.empty(PARTIAL_SLOTS, dtype=dtypes.int32, device="NV").realize()
  v4_map = Tensor(v4_provider.slotmap, device="NV").contiguous().realize()
  v4_chain_times = []
  for iteration in range(args.rounds+3):
    main_t, fixup_t = v4_provider.launch(_buf(v4_output), _buf(v4_partials), _buf(v4_ids), _buf(words), _buf(q8), _buf(scales),
                                         _buf(sums), _buf(v4_map), wait=True)
    if iteration >= 3: v4_chain_times.append((main_t+fixup_t)*1e6)

  got, ref = out.numpy().reshape(M,N), reference.numpy().reshape(M,N)
  diff = np.abs(got-ref)
  words_after, record_after = words.numpy(), record.numpy()
  source_path = pathlib.Path(args.out).with_suffix(".cu")
  if sources: source_path.write_text("\n\n".join(sources))
  binaries = [u.arg for p in programs for u in p.src if u.op is Ops.BINARY and isinstance(u.arg, bytes)]
  cubin_path, sass_path = pathlib.Path(args.out).with_suffix(".cubin"), pathlib.Path(args.out).with_suffix(".sass")
  sass = ""
  if binaries:
    cubin_path.write_bytes(binaries[0])
    nvdisasm = pathlib.Path(__file__).resolve().parents[3]/".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
    env = dict(os.environ, NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{os.environ.get('PATH','')}")
    cp = subprocess.run(["/usr/local/cuda-13.2/bin/cuobjdump", "--dump-resource-usage", "--dump-sass", str(cubin_path)],
                        capture_output=True, text=True, env=env)
    sass = cp.stdout+cp.stderr; sass_path.write_text(sass)
  rec = {"schema":"tinygrad.nv_compiler_q4k_production_gate.v1", "shape":{"M":M,"N":N,"K":K},
         "fixture":{"model":str(model_path),"weight":info.name}, "identity":identity,
         "correctness":{"finite":bool(np.isfinite(got).all()), "reference_finite":bool(np.isfinite(ref).all()),
                        "unwritten_sentinels":int(np.isnan(got).sum()), "nonzero":int(np.count_nonzero(got)),
                        "max_abs":float(diff.max()), "mean_abs":float(diff.mean()),
                        "allclose_rtol2e5_atol2e3":bool(np.allclose(got, ref, rtol=2e-5, atol=2e-3))},
         "readonly":{"words":bool(np.array_equal(words_after, packed_u32_slice(model_path, metadata, info, device="CPU").numpy())),
                     "record":bool(np.array_equal(record_after, record_np))},
         "compiler":{"ordinary_matmul":True,"expanded_global_weight_allocation":False,"programs":len(programs),
                     "signed_imma":any("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in s for s in sources),
                     "candidate_identity_exact":[getattr(getattr(p.src[0].arg,"candidate_context",None),"canonical_identity",None)
                                                 for p in programs] == [identity],
                     "source":str(source_path) if sources else None,
                     "sass":{"path":str(sass_path) if sass else None,"imma":sass.count("IMMA.16832.S8.S8"),
                             "bar":sass.count("BAR.SYNC"),"ldsm":sass.count("LDSM"),"local_load":sass.count("LDL"),
                             "local_store":sass.count("STL")}},
         "timing":{"generated_us":{"min":min(generated_times),"median":statistics.median(generated_times),"samples":generated_times},
                   "v4_static_oracle_us":{"min":min(v4_times),"median":statistics.median(v4_times),"samples":v4_times},
                   "v4_qualified_chain_us":{"min":min(v4_chain_times),"median":statistics.median(v4_chain_times),"samples":v4_chain_times},
                   "generated_over_v4_min":min(generated_times)/min(v4_chain_times)}}
  rec["passed_correctness"] = bool(rec["correctness"]["finite"] and rec["correctness"]["unwritten_sentinels"] == 0 and
    rec["correctness"]["allclose_rtol2e5_atol2e3"] and all(rec["readonly"].values()) and rec["compiler"]["signed_imma"] and
    rec["compiler"]["candidate_identity_exact"])
  rec["passed_performance"] = bool(rec["timing"]["generated_over_v4_min"] <= 1.03)
  rec["passed"] = bool(rec["passed_correctness"] and rec["passed_performance"])
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,sort_keys=True))
  if not rec["passed_correctness"]: raise SystemExit(1)


if __name__ == "__main__": main()
