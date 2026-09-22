#!/usr/bin/env python3
"""Captured 72-real-weight proxy for the compiler-generated Gate-A path."""
from __future__ import annotations

import argparse, hashlib, json, statistics, time
from dataclasses import dataclass
import numpy as np

from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider, Q8ActivationRecordTransform,
  Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops, UOp
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry

M, N, K, TILE_K = 512, 12288, 4096, 64


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
  blocks, halfwords = transform.rows*transform.blocks_per_row, int(transform.block_bytes)//2
  return words.bitcast(dtypes.uint16).reshape(blocks,halfwords).pad(((0,0),(0,128-halfwords))) \
    .reshape(blocks,128,1).expand(blocks,128,2).reshape(transform.rows,transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _activation_carrier(record:Tensor, transform:Q8ActivationRecordTransform) -> Tensor:
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows,transform.k//2) \
    .reshape(transform.rows,transform.k//2,1).expand(transform.rows,transform.k//2,2) \
    .reshape(transform.rows,transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds", type=int, default=7)
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  from tinygrad.llm.generate import load_model_and_tokenizer

  model, _ = load_model_and_tokenizer(args.model, 4608, seed=20260617)
  weights = [p.prefill_packed_weight().contiguous().realize() for block in model.blk for p in (block.ffn_gate, block.ffn_up)]
  expected_words = N*(K//256)*36
  if len(weights) != 72 or any(w.dtype != dtypes.uint32 or w.numel() != expected_words for w in weights):
    raise RuntimeError("proxy requires 72 canonical real Qwen3-8B gate/up Q4_K buffers")

  q8_np = (((np.arange(M*K,dtype=np.int64)*37+11)%255)-127).astype(np.int8).reshape(M,K)
  groups = np.arange(M*(K//32),dtype=np.int64).reshape(M,K//32)
  scales_np = (2.0**((groups%7)-5)).astype(np.float32)
  sums_np = q8_np.reshape(M,K//32,32).astype(np.int32).sum(2).astype(np.float32)
  record_np = np.frombuffer(q8_np.reshape(-1).tobytes()+scales_np.reshape(-1).tobytes()+sums_np.reshape(-1).tobytes(),np.uint32).copy()
  record = Tensor(record_np,device="NV").contiguous().realize()

  wt, at = PackedWeightTransform("Q4_K",N,K), Q8ActivationRecordTransform(M,K)
  wp, apv = Q4KInt8FragmentProvider(wt), Q8Int8FragmentProvider(at)
  accumulator = Q4KQ8GroupAccumulatorContract(wp,apv)
  stride = 80
  geometry = KernelTileGeometry((128,128,TILE_K),(2,4),256,32,
    (KernelLDSWindow("A",0,128*stride,stride),KernelLDSWindow("B",128*stride,256*stride,stride)))
  identity = hashlib.sha256(repr((geometry,wp.identity,apv.identity,accumulator.abi)).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1",identity,geometry,wt,wp,at,apv,accumulator)
  key = warmstart_key({M,N},K,wt.storage_dtype)

  @TinyJit
  def batch(record_arg:Tensor):
    outputs = []
    for words in weights:
      outputs.append(_activation_carrier(record_arg,at).matmul(_weight_carrier(words,wt).transpose(),dtype=dtypes.int)
                     .cast(dtypes.float).contiguous().realize())
    return tuple(outputs)

  times = []
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}):
    for iteration in range(args.rounds+3):
      Device["NV"].synchronize(); started=time.perf_counter_ns(); outputs=batch(record); Device["NV"].synchronize()
      if iteration >= 3: times.append((time.perf_counter_ns()-started)/1e6)
  sample_indices=np.asarray([0,127,128,16383,M*N-1])
  before=[outputs[i].numpy().reshape(-1)[sample_indices].copy() for i in (0,35,71)]
  with warmstart_candidate_state({key:(Opt(OptOps.TC,0,(-1,2,1)),)},{key:context}): outputs=batch(record)
  Device["NV"].synchronize()
  after=[outputs[i].numpy().reshape(-1)[sample_indices] for i in (0,35,71)]
  replay_exact=all(np.array_equal(x,y) for x,y in zip(before,after))
  sample_distinct=all(not np.array_equal(after[i],after[j]) for i in range(3) for j in range(i+1,3))
  def _call_count(linear:UOp) -> int:
    total=0
    for call in linear.src:
      program=call.src[0]
      total += _call_count(program.src[0]) if program.op is Ops.CUSTOM_FUNCTION and program.arg == "graph" else 1
    return total
  captured_calls=_call_count(batch.captured.linear) if batch.captured is not None else 0
  hot_min=min(times); per_call_us=hot_min*1000/72
  # Reverse-order matched graph-wall run of nv_q4_imma_72main_proxy.py.
  v4_in_model_us=513.9353193549646
  rec={"schema":"tinygrad.nv_compiler_q4k_72real_proxy.v1","shape":{"M":M,"N":N,"K":K,"tile_k":TILE_K},
       "real_weight_buffers":len(weights),"captured_calls":captured_calls,"identity":identity,"sample_replay_exact":replay_exact,
       "sample_distinct_real_weights":sample_distinct,
       "sample_finite":bool(all(np.isfinite(x).all() for x in after)),
       "timing":{"r_ms":times,"wall_min_ms":hot_min,"wall_median_ms":statistics.median(times),"per_call_wall_us":per_call_us,
                 "qualified_v4_in_model_per_call_us":v4_in_model_us,"generated_over_v4":per_call_us/v4_in_model_us},
       "v4_authority":"proxy_72real_v4_matched.json reverse-order captured graph wall"}
  rec["passed"]=bool(replay_exact and sample_distinct and captured_calls == 72 and rec["sample_finite"] and per_call_us <= v4_in_model_us)
  import pathlib
  path=pathlib.Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rec,indent=2)+"\n")
  print(json.dumps(rec,sort_keys=True))
  if not rec["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
