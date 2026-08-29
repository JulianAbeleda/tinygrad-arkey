#!/usr/bin/env python3
"""Tiny exact hardware gate for compiler-owned Q4_K/Q8_1 K32 correction."""
from __future__ import annotations

import argparse, hashlib, json, pathlib
from dataclasses import dataclass
import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider, Q8ActivationRecordTransform,
  Q8Int8FragmentProvider, Q4KQ8GroupAccumulatorContract)
from tinygrad.codegen.opt.postrange import warmstart_candidate_state, warmstart_key
from tinygrad.uop.ops import Ops
from extra.llm_research.kernel_vocabulary import KernelLDSWindow, KernelTileGeometry
from extra.llm_research.prefill.q4k_q8_imma_oracle import adversarial_fixture, q4k_q8_block, unpack_scales

M, N, K = 32, 16, 256


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
  # Movement-only carrier with the same expand grammar as the proven packed-B
  # path. The provider replaces every dummy byte, while this spelling keeps K
  # as one logical owner through pm_split_ranges.
  return record.bitcast(dtypes.uint16)[:transform.values_bytes//2].reshape(transform.rows, transform.k//2) \
    .reshape(transform.rows, transform.k//2, 1).expand(transform.rows, transform.k//2, 2) \
    .reshape(transform.rows, transform.k).bitcast(dtypes.float16).cast(dtypes.int8)


def _pack_meta(sc:np.ndarray, mn:np.ndarray) -> bytes:
  meta = np.empty(12, np.uint8)
  meta[:4] = (sc[:4]&63) | ((sc[4:]>>4)<<6)
  meta[4:8] = (mn[:4]&63) | ((mn[4:]>>4)<<6)
  meta[8:12] = (sc[4:]&15) | ((mn[4:]&15)<<4)
  return meta.tobytes()


def _fixture():
  dm0, meta0, payload0, q80, scales0, _ = adversarial_fixture()
  sc0, mn0 = unpack_scales(meta0)
  weights = np.empty((N, 144), np.uint8)
  weight_parts = []
  for n in range(N):
    sc = np.roll(sc0, n%8).astype(np.uint8)
    mn = np.roll(mn0, (3*n)%8).astype(np.uint8)
    dm = (np.float16(float(dm0[0])*(1+n%4)), np.float16(float(dm0[1])*(1+(n//4)%4)))
    payload = np.roll(np.frombuffer(payload0, np.uint8), n*3).copy()
    weights[n, :4] = np.asarray(dm, np.float16).view(np.uint8)
    weights[n, 4:16] = np.frombuffer(_pack_meta(sc, mn), np.uint8)
    weights[n, 16:] = payload
    weight_parts.append(((np.float32(dm[0]), np.float32(dm[1])), weights[n, 4:16].tobytes(), payload.tobytes()))

  q8 = np.stack([np.roll(q80, m*5) for m in range(M)]).astype(np.int8)
  scales = np.stack([np.roll(scales0, m%8) for m in range(M)]).astype(np.float32)
  sums = q8.reshape(M, K//32, 32).astype(np.int32).sum(2).astype(np.float32)
  record_bytes = q8.view(np.uint8).reshape(-1).tobytes() + scales.reshape(-1).tobytes() + sums.reshape(-1).tobytes()
  record = np.frombuffer(record_bytes, np.uint32).copy()
  reference = np.empty((M, N), np.float32)
  for m in range(M):
    for n in range(N): reference[m, n] = q4k_q8_block(*weight_parts[n], q8[m], scales[m], sums[m])
  return weights.view(np.uint32).reshape(-1).copy(), record, reference


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--out", required=True)
  ap.add_argument("--tile-k", type=int, default=32, choices=(32,64,128,256))
  args = ap.parse_args()
  weight_transform = PackedWeightTransform("Q4_K", N, K)
  activation_transform = Q8ActivationRecordTransform(M, K)
  weight_provider, activation_provider = Q4KInt8FragmentProvider(weight_transform), Q8Int8FragmentProvider(activation_transform)
  accumulator = Q4KQ8GroupAccumulatorContract(weight_provider, activation_provider)
  stride = ((args.tile_k + (args.tile_k//16)*4 + 15)//16)*16
  geometry = KernelTileGeometry((M, N, args.tile_k), (1, 1), 32, 32,
    (KernelLDSWindow("A", 0, M*stride, stride),
     KernelLDSWindow("B", M*stride, M*stride+N*stride, stride)))
  identity = hashlib.sha256(repr((geometry, weight_provider.identity, activation_provider.identity, accumulator.abi)).encode()).hexdigest()
  context = _Context("boltbeam.full_kernel_candidate.v1", identity, geometry, weight_transform, weight_provider,
                     activation_transform, activation_provider, accumulator)
  weights_np, record_np, reference = _fixture()
  weights = Tensor(weights_np, device="NV").contiguous().realize()
  record = Tensor(record_np, device="NV").contiguous().realize()
  key = warmstart_key({M, N}, K, weight_transform.storage_dtype)
  from tinygrad.codegen import to_program_cache
  to_program_cache.clear()
  with warmstart_candidate_state({key:(Opt(OptOps.TC, 0, (-1, 2, 1)),)}, {key:context}):
    out = _activation_carrier(record, activation_transform).matmul(_weight_carrier(weights, weight_transform).transpose(),
                                                                    dtype=dtypes.int).cast(dtypes.float).contiguous()
    out.realize()
    programs = list(to_program_cache.values())
  got = out.numpy()
  sources = [u.arg for p in programs for u in p.src if u.op is Ops.SOURCE and isinstance(u.arg, str)]
  source_path = pathlib.Path(args.out).with_suffix(".cu")
  if sources: source_path.write_text("\n\n".join(sources))
  rec = {"schema":"tinygrad.nv_compiler_q4k_group_accumulator_gate.v1", "shape":{"M":M,"N":N,"K":K},
         "identity":identity, "ordinary_matmul":True, "expanded_global_weight_allocation":False,
         "correctness":{"finite":bool(np.isfinite(got).all()), "max_abs":float(np.max(np.abs(got-reference))),
                        "max_rel":float(np.max(np.abs(got-reference)/np.maximum(np.abs(reference), 1e-20))),
                        "nonzero":int(np.count_nonzero(got)), "sentinel_distinct":bool(np.unique(got).size > 32)},
         "source":{"programs":len(programs),
                   "signed_imma":any("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32" in s for s in sources),
                   "candidate_identity_exact":[getattr(getattr(p.src[0].arg, "candidate_context", None), "canonical_identity", None)
                                               for p in programs] == [identity]}}
  rec["passed"] = bool(rec["correctness"]["finite"] and rec["correctness"]["max_abs"] <= 1e-5 and
                       rec["correctness"]["sentinel_distinct"] and rec["source"]["signed_imma"] and
                       rec["source"]["candidate_identity_exact"])
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(rec, indent=2)+"\n")
  print(json.dumps(rec, sort_keys=True))
  if not rec["passed"]: raise SystemExit(1)


if __name__ == "__main__": main()
