#!/usr/bin/env python3
"""CPU-only census of the real Qwen attention quant topology and shared-Q8 ABI.

Reads GGUF metadata only: it never constructs a model, compiler, or GPU device.
"""
from __future__ import annotations

import argparse, json
from collections import Counter

from tinygrad.llm.gguf import gguf_load_metadata

GGML_TYPE = {12:"Q4_K", 14:"Q6_K"}


def census(model:str) -> dict:
  kv, metadata=gguf_load_metadata(model)
  infos={x[0]:x for x in metadata["tensor_infos"]}
  block_count=int(kv["qwen3.block_count"])
  rows=[]
  for block in range(block_count):
    kinds=[]
    for role in ("q","k","v"):
      name=f"blk.{block}.attn_{role}.weight"
      if name not in infos: raise RuntimeError(f"missing {name}")
      kind=GGML_TYPE.get(infos[name][2],f"GGML_TYPE_{infos[name][2]}")
      kinds.append(kind)
    rows.append({"block":block,"q":kinds[0],"k":kinds[1],"v":kinds[2]})
  topologies=Counter(f"{r['q']}/{r['k']}/{r['v']}" for r in rows)
  supported={"Q4_K/Q4_K/Q4_K","Q4_K/Q4_K/Q6_K"}
  return {"schema":"tinygrad.shared_q8_real_topology_census.v1", "model":model,
    "block_count":block_count,"topologies":dict(sorted(topologies.items())),"blocks":rows,
    "shared_q8_abi":{"provider":"Q8_1 uint32[1152] (1024 int8x4 packets + 128 d|s metadata)",
      "q4_consumer":"_emit_q4 reads the common packet ABI","q6_consumer":"_emit_q6 reads the common packet ABI",
      "supported_topologies":sorted(supported),"all_blocks_supported":set(topologies).issubset(supported),
      "providers_per_block":1,"copies_between_provider_and_consumers":0},
    "evidence":"CPU metadata/static ABI census; no correctness or performance claim"}


if __name__ == "__main__":
  ap=argparse.ArgumentParser(); ap.add_argument("--model",default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--out"); args=ap.parse_args()
  payload=census(args.model); rendered=json.dumps(payload,indent=2,sort_keys=True)+"\n"
  if args.out:
    from pathlib import Path
    Path(args.out).write_text(rendered)
  else: print(rendered,end="")
