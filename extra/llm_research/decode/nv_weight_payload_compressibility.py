#!/usr/bin/env python3
"""Measure offline lossless compressibility of exact GGUF projection payloads."""
from __future__ import annotations

import argparse, collections, json, mmap, pathlib, sys, zlib
ROOT=pathlib.Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from tinygrad.llm.gguf import gguf_load_metadata
from tinygrad.llm.gguf_memory_scan import gguf_tensor_spans
from extra.llm_research.decode.nv_byte_topology_wall_audit import weight_role


def main()->int:
  ap=argparse.ArgumentParser();ap.add_argument("--model",type=pathlib.Path,required=True);ap.add_argument("--level",type=int,default=1)
  ap.add_argument("--out",type=pathlib.Path,required=True);args=ap.parse_args()
  _kv,meta=gguf_load_metadata(args.model);spans=gguf_tensor_spans(meta,args.model.stat().st_size)
  wanted={"vocab_output","attn_q","attn_k","attn_v","attn_o","ffn_gate","ffn_up","ffn_down"}
  roles=collections.defaultdict(lambda:{"payload_bytes":0,"zlib_bytes":0,"tensors":0})
  with args.model.open("rb") as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mm:
    for span in spans:
      role=weight_role(span.name)
      if role not in wanted or span.payload_bytes is None:continue
      raw=mm[span.absolute_offset:span.absolute_offset+span.payload_bytes];packed=zlib.compress(raw,args.level)
      row=roles[role];row["payload_bytes"]+=len(raw);row["zlib_bytes"]+=len(packed);row["tensors"]+=1
  total_raw=sum(x["payload_bytes"] for x in roles.values());total_z=sum(x["zlib_bytes"] for x in roles.values())
  result={"schema":"tinygrad.nv_weight_payload_compressibility.v1","codec":"zlib","level":args.level,
    "warning":"offline feasibility diagnostic; no optimal-codec, GPU-decodable representation, or runtime speed claim",
    "payload_bytes":total_raw,"compressed_bytes":total_z,"saving_bytes":total_raw-total_z,"saving_pct":100*(1-total_z/total_raw),
    "roles":{k:{**v,"saving_pct":100*(1-v["zlib_bytes"]/v["payload_bytes"])} for k,v in sorted(roles.items())}}
  args.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result,indent=2,sort_keys=True));return 0


if __name__=="__main__":raise SystemExit(main())
