#!/usr/bin/env python3
"""Diagnostic census of real CUDA decode graph call/buffer ABIs.

Installs an observational CUDAGraph subclass in-process.  The production graph
constructor and execution remain unchanged; after each graph is built the
probe records program identities, argument shapes/dtypes, read/write slots,
buffer sizes/offsets, and metadata.  Pointer values are intentionally omitted.
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  p.add_argument("--depth", type=int, default=512)
  p.add_argument("--out", required=True)
  args = p.parse_args()

  from tinygrad.device import Device
  from tinygrad.engine.realize import get_call_arg_uops
  from tinygrad.runtime.graph.cuda import CUDAGraph
  from tinygrad.uop.ops import Ops

  groups = []

  class ObservingCUDAGraph(CUDAGraph):
    def __init__(self, linear, input_uops=()):
      super().__init__(linear, input_uops)
      calls = []
      for j, ((_, ast, bufs, device_vars), metadata) in enumerate(zip(self.calls, self.call_metadata)):
        call = self.linear.src[j]
        arg_uops = get_call_arg_uops(call)
        calls.append({"ordinal": j, "op": ast.op.name, "name": ast.arg.name if ast.op is Ops.PROGRAM else ast.op.name,
                      "function_name": ast.arg.function_name if ast.op is Ops.PROGRAM else None,
                      "outs": list(ast.arg.outs) if ast.op is Ops.PROGRAM else [0],
                      "ins": list(ast.arg.ins) if ast.op is Ops.PROGRAM else ([1] if len(bufs) > 1 else []),
                      "metadata": [repr(x) for x in metadata], "device_vars": device_vars,
                      "args": [{"slot": k, "shape": [str(x) for x in u.shape], "dtype": str(u.dtype), "device": str(u.device),
                                "nbytes": b.nbytes, "offset": b.offset, "base_nbytes": b.base.nbytes}
                               for k, (u, b) in enumerate(zip(arg_uops, bufs))]})
      groups.append({"group": len(groups), "calls": calls, "call_count": len(calls)})

  Device["CUDA"].graph = ObservingCUDAGraph
  from tinygrad.helpers import Context
  from tinygrad.llm.model import Transformer
  model, _ = Transformer.from_gguf(args.model, 4608)
  gen = model.generate([1]*args.depth, chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
    # Prime constructs the decode groups; the next replay proves the unchanged
    # production graph remains executable after observation.
    first = int(next(gen)); second = int(next(gen))
  gen.close()

  model_path = pathlib.Path(args.model)
  h = hashlib.sha256()
  with model_path.open("rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
  out = {"schema": "tinygrad.cuda_decode_graph_call_abi.v1", "evidence": "OBSERVATIONAL_PRODUCTION_GRAPH_CONSTRUCTION",
         "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
         "model": {"path": str(model_path), "sha256": h.hexdigest(), "bytes": model_path.stat().st_size},
         "depth": args.depth, "first_tokens": [first, second], "group_count": len(groups),
         "call_count": sum(x["call_count"] for x in groups), "groups": groups}
  pathlib.Path(args.out).write_text(json.dumps(out, indent=2)+"\n")
  print(json.dumps({"group_count": out["group_count"], "call_count": out["call_count"], "first_tokens": out["first_tokens"]}))


if __name__ == "__main__": main()
