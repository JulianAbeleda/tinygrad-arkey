"""Census CLI: emit the graph-topology motif report for a real decode block.

Captures one TransformerBlock's traced body (the @function uret DAG, before realization)
and prints the three generic motifs (shared-input multi-reduction, sole pointwise consumer,
immediate reduction). Run on the NV or AMD machine; the analysis itself never consults
backend/architecture strings.

Usage:
  python3 extra/llm_research/decode/graph_topology_census.py --depth 512 --block 0 \
    [--use-flash] [--out X.json]
"""
from __future__ import annotations

import argparse, json, sys

sys.path.insert(0, "/home/ubuntu/tinygrad-arkey")

from tinygrad import Tensor, UOp
from tinygrad.helpers import Context
from tinygrad.llm.model import Transformer
import tinygrad.llm.model as tgm
from tinygrad.uop.ops import Ops

from extra.llm_research.decode.graph_topology import analyze_graph_topology

MODEL = "/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf"


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--block", type=int, default=0)
  ap.add_argument("--use-flash", action="store_true")
  ap.add_argument("--fusions-open", action="store_true")
  ap.add_argument("--kv-open", action="store_true")
  ap.add_argument("--out", type=str, default=None)
  args = ap.parse_args()

  tgm._CUSTOM_KERNEL_PREFILL_ATTN_PROMOTED_TARGETS = frozenset()
  if args.kv_open:
    import tinygrad.llm.model_route_plan as mrp
    mrp.decode_kv_store_fusion_promoted = lambda target: target == ("NV", "sm_120")
    tgm.decode_kv_store_fusion_promoted = mrp.decode_kv_store_fusion_promoted
  if not args.fusions_open:
    import tinygrad.llm.model_route_plan as mrp
    for name in ("decode_norm_fusion_promoted", "decode_epilogue_fusion_promoted",
                 "decode_q4k_epilogue_fusion_promoted", "decode_q4k_w1w3_fusion_promoted",
                 "decode_kv_store_fusion_promoted", "decode_flash_combine_fusion_promoted"):
      setattr(mrp, name, lambda t: False)
      setattr(tgm, name, lambda t: False)

  model, _kv = Transformer.from_gguf(MODEL, 4608)
  prompt = [1] * args.depth
  gen = model.generate(prompt.copy(), chunk_size=32, temperature=0.0)
  with Context(DEBUG=0):
    next(gen)
    next(gen)
  gen.close()

  blk = model.blk[args.block]
  blk._use_flash = args.use_flash
  blk._is_prefill = False
  x = Tensor.empty(1, 1, model.config.dim, device="NV")
  sp = UOp.variable("start_pos", 0, model.config.max_context - 1).bind(args.depth)
  out = blk(x, sp)
  call = out.uop
  while call.op not in (Ops.CALL, Ops.FUNCTION) and call.src:
    call = call.src[0]
  if call.op not in (Ops.CALL, Ops.FUNCTION):
    raise RuntimeError("cannot find function body")
  report = analyze_graph_topology(call.src[0])
  report["label"] = (f"block{args.block} depth={args.depth} flash={args.use_flash} "
                     f"{'OPEN' if args.fusions_open else 'CLOSED'} kv={args.kv_open}")
  print(json.dumps(report, indent=2, default=str))
  if args.out:
    with open(args.out, "w") as f:
      json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
  main()
