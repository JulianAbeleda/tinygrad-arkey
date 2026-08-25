"""Hermetic tests for the graph-topology motif analyzer.

Builds a synthetic traced body (UOp DAG) that contains the three generic motifs from
docs/what-makes-inference-fast.md section 2A.1 and asserts the analyzer reports them.
The synthetic CALL/AFTER construction mirrors the shape real custom kernels produce
(custom_kernel -> fxn(*placeholders).call(*contig_srcs) -> src.after(kernel)).
No GPU or model is needed.
"""
from __future__ import annotations

import unittest

from tinygrad import Tensor, dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.llm_research.decode.graph_topology import analyze_graph_topology


def _synthetic_body() -> UOp:
  """Build a body with all three motifs:
  - shared input: normed (a MUL) read by gemv_a and gemv_b (two CALL kernels)
  - sole pointwise consumer: gemv_a output (AFTER) read by exactly one MUL
  - immediate reduction: REDUCE over gemv_b output (AFTER)
  """
  dev = "NV"
  x = UOp.new_buffer(dev, 8, dtypes.float16).reshape(1, 1, 8)
  w_a = UOp.new_buffer(dev, 16, dtypes.float16).reshape(8, 2)
  w_b = UOp.new_buffer(dev, 16, dtypes.float16).reshape(8, 2)
  scale = UOp.const(dtypes.float16, 1.0, shape=(1, 1, 8))
  normed = x.alu(Ops.MUL, scale)  # MUL — the shared input

  def gemv(inp: UOp, w: UOp) -> UOp:
    # custom_kernel shape: fxn(*placeholders).call(*contig_srcs) -> AFTER wrapper.
    out = UOp.new_buffer(dev, 2, dtypes.float16).reshape(1, 1, 2)
    kernel = UOp.sink(inp, w).call(out, inp, w)
    return out.after(kernel)

  out_a = gemv(normed, w_a)
  out_b = gemv(normed, w_b)
  post = out_a.alu(Ops.MUL, UOp.const(dtypes.float16, 1.0, shape=(1, 1, 2)))  # sole pointwise consumer of out_a
  red = out_b.reduce(arg=(Ops.ADD, (2,)))  # immediate reduction over gemv_b output
  return UOp.sink(post, red)


class TestGraphTopology(unittest.TestCase):
  def test_shared_input_multi_reduction(self) -> None:
    report = analyze_graph_topology(_synthetic_body())
    self.assertEqual(report["compute_ops"], 3)  # two gemv CALLs + the immediate REDUCE
    self.assertEqual(len(report["shared_inputs"]), 1)
    group = report["shared_inputs"][0]
    self.assertEqual(group["input"], "MUL")
    self.assertEqual(group["count"], 2)
    self.assertEqual(group["reader_kinds"], ["gemv", "gemv"])

  def test_sole_pointwise_consumer(self) -> None:
    report = analyze_graph_topology(_synthetic_body())
    chains = [c for c in report["sole_consumers"] if c["consumer"] == "MUL"]
    self.assertEqual(len(chains), 1)
    self.assertEqual(chains[0]["producer_kind"], "gemv")

  def test_immediate_reduction(self) -> None:
    report = analyze_graph_topology(_synthetic_body())
    self.assertEqual(len(report["immediate_reduces"]), 1)
    self.assertEqual(report["immediate_reduces"][0]["input_from"][0], "CALL")


if __name__ == "__main__":
  unittest.main()
