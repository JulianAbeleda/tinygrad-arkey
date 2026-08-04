"""Graph-topology analysis over a traced LLM UOp body.

Implements the three generic motifs from docs/what-makes-a-token-fast-20260731.md section 2A.1
as pure graph facts over the semantic model graph (the traced @function body, before
kernelization):

  1. shared-input multi-reduction   — one tensor read by >=2 GEMV/reduce ops
  2. producer + sole pointwise consumer — a GEMV/reduce output read by exactly one elementwise op
  3. indexed producers + immediate reduction — a REDUCE consuming a produced compute op directly

The analysis is target-agnostic: it never consults backend/architecture strings or route roles.
It is also kernelization-agnostic: the same body is analyzed whether or not fusions are open,
so a landed fusion (e.g. w1+w3, kv-store) removes the motif from the report.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from tinygrad.uop.ops import Ops, UOp

# Pure layout/view wrappers: transparent when unwrapping producer/consumer chains.
VIEW_OPS = frozenset({
  Ops.RESHAPE, Ops.CAST, Ops.CONTIGUOUS, Ops.MEMORY_SEMANTIC, Ops.SLICE, Ops.SHRINK,
  Ops.EXPAND, Ops.PAD, Ops.PERMUTE, Ops.AFTER, Ops.INDEX, Ops.BIND,
  Ops.FLIP, Ops.STACK, Ops.VCAT,
})

# Ops that do real arithmetic across a reduction or matrix unit: "reduce-class".
REDUCE_OPS = frozenset({Ops.REDUCE, Ops.WMMA, Ops.SHAPED_WMMA, Ops.CONTRACT, Ops.SCOPED_REDUCE})

# Elementwise arithmetic ops.
ELEM_OPS = frozenset({
  Ops.ADD, Ops.MUL, Ops.SUB, Ops.EXP2, Ops.MAX, Ops.CMPLT, Ops.CMPEQ, Ops.NEG, Ops.SQRT,
  Ops.LOG2, Ops.WHERE, Ops.POW, Ops.FDIV, Ops.RECIPROCAL, Ops.TRUNC, Ops.SIN, Ops.SHR, Ops.SHL,
})


def op_kind(u: UOp) -> str:
  """Classify a UOp into gemv (custom kernel program), reduce, elem, sink, leaf, other."""
  if u.op is Ops.CALL:
    return "gemv"
  if u.op in REDUCE_OPS:
    return "reduce"
  if u.op in ELEM_OPS:
    return "elem"
  if u.op is Ops.SINK:
    return "sink"
  if u.op in (Ops.BUFFER, Ops.PARAM, Ops.UNIQUE, Ops.DEVICE, Ops.CONST, Ops.DEFINE_VAR, Ops.RANGE,
              Ops.LOAD, Ops.STORE, Ops.INDEX, Ops.SPECIAL, Ops.IF, Ops.BARRIER, Ops.SOURCE, Ops.INS):
    return "leaf"
  return "other"


def is_compute(u: UOp) -> bool:
  return op_kind(u) in ("gemv", "reduce")


def is_elem(u: UOp) -> bool:
  return op_kind(u) == "elem"


def unwrap_producer(u: UOp) -> UOp:
  """Follow the view chain to the producing compute op (or leaf)."""
  if u.op is Ops.AFTER:
    # AFTER(kernel, buffer): the kernel call is among the srcs.
    for s in u.src:
      if s.op in (Ops.CALL, Ops.FUNCTION):
        return s
    u = u.src[0]
  while u.op in VIEW_OPS and len(u.src):
    u = u.src[0]
  return u


def base_inputs(compute: UOp) -> list[UOp]:
  """The compute op's non-view inputs, unwrapped to their producing ops."""
  srcs = list(compute.src)
  if compute.op is Ops.CALL:
    # CALL(src=(sink_body, out_buf, x, w, ...)) — drop the sink and output buffer.
    srcs = srcs[2:]
  out: list[UOp] = []
  for s in srcs:
    p = unwrap_producer(s)
    if p not in out:
      out.append(p)
  return out


def _shape_str(u: UOp) -> str:
  try:
    return f"{list(u.shape)}"
  except RuntimeError:
    return ""


def analyze_graph_topology(body: UOp) -> dict[str, Any]:
  """Detect the three generic motifs in a traced UOp body.

  Returns a dict with:
    compute_ops      — count of GEMV/reduce ops in the body
    shared_inputs    — motif 1 groups (input op, shape, reader kinds, count)
    sole_consumers   — motif 2 chains (producer kind, consumer op)
    immediate_reduces — motif 3 (reduce over a produced compute op)
  """
  nodes = body.toposort()
  consumers: dict[UOp, list[UOp]] = defaultdict(list)
  for u in nodes:
    for s in u.src:
      consumers[s].append(u)

  compute_ops = [u for u in nodes if is_compute(u)]
  names = {u: f"{op_kind(u)}_{i}" for i, u in enumerate(compute_ops)}

  # Motif 1: shared-input multi-reduction.
  shared: dict[UOp, list[UOp]] = defaultdict(list)
  for c in compute_ops:
    for p in base_inputs(c):
      if p.op is not Ops.BUFFER and not is_compute(p) and p.op not in VIEW_OPS and p.op is not Ops.CONST:
        shared[p].append(c)
  shared_inputs = []
  for p, cs in shared.items():
    if len(cs) >= 2:
      shared_inputs.append({
        "input": p.op.name,
        "input_shape": _shape_str(p),
        "readers": [names[c] for c in cs],
        "reader_kinds": [op_kind(c) for c in cs],
        "count": len(cs),
      })

  # Motif 2: producer + sole pointwise consumer.
  sole_consumers = []
  for u in compute_ops:
    # For CALLs the produced value is the AFTER wrapper; for reduces it is the uop itself.
    produced: list[UOp] = []
    if u.op is Ops.CALL:
      produced = [v for v in nodes if v.op is Ops.AFTER and u in v.src]
    else:
      produced = [u]
    if len(produced) != 1:
      continue
    cons = [x for x in consumers.get(produced[0], []) if x is not u]
    unwrapped = [unwrap_producer(c) for c in cons]
    elem_cons = [c for c in unwrapped if is_elem(c) and c is not u]
    if len(elem_cons) == 1 and len(unwrapped) == 1:
      sole_consumers.append({
        "producer": names.get(u, u.op.name),
        "producer_kind": op_kind(u),
        "consumer": elem_cons[0].op.name,
      })

  # Motif 3: indexed producers + immediate reduction.
  immediate_reduces = []
  for u in nodes:
    if u.op is not Ops.REDUCE:
      continue
    srcs = [unwrap_producer(s) for s in u.src]
    if any(is_compute(s) for s in srcs if s.op is not Ops.BUFFER):
      immediate_reduces.append({
        "reduce_shape": _shape_str(u),
        "input_from": [f"{s.op.name}{_shape_str(s)}" for s in srcs],
      })

  return {
    "compute_ops": len(compute_ops),
    "shared_inputs": shared_inputs,
    "sole_consumers": sole_consumers,
    "immediate_reduces": immediate_reduces,
  }
