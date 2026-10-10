"""Decode fusions as a search space: one launch that replaces the kernels of several adjacent decode-graph nodes.

A fusion family is a registered emitter (semantic_kernel_lowering.build_registered_llm_emitter) that computes the
output of a connected set of decode-graph nodes in one kernel. Nodes are named in the caller's decode-graph
vocabulary: a weight node by its tensor role (ffn_gate, ffn_up, ffn_down, attention_o, attention_v), the others by
what they compute (silu_mul, ffn_residual). The caller judges adjacency with its own graph and its own reference;
this module only says, per family:

  covers     the nodes the one launch replaces
  operands   what the kernel reads after its output, in parameter order: "weight:<node>" (the node's packed GGUF
             tensor) or "value:<name>" (a vector: a graph input such as ffn_norm, or another node's output)
  rows       the coupled schedule rows (one per emitter parameter combination), in decode_emitter_space's spelling
  validate   whether the emitter's own validate accepts a shape and a row (the emitter is constructed, not compiled)

and builds the fused program and each covered node's program alone (a weight node through its single-role decode
emitter at the row the model's decode installs, an elementwise node through tinygrad's scheduler), on the recorded
tensors' packed bytes. Nothing here names a backend, a chip or a model shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from extra.llm_research import decode_emitter_space as single

EMITTER_PLAN_KIND = single.EMITTER_PLAN_KIND
BLOCK_BYTES = {"Q4_K": 144, "Q6_K": 210}
# what an elementwise node computes, as tinygrad Tensor ops on its value operands (in the order the node reads them)
ELEMENTWISE: dict[str, Callable[..., Any]] = {
  "silu_mul": lambda a, b: a.silu() * b,
  "add": lambda a, b: a + b,
}


@dataclass(frozen=True)
class Node:
  name: str
  op: str  # "gemv" or an ELEMENTWISE key
  reads: tuple[str, ...]  # the values it reads: "weight:<node>"/"value:<name>" for a gemv, "value:<name>" otherwise


@dataclass(frozen=True)
class FusionFamily:
  family: str  # the registry family (semantic_kernel_lowering.build_registered_llm_emitter)
  covers: tuple[Node, ...]  # the nodes the one launch replaces, in dataflow order; the last one's value is the output
  quant: str
  storage: str  # the dtype every weight operand's GGUF bytes are viewed as: uint32 words or uint16 halfs
  operands: tuple[str, ...]  # what the fused kernel reads after its output, in parameter order
  value_dtypes: Mapping[str, str]  # the dtype of each value operand the fused kernel reads ("half" or "float")
  rows: tuple[dict[str, Any], ...]
  one_subgroup: bool
  parameters: Callable[[Mapping[str, Any], int, int], dict[str, Any]]  # schedule, rows, k -> registry parameters

  def to_json(self) -> dict[str, Any]:
    return {"family": self.family, "covers": [n.name for n in self.covers], "quant": self.quant, "storage": self.storage,
            "operands": list(self.operands), "value_dtypes": dict(self.value_dtypes), "out_dtype": "float",
            "nodes": [{"node": n.name, "op": n.op, "reads": list(n.reads)} for n in self.covers],
            "coupled_rows": [dict(r) for r in self.rows], "threads_within_subgroup": self.one_subgroup}

  def weights(self) -> tuple[str, ...]:
    return tuple(o.split(":", 1)[1] for o in self.operands if o.startswith("weight:"))


def _w1w3_parameters(schedule:Mapping[str, Any], rows:int, k:int) -> dict[str, Any]:
  width, space = schedule["memory"]["b"]["vector_width"], schedule["memory"]["a"]["space"]
  style = {(4, "global"): "scalar", (16, "global"): "vector", (16, "shared"): "quad"}.get((width, space))
  if style is None: raise ValueError(f"no fused gate/up load style reads b {width} wide with a in {space} memory")
  tile, threads = schedule["tile"]["n"], schedule["launch"]["threads"]
  if (style == "quad") != ((tile, threads) == (16, 128)) or (style != "quad" and (tile, threads) != (1, 32)):
    raise ValueError(f"the {style} style does not run {tile} rows a block on {threads} threads")
  return {"rows": rows, "k": k, "load_style": style, "store_fp16": False}


FAMILIES: tuple[FusionFamily, ...] = (
  FusionFamily("q4_w1w3.v1",
               (Node("ffn_gate", "gemv", ("weight:ffn_gate", "value:ffn_norm")),
                Node("ffn_up", "gemv", ("weight:ffn_up", "value:ffn_norm")),
                Node("silu_mul", "silu_mul", ("value:ffn_gate", "value:ffn_up"))),
               "Q4_K", "uint32", ("weight:ffn_gate", "weight:ffn_up", "value:ffn_norm"), {"ffn_norm": "half"},
               (single._row(32, 1, 4), single._row(32, 1, 16), single._row(128, 16, 16, "shared")), False,
               _w1w3_parameters),
)
BY_FAMILY = {f.family: f for f in FAMILIES}


def family_of(candidate:Mapping[str, Any]) -> FusionFamily | None:
  return BY_FAMILY.get(((candidate.get("schedule") or {}).get("compute") or {}).get("family"))


def validate(family:FusionFamily, row:Mapping[str, Any], rows:int, k:int) -> str | None:
  """None when the emitter accepts this shape and row, else its own words. The emitter is constructed (its validate
  runs), not compiled."""
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  schedule = _schedule(row)
  try: build_registered_llm_emitter(family.family, family.parameters(schedule, rows, k))
  except (ValueError, AssertionError) as exc: return str(exc) or type(exc).__name__
  return None


def shape_verdicts(shapes:list[Mapping[str, Any]]) -> list[dict[str, Any]]:
  """For each family and each asked shape {"n", "k"}: every coupled row's validate verdict."""
  out = []
  for family in FAMILIES:
    for shape in shapes:
      n, k = int(shape["n"]), int(shape["k"])
      out.append({"family": family.family, "shape": {"n": n, "k": k},
                  "rows": [{"row": dict(row), "refused": validate(family, row, n, k)} for row in family.rows]})
  return out


def _schedule(row:Mapping[str, Any]) -> dict[str, Any]:
  """A coupled row as the nested schedule the parameter functions read."""
  out: dict[str, Any] = {}
  for path, value in row.items():
    parts = path.split(".")[1:] if path.startswith("schedule.") else path.split(".")
    cur = out
    for part in parts[:-1]: cur = cur.setdefault(part, {})
    cur[parts[-1]] = value
  return out


def _words(raw, storage:str, device:str):
  import numpy as np
  from tinygrad import Tensor, dtypes
  view, dtype = (np.uint32, dtypes.uint32) if storage == "uint32" else (np.uint16, dtypes.uint16)
  return Tensor(raw.view(view), dtype=dtype, device=device).realize()


def values(names:tuple[str, ...], sizes:Mapping[str, int], dtypes_by_name:Mapping[str, str], device:str) -> dict[str, Any]:
  """Deterministic value vectors (the fixture rule: small, finite, nonzero), one per named value."""
  import numpy as np
  from tinygrad import Tensor, dtypes
  out = {}
  for i, name in enumerate(names):
    n = sizes[name]
    data = np.array([1.0 + ((j + 3 * i) % 7) / 16 - (0.5 if (j + i) % 5 == 0 else 0.0) for j in range(n)], dtype=np.float32)
    half = dtypes_by_name.get(name, "float") == "half"
    out[name] = Tensor(data.astype(np.float16) if half else data, dtype=dtypes.float16 if half else dtypes.float32, device=device).realize()
  return out


def build(family:FusionFamily, schedule:Mapping[str, Any], rows:int, k:int, raws:Mapping[str, Any], device:str):
  """The fused program at the anchor's exact shape: (output tensor, the operand tensors by name in parameter order)."""
  from tinygrad import Tensor, dtypes
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  emitter = build_registered_llm_emitter(family.family, family.parameters(schedule, rows, k))
  inputs = {}
  vals = values(tuple(o.split(":", 1)[1] for o in family.operands if o.startswith("value:")),
                _value_sizes(family, rows, k), family.value_dtypes, device)
  for operand in family.operands:
    kind, name = operand.split(":", 1)
    if kind == "weight":
      raw = raws[name]
      if raw.nbytes != rows * k // 256 * BLOCK_BYTES[family.quant]: raise ValueError(f"{raw.nbytes} weight bytes are not {rows}x{k} {family.quant}")
      inputs[operand] = _words(raw, family.storage, device)
    else: inputs[operand] = vals[name]
  out = Tensor.empty(rows, dtype=dtypes.float32, device=device).uop_program(*inputs.values(), fxn=emitter)[0]
  return out, inputs


def _value_sizes(family:FusionFamily, rows:int, k:int) -> dict[str, int]:
  """The length of every value a node of the family reads: a gemv's input is k long, everything else is a row."""
  sizes: dict[str, int] = {}
  for node in family.covers:
    for read in node.reads:
      kind, name = read.split(":", 1)
      if kind == "value": sizes[name] = k if node.op == "gemv" else rows
  return sizes


def unfused(family:FusionFamily, rows:int, k:int, raws:Mapping[str, Any], device:str, backend:str | None,
            architecture:str | None) -> list[tuple[str, Any, dict[str, Any]]]:
  """Each covered node's program alone: (node, output tensor, operand tensors by name). A weight node runs its
  single-role decode emitter at the row the model's decode installs; an elementwise node runs tinygrad's scheduler."""
  from tinygrad import Tensor, dtypes
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  sizes = _value_sizes(family, rows, k)
  # a single-role emitter reads its activation in fp16; a node's output is fp32; a graph input as the fused kernel reads it
  gemv_reads = {r.split(":", 1)[1] for n in family.covers if n.op == "gemv" for r in n.reads if r.startswith("value:")}
  dtypes_by_name = {name: "half" if name in gemv_reads else family.value_dtypes.get(name, "float") for name in sizes}
  vals = values(tuple(sorted(sizes)), sizes, dtypes_by_name, device)
  out = []
  for node in family.covers:
    if node.op == "gemv":
      weight = next(r.split(":", 1)[1] for r in node.reads if r.startswith("weight:"))
      value = next(r for r in node.reads if r.startswith("value:"))
      one = next(f for f in single.FAMILIES if f.quant == family.quant)
      row = single.installed_row(one, backend, architecture)
      emitter = build_registered_llm_emitter(one.family, one.parameters(_schedule(row), rows, k))
      inputs = {f"weight:{weight}": _words(raws[weight], one.storage, device), value: vals[value.split(":", 1)[1]]}
      result = Tensor.empty(rows, dtype=dtypes.float32, device=device).uop_program(*inputs.values(), fxn=emitter)[0]
    else:
      inputs = {read: vals[read.split(":", 1)[1]] for read in node.reads}
      result = ELEMENTWISE[node.op](*(t.float() for t in inputs.values())).contiguous()
    out.append((node.name, result, inputs))
  return out


def reference(family:FusionFamily, rows:int, k:int, raws:Mapping[str, Any], vals:Mapping[str, Any]):
  """The covered nodes evaluated one after another with the canonical packed reference and numpy (float32)."""
  import numpy as np
  from tinygrad import Tensor
  from extra.llm_research.layout import q4_k_reference, q6_k_reference
  deq = q4_k_reference if family.quant == "Q4_K" else q6_k_reference
  got: dict[str, Any] = {name: t.numpy().astype(np.float32) for name, t in vals.items()}
  for node in family.covers:
    if node.op == "gemv":
      weight = next(r.split(":", 1)[1] for r in node.reads if r.startswith("weight:"))
      x = got[next(r.split(":", 1)[1] for r in node.reads if r.startswith("value:"))]
      w = deq(Tensor(raws[weight]), rows * k).numpy().reshape(rows, k).astype(np.float32)
      got[node.name] = w @ x
    elif node.op == "silu_mul":
      a, b = (got[r.split(":", 1)[1]] for r in node.reads)
      got[node.name] = a / (1.0 + np.exp(-a)) * b
    else:
      a, b = (got[r.split(":", 1)[1]] for r in node.reads)
      got[node.name] = a + b
  return got[family.covers[-1].name]


__all__ = ["BY_FAMILY", "ELEMENTWISE", "EMITTER_PLAN_KIND", "FAMILIES", "FusionFamily", "Node", "build", "family_of",
           "reference", "shape_verdicts", "unfused", "validate", "values"]
