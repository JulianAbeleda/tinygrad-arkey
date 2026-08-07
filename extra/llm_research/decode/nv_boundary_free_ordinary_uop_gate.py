#!/usr/bin/env python3
"""Phase-0 construction gate for boundary-free, ordinary-UOp NV decode routes.

This is intentionally a construction gate, not another custom-kernel benchmark.
For each fusion/dataflow population it establishes the exact ordinary scheduler
topology for both a realized input and a lazy producer view.  A construction may
advance only if it produces one replayable ordinary program in both cases
without a custom-program boundary or CONTIGUOUS.

The population keys are the ledger's ``POP_*`` constants
(``nv_fusion_population_ledger.py``), so a gate ``--out`` joins onto a ledger
``--out`` on the population key to produce the capability column mechanically.

Schema: ``tinygrad.nv_boundary_free_ordinary_uop_gate.v2`` (same family as v1;
the v1 top-level ``{contract, baseline, verdict, reason}`` envelope moved
per-population).
"""
from __future__ import annotations

import argparse, hashlib, json, pathlib
from typing import Callable

from tinygrad import Tensor, dtypes, nn
from tinygrad.uop.ops import Ops

from extra.llm_research.decode.nv_fusion_population_ledger import (
  POP_FLASH, POP_NORMS, POP_OTHER, POP_Q8PACK, POP_QUANT, POP_RESIDUAL, POP_ROPE_KV,
  POP_VOCAB, load as load_dag,
)

DIM = 4096
SCHEMA = "tinygrad.nv_boundary_free_ordinary_uop_gate.v2"


def _programs(out: Tensor) -> list[str]:
  linear, _ = out.linear_with_vars()
  return [x.src[0].arg.name for x in linear.src]


def _ordinary(x: Tensor) -> Tensor:
  norm = nn.RMSNorm(DIM, eps=1e-6)
  norm.weight = Tensor.randn(DIM, dtype=dtypes.float16, device=x.device).realize()
  norm._rmsnorm_native_promoted = False
  return norm(x)


def _flash(x: Tensor) -> Tensor:
  # flash score/PV population: one token (head dim 128) against a 32-row KV
  # cache tile, mirroring flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128.
  q = x.reshape(1, 128).cast(dtypes.float32)
  k = Tensor.randn(32, 128, dtype=dtypes.float16, device=x.device).realize()
  v = Tensor.randn(32, 128, dtype=dtypes.float16, device=x.device).realize()
  scores = (q * k).sum(axis=1)
  probs = (scores - scores.max()).exp()
  return (probs.reshape(1, 32) @ v).reshape(128)


def _residual(x: Tensor) -> Tensor:
  # residual/cast/contiguous population: block output plus residual cast,
  # then contiguous (the M4 fold target, E_32_32_4 family).
  residual = Tensor.randn(1, 1, DIM, dtype=dtypes.float16, device=x.device).realize()
  return (x.reshape(1, 1, DIM) + residual.cast(dtypes.float32)).contiguous()


def _vocab(x: Tensor) -> Tensor:
  # vocab sampler population: argmax over the ledger's 1187-row sampler shape.
  return x.reshape(1, 1187).argmax(axis=1)


def _rope_kv(x: Tensor) -> Tensor:
  # rope/kv population: rotary embedding of one head vector (elementwise).
  cos = Tensor.randn(64, dtype=dtypes.float16, device=x.device).realize()
  sin = Tensor.randn(64, dtype=dtypes.float16, device=x.device).realize()
  h = x.reshape(64, 2)
  ev, od = h[:, 0], h[:, 1]
  return Tensor.stack(ev * cos - od * sin, ev * sin + od * cos, dim=1).reshape(128)


def _quant(x: Tensor) -> Tensor:
  # quant core population: the ordinary fp16 spelling of the q4k GEMV.
  w = Tensor.randn(DIM, DIM, dtype=dtypes.float16, device=x.device).realize()
  return (x.reshape(1, DIM) @ w).reshape(1, DIM)


def _q8pack(x: Tensor) -> Tensor:
  # llama q8 pack population: scale + int8 cast.  Attribution-only in the
  # native graph (llama.cpp hides quantize_q8_1 work we do not perform).
  return (x * 127.0).cast(dtypes.int8)


# population -> (construction, contract_shape, realized base shape); None
# construction means the population has no defined construction.
CONSTRUCTIONS: dict[str, tuple[Callable[[Tensor], Tensor] | None, list[int], tuple[int, ...]]] = {
  POP_NORMS: (_ordinary, [1, DIM], (1, DIM)),
  POP_FLASH: (_flash, [32, 128], (1, 128)),
  POP_RESIDUAL: (_residual, [1, 1, DIM], (1, DIM)),
  POP_VOCAB: (_vocab, [1, 1187], (1, 1187)),
  POP_ROPE_KV: (_rope_kv, [1, 128], (1, 128)),
  POP_QUANT: (_quant, [1, DIM], (1, DIM)),
  POP_Q8PACK: (_q8pack, [1, DIM], (1, DIM)),
  POP_OTHER: (None, [], (1, 1)),
}


def _contains_op(x: Tensor, op: Ops) -> bool:
  return any(u.op is op for u in x.uop.toposort())


def _verdict(rows: dict[str, dict]) -> tuple[str, str]:
  any_custom = any(r["contains_custom_kernel"] for r in rows.values())
  any_contig = any(r["contains_contiguous"] for r in rows.values())
  counts = sorted({r["program_count"] for r in rows.values()})
  if counts == [1] and not any_custom and not any_contig:
    return "PASS", "one replayable ordinary program; no custom-program boundary, no CONTIGUOUS"
  blockers = []
  if any_custom: blockers.append("custom-program boundary")
  if any_contig: blockers.append("CONTIGUOUS materialization")
  if counts != [1]: blockers.append(f"split into {counts} programs (reduction + dependent epilogue)")
  return "CONSTRUCTION_GAP", "construction not expressible as one ordinary program: " + "; ".join(blockers)


def run(population: str | None = None, dag: dict | None = None) -> dict:
  pops = list(CONSTRUCTIONS) if population in (None, "all") else [population]
  populations = {}
  for pop in pops:
    fxn, contract_shape, base_shape = CONSTRUCTIONS[pop]
    base = Tensor.randn(*base_shape, dtype=dtypes.float16).realize()
    rows = {}
    for label, x in (("realized", base), ("lazy_add", base + base)):
      if fxn is None:
        rows[label] = None
      else:
        out = fxn(x)
        programs = _programs(out)
        rows[label] = {
          "programs": programs,
          "program_count": len(programs),
          "contains_custom_kernel": _contains_op(out, Ops.CUSTOM),
          "contains_contiguous": _contains_op(out, Ops.CONTIGUOUS),
        }
    entry: dict = {"contract_shape": contract_shape, "baseline": rows}
    if fxn is None:
      entry["verdict"] = "NO_CONSTRUCTION"
      entry["reason"] = "unclassified fallback population; no construction defined"
    else:
      entry["verdict"], entry["reason"] = _verdict(rows)
    populations[pop] = entry
  capture: dict = {"construction_count": len(pops)}
  if dag is not None:
    capture.update({
      "dag_node_count": len(dag["nodes"]),
      "dag_edge_count": len(dag["edges"]),
      "dag_name_digest": hashlib.sha256("\n".join(n["name"] for n in dag["nodes"]).encode()).hexdigest(),
    })
  return {"schema": SCHEMA, "capture": capture, "populations": populations}


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--dag", default=None, help="optional duration-bearing DAG (nv_dependency_closed_cut) for the envelope capture")
  ap.add_argument("--out", required=True)
  ap.add_argument("--population", default="all", choices=["all", *CONSTRUCTIONS])
  args = ap.parse_args()
  dag = load_dag(args.dag) if args.dag else None
  result = run(args.population, dag)
  path = pathlib.Path(args.out); path.parent.mkdir(parents=True, exist_ok=True)
  path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  header = f"{'population':26s} {'verdict':16s} {'contract_shape':24s} reason"
  print(header)
  print("-" * len(header))
  for pop in sorted(result["populations"]):
    entry = result["populations"][pop]
    print(f"{pop:26s} {entry['verdict']:16s} {str(entry['contract_shape']):24s} {entry['reason']}")
  print(f"\ncapture: {result['capture']}")


if __name__ == "__main__": main()
