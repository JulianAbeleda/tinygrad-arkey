#!/usr/bin/env python3
"""Turn a BoltBeam prefill roofline report into a tinygrad packed-prefill route backlog.

This is intentionally analysis/codegen-facing, not a benchmark runner. The 14B parity trace showed that the hot
memory-safe path is no longer a broad lifecycle problem: Q4_K/Q6_K packed prefill GEMMs run with a narrow 32/64-thread
scalar-ish topology while llama's quantized matmul family runs a much wider tiled workgroup. This script preserves
that finding as data so the implementation work can be routed through a generated schedule/spec path instead of
one-off kernel edits.
"""
from __future__ import annotations

import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LLAMA_CLASS_Q4_THREADS = 256


@dataclass(frozen=True)
class PackedPrefillWorkItem:
  role: str
  quant: str
  m: int
  n: int
  k: int
  current_us: float
  current_gbs: float
  target_gbs: float | None
  reclaim_us: float | None
  global_size: tuple[int, int, int] | None
  local_size: tuple[int, int, int] | None
  workgroup_threads: int | None
  vgpr: int | None
  sgpr: int | None
  lds_bytes: int | None
  scratch_bytes: int | None

  @property
  def shape(self) -> str:
    return f"[{self.m},{self.n},{self.k}]"

  @property
  def route_id(self) -> str:
    return f"prefill_{self.quant.lower()}_generated_tile_{self.role}_{self.m}_{self.n}_{self.k}"

  @property
  def priority(self) -> tuple[int, float]:
    reclaim = self.reclaim_us if self.reclaim_us is not None else 0.0
    return (1 if self.quant == "Q4_K" else 0, reclaim)

  @property
  def diagnosis(self) -> str:
    if self.quant == "Q4_K":
      return "generated tiled Q4_K packed-prefill matmul; match llama-class workgroup/token tiling"
    return "same generated packed-prefill topology after Q4 proves the schedule; Q6 target rate needs trusted baseline"


def _triple(v: Any) -> tuple[int, int, int] | None:
  if isinstance(v, list) and len(v) == 3 and all(isinstance(x, int) for x in v): return (v[0], v[1], v[2])
  return None


def _item(row: dict[str, Any]) -> PackedPrefillWorkItem:
  res = row.get("resources") or {}
  shape = row["shape"]
  return PackedPrefillWorkItem(
    role=row["role"], quant=row["quant"], m=shape[0], n=shape[1], k=shape[2],
    current_us=float(row["wall_us"]), current_gbs=float(row["current_eff_gbs"]),
    target_gbs=float(row["trusted_target_gbs"]) if row.get("trusted_target_gbs") is not None else None,
    reclaim_us=float(row["trusted_reclaimable_us"]) if row.get("trusted_reclaimable_us") is not None else None,
    global_size=_triple(res.get("global_size")), local_size=_triple(res.get("local_size")),
    workgroup_threads=res.get("workgroup_threads"), vgpr=res.get("vgpr"), sgpr=res.get("sgpr"),
    lds_bytes=res.get("lds_bytes"), scratch_bytes=res.get("scratch_bytes"))


def load_items(path: Path) -> list[PackedPrefillWorkItem]:
  report = json.loads(path.read_text())
  items = [_item(row) for row in report.get("hot_rows", [])]
  return sorted(items, key=lambda x: x.priority, reverse=True)


def schedule_requirements(items: list[PackedPrefillWorkItem]) -> dict[str, Any]:
  q4 = [x for x in items if x.quant == "Q4_K"]
  q6 = [x for x in items if x.quant == "Q6_K"]
  q4_reclaim = sum(x.reclaim_us or 0.0 for x in q4)
  return {
    "primary_target": q4[0].__dict__ if q4 else None,
    "q4_reclaimable_us": q4_reclaim,
    "route_family": "generated_packed_prefill_tile",
    "default": "off",
    "route_env": "PREFILL_QK_GENERATED_TILE=1",
    "strict_env": "PREFILL_ROUTE_STRICT=1",
    "implementation_order": [x.route_id for x in items],
    "required_codegen_axes": {
      "current": "row GLOBAL, token mostly UPCAST/serial, q4 lane4 REDUCE, kblock REDUCE",
      "target": "row tile GLOBAL/LOCAL, token tile GLOBAL/LOCAL, q4 lane4 LOCAL/cooperative, kblock REDUCE",
      "reason": "the current REDUCE-lane direct kernel is a scalar packed-load GEMM; llama-class throughput needs a cooperative tile topology",
    },
    "first_shape": q4[0].shape if q4 else "",
    "first_gate": {
      "correctness": "candidate output must match current lossless direct-packed Q4_K within fp reassociation tolerance",
      "route": "strict mode must show only the generated-tile kernel for the selected tensor",
      "speed": "close candidates below 5 GB/s on ffn_gate_up; continue search toward 10+ then llama-class 26 GB/s",
    },
    "q6_policy": "do not lead with Q6_K; add it after Q4_K ffn_gate_up/attn_qo proves the topology",
    "non_goals": ["do not chase elementwise/lifecycle fusion before packed GEMM leaves the roofline floor",
                  "do not promote lossy Q8 activation routes as the default parity route",
                  "do not add source-string or external handwritten kernels"],
    "hot_role_counts": {"Q4_K": len(q4), "Q6_K": len(q6)},
  }


def write_markdown(items: list[PackedPrefillWorkItem], req: dict[str, Any]) -> str:
  lines = [
    "# Packed Prefill Generated-Tile Scope",
    "",
    "This backlog is derived from a BoltBeam practical roofline report. It ranks tinygrad work by reclaimable",
    "pp512 time against llama's measured Q4 packed-matmul rate, not by broad code ownership.",
    "",
    "## Conclusion",
    "",
    "The next tinygrad work is a generated packed-prefill tile route. The current Q4_K direct-output path is a",
    "fit/safety floor, but its topology is the bottleneck: 32/64-thread workgroups, no LDS, no scratch, and a",
    "Q4 lane axis reduced inside the output element. The route needs an explicit cooperative token/row/lane tile.",
    "",
    "## Generated Schedule Requirements",
    "",
    f"- Route family: `{req['route_family']}`",
    f"- Default state: `{req['default']}` via `{req['route_env']}`",
    f"- Strict gate: `{req['strict_env']}` must fail on hidden fallback",
    f"- Current axes: {req['required_codegen_axes']['current']}",
    f"- Target axes: {req['required_codegen_axes']['target']}",
    f"- First shape: `{req['first_shape']}`",
    "",
    "## Ranked Work",
    "",
    "| priority | role | quant | shape | current us | current GB/s | target GB/s | reclaim us | launch resources | route id |",
    "|---:|---|---|---|---:|---:|---:|---:|---|---|",
  ]
  for i, x in enumerate(items, 1):
    res = f"global={x.global_size}, local={x.local_size}, threads={x.workgroup_threads}, vgpr={x.vgpr}, sgpr={x.sgpr}, lds={x.lds_bytes}, scratch={x.scratch_bytes}"
    lines.append("| {} | {} | {} | `{}` | {:.3f} | {:.3f} | {} | {} | {} | `{}` |".format(
      i, x.role, x.quant, x.shape, x.current_us, x.current_gbs,
      "" if x.target_gbs is None else f"{x.target_gbs:.3f}",
      "" if x.reclaim_us is None else f"{x.reclaim_us:.3f}", res, x.route_id))
  lines += [
    "",
    "## Implementation Path",
    "",
    "1. Add a `PackedPrefillTileSpec` data object for Q4_K with row tile, token tile, lane tile, k-block policy,",
    "   accumulator dtype, output layout, and strict role/shape guards.",
    "2. Lower that spec through a generated UOp emitter. The first emitter should keep lossless fp32 accumulation and",
    "   direct `[tokens, rows]` output; an external lane-partial probe is acceptable only as a short-lived microgate.",
    "3. Wire `tinygrad/llm/prefill_routes.py` behind `PREFILL_QK_GENERATED_TILE=1`, with tensor-role filters so the",
    "   first target can be only `ffn_gate_up`.",
    "4. Add route-manifest metadata with provenance `machine_authored_generated` once the emitter is spec-driven.",
    "5. Gate ffn_gate_up first, then attn_qo and ffn_down Q4_K. Add Q6_K only after the Q4 topology moves.",
    "",
    "## Exhaustion Rule",
    "",
    "Close a candidate quickly if the bound hot-row kernel stays in the ~2 GB/s class. Continue only when the generated",
    "tile changes the substrate class, visible as wider workgroups/cooperative lanes and a multi-x per-kernel GB/s move.",
    "",
  ]
  return "\n".join(lines)


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("report", type=Path, help="BoltBeam prefill_practical_roofline.json")
  ap.add_argument("--json", type=Path, help="write machine-readable scope")
  ap.add_argument("--markdown", type=Path, help="write markdown scope")
  args = ap.parse_args()
  items = load_items(args.report)
  req = schedule_requirements(items)
  payload = {"schema": "tinygrad.packed_prefill_tile_scope.v1", "requirements": req,
             "work_items": [x.__dict__ | {"shape_str": x.shape, "route_id": x.route_id,
                                          "diagnosis": x.diagnosis} for x in items]}
  if args.json:
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
  md = write_markdown(items, req)
  if args.markdown:
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(md + "\n")
  if not args.json and not args.markdown:
    print(md)


if __name__ == "__main__":
  main()
