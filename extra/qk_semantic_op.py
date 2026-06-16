#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib
from dataclasses import asdict, dataclass
from typing import Any

from extra.qk_packed_tile import PackedQKTile, load_tile, tile_from_semantic_row

ARTIFACT_KIND = "qk_packed_semantic_op_contract"
DEFAULT_OUT = pathlib.Path("bench/qk-packed-semantic-op-20260613")
DEFAULT_DESCRIPTORS = (
  pathlib.Path("bench/qk-ansor-transition-20260612/descriptors/8b.json"),
  pathlib.Path("bench/qk-ansor-transition-20260612/descriptors/14b.json"),
)

QK_BLOCK_DOT = "QK_BLOCK_DOT"
LOWERING_AMD_RENDERER_PATTERN = "amd_renderer_pattern"
LOWERING_RAW_CUSTOM_KERNEL = "raw_custom_kernel"


from extra.qk_paths import portable_path as _portable


@dataclass(frozen=True)
class QKSemanticOpContract:
  name: str
  granularity: str
  format: str
  load_tile: str
  input_operands: tuple[str, ...]
  output_dtype: str
  scheduler_visible_axes: tuple[str, ...]
  hidden_allowed: tuple[str, ...]
  hidden_forbidden: tuple[str, ...]
  lowering_target: str
  runtime_lowering_exists: bool


def qk_block_dot_contract(tile:PackedQKTile, *, load_tile_name:str="u32x4_aligned",
                          lowering_target:str=LOWERING_AMD_RENDERER_PATTERN) -> QKSemanticOpContract:
  if tile.format.name != "Q4_K":
    raise ValueError(f"{QK_BLOCK_DOT} contract currently supports Q4_K only, got {tile.format.name}")
  selected = load_tile(tile, load_tile_name)
  if selected.name != "u32x4_aligned":
    raise ValueError(f"{QK_BLOCK_DOT} first lowering requires u32x4_aligned, got {selected.name}")
  if lowering_target == LOWERING_RAW_CUSTOM_KERNEL:
    raise ValueError("raw custom full-kernel lowering is explicitly closed; use a renderer/core semantic lowering")
  if lowering_target != LOWERING_AMD_RENDERER_PATTERN:
    raise ValueError(f"unsupported lowering target {lowering_target!r}")
  return QKSemanticOpContract(
    name=QK_BLOCK_DOT,
    granularity="one packed quant block and matching activation block produce one float32 contribution",
    format=tile.format.name,
    load_tile=selected.name,
    input_operands=("qk_packed_tile_words", "activation_fp16_block", "block_index", "row_index"),
    output_dtype="float32",
    scheduler_visible_axes=("row", "k_block", "split_part", "lane_or_reduce"),
    hidden_allowed=(
      "format-specific Q4_K scale/min unpack",
      "Q4_K nibble extraction",
      "lane mapping inside one 256-element block",
      "target load intrinsic or vector-load spelling",
    ),
    hidden_forbidden=(
      "row loop",
      "K-block loop",
      "split-K partial output layout",
      "partial reduction kernel",
      "full GEMV kernel body",
      "runtime policy selection",
    ),
    lowering_target=lowering_target,
    runtime_lowering_exists=False,
  )


def op_contract_json(contract:QKSemanticOpContract, *, tensor:str, role:str, shape:dict[str, int]) -> dict[str, Any]:
  return {
    "tensor": tensor,
    "role": role,
    "shape": shape,
    "contract": json.loads(json.dumps(asdict(contract))),
    "correctness_oracle": "extra.qk_layout.q4_k_reference plus random-activation GEMV compare on AMD",
    "promotion_status": "design_only_no_runtime_lowering",
  }


def promotion_gates() -> list[dict[str, Any]]:
  return [
    {"gate": "reference_unpack", "requirement": "bit-exact Q4_K unpack against extra.qk_layout.q4_k_reference"},
    {"gate": "amd_gemv_numeric", "requirement": "random fp16 activation GEMV compare against current v1 primitive"},
    {"gate": "source_width", "requirement": "generated source records intended packed load spelling"},
    {"gate": "target_width", "requirement": "DEBUG=7 target block contains wide/coalesced load evidence"},
    {"gate": "scheduler_shape", "requirement": "target workgroup shape preserves v1-like schedulable row/K parallelism; reject workgroup-size 1"},
    {"gate": "target_body_size", "requirement": "target instruction count must not exceed 2x comparable v1 kernel without a measured win"},
    {"gate": "microbench", "requirement": "repeated dominant-shape median gain >= 10% before full decode"},
    {"gate": "full_decode", "requirement": "8B and 14B confirmation reruns accept; 32B optional only after promise"},
    {"gate": "greedy_ab", "requirement": "end-to-end greedy output A/B passes"},
  ]


def build_contract_report(descriptor_paths:list[pathlib.Path], *, repo:pathlib.Path=pathlib.Path.cwd()) -> dict[str, Any]:
  rows, skipped = [], []
  seen: set[tuple[str, str, str]] = set()
  for path in descriptor_paths:
    descriptor = json.loads(path.read_text())
    model = descriptor.get("model_label") or path.stem.upper()
    for row in descriptor["descriptors"]:
      tile = tile_from_semantic_row(row)
      key = (str(model), tile.tensor, tile.format.name)
      if key in seen: continue
      seen.add(key)
      if tile.format.name != "Q4_K":
        skipped.append({
          "model": model,
          "tensor": tile.tensor,
          "role": tile.role,
          "format": tile.format.name,
          "reason": "first semantic op contract scopes Q4_K only; Q6_K vector/load layout is separate work",
        })
        continue
      try:
        contract = qk_block_dot_contract(tile)
      except ValueError as exc:
        skipped.append({
          "model": model,
          "tensor": tile.tensor,
          "role": tile.role,
          "format": tile.format.name,
          "reason": str(exc),
        })
        continue
      rows.append({
        "model": model,
        **op_contract_json(contract, tensor=tile.tensor, role=tile.role, shape={"rows": tile.rows, "cols": tile.cols}),
      })
  return {
    "kind": ARTIFACT_KIND,
    "schema_version": 1,
    "summary": {
      "decision": "semantic_op_contract_defined_no_runtime_lowering",
      "descriptors": [_portable(path, repo) for path in descriptor_paths],
      "q4_contract_rows": len(rows),
      "skipped_rows": len(skipped),
      "runtime_lowering_exists": False,
      "run_microbench": False,
      "run_full_decode": False,
      "next_step": "minimal compile gate for QK_BLOCK_DOT renderer/core lowering",
    },
    "op_name": QK_BLOCK_DOT,
    "contract_rows": rows,
    "skipped_rows": skipped,
    "promotion_gates": promotion_gates(),
    "non_goals": [
      "raw Ops.CUSTOM full-kernel variants",
      "parts/LOCAL-only sweeps",
      "row-group broadening",
      "full decode before repeated microbench clears the gate",
      "32B before 8B/14B promise",
    ],
  }


def contract_report_markdown(report:dict[str, Any]) -> str:
  lines = [
    "# Packed QK Semantic Op Contract",
    "",
    f"Decision: `{report['summary']['decision']}`",
    "",
    "This artifact defines the next compiler-facing contract only. It does not add",
    "a runtime lowering, benchmark result, or full-decode claim.",
    "",
    "## Summary",
    "",
    f"- op: `{report['op_name']}`",
    f"- Q4 contract rows: `{report['summary']['q4_contract_rows']}`",
    f"- skipped rows: `{report['summary']['skipped_rows']}`",
    f"- runtime lowering exists: `{report['summary']['runtime_lowering_exists']}`",
    f"- next step: `{report['summary']['next_step']}`",
    "",
    "## Contract Rows",
    "",
    "| model | tensor | role | shape | load tile | lowering target | status |",
    "|---|---|---|---:|---|---|---|",
  ]
  for row in report["contract_rows"]:
    contract = row["contract"]
    shape = row["shape"]
    lines.append(
      f"| `{row['model']}` | `{row['tensor']}` | `{row['role']}` | "
      f"`{shape['rows']}x{shape['cols']}` | `{contract['load_tile']}` | "
      f"`{contract['lowering_target']}` | `{row['promotion_status']}` |"
    )
  lines += [
    "",
    "## Hidden Boundary",
    "",
    "The semantic op may hide Q4_K scale/min unpack, nibble extraction, and a",
    "target load intrinsic spelling inside one block. It must not hide the row",
    "loop, K-block loop, split-K output layout, partial reduction, full GEMV body,",
    "or runtime policy selection.",
    "",
    "## Promotion Gates",
    "",
    "| gate | requirement |",
    "|---|---|",
  ]
  for gate in report["promotion_gates"]:
    lines.append(f"| `{gate['gate']}` | {gate['requirement']} |")
  if report["skipped_rows"]:
    lines += ["", "## Skipped Rows", "", "| model | tensor | format | reason |", "|---|---|---|---|"]
    for row in report["skipped_rows"]:
      lines.append(f"| `{row['model']}` | `{row['tensor']}` | `{row['format']}` | {row['reason']} |")
  lines.append("")
  return "\n".join(lines)


def write_report(report:dict[str, Any], out:pathlib.Path) -> None:
  out.mkdir(parents=True, exist_ok=True)
  (out / "semantic-op-contract.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
  md = contract_report_markdown(report)
  (out / "semantic-op-contract.md").write_text(md)
  (out / "README.md").write_text(md)


def main() -> int:
  parser = argparse.ArgumentParser(description="Build static packed-QK semantic op contract artifact")
  parser.add_argument("--descriptor", type=pathlib.Path, action="append", default=None)
  parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
  args = parser.parse_args()
  descriptors = args.descriptor if args.descriptor is not None else list(DEFAULT_DESCRIPTORS)
  report = build_contract_report(descriptors, repo=pathlib.Path.cwd())
  write_report(report, args.out)
  print(contract_report_markdown(report))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
