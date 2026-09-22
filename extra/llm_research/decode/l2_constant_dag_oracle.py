#!/usr/bin/env python3
"""L2 constant-DAG llama-kernel oracle config builder (CPU-only).

Diagnostic lane L, gate G-B3-LQ / G-B3-LO harness construction per
docs/task_workflow/input/nv-decode-overlap-route-b3-exhaustive-execution-scope-20260804.md
section 14.4.  This tool is CPU-only and never initializes CUDA: it turns the
aligned physical capture manifest (arms.physical: nodes, edges, groups) plus
the llama quantized-oracle ABI report into a frozen L2 config.  A later GPU
agent can then hold the physical DAG constant while swapping ONLY the dominant
Q4_K/Q6_K MMV kernels for llama's compiled mul_mat_vec_q, and measure the
surviving wall gap.

Config schema tinygrad.route_b3.l2_config.v1:

  source_dag_hash            sha256 over the canonical (sorted) physical edges
  per_group                  every physical node grouped by graph group
  mmv_swap_table             node id -> oracle variant + block layout + ABI template
  preserved_physical_edges   the full physical edge list, unchanged
  predicted_swap_count       number of oracle-swapped MMV nodes

Kernel class vocabulary follows extra/llm_research/decode/route_kernel_census.py
(read-only reference): q4k_gemv / q6k_gemv are the MMV classes; the vocab-head
q6k kernels (class vocab_head) stay untouched because they sit on a different
semantic boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile


SCHEMA = "tinygrad.route_b3.l2_config.v1"
ORACLE_SCHEMA = "tinygrad.llama_cuda_quantized_oracle.v1"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPTURE = REPO_ROOT / "docs/task_workflow/output/nv-decode-overlap-b3-2-aligned-capture-manifest-20260804.json"
DEFAULT_ORACLE = REPO_ROOT / "scratchpad/llama_cuda_quantized_oracle_dump/llama_cuda_quantized_oracle_v1.json"

MMV_CLASSES = ("q4k_gemv", "q6k_gemv")
VARIANT_FOR_CLASS = {"q4k_gemv": "Q4_K", "q6k_gemv": "Q6_K"}
CLASS_FOR_VARIANT = {variant: cls for cls, variant in VARIANT_FOR_CLASS.items()}

QK8_1 = 32
SRC1_ROW_PADDING = 512
MMVQ_MAX_BATCH_SIZE = 8
EXPECTED_ARG_COUNT = 19
FUSION_ARGS_SIZE = 32

HASH64 = re.compile(r"_[0-9a-f]{64}\b")


def canonical_name(name: str) -> str:
  return HASH64.sub("", name).strip()


def classify(name: str) -> str:
  """Kernel class from the route_kernel_census vocabulary (order matters)."""
  clean = canonical_name(name)
  if clean.startswith("flash_"):
    return "flash_decode_attention"
  if clean.startswith("q4k_g3_lanemap_gemv"):
    return "q4k_gemv"
  if "151936" in clean or clean.startswith("q6k_vocab_scalar_reduce"):
    return "vocab_head"
  if clean.startswith("q6k_gen_coop") or clean.startswith("q6k_gen_partial"):
    return "q6k_gemv"
  if clean.startswith("decode_kv_rope_store"):
    return "kv_store"
  if "1187" in clean:
    return "scatter"
  if clean.startswith("E_") or clean.startswith("r_"):
    return "elementwise_fusion"
  return "other"


# 19-arg launch ABI of llama's mul_mat_vec_q, mirroring the argument order the
# quantized-oracle report pins (vx_ptr ... ids_stride, uint3 fastdiv params).
KERNEL_ARGUMENT_ORDER = [
  ("vx_ptr", "const void *", "src0 quantized data (block_q4_K/block_q6_K)"),
  ("vy_ptr", "const void *", "src1 quantized to block_q8_1 staging buffer"),
  ("ids_ptr", "const int32_t *", "MUL_MAT_ID row ids, nullptr for MUL_MAT"),
  ("fusion", "ggml_cuda_mm_fusion_args_device", "passed by value (32 bytes)"),
  ("dst_ptr", "float *", "output f32 buffer"),
  ("ncols_x", "uint32_t", "src0->ne[0], contracted K in elements"),
  ("nchannels_y", "uint3", "fastdiv table for src1 channel index"),
  ("stride_row_x", "uint32_t", "src0 rows per channel (block units)"),
  ("stride_col_y", "uint32_t", "q8_1 blocks per quantized src1 row"),
  ("stride_col_dst", "uint32_t", "dst rows per column in floats"),
  ("channel_ratio", "uint3", "fastdiv(dst ne2 / src0 ne2) as (mp,L,d)"),
  ("stride_channel_x", "uint32_t", "src0 channels per sample (block units)"),
  ("stride_channel_y", "uint32_t", "src1 channels per sample (q8_1 blocks)"),
  ("stride_channel_dst", "uint32_t", "dst channels per sample in floats"),
  ("sample_ratio", "uint3", "fastdiv(dst ne3 / src0 ne3) as (mp,L,d)"),
  ("stride_sample_x", "uint32_t", "src0 samples stride (block units)"),
  ("stride_sample_y", "uint32_t", "src1 samples stride (q8_1 blocks)"),
  ("stride_sample_dst", "uint32_t", "dst samples stride in floats"),
  ("ids_stride", "uint32_t", "ids row stride, 0 when ids is null"),
]


def argument_order_list() -> list[dict[str, object]]:
  return [
    {"index": index, "name": name, "ctype": ctype, "note": note}
    for index, (name, ctype, note) in enumerate(KERNEL_ARGUMENT_ORDER, start=1)
  ]


def _fixture_kernel(variant: str, ggml_type: int, size_bytes: int, vdr: int, canonical_entry: str) -> dict[str, object]:
  if variant == "Q4_K":
    fields = [
      {"name": "d", "ctype": "ggml_half", "offset": 0, "size": 2},
      {"name": "dmin", "ctype": "ggml_half", "offset": 2, "size": 2},
      {"name": "scales", "ctype": "uint8_t[12]", "offset": 4, "size": 12},
      {"name": "qs", "ctype": "uint8_t[128]", "offset": 16, "size": 128},
    ]
  else:
    fields = [
      {"name": "ql", "ctype": "uint8_t[128]", "offset": 0, "size": 128},
      {"name": "qh", "ctype": "uint8_t[64]", "offset": 128, "size": 64},
      {"name": "scales", "ctype": "int8_t[16]", "offset": 192, "size": 16},
      {"name": "d", "ctype": "ggml_half", "offset": 208, "size": 2},
    ]
  return {
    "quant_kind": variant,
    "ggml_type": ggml_type,
    "block_layout": {"size_bytes": size_bytes, "fields": fields},
    "canonical_entry": canonical_entry,
    "abi": {
      "qk": 256,
      "qr": 2,
      "qi": 32,
      "vdr": vdr,
      "blck_size": 256,
      "block_size_bytes": size_bytes,
      "nwarps_generic": [
        {"ncols_dst": ncols, "nwarps": 4 if ncols <= 4 else 2}
        for ncols in range(1, MMVQ_MAX_BATCH_SIZE + 1)
      ],
      "rows_per_block_default": 1,
    },
  }


FIXTURE_ORACLE: dict[str, object] = {
  "schema": ORACLE_SCHEMA,
  "mode": "fixture",
  "library": "embedded-fixture",
  "fusion_args": {"struct": "ggml_cuda_mm_fusion_args_device", "size_bytes": FUSION_ARGS_SIZE},
  "kernels": {
    "Q4_K": _fixture_kernel(
      "Q4_K", 12, 144, 2,
      "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
    ),
    "Q6_K": _fixture_kernel(
      "Q6_K", 14, 210, 1,
      "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj",
    ),
  },
  "launch_wrapper": {
    "host_dispatch_symbol": (
      "_Z23ggml_cuda_mul_mat_vec_qR25ggml_backend_cuda_contextPK11ggml_tensor"
      "S3_S3_PS1_PK29ggml_cuda_mm_fusion_args_host"
    ),
    "kernel_argument_order": argument_order_list(),
    "grid": "dim3(ceil(nrows_x / rows_per_block), nchannels_dst, nsamples_dst)",
    "block": "dim3(warp_size, nwarps, 1)",
    "notes": "fixture ABI mirrors llama_cuda_quantized_oracle_v1.json",
  },
}


def canonical_physical_edges(edges: list[dict[str, object]]) -> list[tuple[int, int, str]]:
  return sorted((int(edge["from"]), int(edge["to"]), str(edge["kind"])) for edge in edges)


def dag_hash(edges: list[dict[str, object]]) -> str:
  payload = json.dumps(canonical_physical_edges(edges), separators=(",", ":")).encode("ascii")
  return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def abi_arg_template(report: dict[str, object], variant: str) -> dict[str, object]:
  kernel = report["kernels"][variant]
  launch = report["launch_wrapper"]
  return {
    "kernel_template": "mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>",
    "canonical_entry": kernel["canonical_entry"],
    "arg_count": len(launch["kernel_argument_order"]),
    "argument_order": [
      {"index": arg["index"], "name": arg["name"], "ctype": arg["ctype"], "note": arg["note"]}
      for arg in launch["kernel_argument_order"]
    ],
    "fusion_args": {
      "struct": report["fusion_args"]["struct"],
      "size_bytes": report["fusion_args"]["size_bytes"],
    },
    "src1": {
      "quant": "block_q8_1",
      "qk8_1": QK8_1,
      "row_padding": SRC1_ROW_PADDING,
      "note": "src1 must be quantized to block_q8_1 with rows padded to 512 elements",
    },
    "grid": launch["grid"],
    "block": launch["block"],
    "nwarps_generic": kernel["abi"]["nwarps_generic"],
  }


def block_layout_summary(report: dict[str, object], variant: str) -> dict[str, object]:
  kernel = report["kernels"][variant]
  block = kernel["block_layout"]
  return {
    "name": f"block_{variant.lower()}",
    "size_bytes": block["size_bytes"],
    "qk": kernel["abi"]["qk"],
    "qr": kernel["abi"]["qr"],
    "qi": kernel["abi"]["qi"],
    "vdr": kernel["abi"]["vdr"],
    "fields": [
      {"name": field["name"], "ctype": field["ctype"], "offset": field["offset"], "size": field["size"]}
      for field in block["fields"]
    ],
  }


def swap_entry(node_id: int, variant: str, report: dict[str, object]) -> dict[str, object]:
  return {
    "node_id": node_id,
    "variant": variant,
    "block_layout": block_layout_summary(report, variant),
    "abi_arg_template": abi_arg_template(report, variant),
  }


def build_config(
  capture: dict[str, object],
  oracle: dict[str, object],
  source_capture: str | None = None,
  source_capture_sha256: str | None = None,
  oracle_report: str | None = None,
  oracle_report_sha256: str | None = None,
) -> dict[str, object]:
  """Build an L2 config from a capture dict and an oracle ABI report dict."""
  physical = capture["arms"]["physical"]
  nodes = physical["nodes"]
  edges = physical["edges"]

  per_group: dict[str, list[dict[str, object]]] = {}
  node_by_id: dict[int, dict[str, object]] = {}
  for node in nodes:
    node_id = int(node["id"])
    group_id = int(node["group_id"])
    metadata = node.get("metadata") or {}
    entry = {
      "id": node_id,
      "name": str(node["name"]),
      "class": classify(str(node["name"])),
      "group": group_id,
      "identity_sha256": metadata.get("identity_sha256"),
    }
    per_group.setdefault(str(group_id), []).append(entry)
    node_by_id[node_id] = entry

  swap_table: dict[str, dict[str, object]] = {}
  for node in sorted(nodes, key=lambda item: int(item["id"])):
    variant = VARIANT_FOR_CLASS.get(classify(str(node["name"])))
    if variant is not None:
      node_id = int(node["id"])
      swap_table[str(node_id)] = swap_entry(node_id, variant, oracle)

  class_counts: dict[str, int] = {}
  for node in nodes:
    cls = classify(str(node["name"]))
    class_counts[cls] = class_counts.get(cls, 0) + 1

  return {
    "schema": SCHEMA,
    "source_capture": source_capture,
    "source_capture_sha256": source_capture_sha256,
    "oracle_report": oracle_report,
    "oracle_report_sha256": oracle_report_sha256,
    "source_dag_hash": dag_hash(edges),
    "node_count": len(nodes),
    "edge_count": len(edges),
    "group_count": len(per_group),
    "per_group": {group: sorted(items, key=lambda item: int(item["id"])) for group, items in per_group.items()},
    "mmv_swap_table": swap_table,
    "preserved_physical_edges": [
      {"from": int(edge["from"]), "to": int(edge["to"]), "kind": str(edge["kind"]), "crosses_group": bool(edge["crosses_group"])}
      for edge in edges
    ],
    "predicted_swap_count": len(swap_table),
    "class_counts": {cls: class_counts[cls] for cls in sorted(class_counts)},
    "notes": "physical DAG held constant; only dominant Q4_K/Q6_K MMV kernels swap to llama mul_mat_vec_q",
  }


def build_synthetic_capture() -> dict[str, object]:
  """Hermetic fixture: 12 nodes, 2 groups, 4 MMV nodes (3 Q4_K + 1 Q6_K)."""
  spec = [
    (0, "E_32_32_4_synthetic_a", 0),
    (1, "q4k_g3_lanemap_gemv_4096_4096", 0),
    (2, "r_16_256_synthetic_b", 0),
    (3, "flash_block_tiled_xlane_score_pv_tile_whole_cache_32_128", 0),
    (4, "q4k_g3_lanemap_gemv_12288_4096", 0),
    (5, "E_8_2_16_4_synthetic_c", 0),
    (6, "q6k_gen_coop_4096_12288", 1),
    (7, "E_128_32_3_synthetic_d", 1),
    (8, "q4k_g3_lanemap_gemv_1024_4096", 1),
    (9, "r_32_32_4_synthetic_e", 1),
    (10, "decode_kv_rope_store_1024", 1),
    (11, "E_4_2_8_16_synthetic_f", 1),
  ]
  nodes = [
    {
      "id": node_id,
      "name": name,
      "group_id": group_id,
      "duration_us": 0.0,
      "metadata": {"identity_sha256": hashlib.sha256(name.encode("ascii")).hexdigest()},
    }
    for node_id, name, group_id in spec
  ]
  groups = {node_id: group_id for node_id, _, group_id in spec}
  edge_spec = [
    (0, 1, "RAW"), (1, 2, "RAW"), (2, 3, "RAW"), (2, 4, "RAW"),
    (3, 5, "RAW"), (4, 5, "RAW"), (5, 6, "WAR"), (6, 7, "RAW"),
    (7, 8, "RAW"), (8, 9, "RAW"), (8, 10, "RAW"), (9, 11, "RAW"),
    (10, 11, "RAW"), (0, 6, "WAW"),
  ]
  edges = [
    {"from": frm, "to": to, "kind": kind, "crosses_group": groups[frm] != groups[to]}
    for frm, to, kind in edge_spec
  ]
  return {"arms": {"physical": {"nodes": nodes, "edges": edges}}}


def validate_config(config: dict[str, object]) -> dict[str, object]:
  """Check L2 invariants: edges preserved, swap nodes present, groups intact."""
  issues: list[str] = []
  if config.get("schema") != SCHEMA:
    issues.append(f"schema mismatch: {config.get('schema')!r}")

  dag_hash_value = config.get("source_dag_hash")
  if not isinstance(dag_hash_value, str) or len(dag_hash_value) != 64 or any(char not in "0123456789abcdef" for char in dag_hash_value.lower()):
    issues.append(f"invalid source_dag_hash: {dag_hash_value!r}")

  per_group = config.get("per_group")
  swap_table = config.get("mmv_swap_table")
  edges = config.get("preserved_physical_edges")
  if not isinstance(per_group, dict) or not per_group:
    issues.append("per_group missing or empty")
    return {"valid": not issues, "issues": issues}

  node_by_id: dict[int, dict[str, object]] = {}
  for group_name, group in per_group.items():
    if not isinstance(group, list):
      issues.append(f"group {group_name} is not a node list")
      continue
    for node in group:
      if not isinstance(node, dict):
        issues.append(f"group {group_name} contains a non-dict node")
        continue
      node_id = node.get("id")
      if not isinstance(node_id, int):
        issues.append(f"group {group_name} node without int id: {node!r}")
        continue
      if node_id in node_by_id:
        issues.append(f"duplicate node id {node_id} across groups")
      node_by_id[node_id] = node
      if node.get("group") != int(group_name):
        issues.append(f"node {node_id} group {node.get('group')!r} does not match group key {group_name}")
      if not node.get("name") or not node.get("class"):
        issues.append(f"node {node_id} missing name or class")

  if not isinstance(swap_table, dict):
    issues.append("mmv_swap_table missing")
  else:
    for swap_key, entry in swap_table.items():
      if not isinstance(entry, dict):
        issues.append(f"swap entry {swap_key} is not a dict")
        continue
      node_id = entry.get("node_id")
      if not isinstance(node_id, int):
        issues.append(f"swap entry {swap_key} missing int node_id")
        continue
      node = node_by_id.get(node_id)
      if node is None:
        issues.append(f"swap node {node_id} not present in per_group node lists")
        continue
      node_class = node.get("class")
      variant = entry.get("variant")
      if node_class not in MMV_CLASSES:
        issues.append(f"swap node {node_id} has non-MMV class {node_class!r}")
      if variant != VARIANT_FOR_CLASS.get(node_class):
        issues.append(f"swap node {node_id} variant {variant!r} does not match class {node_class!r}")
      block_layout = entry.get("block_layout") or {}
      if block_layout.get("size_bytes") not in (144, 210):
        issues.append(f"swap node {node_id} unexpected block size {block_layout.get('size_bytes')!r}")
      abi = entry.get("abi_arg_template") or {}
      if abi.get("arg_count") != EXPECTED_ARG_COUNT:
        issues.append(f"swap node {node_id} ABI arg_count {abi.get('arg_count')!r} != {EXPECTED_ARG_COUNT}")
      if (abi.get("fusion_args") or {}).get("size_bytes") != FUSION_ARGS_SIZE:
        issues.append(f"swap node {node_id} fusion args size != {FUSION_ARGS_SIZE}")
      if (abi.get("src1") or {}).get("row_padding") != SRC1_ROW_PADDING:
        issues.append(f"swap node {node_id} src1 row padding != {SRC1_ROW_PADDING}")

  if not isinstance(edges, list):
    issues.append("preserved_physical_edges missing")
  else:
    if len(edges) != config.get("edge_count"):
      issues.append(f"preserved edge count {len(edges)} != edge_count {config.get('edge_count')}")
    valid_kinds = ("RAW", "WAR", "WAW")
    for edge in edges:
      frm, to, kind = edge.get("from"), edge.get("to"), edge.get("kind")
      if not isinstance(frm, int) or not isinstance(to, int) or frm not in node_by_id or to not in node_by_id:
        issues.append(f"edge {frm!r}->{to!r} references an unknown node")
      if kind not in valid_kinds:
        issues.append(f"edge {frm!r}->{to!r} has unknown kind {kind!r}")
    if dag_hash_value and isinstance(dag_hash_value, str) and dag_hash(edges) != dag_hash_value:
      issues.append("recomputed source_dag_hash does not match preserved_physical_edges")

  if len(node_by_id) != config.get("node_count"):
    issues.append(f"node_count {config.get('node_count')} != distinct nodes {len(node_by_id)}")
  if len(per_group) != config.get("group_count"):
    issues.append(f"group_count {config.get('group_count')} != per_group entries {len(per_group)}")
  if not isinstance(swap_table, dict) or config.get("predicted_swap_count") != len(swap_table):
    issues.append("predicted_swap_count does not match mmv_swap_table size")
  return {"valid": not issues, "issues": issues}


def write_json(path: Path, payload: dict[str, object]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
  try:
    with os.fdopen(descriptor, "w") as handle:
      json.dump(payload, handle, indent=1, sort_keys=True)
      handle.write("\n")
      handle.flush()
      os.fsync(handle.fileno())
    os.replace(temporary, path)
  except BaseException:
    try:
      os.unlink(temporary)
    except OSError:
      pass
    raise


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="CPU-only L2 constant-DAG llama-kernel oracle config builder")
  parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE, help="aligned capture manifest JSON")
  parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE, help="llama quantized-oracle ABI report JSON")
  parser.add_argument("--out", type=Path, help="write the L2 config JSON here")
  parser.add_argument("--synthetic", action="store_true", help="build from the hermetic 12-node fixture DAG")
  parser.add_argument("--validate", type=Path, metavar="CONFIG", help="validate an L2 config and exit")
  args = parser.parse_args(argv)

  if args.validate is not None:
    config = json.loads(args.validate.read_text())
    result = validate_config(config)
    print(json.dumps(result, sort_keys=True, indent=1))
    return 0 if result["valid"] else 1

  if args.synthetic:
    config = build_config(
      build_synthetic_capture(),
      FIXTURE_ORACLE,
      source_capture="synthetic-fixture-12n-2g-4mmv",
      oracle_report="embedded-llama-quantized-oracle-abi",
    )
  else:
    if not args.capture.is_file():
      raise FileNotFoundError(f"capture manifest not found: {args.capture}")
    if not args.oracle.is_file():
      raise FileNotFoundError(f"oracle report not found: {args.oracle}")
    config = build_config(
      json.loads(args.capture.read_text()),
      json.loads(args.oracle.read_text()),
      source_capture=str(args.capture.resolve()),
      source_capture_sha256=sha256_file(args.capture),
      oracle_report=str(args.oracle.resolve()),
      oracle_report_sha256=sha256_file(args.oracle),
    )

  result = validate_config(config)
  if not result["valid"]:
    print(json.dumps(result, sort_keys=True, indent=1), file=sys.stderr)
    return 1
  if args.out is not None:
    write_json(args.out, config)
  else:
    print(json.dumps(config, indent=1, sort_keys=True))
  print(
    f"predicted_swap_count={config['predicted_swap_count']} "
    f"node_count={config['node_count']} edge_count={config['edge_count']} "
    f"group_count={config['group_count']} source_dag_hash={config['source_dag_hash']}",
    file=sys.stderr,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
