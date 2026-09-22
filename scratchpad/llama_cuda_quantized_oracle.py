#!/usr/bin/env python3
"""CPU-only artifact/ABI preparation for llama.cpp Q4_K/Q6_K quantized GEMV kernels.

The float launcher probe (llama_cuda_binary_kernel_probe.py) proves that a
compiled llama.cpp kernel can be launched directly from tinygrad.  The Q4_K and
Q6_K GEMV kernels (mul_mat_vec_q<...>) are different: they are STB_LOCAL inside
embedded cubins of libggml-cuda, so the dynamic symbol table only exposes the
host dispatcher ggml_cuda_mul_mat_vec_q.  A live launch needs the embedded
cubin extracted and the local kernel entry name plus parameter ABI pinned down.

This tool is deliberately CPU-only.  --inspect-only (the default) enumerates
the embedded cubins and their local symbol tables with cuobjdump, resolves the
Q4_K/Q6_K mul_mat_vec_q entries, and emits a stable JSON report; it never
initializes CUDA.  --dump additionally writes the extracted cubin(s) and the
machine-readable ABI report JSON, re-verifying the local entry names with nm on
the extracted file.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


SCHEMA = "tinygrad.llama_cuda_quantized_oracle.v1"
DEFAULT_LLAMA_ROOT = Path("/home/ubuntu/env/llama.cpp")
DEFAULT_CUOBJDUMP = Path("/usr/local/cuda-13.2/bin/cuobjdump")
GGML_CUDA = "ggml/src/ggml-cuda"

KERNEL_PREFIX = "_Z13mul_mat_vec_q"
MOE_KERNEL_PREFIX = "_Z17mul_mat_vec_q_moe"
DISPATCHER_SYMBOL = (
  "_Z23ggml_cuda_mul_mat_vec_qR25ggml_backend_cuda_contextPK11ggml_tensor"
  "S3_S3_PS1_PK29ggml_cuda_mm_fusion_args_host"
)
QUANTIZE_Q8_1_SYMBOL = (
  "_Z22quantize_row_q8_1_cudaPKfPKiPv9ggml_typellllllllP11CUstream_st"
)

# Constants lifted from ggml-common.h / ggml-cuda headers of the llama.cpp tree
# under /home/ubuntu/env/llama.cpp (see git commit ac4cddeb0, tagged b4818).
QK_K = 256
K_SCALE_SIZE = 12
QR4_K = 2
QI4_K = QK_K // (4 * QR4_K)
QR6_K = 2
QI6_K = QK_K // (4 * QR6_K)
QK8_1 = 32
MATRIX_ROW_PADDING = 512
MMVQ_MAX_BATCH_SIZE = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q6_K = 14

# vecdotq.cuh: VDR_Q4_K_Q8_1_MMVQ / VDR_Q6_K_Q8_1_MMVQ.
VDR_MMVQ = {GGML_TYPE_Q4_K: 2, GGML_TYPE_Q6_K: 1}

# mmvq.cu mul_mat_vec_q<type, ncols_dst, has_fusion, small_k> argument order,
# exactly as declared (const void*, const void*, const int32_t*, fusion,
# float*, then scalars and fastdiv-encoded uint3 triples).
KERNEL_ARGUMENTS = [
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

# C struct field order from ggml-common.h.  Offsets are computed, not copied.
BLOCK_Q4_K_FIELDS = [
  ("d", "ggml_half", 2, "super-block scale for quantized scales"),
  ("dmin", "ggml_half", 2, "super-block scale for quantized mins"),
  ("scales", "uint8_t[%d]" % K_SCALE_SIZE, K_SCALE_SIZE, "scales and mins, quantized with 6 bits"),
  ("qs", "uint8_t[%d]" % (QK_K // 2), QK_K // 2, "4-bit quants"),
]
BLOCK_Q6_K_FIELDS = [
  ("ql", "uint8_t[%d]" % (QK_K // 2), QK_K // 2, "quants, lower 4 bits"),
  ("qh", "uint8_t[%d]" % (QK_K // 4), QK_K // 4, "quants, upper 2 bits"),
  ("scales", "int8_t[%d]" % (QK_K // 16), QK_K // 16, "scales, quantized with 8 bits"),
  ("d", "ggml_half", 2, "super-block scale"),
]

GLU_OPS = [
  "GGML_GLU_OP_REGLU", "GGML_GLU_OP_GEGLU", "GGML_GLU_OP_SWIGLU",
  "GGML_GLU_OP_SWIGLU_OAI", "GGML_GLU_OP_GEGLU_ERF", "GGML_GLU_OP_GEGLU_QUICK",
]


class FusionArgs(ctypes.Structure):
  # Mirrors ggml_cuda_mm_fusion_args_device in ggml-cuda/common.cuh.
  _fields_ = [
    ("x_bias", ctypes.c_void_p),
    ("gate", ctypes.c_void_p),
    ("gate_bias", ctypes.c_void_p),
    ("glu_op", ctypes.c_int),
  ]


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_artifacts(root: Path = DEFAULT_LLAMA_ROOT) -> tuple[Path, Path]:
  library = Path(os.environ.get("LLAMA_CUDA_LIB", root / "build-cuda/bin/libggml-cuda.so.0.14.0"))
  source = root / f"{GGML_CUDA}/ggml-cuda.cu"
  if not library.is_file():
    raise FileNotFoundError(f"llama CUDA library not found: {library}")
  if not source.is_file():
    raise FileNotFoundError(f"llama CUDA source not found: {source}")
  return library.resolve(), source.resolve()


def find_cuobjdump(candidate: Path = DEFAULT_CUOBJDUMP) -> Path | None:
  return candidate if candidate.is_file() else None


def cuobjdump_version(cuobjdump: Path) -> str:
  result = subprocess.run([str(cuobjdump), "--version"], check=True, capture_output=True, text=True)
  return result.stdout.splitlines()[0].strip() if result.stdout.splitlines() else "unknown"


def parse_cubin_list(list_elf_text: str) -> list[dict[str, object]]:
  cubins: list[dict[str, object]] = []
  for line in list_elf_text.splitlines():
    line = line.strip()
    if not line.startswith("ELF file"):
      continue
    header, name = line.split(":", 1)
    ordinal = int(header.split()[-1])
    name = name.strip()
    cubins.append({"ordinal": ordinal, "name": name, "arch": name.rsplit(".", 1)[0].rsplit("_", 1)[-1]})
  return cubins


def parse_symbol_blocks(dump_text: str) -> list[dict[str, object]]:
  """Associate --dump-elf-symbols output with embedded cubin ordinals.

  cuobjdump prints one 'Fatbin elf code:' block per cubin, in the same order as
  --list-elf.  Each block's local symbols follow a 'symbols:' header.
  """
  blocks: list[dict[str, object]] = []
  current: dict[str, object] | None = None
  in_symbols = False
  for raw in dump_text.splitlines():
    line = raw.strip()
    if line.startswith("Fatbin elf code:"):
      current = {"ordinal": len(blocks) + 1, "symbols": []}
      blocks.append(current)
      in_symbols = False
    elif current is None:
      continue
    elif line == "symbols:":
      in_symbols = True
    elif in_symbols and line.startswith("STT_"):
      name = line.split()[-1]
      current["symbols"].append({"binding": "local" if "STB_LOCAL" in line else "other", "name": name})
    elif in_symbols and line.startswith("Fatbin"):
      in_symbols = False
  return blocks


def classify_kernels(blocks: list[dict[str, object]], cubins: list[dict[str, object]]) -> dict[str, object]:
  by_ordinal = {int(c["ordinal"]): c for c in cubins}
  groups: dict[str, object] = {}
  for block in blocks:
    ordinal = int(block["ordinal"])
    for entry in block["symbols"]:
      name = str(entry["name"])
      if "mul_mat_vec_q" not in name:
        continue
      for kind, ggml_type in (("Q4_K", GGML_TYPE_Q4_K), ("Q6_K", GGML_TYPE_Q6_K)):
        if f"ggml_type{ggml_type}E" not in name:
          continue
        group = groups.setdefault(kind, {"quant_kind": kind, "ggml_type": ggml_type, "cubin": by_ordinal[ordinal], "symbols": []})
        group["symbols"].append({"name": name, "moe": name.startswith(MOE_KERNEL_PREFIX)})
  for kind, group in groups.items():
    plain = [s["name"] for s in group["symbols"] if not s["moe"]]
    canonical = next(
      (n for n in plain if n.startswith(KERNEL_PREFIX) and f"ELi1ELb0ELb0EEv" in n), None
    )
    group["kernel_count"] = len(group["symbols"])
    group["plain_kernel_count"] = len(plain)
    group["canonical_entry"] = canonical
    group["local_kernel_symbols"] = sorted(set(plain))
  return groups


def block_layout(fields: list[tuple[str, str, int, str]]) -> dict[str, object]:
  offset = 0
  out: dict[str, object] = {"fields": [], "size_bytes": 0}
  for name, ctype, size, meaning in fields:
    out["fields"].append({"name": name, "ctype": ctype, "offset": offset, "size": size, "meaning": meaning})
    offset += size
  out["size_bytes"] = offset
  return out


def kernel_abi(kind: str, ggml_type: int) -> dict[str, object]:
  qk, qr, qi = QK_K, (QR4_K if kind == "Q4_K" else QR6_K), (QI4_K if kind == "Q4_K" else QI6_K)
  nwarps_generic = [
    {"ncols_dst": ncols_dst, "nwarps": 4 if ncols_dst <= 4 else 2}
    for ncols_dst in range(1, MMVQ_MAX_BATCH_SIZE + 1)
  ]
  return {
    "quant_kind": kind,
    "ggml_type": ggml_type,
    "qk": qk,
    "qr": qr,
    "qi": qi,
    "vdr": VDR_MMVQ[ggml_type],
    "vec_dot": f"vec_dot_{kind.lower()}_q8_1",
    "block_size_bytes": block_layout(BLOCK_Q4_K_FIELDS if kind == "Q4_K" else BLOCK_Q6_K_FIELDS)["size_bytes"],
    "blck_size": QK_K,
    "nwarps_generic": nwarps_generic,
    "rows_per_block_default": 1,
  }


def logical_tensor_mapping() -> list[dict[str, str]]:
  ts0 = "ggml_type_size(src0->type)"
  return [
    {"param": "ncols_x", "logical": "src0->ne[0]", "note": "contracted K dimension in elements"},
    {"param": "nrows_x", "logical": "src0->ne[1]", "note": "rows of src0; grid.x = ceil(nrows_x / rows_per_block)"},
    {"param": "stride_row_x", "logical": f"src0->nb[1] / {ts0}", "note": "src0 row stride in block units"},
    {"param": "nchannels_x", "logical": "src0->ne[2]", "note": "src0 channel count"},
    {"param": "stride_channel_x", "logical": f"src0->nb[2] / {ts0}", "note": "src0 channel stride in block units"},
    {"param": "nsamples_x", "logical": "src0->ne[3]", "note": "src0 sample count"},
    {"param": "stride_sample_x", "logical": f"src0->nb[3] / {ts0}", "note": "src0 sample stride in block units"},
    {"param": "ncols_dst", "logical": "dst->ne[1]", "note": "dst rows (must be <= MMVQ_MAX_BATCH_SIZE=8)"},
    {"param": "stride_col_dst", "logical": "dst->nb[1] / 4", "note": "dst row stride in floats"},
    {"param": "nchannels_y", "logical": "src1->ne[2]", "note": "src1 channel count"},
    {"param": "stride_col_y", "logical": "GGML_PAD(src1->ne[0], 512) / QK8_1", "note": "q8_1 blocks per quantized src1 row"},
    {"param": "stride_channel_y", "logical": "src1->ne[1] * stride_col_y", "note": "q8_1 blocks per src1 channel"},
    {"param": "stride_sample_y", "logical": "src1->ne[2] * stride_channel_y", "note": "q8_1 blocks per src1 sample"},
    {"param": "nsamples_dst", "logical": "dst->ne[3]", "note": "dst sample count"},
    {"param": "stride_sample_dst", "logical": "dst->nb[3] / 4", "note": "dst sample stride in floats"},
    {"param": "channel_ratio", "logical": "fastdiv(dst->ne[2] / src0->ne[2])", "note": "encoded uint3(mp,L,d), zero when ids is set"},
    {"param": "sample_ratio", "logical": "fastdiv(dst->ne[3] / src0->ne[3])", "note": "encoded uint3(mp,L,d)"},
    {"param": "ids_stride", "logical": "ids ? ids->nb[1] / 4 : 0", "note": "zero for plain MUL_MAT"},
  ]


def build_report(
  library: Path,
  source: Path,
  related_sources: list[tuple[Path, str]],
  cuobjdump_path: Path | None,
  cubins: list[dict[str, object]],
  groups: dict[str, object],
  mode: str,
  dynamic_symbols: dict[str, bool],
) -> dict[str, object]:
  layouts = {
    "Q4_K": block_layout(BLOCK_Q4_K_FIELDS),
    "Q6_K": block_layout(BLOCK_Q6_K_FIELDS),
  }
  kernels: dict[str, object] = {}
  for kind in ("Q4_K", "Q6_K"):
    group = groups.get(kind)
    if group is None:
      continue
    kernels[kind] = {
      "quant_kind": kind,
      "ggml_type": group["ggml_type"],
      "block_layout": layouts[kind],
      "abi": kernel_abi(kind, group["ggml_type"]),
      "cubin": group["cubin"],
      "kernel_count": group["kernel_count"],
      "local_kernel_symbols": group["local_kernel_symbols"],
      "canonical_entry": group["canonical_entry"],
      "template_params": {
        "kernel": "mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>",
        "type": f"ggml_type{group['ggml_type']}",
        "ncols_dst_variants": list(range(1, MMVQ_MAX_BATCH_SIZE + 1)),
        "has_fusion_variants": [False, True],
        "small_k_variants": [False, True],
      },
    }

  fusion_fields = [{"name": name, "ctype": "void *" if ctype == ctypes.c_void_p else "int32_t"}
                   for name, ctype in FusionArgs._fields_]
  fusion_size = ctypes.sizeof(FusionArgs)

  cuobjdump = {
    "path": str(cuobjdump_path) if cuobjdump_path else None,
    "present": cuobjdump_path is not None,
  }
  if cuobjdump_path:
    try:
      cuobjdump["version"] = cuobjdump_version(cuobjdump_path)
    except (OSError, subprocess.SubprocessError):
      cuobjdump["version"] = "unavailable"

  reasons: list[str] = []
  caveats: list[str] = []
  if cuobjdump_path is None:
    caveats.append("cuobjdump not found; cubin enumeration disabled")
  for kind in ("Q4_K", "Q6_K"):
    group = groups.get(kind)
    if group is None:
      caveats.append(f"{kind} mul_mat_vec_q kernel group not found in embedded cubins")
    else:
      reasons.append(f"{kind} mul_mat_vec_q kernel group found in embedded cubin {group['cubin']['name']} ({len(group['local_kernel_symbols'])} local entries)")
  if dynamic_symbols.get("dispatcher"):
    reasons.append("host dispatcher ggml_cuda_mul_mat_vec_q present in dynamic symbol table")
  else:
    caveats.append("host dispatcher ggml_cuda_mul_mat_vec_q missing from dynamic symbol table")
  if dynamic_symbols.get("quantize_q8_1"):
    reasons.append("src1 row quantizer quantize_row_q8_1_cuda present in dynamic symbol table")
  if fusion_size == 32:
    reasons.append(f"ggml_cuda_mm_fusion_args_device layout is {fusion_size} bytes on this host")
  caveats.append(
    "kernels are STB_LOCAL inside the embedded cubin; live launch requires loading the extracted cubin "
    "and resolving the local entry name, and src1 must be quantized to block_q8_1 with rows padded to 512"
  )

  binary_reuse_candidate = (
    cuobjdump_path is not None
    and all(kind in groups and groups[kind]["canonical_entry"] for kind in ("Q4_K", "Q6_K"))
    and fusion_size == 32
    and dynamic_symbols.get("dispatcher", False)
  )

  sources = [{"path": str(source), "sha256": sha256(source), "role": "primary (ggml-cuda.cu)"}]
  for path, role in related_sources:
    if path.is_file():
      sources.append({"path": str(path), "sha256": sha256(path), "role": role})

  return {
    "schema": SCHEMA,
    "mode": mode,
    "library": str(library),
    "library_sha256": sha256(library),
    "sources": sources,
    "cuobjdump": cuobjdump,
    "cubins": cubins,
    "cubin_count": len(cubins),
    "kernels": kernels,
    "fusion_args": {
      "struct": "ggml_cuda_mm_fusion_args_device",
      "size_bytes": fusion_size,
      "fields": fusion_fields,
      "glu_op_enum": {name: index for index, name in enumerate(GLU_OPS)},
    },
    "launch_wrapper": {
      "host_dispatch_symbol": DISPATCHER_SYMBOL,
      "dynamic_symbol_present": dynamic_symbols.get("dispatcher", False),
      "kernel_argument_order": [
        {"index": i + 1, "name": name, "ctype": ctype, "note": note}
        for i, (name, ctype, note) in enumerate(KERNEL_ARGUMENTS)
      ],
      "grid": "dim3(ceil(nrows_x / rows_per_block), nchannels_dst, nsamples_dst)",
      "block": "dim3(warp_size, nwarps, 1)",
      "notes": "vdr from get_vdr_mmvq(type); this tree has no qk_peel template "
               "parameter, the equivalent is the small_k flag plus vdr",
    },
    "logical_tensor_mapping": {
      "case": "MUL_MAT without ids (direct-launch target)",
      "params": logical_tensor_mapping(),
    },
    "binary_reuse_candidate": binary_reuse_candidate,
    "reasons": reasons,
    "caveats": caveats,
  }


def validate_report(report: dict[str, object]) -> dict[str, object]:
  required = {
    "schema", "mode", "library", "library_sha256", "sources", "cuobjdump",
    "cubins", "cubin_count", "kernels", "fusion_args", "launch_wrapper",
    "logical_tensor_mapping", "binary_reuse_candidate", "reasons",
  }
  missing = sorted(required - set(report))
  if missing:
    raise ValueError(f"report missing required fields: {missing}")
  if report["schema"] != SCHEMA:
    raise ValueError(f"unexpected schema {report['schema']!r}")
  for kind in ("Q4_K", "Q6_K"):
    kernel = report["kernels"].get(kind)
    if kernel is None:
      raise ValueError(f"report missing kernel group {kind}")
    for field in ("quant_kind", "ggml_type", "block_layout", "abi", "cubin", "local_kernel_symbols", "canonical_entry"):
      if field not in kernel:
        raise ValueError(f"kernel group {kind} missing field {field}")
  if not isinstance(report["cubins"], list) or not isinstance(report["kernels"]["Q4_K"]["local_kernel_symbols"], list):
    raise ValueError("report schema violated for cubins or local_kernel_symbols")
  return report


def inspect(library: Path, source: Path, cuobjdump_path: Path | None = None) -> dict[str, object]:
  """CPU-only artifact/ABI inspection.  Never initializes CUDA."""
  lib = ctypes.CDLL(str(library), mode=ctypes.RTLD_LOCAL)
  dynamic_symbols = {
    "dispatcher": hasattr(lib, DISPATCHER_SYMBOL),
    "quantize_q8_1": hasattr(lib, QUANTIZE_Q8_1_SYMBOL),
  }
  if cuobjdump_path is None:
    cuobjdump_path = find_cuobjdump()
  cubins: list[dict[str, object]] = []
  groups: dict[str, object] = {}
  if cuobjdump_path is not None:
    list_elf = subprocess.run(
      [str(cuobjdump_path), "--list-elf", str(library)], check=True, capture_output=True, text=True
    )
    cubins = parse_cubin_list(list_elf.stdout)
    symbols = subprocess.run(
      [str(cuobjdump_path), "--all-fatbin", "--dump-elf-symbols", str(library)],
      check=True, capture_output=True, text=True,
    )
    groups = classify_kernels(parse_symbol_blocks(symbols.stdout), cubins)

  related_sources = [
    (source.parent / "mmvq.cu", "kernel + launch wrapper"),
    (source.parent / "mmvq.cuh", "mmvq interface"),
    (source.parent / "vecdotq.cuh", "vec_dot + VDR constants"),
    (source.parent / "common.cuh", "fusion args struct"),
    (source.parents[2] / "ggml-common.h", "block_q4_K/block_q6_K layouts"),
  ]
  report = build_report(library, source, related_sources, cuobjdump_path, cubins, groups, "inspect", dynamic_symbols)
  return validate_report(report)


def verify_cubin_symbols(cubin_path: Path, expected: list[str]) -> dict[str, object]:
  """Verify the STB_LOCAL kernel names inside an extracted cubin file."""
  found: set[str] = set()
  if _have_binary("nm"):
    result = subprocess.run(["nm", str(cubin_path)], check=True, capture_output=True, text=True)
    for line in result.stdout.splitlines():
      parts = line.split()
      if len(parts) >= 2 and parts[1] in ("t", "T"):
        found.add(parts[-1])
    tool = "nm"
  else:
    result = subprocess.run(
      [str(DEFAULT_CUOBJDUMP), "--dump-elf-symbols", str(cubin_path)], check=True, capture_output=True, text=True
    )
    for block in parse_symbol_blocks(result.stdout):
      found.update(str(s["name"]) for s in block["symbols"])
    tool = "cuobjdump"
  missing = [name for name in expected if name not in found]
  return {"tool": tool, "found_local": sorted(found & set(expected)), "missing": missing}


def _have_binary(name: str) -> bool:
  return subprocess.run(["sh", "-c", f"command -v {name} >/dev/null 2>&1"], check=False).returncode == 0


def dump_artifacts(report: dict[str, object], dump_dir: Path) -> dict[str, object]:
  dump_dir.mkdir(parents=True, exist_ok=True)
  written: dict[str, object] = {"cubins": [], "report": None}
  for kind in ("Q4_K", "Q6_K"):
    cubin_name = str(report["kernels"][kind]["cubin"]["name"])
    cubin_path = dump_dir / cubin_name
    if not cubin_path.is_file():
      subprocess.run(
        [str(DEFAULT_CUOBJDUMP), "-xelf", cubin_name, str(report["library"])],
        check=True, capture_output=True, text=True, cwd=str(dump_dir),
      )
      if not cubin_path.is_file():
        raise FileNotFoundError(f"cuobjdump did not produce {cubin_path}")
    expected = sorted(set([str(name) for name in report["kernels"][kind]["local_kernel_symbols"]] +
                         [str(report["kernels"][kind]["canonical_entry"])]))
    verification = verify_cubin_symbols(cubin_path, expected)
    if verification["missing"]:
      raise RuntimeError(f"extracted cubin {cubin_path} missing expected local symbols: {verification['missing']}")
    written["cubins"].append({
      "kind": kind, "path": str(cubin_path), "sha256": sha256(cubin_path),
      "expected_symbols": len(expected), "verified": verification,
    })

  report_path = dump_dir / "llama_cuda_quantized_oracle_v1.json"
  report = dict(report)
  report["mode"] = "dump"
  written["report"] = str(report_path)
  report["dump"] = written
  with report_path.open("w") as handle:
    json.dump(report, handle, sort_keys=True, indent=2)
  return validate_report(report)


def exit_code(report: dict[str, object], mode: str) -> int:
  return 0 if (report.get("binary_reuse_candidate") and mode in ("inspect", "dump")) else 1


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description="CPU-only Q4_K/Q6_K GEMV cubin/ABI oracle")
  parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
  parser.add_argument("--cuobjdump", type=Path, default=DEFAULT_CUOBJDUMP)
  parser.add_argument("--inspect-only", action="store_true", help="CPU-only inspection (the default)")
  parser.add_argument("--dump", action="store_true", help="extract cubins and write the ABI report JSON")
  parser.add_argument("--dump-dir", type=Path, default=Path(__file__).resolve().parent / "llama_cuda_quantized_oracle_dump")
  args = parser.parse_args(argv)

  library, source = resolve_artifacts(args.llama_root)
  cuobjdump_path = find_cuobjdump(args.cuobjdump)
  report = inspect(library, source, cuobjdump_path)
  mode = "dump" if args.dump else "inspect"
  if mode == "dump":
    report = dump_artifacts(report, args.dump_dir)
  print(json.dumps(report, sort_keys=True))
  return exit_code(report, mode)


if __name__ == "__main__":
  sys.exit(main())
