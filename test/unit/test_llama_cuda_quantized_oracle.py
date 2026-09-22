"""Hermetic CPU-only tests for the Q4_K/Q6_K compiled-kernel oracle.

The oracle (scratchpad/llama_cuda_quantized_oracle.py) prepares the llama.cpp
quantized GEMV kernels for a future tinygrad live launch.  Its inspect path is
pure subprocess + ctypes and must never initialize CUDA.  These tests cover the
JSON schema contract, the cubin/symbol parsing on synthetic fixture text, and
the real library inspection/dump paths when libggml-cuda is present.
"""
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
ORACLE_PATH = REPO / "scratchpad" / "llama_cuda_quantized_oracle.py"


def _load_oracle():
  spec = importlib.util.spec_from_file_location("llama_cuda_quantized_oracle", ORACLE_PATH)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


@pytest.fixture(scope="module")
def oracle():
  return _load_oracle()


def _fixture_report():
  """Minimal but schema-valid report dict (no filesystem/library needed)."""
  return {
    "schema": "tinygrad.llama_cuda_quantized_oracle.v1",
    "mode": "inspect",
    "library": "/nonexistent/libggml-cuda.so.0.14.0",
    "library_sha256": "0" * 64,
    "sources": [{"path": "/nonexistent", "sha256": "0" * 64, "role": "primary"}],
    "cuobjdump": {"path": None, "present": False},
    "cubins": [],
    "cubin_count": 0,
    "kernels": {
      "Q4_K": {
        "quant_kind": "Q4_K", "ggml_type": 12,
        "block_layout": {"size_bytes": 144}, "abi": {},
        "cubin": {"ordinal": 1, "name": "x.cubin"},
        "local_kernel_symbols": ["_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEv"],
        "canonical_entry": "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEv",
      },
      "Q6_K": {
        "quant_kind": "Q6_K", "ggml_type": 14,
        "block_layout": {"size_bytes": 210}, "abi": {},
        "cubin": {"ordinal": 1, "name": "x.cubin"},
        "local_kernel_symbols": ["_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEv"],
        "canonical_entry": "_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEv",
      },
    },
    "fusion_args": {"struct": "ggml_cuda_mm_fusion_args_device", "size_bytes": 32, "fields": [], "glu_op_enum": {}},
    "launch_wrapper": {"host_dispatch_symbol": "", "dynamic_symbol_present": False, "kernel_argument_order": []},
    "logical_tensor_mapping": {"case": "MUL_MAT without ids", "params": []},
    "binary_reuse_candidate": False,
    "reasons": [],
  }


LIST_ELF_FIXTURE = """\
ELF file    1: libggml-cuda.so.0.14.1.sm_120a.cubin
ELF file    2: libggml-cuda.so.0.14.2.sm_120a.cubin
ELF file    3: libggml-cuda.so.0.14.3.sm_120a.cubin
"""

SYMBOL_DUMP_FIXTURE = """\
Fatbin ptx code:
================
arch = sm_120a

Fatbin elf code:
================
arch = sm_120a
code version = [1,8]
host = linux
compile_size = 64bit
compressed

symbols:
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z7acc_f32PKfS0_Pflllllllll
STT_OBJECT       STB_LOCAL  STV_DEFAULT    iq2s_grid

Fatbin ptx code:
================
arch = sm_120a

Fatbin elf code:
================
arch = sm_120a
code version = [1,8]
host = linux
compile_size = 64bit
compressed

symbols:
STT_OBJECT       STB_LOCAL  STV_DEFAULT    kvalues_mxfp4

Fatbin ptx code:
================
arch = sm_120a

Fatbin elf code:
================
arch = sm_120a
code version = [1,8]
host = linux
compile_size = 64bit
compressed

symbols:
STT_OBJECT       STB_LOCAL  STV_DEFAULT    kvalues_mxfp4
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z13mul_mat_vec_qIL9ggml_type12ELi8ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z13mul_mat_vec_qIL9ggml_type14ELi8ELb0ELb0EEvPKvS2_PKi31ggml_cuda_mm_fusion_args_devicePfj5uint3jjjS7_jjjS7_jjjj
STT_FUNC         STB_LOCAL  STO_ENTRY      _Z17mul_mat_vec_q_moeIL9ggml_type12ELi2EEvPKvS2_PKiPfj5uint3jjjjjjjjj
"""


def test_synthetic_cubin_list_and_symbol_block_parsing(oracle):
  cubins = oracle.parse_cubin_list(LIST_ELF_FIXTURE)
  assert [int(c["ordinal"]) for c in cubins] == [1, 2, 3]
  assert cubins[-1]["name"] == "libggml-cuda.so.0.14.3.sm_120a.cubin"
  blocks = oracle.parse_symbol_blocks(SYMBOL_DUMP_FIXTURE)
  assert [int(b["ordinal"]) for b in blocks] == [1, 2, 3]
  groups = oracle.classify_kernels(blocks, cubins)
  assert set(groups) == {"Q4_K", "Q6_K"}
  # STB_LOCAL symbols inside an embedded cubin, never in the dynamic table.
  assert groups["Q4_K"]["cubin"]["name"] == "libggml-cuda.so.0.14.3.sm_120a.cubin"
  assert groups["Q4_K"]["cubin"]["ordinal"] == 3
  assert groups["Q4_K"]["canonical_entry"].startswith("_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0ELb0E")
  assert groups["Q6_K"]["canonical_entry"].startswith("_Z13mul_mat_vec_qIL9ggml_type14ELi1ELb0ELb0E")
  assert all("mul_mat_vec_q" in name for name in groups["Q4_K"]["local_kernel_symbols"])
  assert all("ggml_type14" in name for name in groups["Q6_K"]["local_kernel_symbols"])


def test_synthetic_report_schema_validation_and_json_roundtrip(oracle):
  report = oracle.validate_report(_fixture_report())
  assert report["schema"] == "tinygrad.llama_cuda_quantized_oracle.v1"
  assert report["kernels"]["Q4_K"]["block_layout"]["size_bytes"] == 144
  assert report["kernels"]["Q6_K"]["block_layout"]["size_bytes"] == 210
  assert json.loads(json.dumps(report, sort_keys=True)) == report
  broken = dict(_fixture_report())
  del broken["kernels"]
  with pytest.raises(ValueError, match="missing required fields"):
    oracle.validate_report(broken)
  broken_kernel = _fixture_report()
  del broken_kernel["kernels"]["Q6_K"]["canonical_entry"]
  with pytest.raises(ValueError, match="Q6_K"):
    oracle.validate_report(broken_kernel)


def test_synthetic_block_layout_offsets(oracle):
  q4 = oracle.block_layout(oracle.BLOCK_Q4_K_FIELDS)
  assert [(f["name"], f["offset"]) for f in q4["fields"]] == [("d", 0), ("dmin", 2), ("scales", 4), ("qs", 16)]
  assert q4["size_bytes"] == 144
  q6 = oracle.block_layout(oracle.BLOCK_Q6_K_FIELDS)
  assert [(f["name"], f["offset"]) for f in q6["fields"]] == [("ql", 0), ("qh", 128), ("scales", 192), ("d", 208)]
  assert q6["size_bytes"] == 210


def test_abi_constants_match_llama_source(oracle):
  assert oracle.QK_K == 256 and oracle.QI4_K == 32 and oracle.QI6_K == 32
  assert oracle.VDR_MMVQ == {oracle.GGML_TYPE_Q4_K: 2, oracle.GGML_TYPE_Q6_K: 1}
  assert oracle.ctypes.sizeof(oracle.FusionArgs) == 32
  assert len(oracle.KERNEL_ARGUMENTS) == 19


def _real_artifacts(oracle):
  try:
    library, source = oracle.resolve_artifacts()
  except FileNotFoundError as exc:
    pytest.skip(str(exc))
  return library, source


def test_inspect_real_library_finds_q4k_q6k_kernels(oracle):
  library, source = _real_artifacts(oracle)
  report = oracle.inspect(library, source)
  assert report["schema"] == "tinygrad.llama_cuda_quantized_oracle.v1"
  assert report["mode"] == "inspect"
  assert report["library_sha256"]
  assert report["cubin_count"] > 0
  for kind in ("Q4_K", "Q6_K"):
    kernel = report["kernels"][kind]
    assert isinstance(kernel["cubin"]["ordinal"], int)
    assert kernel["cubin"]["name"].endswith(".cubin")
    assert len(kernel["local_kernel_symbols"]) > 0
    assert kernel["canonical_entry"]
    assert kernel["canonical_entry"] in kernel["local_kernel_symbols"]
    assert f"ggml_type{kernel['ggml_type']}" in kernel["canonical_entry"]
    assert all(name.startswith("_Z13mul_mat_vec_q") for name in kernel["local_kernel_symbols"])
  # This build keeps both quant kinds in one mmvq.cu cubin.
  assert report["kernels"]["Q4_K"]["cubin"]["name"] == report["kernels"]["Q6_K"]["cubin"]["name"]


def test_inspect_real_library_json_and_exit_code_contract(oracle):
  library, source = _real_artifacts(oracle)
  report = oracle.inspect(library, source)
  assert json.loads(json.dumps(report, sort_keys=True)) == report
  assert report["fusion_args"]["size_bytes"] == 32
  assert report["launch_wrapper"]["kernel_argument_order"][-1]["name"] == "ids_stride"
  assert report["binary_reuse_candidate"] is True
  assert any("cubin libggml-cuda.so.0.14." in reason for reason in report["reasons"])
  assert oracle.exit_code(report, "inspect") == 0


def test_dump_real_library_extracts_cubin_and_nm_verifies_symbols(oracle, tmp_path):
  if oracle.find_cuobjdump() is None:
    pytest.skip("cuobjdump not present")
  library, source = _real_artifacts(oracle)
  report = oracle.inspect(library, source)
  dumped = oracle.dump_artifacts(report, tmp_path)
  assert dumped["mode"] == "dump"
  assert (tmp_path / "llama_cuda_quantized_oracle_v1.json").is_file()
  for entry in dumped["dump"]["cubins"]:
    cubin_path = Path(entry["path"])
    assert cubin_path.is_file()
    assert entry["sha256"]
    assert entry["verified"]["missing"] == []
    assert entry["verified"]["tool"] == "nm"
    assert len(entry["verified"]["found_local"]) == entry["expected_symbols"]
