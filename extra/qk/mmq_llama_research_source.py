"""Research-only vendored source metadata for llama.cpp MMQ.

This keeps a stable copy of the upstream HIP/CUDA MMQ source for reduction
work. It is not imported by production dispatch and is not a selectable atom.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = ROOT / "extra/qk/research/llama_mmq"
VENDORED_MMQ_CUH = RESEARCH_DIR / "mmq.cuh"
VENDORED_LICENSE = RESEARCH_DIR / "LICENSE.llama.cpp"
SOURCE_CLONE_MMQ_CUH = Path("/home/ubuntu/env/llama.cpp/ggml/src/ggml-cuda/mmq.cuh")
SOURCE_CLONE_COMMIT = "ac4cddeb0dbd778f650bf568f6f08344a06abe3a"
VENDORED_MMQ_CUH_SHA256 = "6d153a9d6f293a4ff5f11e7886a48bf765b21d74075d73b2097a2b2a9149de6f"


def _sha256(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def llama_mmq_research_source_manifest() -> dict[str, Any]:
  vendored_hash = _sha256(VENDORED_MMQ_CUH)
  clone_hash = _sha256(SOURCE_CLONE_MMQ_CUH) if SOURCE_CLONE_MMQ_CUH.exists() else None
  return {
    "schema": "llama-mmq-research-source.v1",
    "status": "research_source_copy",
    "production_dispatch_changed": False,
    "selectable_backend": False,
    "source_project": "llama.cpp/ggml",
    "source_license": "MIT",
    "source_clone_commit": SOURCE_CLONE_COMMIT,
    "source_clone_path": str(SOURCE_CLONE_MMQ_CUH),
    "vendored_path": str(VENDORED_MMQ_CUH.relative_to(ROOT)),
    "vendored_license_path": str(VENDORED_LICENSE.relative_to(ROOT)),
    "vendored_sha256": vendored_hash,
    "expected_sha256": VENDORED_MMQ_CUH_SHA256,
    "source_clone_sha256": clone_hash,
    "matches_source_clone": clone_hash == vendored_hash if clone_hash is not None else None,
    "anchors": [
      "MMQ_NWARPS",
      "block_q8_1_mmq",
      "load_tiles_q4_K",
      "vec_dot_q4_K_q8_1_impl_mmq",
      "mmq_write_back_mma",
      "mul_mat_q_process_tile",
      "launch_mul_mat_q",
      "mul_mat_q_case",
    ],
  }
