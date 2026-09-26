"""tinygrad keeps only the exporter half of NCU measurement: the cubin + launch-spec record BoltBeam replays."""
import json, subprocess, sys
from pathlib import Path

import pytest

import extra.llm_research.decode.nv_cubin_capture as capture

ROOT = Path(__file__).resolve().parents[2]


def test_write_capture_emits_the_producer_contract(tmp_path, monkeypatch):
  monkeypatch.setattr(capture, "CAPTURED", {"E_2_137": {
    "name": "E_2_137", "cubin": b"\x7fELF", "cubin_sha256": capture._sha(b"\x7fELF"), "regs_usage": 128, "shmem_usage": 66560,
    "lcmem_usage": 0, "max_threads": 256, "shared_mem": 65536,
    "calls": [{"global_size": [137, 2, 1], "local_size": [32, 2, 4], "vals": [], "n_bufs": 3,
               "buf_meta": [{"va_addr": 1, "size": 64}, {"va_addr": 2, "size": 64}, {"va_addr": 3, "size": 64}]}]}})
  doc = capture.write_capture(tmp_path / "cap.json", ["E_"], {"script": "x.py", "script_args": []})
  [row] = doc["captured"]
  assert doc["schema"] == "tinygrad.nv_cubin_capture.v1" and doc["verdict"] == "CAPTURED"
  assert row["shared_mem"] == 65536 and Path(row["cubin_path"]).read_bytes() == b"\x7fELF"
  assert json.loads((tmp_path / "cap.json").read_text()) == doc
  try:
    from extra.llm_research.boltbeam_checkout import require_boltbeam
    require_boltbeam("boltbeam.collectors.cubin_launch")
  except ImportError as exc: pytest.skip(f"BoltBeam checkout unavailable: {exc}")
  from boltbeam.collectors.cubin_launch import launch_spec_from_capture   # the consumer accepts what we emit
  spec = launch_spec_from_capture(doc, "E_2_137")
  assert spec["grid"] == [137, 2, 1] and spec["shared_mem"] == 65536


def test_the_launcher_is_a_thin_caller_of_boltbeam():
  src = (ROOT / "extra/llm_research/decode/nv_cubin_ncu_launcher.py").read_text()
  assert "boltbeam.collectors.cubin_launch" in src and "cuLaunchKernel(" not in src
  audit = ROOT / "docs/nemotron-vllm-parity/bench/audit"
  assert not (audit / "ncu_kernel_counters.py").exists()
  assert all("boltbeam.cli ncu-collect" in (audit / f).read_text() for f in ("oncu.sh", "vncu.sh"))
