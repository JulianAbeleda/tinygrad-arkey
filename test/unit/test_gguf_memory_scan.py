from pathlib import Path

import pytest

from tinygrad.llm.gguf_memory_scan import CandidateWorkspace, RuntimeGeometry, scan_selected_gguf_memory, selected_gguf_backing_bytes
from tinygrad.llm.memory_ledger import AllocationKind, ScannedMemoryBudget, AllocationProvenance


def _metadata(_path):
  # GGUF dimensions are stored least-significant first and become reversed runtime shapes.
  return {"general.name": "fixture"}, {"data_start": 128, "tensor_infos": [
    ("q4", (256, 2), 12, 0),       # 2 * 144 bytes
    ("f16", (4, 8), 1, 320),       # 64 bytes; final span includes trailing padding
  ]}


def _geometry(**overrides):
  values = dict(num_blocks=2, n_kv_heads=3, head_dim=4, max_context=5, prefill_ubatch=6,
                batch_size=1, kv_element_bytes=2, runtime_persistent_bytes=11,
                peak_prefill_activation_bytes=12, peak_prefill_output_bytes=13, peak_prefill_scratch_bytes=14)
  values.update(overrides)
  return RuntimeGeometry(**values)


def test_path_loader_backing_uses_one_allocator_aligned_whole_file(tmp_path):
  path = tmp_path/"selected.gguf"
  path.write_bytes(b"x" * 129)
  assert selected_gguf_backing_bytes(path, 64) == 192
  assert selected_gguf_backing_bytes(path, None) is None


def test_builds_tensor_spans_and_all_runtime_classes_from_injected_metadata(tmp_path):
  scan = scan_selected_gguf_memory(tmp_path/"chosen.gguf", _geometry(), [CandidateWorkspace("direct", 15, "measured")],
    allocation_alignment=64, resident_copies=1, metadata_loader=_metadata, file_size=512)
  assert [(x.name, x.shape, x.payload_bytes, x.span_bytes) for x in scan.tensor_spans] == [
    ("q4", (2, 256), 288, 320), ("f16", (8, 4), 64, 64)]
  by_name = {x.name: x for x in scan.ledger.allocations}
  assert by_name["tensor:q4"].bytes == 320
  assert by_name["tensor:f16"].bytes == 64
  assert by_name["kv_cache"].bytes == 2*2*1*3*5*4*2
  assert {x.kind for x in scan.ledger.allocations} == set(AllocationKind)
  assert f"selected path={tmp_path/'chosen.gguf'}" in by_name["tensor:q4"].provenance.detail


def test_unknown_alignment_copy_and_runtime_facts_fail_closed(tmp_path):
  ledger = scan_selected_gguf_memory(tmp_path/"chosen.gguf", _geometry(batch_size=None, runtime_persistent_bytes=None),
    [CandidateWorkspace("direct", None, "backend did not report workspace")], metadata_loader=_metadata, file_size=512).ledger
  tensors = [x for x in ledger.allocations if x.kind is AllocationKind.GGUF_TENSOR]
  assert all(x.alignment is None and x.copies is None and x.bytes is None for x in tensors)
  decision = ledger.decide(ScannedMemoryBudget(10_000, 0, AllocationProvenance("test", "known budget")), "direct")
  assert not decision.admitted and decision.peak_bytes is None
  assert "unknown allocation bytes: tensor:q4" in decision.reasons
  assert "unknown allocation bytes: kv_cache" in decision.reasons
  assert "unknown allocation bytes: runtime_persistent" in decision.reasons
  assert "unknown allocation bytes: workspace:direct" in decision.reasons


def test_quantized_payload_requires_complete_physical_blocks(tmp_path):
  def bad(_path): return {}, {"data_start": 64, "tensor_infos": [("bad", (255,), 12, 0)]}
  scan = scan_selected_gguf_memory(tmp_path/"bad.gguf", _geometry(), [CandidateWorkspace("x", 0, "none")],
                                  allocation_alignment=32, resident_copies=1, metadata_loader=bad, file_size=300)
  assert scan.tensor_spans[0].payload_bytes is None
  assert scan.ledger.allocations[0].bytes is None


def test_payload_cannot_exceed_offset_derived_span(tmp_path):
  def bad(_path): return {}, {"data_start": 64, "tensor_infos": [("a", (32,), 1, 0), ("b", (1,), 1, 16)]}
  with pytest.raises(ValueError, match="payload .* exceeds its file span"):
    scan_selected_gguf_memory(tmp_path/"bad.gguf", _geometry(), [], metadata_loader=bad, file_size=100)


def test_kv_quant_scale_geometry_is_explicit():
  geo = _geometry(kv_element_bytes=1, kv_scales_per_token=2, kv_scale_element_bytes=2)
  assert geo.kv_bytes == 2*2*1*3*5*4 + 2*1*5*2*2
  assert _geometry(kv_element_bytes=1, kv_scales_per_token=2, kv_scale_element_bytes=None).kv_bytes is None


def test_supplied_metadata_is_not_rescanned(tmp_path):
  def forbidden(_path): raise AssertionError("selected GGUF metadata was reopened")
  scan = scan_selected_gguf_memory(tmp_path/"chosen.gguf", _geometry(), [CandidateWorkspace("direct", 0, "proven")],
    allocation_alignment=64, resident_copies=1, metadata=_metadata(None), metadata_loader=forbidden, file_size=512)
  assert len(scan.tensor_spans) == 2


def test_real_local_gguf_metadata_read_only_when_available():
  paths = sorted((*Path("/home/ubuntu/env/llama.cpp/models").glob("*.gguf"), *Path("/home/ubuntu/models").glob("*.gguf")),
                 key=lambda p: p.stat().st_size)
  if not paths: pytest.skip("no local GGUF metadata fixture")
  scans = (scan_selected_gguf_memory(path, _geometry(), [CandidateWorkspace("metadata-only", None, "not measured")])
           for path in paths)
  scan = next((item for item in scans if item.tensor_spans), None)
  if scan is None: pytest.skip("local GGUF files contain no tensors")
  assert scan.model_path in paths
  assert scan.tensor_spans
  assert all(x.absolute_offset >= 0 and x.relative_offset >= 0 for x in scan.tensor_spans)
  assert all(x.provenance.source == "gguf_load_metadata tensor table" for x in scan.ledger.allocations
             if x.kind is AllocationKind.GGUF_TENSOR)
