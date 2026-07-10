from extra.qk.mmq_llama_research_source import (
  VENDORED_MMQ_CUH_SHA256, llama_mmq_research_source_manifest,
)


def test_llama_mmq_research_source_is_quarantined_and_hash_pinned():
  manifest = llama_mmq_research_source_manifest()

  assert manifest["schema"] == "llama-mmq-research-source.v1"
  assert manifest["status"] == "research_source_copy"
  assert manifest["production_dispatch_changed"] is False
  assert manifest["selectable_backend"] is False
  assert manifest["source_license"] == "MIT"
  assert manifest["vendored_sha256"] == VENDORED_MMQ_CUH_SHA256
  assert manifest["expected_sha256"] == VENDORED_MMQ_CUH_SHA256
  assert manifest["matches_source_clone"] is True
  assert "mul_mat_q_process_tile" in manifest["anchors"]
  assert "mmq_write_back_mma" in manifest["anchors"]
