from pathlib import Path

import pytest

from extra.qk.mmq_llama_source_extract import (
  DEFAULT_LLAMA_ROOT, PINNED_LLAMA_REVISION, SOURCE_FILES, SourceContractError,
  extract_from_sources, extract_llama_checkout,
)


def _sources() -> dict[str, str]:
  return {name: (DEFAULT_LLAMA_ROOT / name).read_text() for name in SOURCE_FILES}


def _mutate(sources: dict[str, str], file: str, old: str, new: str) -> dict[str, str]:
  assert sources[file].count(old) == 1
  return {**sources, file: sources[file].replace(old, new)}


def test_extracts_pinned_source_contract_and_typed_proofs():
  got = extract_llama_checkout()
  assert got.revision == PINNED_LLAMA_REVISION
  assert (got.tile_rows, got.k_epoch, got.waves, got.wave_size, got.threads) == (128, 256, 8, 32, 256)
  assert (got.q8_record_values, got.q8_record_bytes, got.q4_block_values, got.q4_row_bytes) == (128, 144, 256, 144)
  assert got.q8_ds4_order == ("d0", "s0", "d1", "s1", "d2", "s2", "d3", "s3")
  assert got.lds_order == ("ids", "q8 tile_y", "q4 tile_x")
  assert got.barriers_per_epoch == 4 and got.wmma_calls == 2
  assert got.wmma_signed_controls == ((True, True, True), (True, True, True))
  assert got.grid.tile_rows.startswith("(args.nrows_x")
  assert got.proof_for("q4_correction").digest


@pytest.mark.parametrize(("file", "old", "new"), (
  (SOURCE_FILES[1], "#define MMQ_ITER_K             256", "#define MMQ_ITER_K             128"),
  (SOURCE_FILES[1], "return MMQ_Q8_1_DS_LAYOUT_DS4;", "return MMQ_Q8_1_DS_LAYOUT_D4;"),
  (SOURCE_FILES[1], "tile_y[l] = by0[l];", "tile_y[l] = by0[l + 1];"),
  (SOURCE_FILES[1], "vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);", "vec_dot(tile_x, tile_y, sum, 0);"),
  (SOURCE_FILES[2], "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a_vec[1], true, b_vec[1], acc[0], true)", "__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, a_vec[1], true, b_vec[1], acc[0], true)"),
  (SOURCE_FILES[3], "y[ib].ds4[iqs/32] = make_half2(d, sum);", "y[ib].ds4[iqs/32] = make_half2(sum, d);"),
))
def test_mutated_authoritative_snippets_fail_closed(file: str, old: str, new: str):
  sources = _sources()
  # Some anchors occur elsewhere; select a unique contract spelling where necessary.
  if old == "return MMQ_Q8_1_DS_LAYOUT_DS4;":
    old = "case GGML_TYPE_Q4_K:\n        case GGML_TYPE_Q5_K:\n            return MMQ_Q8_1_DS_LAYOUT_DS4;"
    new = old.replace("DS4", "D4")
  elif old == "tile_y[l] = by0[l];":
    assert sources[file].count(old) == 2
    sources = {**sources, file: sources[file].replace(old, new, 1)}
    with pytest.raises(SourceContractError): extract_from_sources(sources)
    return
  with pytest.raises(SourceContractError): extract_from_sources(_mutate(sources, file, old, new))


def test_missing_or_wrong_source_anchor_fails_closed():
  sources = _sources()
  sources.pop(SOURCE_FILES[2])
  with pytest.raises(SourceContractError, match="missing authoritative"):
    extract_from_sources(sources)


def test_wrong_checkout_revision_fails_before_parsing(tmp_path: Path):
  # A copied source tree is not authoritative merely because its bytes happen to match.
  with pytest.raises(SourceContractError, match="cannot establish llama.cpp revision"):
    extract_llama_checkout(tmp_path)
