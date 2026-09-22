"""Hermetic CPU tests for the M4 epi_resadd kernel microgate harness pieces."""
from extra.llm_research.decode.m4_resadd_kernel_microgate import (
  COPY_PREFIX, FUSED, LEGACY, LAYERS, extract_rows, microgate_verdict, parse_census_log,
)


def _log_line(seq: int, name: str, us: float) -> str:
  return f"*** NV         1 {name:<60} arg  1 mem   1.00 GB tm     {us:>10.2f}us/    1.00ms"


def test_parse_census_log_extracts_us_and_names():
  log = "\n".join([
    _log_line(1, FUSED, 9.73), _log_line(2, FUSED, 9.81), _log_line(3, FUSED, 9.65),
    _log_line(4, LEGACY, 9.28), _log_line(5, LEGACY, 9.40),
    _log_line(6, "E_32_32_4_86a23e1a5cd1cbd6101066fd85449138b653e9ecbb53d1d704f32aa470cd6f2b", 1.5),
    _log_line(7, "E_32_32_4_86a23e1a5cd1cbd6101066fd85449138b653e9ecbb53d1d704f32aa470cd6f2b", 1.6),
    _log_line(8, "E_32_32_4_02a9738c0547f555d270a4d68dcef46d880b70889ac4f0845db19755a52f69d5", 1.65),
  ])
  per = parse_census_log(log)
  assert per[FUSED] == [9.73, 9.81, 9.65]
  assert per[LEGACY] == [9.28, 9.40]
  assert len(per) == 4


def test_parse_census_log_handles_ms():
  line = "*** NV         1 some_kernel_4096_4096 arg  2 mem   5.03 GB tm   3019.63ms/  3019.63ms"
  per = parse_census_log(line + "\n")
  assert per["some_kernel_4096_4096"] == [3019.63 * 1e-3]


def test_extract_rows_counts_and_medians():
  per = {
    FUSED: [9.73, 9.81, 9.65],
    LEGACY: [9.28, 9.40],
    "E_32_32_4_86a2aa": [1.5, 1.6],
    "E_32_32_4_02a9bb": [1.65],
  }
  rows = extract_rows(per)
  assert rows["fused_count"] == 3 and rows["fused_median_us"] == 9.73
  assert rows["legacy_count"] == 2 and rows["legacy_median_us"] == 9.34
  assert rows["copy_count"] == 2 and rows["copy_median_us"] == 1.55


def test_extract_rows_missing_census_is_none_not_crash():
  rows = extract_rows({})
  assert rows["fused_median_us"] is None and rows["legacy_median_us"] is None
  assert rows["copy_count"] == 0


def test_microgate_verdict_pass_when_fused_cheap_and_copies_heavy():
  # Fused ~= legacy (+4.8%), copies 72x1.5us: ceiling measured 108 - 36*0.45 = +91.8us.
  rows = {"fused_median_us": 9.73, "legacy_median_us": 9.28,
          "copy_count": 72, "copy_median_us": 1.5}
  v = microgate_verdict(rows)
  assert v["verdict"] == "PASS"
  assert v["fused_vs_legacy_pct"] == 4.85
  assert v["copy_mass_measured_us"] == 108.0
  assert v["copy_free_ceiling_measured_us"] == 91.8
  assert v["copy_free_ceiling_book_us"] == 36.72


def test_microgate_verdict_fails_on_material_kernel_regression():
  # Fused 3.74x legacy (the ffn_down recompute defect shape): regression is material.
  rows = {"fused_median_us": 98.16, "legacy_median_us": 26.23,
          "copy_count": 72, "copy_median_us": 1.5}
  v = microgate_verdict(rows)
  assert v["verdict"] == "FAIL"
  assert v["regress_ok"] is False
  assert v["fused_vs_legacy_pct"] > 270


def test_microgate_verdict_fails_on_negative_ceiling():
  # A fused kernel only slightly slower than legacy but with no copies to remove.
  rows = {"fused_median_us": 9.73, "legacy_median_us": 9.28, "copy_count": 0, "copy_median_us": 0.0}
  v = microgate_verdict(rows)
  assert v["verdict"] == "FAIL"
  assert v["regress_ok"] is True
  assert v["copy_free_ceiling_measured_us"] < 0


def test_microgate_verdict_no_census_is_fail_closed():
  v = microgate_verdict({"fused_median_us": None, "legacy_median_us": 9.28,
                         "copy_count": 0, "copy_median_us": 0.0})
  assert v["verdict"] == "NO-CENSUS"


def test_microgate_constants_match_scope():
  assert LAYERS == 36
  assert FUSED == "q4k_g3_lanemap_gemv_epi_resadd_4096_4096"
  assert LEGACY == "q4k_g3_lanemap_gemv_4096_4096"
  assert COPY_PREFIX == "E_32_32_4_86a2"
