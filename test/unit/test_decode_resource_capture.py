import pytest

from extra.qk.decode.decode_resource_capture import capture_report, capture_row


def test_decode_resource_capture_fixture_is_fail_closed_and_complete():
  row = capture_row(kernel_name="flash_block_32_128", source="kernel source",
                    code_object=b"final hsaco", expected_names=("flash_block_32_128",),
                    workgroup=(128, 1, 1), grid=(384, 1, 1),
                    metadata={"vgpr": 54, "sgpr": 29, "lds_bytes": 8192, "scratch_bytes": 0,
                              "vgpr_spills": 0, "sgpr_spills": 0})
  report = capture_report((row,), geometry={"Hq": 32, "Hkv": 8, "Hd": 128, "Tc": 512})
  assert report["_schema"] == "decode-resource-capture.v1"
  assert report["positive_expected_name_matches"] == 1
  assert report["rows"][0]["code_object_bytes"] == len(b"final hsaco")


def test_decode_resource_capture_rejects_missing_positive_match():
  with pytest.raises(ValueError, match="positive control"):
    capture_row(kernel_name="actual", source="source", code_object=b"binary",
                expected_names=("expected",), workgroup=(1, 1, 1), grid=(1, 1, 1),
                metadata={"vgpr": 1, "sgpr": 1, "lds_bytes": 0, "scratch_bytes": 0,
                          "vgpr_spills": 0, "sgpr_spills": 0})
