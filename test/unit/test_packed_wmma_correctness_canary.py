import numpy as np
import pytest

from extra.qk.prefill.packed_wmma_correctness_canary import M, N, K, build_artifact, candidate_payload
from extra.qk.runtime_specs import full_kernel_workload


@pytest.mark.parametrize("quant_format,dtype,packed_bytes", (
  ("Q4_K", np.uint32, N*K//256*144),
  ("Q6_K", np.uint16, N*K//256*210),
))
def test_canary_artifact_has_exact_packed_abi_and_full_nonconstant_reference(tmp_path, quant_format, dtype, packed_bytes):
  path = tmp_path / f"{quant_format}.npz"
  summary = build_artifact(quant_format, str(path))
  with np.load(path, allow_pickle=False) as row:
    assert set(row.files) == {"a", "b", "reference"}
    assert row["a"].shape == (M,K) and row["a"].dtype == np.float16
    assert row["b"].ndim == 1 and row["b"].dtype == np.dtype(dtype) and row["b"].nbytes == packed_bytes
    assert row["reference"].shape == (M,N) and row["reference"].dtype == np.float16
    assert np.ptp(row["a"]) > 0 and np.ptp(row["b"]) > 0 and np.ptp(row["reference"]) > 0
    assert np.all(np.isfinite(row["reference"]))
    assert np.count_nonzero(row["a"]) == M
  assert summary["packed_bytes"] == packed_bytes and summary["reference_elements"] == M*N


def test_canary_artifact_generation_is_shape_driven(tmp_path):
  shape = (16, 128, 256)
  path = tmp_path / "small-q4.npz"
  summary = build_artifact("Q4_K", str(path), shape)
  with np.load(path, allow_pickle=False) as row:
    assert row["a"].shape == (shape[0], shape[2])
    assert row["reference"].shape == shape[:2]
    assert row["b"].nbytes == shape[1] * (shape[2] // 256) * 144
  assert summary["shape"] == list(shape)


def test_canary_candidate_resolution_reuses_role_template_for_14b():
  workload = full_kernel_workload(candidate_payload("qwen3_14b_q4k_m_gfx1100", "ffn_down"))
  assert workload.profile == "qwen3_14b_q4k_m_gfx1100"
  assert workload.role == "ffn_down" and workload.shape == (512,5120,17408)
