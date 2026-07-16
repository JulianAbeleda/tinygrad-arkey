import json
from pathlib import Path

import numpy as np

from extra.qk.layout import q6_k_reference
from extra.qk.prefill.q6_direct_packed_qualification import make_finite_q6k_bytes, q6k_dequantize_selected_positions
from extra.qk.prefill.six_row_policy_artifact import missing_qualification_evidence
from tinygrad import Tensor


def test_selected_q6_decoder_matches_canonical_reference():
  n, k = 3, 512
  raw = make_finite_q6k_bytes(n, k, 7)
  positions = np.array([0, 15, 16, 63, 127, 128, 191, 255, 256, 511])
  got = q6k_dequantize_selected_positions(raw, positions)
  reference = q6_k_reference(Tensor(raw.reshape(-1)), n*k).reshape(n, k).numpy()[:, positions]
  np.testing.assert_array_equal(got, reference)


def test_q6_fixture_is_deterministic_and_finite():
  first = make_finite_q6k_bytes(2, 512, 3); second = make_finite_q6k_bytes(2, 512, 3)
  np.testing.assert_array_equal(first, second)
  assert np.isfinite(q6k_dequantize_selected_positions(first, np.arange(512))).all()


def test_committed_evidence_advances_exactly_the_two_q6_rows():
  root = Path("bench/prefill-pure-full-kernel")
  inventory = json.loads((root/"qwen3-14b-mixed-quant-candidate-inventory-v1.json").read_text())
  evidence = [json.loads((root/name).read_text()) for name in (
    "q6-direct-packed-attn-kv-512x1024x5120-20260716.json",
    "q6-direct-packed-ffn-down-512x5120x17408-20260716.json")]
  missing = missing_qualification_evidence(inventory, q6_evidence=evidence)
  assert len(missing) == 4 and all(row.startswith("Q4_K:") for row in missing)
