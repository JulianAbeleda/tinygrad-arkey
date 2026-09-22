"""Hermetic proof of the parity path: fusion first, launch hiding is a scheduler substrate.

Pins three claims against committed evidence, read-only (no GPU, no codegen):

1. The llama single-pass flash kernel body is FLAT versus the legacy tile, so a
   flash kernel rewrite is not a tok/s lever.
2. llama achieves ~946 us of in-graph overlap on ONE stream while tinygrad runs
   seven streams with ZERO overlap; the only llama-hidden mass tinygrad has not
   already fused away is the flash pair (~143 us).
3. The tok/s ladder is internally consistent: closing all measured class deltas
   (fusion) lands at ~220 tok/s, llama's wall is ~245 tok/s, and the residual
   ~477 us is launch hiding, not kernel body.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs" / "task_workflow" / "evidence"


def _load(name: str) -> dict:
  return json.loads((EVIDENCE / name).read_text())


LLAMA = _load("nv-llama-d512-node-ledger-20260812.json")
TINYGRAD = _load("nv-tinygrad-d512-node-ledger-20260813.json")
MICROGATE = _load("nv-flash-vec-llama-microgate-20260813.json")
PRIME = _load("nv-tinygrad-prime-gap-table-20260812.json")


def test_flash_kernel_body_rewrite_is_flat():
  t = MICROGATE["timing"]
  # 1 us x 36 flash nodes = 36 us, below the +50 us promotion gate: the body
  # rewrite has no promotion-sized wall effect even before noise is considered.
  assert t["candidate_median"] == pytest.approx(t["control_midpoint_median"], abs=1.0)
  assert abs(t["delta"]) < 1.0
  for row in MICROGATE["correctness"]:
    assert row["finite"] is True
    assert row["max_abs"] < 1e-3


def test_llama_overlap_is_ingraph_not_multistream():
  # llama hides work with ONE stream (in-graph concurrency); tinygrad has seven
  # streams and hides none. The substrate is graph dependency structure, not
  # stream count, so adding streams is not the lever.
  assert LLAMA["median"]["streams"] == 1.0
  assert LLAMA["median"]["overlap_mass_us"] > 900.0
  assert TINYGRAD["median"]["streams"] > 1
  assert TINYGRAD["median"]["overlap_mass_us"] == 0.0
  # overlap_mass is exactly the span-hidden mass: span = node_sum - overlap_mass
  # + internal gap. This proves hiding reduces exposed time rather than node-sum.
  assert LLAMA["median"]["span_us"] == pytest.approx(
    LLAMA["median"]["node_sum_us"] - LLAMA["median"]["overlap_mass_us"] + LLAMA["median"]["internal_gap_us"],
    abs=2.0)


def test_flash_is_the_only_unfused_llama_hidden_mass():
  big_hidden = {
    cls: LLAMA["classes"][cls]["hidden_behind_mmq_us"]
    for cls in LLAMA["classes"]
    if LLAMA["classes"][cls].get("hidden_behind_mmq_us", 0.0) > 50.0
  }
  assert set(big_hidden) == {"quantize_q8_1", "rms_norm", "flash_score", "flash_combine"}

  # quantize and rope are folded into tinygrad's GEMV epilogues and norms are
  # fused ahead of llama, so those hidden masses are not recoverable levers.
  assert "quantize_q8_1" not in TINYGRAD["classes"]
  assert "rope" not in PRIME["classes"]
  assert PRIME["classes"]["rmsnorm"]["sum_us"] < LLAMA["classes"]["rms_norm"]["node_sum_us"]

  # The flash pair is the only llama-hidden mass still fully exposed in
  # tinygrad: ~143 us (57.8 score + 85.2 combine) behind MMQ, zero hidden here.
  flash_hidden_llama = (LLAMA["classes"]["flash_score"]["hidden_behind_mmq_us"]
                        + LLAMA["classes"]["flash_combine"]["hidden_behind_mmq_us"])
  assert flash_hidden_llama == pytest.approx(57.825 + 85.216, abs=1.0)
  assert flash_hidden_llama == pytest.approx(143.0, abs=5.0)
  assert TINYGRAD["classes"]["flash_score"]["hidden_behind_gemv_us"] == 0.0
  assert TINYGRAD["classes"]["flash_combine"]["hidden_behind_gemv_us"] == 0.0


def test_tok_s_ladder_fusion_ceiling_then_launch_hiding():
  baseline_ms = 5.2031       # production wall (gap attribution 08-12)
  fusion_savings_us = 652.4  # all measured class deltas at 1:1 removal
  llama_ms = 4.0741          # pinned llama pair wall

  fusion_ceiling_ms = baseline_ms - fusion_savings_us / 1000.0
  launch_hiding_us = (fusion_ceiling_ms - llama_ms) * 1000.0

  assert 1000.0 / baseline_ms == pytest.approx(192.19, abs=0.01)
  assert 1000.0 / fusion_ceiling_ms == pytest.approx(219.7, abs=0.2)
  assert 1000.0 / llama_ms == pytest.approx(245.45, abs=0.02)

  # Fusion alone cannot reach parity; the residual is launch hiding (~0.48 ms),
  # which is a scheduler substrate, not a kernel-body delta.
  assert 1000.0 / fusion_ceiling_ms < 1000.0 / llama_ms
  assert launch_hiding_us == pytest.approx(477.0, abs=10.0)
