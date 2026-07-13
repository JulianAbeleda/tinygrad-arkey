"""Regression pin for P0-1 (false dispatch evidence).

# PINS P0-1 (fixed in C1)

The scope doc `pure-register-direct-l2-completion-scope-20260712.md` finding
P0-1 requires the hardware executor / canary to encode a *typed* dispatch state
(`not_attempted`, `attempted`, `submitted`, `completed`, `failed`, `timed_out`,
`device_lost`) instead of a bare boolean `dispatch_performed=False`, which can
falsely claim a dispatch was not performed after a callback that may have
dispatched.

These tests are CPU-only: they exercise the blocked/prepared code paths of the
real executor symbols, which construct no runtime and dispatch nothing. They
FAIL against current code (which emits the bare boolean) and will pass once the
typed dispatch state replaces it.
"""
from __future__ import annotations

from extra.qk.prefill.attn_qo_direct_l2_hardware_executor_20260712 import (
  exact_pair_metadata, run_hardware_executor,
)

# The required typed dispatch states from finding P0-1.
REQUIRED_DISPATCH_STATES = frozenset({
  "not_attempted", "attempted", "submitted", "completed", "failed",
  "timed_out", "device_lost",
})


def _assert_typed_dispatch_state(result: dict, where: str) -> None:
  # PINS P0-1: the bare boolean must be gone, replaced by a typed state field.
  assert "dispatch_performed" not in result, (
    f"{where} still reports the bare boolean dispatch_performed="
    f"{result.get('dispatch_performed')!r}; P0-1 requires a typed dispatch state")
  state = result.get("dispatch_state")
  assert state in REQUIRED_DISPATCH_STATES, (
    f"{where} must carry a typed dispatch_state in {sorted(REQUIRED_DISPATCH_STATES)}, "
    f"got dispatch_state={state!r}")


def test_prepared_pair_metadata_uses_typed_dispatch_state():
  # CPU-only: exact_pair_metadata() generates the pair on the host, no GPU.
  _assert_typed_dispatch_state(exact_pair_metadata(), "exact_pair_metadata()")


def test_blocked_executor_uses_typed_dispatch_state():
  # CPU-only: no opt-in => blocked preflight, no runtime constructed, no dispatch.
  result = run_hardware_executor(candidate=None, compile_artifact=None,
                                 route_binding=None, stage_dispatch=None,
                                 paired_benchmark=None, enable_value=None)
  assert result.get("status") == "blocked"
  _assert_typed_dispatch_state(result, "run_hardware_executor() blocked result")
