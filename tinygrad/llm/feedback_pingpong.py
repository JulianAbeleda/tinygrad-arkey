"""Fail-closed qualification helpers for the two-capture decode feedback ring."""
from __future__ import annotations
from typing import Any

from tinygrad import Tensor


def _sampled_return(ret:Any) -> Tensor:
  sampled = ret[0] if isinstance(ret, tuple) else ret
  if not isinstance(sampled, Tensor): raise ValueError("ping-pong capture has no Tensor sampled return")
  return sampled


def pingpong_capture_contract(jits:tuple[Any, Any]) -> dict[str, Any]:
  """Describe an already-warmed pair, rejecting every ambiguous alias case.

  This never changes CapturedJit behavior.  It is a qualification oracle used
  after both arms have captured; a production experiment must fall back to the
  ordinary single capture unless ``admitted`` is true.
  """
  if not isinstance(jits, tuple) or len(jits) != 2: raise ValueError("ping-pong needs exactly two captures")
  captured = tuple(getattr(jit, "captured", None) for jit in jits)
  if any(cap is None for cap in captured):
    return {"admitted": False, "reason": "capture_not_warm"}
  returns = tuple(_sampled_return(cap.ret) for cap in captured)
  bases = tuple(ret.uop.buf_uop for ret in returns)
  if bases[0] is bases[1]: return {"admitted": False, "reason": "return_buffers_alias"}
  if any(cap._written_input_shadows for cap in captured):
    return {"admitted": False, "reason": "written_input_shadow_present"}
  contracts = tuple(tuple((view.key, variables, dtype, device) for view, variables, dtype, device in cap.expected_input_info)
                    for cap in captured)
  if contracts[0] != contracts[1]: return {"admitted": False, "reason": "input_contract_mismatch"}
  return {"admitted": True, "reason": "distinct_fixed_returns_and_read_only_inputs",
          "return_shape": list(returns[0].shape), "return_dtype": str(returns[0].dtype), "return_device": str(returns[0].device),
          "program_counts": [len(cap.linear.src) for cap in captured], "written_input_shadows": [0, 0]}
