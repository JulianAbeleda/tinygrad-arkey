"""Live compile, check and measure for BoltBeam flash decode candidates on the GPU tinygrad opened.

These are the three ``*_fn`` hooks of search_provider.FlashAdapter. Each receives (payload, descriptor), where
the descriptor is the candidate's canonical ``{"tile", "combine"}``. Nothing here is vendor-specific:
- the kernel is emitted from the tile exactly as decode/nv_flash_geometry_search.py emits it,
- it is compiled by the opened device's renderer, the way search_provider.MetalAdapter compiles,
- it is timed on the device (``time_call`` with ``wait=True`` reads the GPU's own timestamps), with the
  device's cache invalidated before every launch, which is BoltBeam's cold-L2 regime
  (boltbeam/plan/flash_measure.py: flush between launches, zero warmups).
The live token count ``Tc`` is the workload, not the geometry, so it arrives in ``payload["execution"]``
as ``context_tokens`` and never in the candidate.
"""
from __future__ import annotations

import hashlib, statistics
from dataclasses import dataclass, field
from typing import Any, Mapping

from extra.llm_research.search_provider import ProtocolError

CONTROL_SPLIT = 48  # the production tile (nv_flash_geometry_search.py CONTROL_NAME) is the check oracle
CHECK_TOLERANCE = {"atol": 2e-3, "rtol": 2e-3}  # the geometry search's tolerance against the control
FLUSH_MODES = {True: "device_invalidate_caches", False: "4MiB_write_fallback"}


def context_tokens(payload: Mapping[str, Any]) -> int:
  """``execution.context_tokens``: the live token count the tile is emitted for; required, positive."""
  execution = payload.get("execution")
  value = execution.get("context_tokens") if isinstance(execution, Mapping) else None
  if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
    raise ProtocolError("admission_rejected", "execution.context_tokens must be a positive int")
  return value


def _spec(tile: Mapping[str, Any], combine: Mapping[str, Any] | None, *, fused_combine: bool):
  from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
  combine = combine or {}
  return describe_flash_decode_attention(
    tile["Hq"], tile["Hd"], tile["Hkv"], tile["MAXC"], tile["split_count"], staging=tile["staging"],
    fused_combine=fused_combine, quant=tile["quant"], rope=tile["rope"], combine_stride=combine.get("stride"),
    query_group_size=tile["query_group_size"], stage_width=tile["stage_width"], token_block=tile["token_block"],
    lane_width=tile["lane_width"], score_group_width=tile["score_group_width"], warps=tile["warps"],
    reduce_structure=tile["reduce_structure"], dot_pair_width=tile["dot_pair_width"],
    combine_lane_width=combine.get("lane_width"), combine_fp16=bool(combine.get("output_fp16", False)))


def _control_spec(tile: Mapping[str, Any]):
  """The production tile for the candidate's shape: split 48 and the emitter's default geometry, the same
  control nv_flash_geometry_search.py checks against. Only the shape and the KV format come from the candidate."""
  from tinygrad.llm.flash_decode_attention import describe_flash_decode_attention
  return describe_flash_decode_attention(tile["Hq"], tile["Hd"], tile["Hkv"], tile["MAXC"], CONTROL_SPLIT,
                                         staging=tile["staging"], quant=tile["quant"], rope=tile["rope"], fused_combine=True)


def _sha256(value: bytes | str) -> str:
  return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass
class LiveFlash:
  """One opened device; inputs and compiled tiles are cached per (descriptor, Tc)."""
  device: str
  _inputs: dict[tuple, tuple] = field(default_factory=dict, repr=False)
  _tiles: dict[str, Any] = field(default_factory=dict, repr=False)

  def inputs(self, tile: Mapping[str, Any]):
    """Deterministic q and KV cache for the tile's shape, the geometry search's own fixture."""
    key = (tile["Hq"], tile["Hd"], tile["Hkv"], tile["MAXC"])
    if key not in self._inputs:
      import numpy as np
      from tinygrad import Tensor
      Hq, Hd, Hkv, MAXC = key
      rng = np.random.default_rng(20260813)
      q = rng.normal(0, .2, Hq * Hd).astype(np.float16)
      cache = rng.normal(0, .2, (2, 1, Hkv, MAXC, Hd)).astype(np.float16)
      self._inputs[key] = (Tensor(q, device=self.device).contiguous().realize(),
                           Tensor(cache, device=self.device).contiguous().realize())
    return self._inputs[key]

  def _program(self, spec, tc: int, fused: bool):
    from tinygrad import dtypes
    from tinygrad.llm.kernel_program import KernelProgram, KernelProgramProvenance, OutputSpec
    from tinygrad.uop.ops import UOp
    tile = spec.tile
    if not fused:
      return KernelProgram("research.flash_live", tile.kernel_name, KernelProgramProvenance.RESEARCH_ONLY,
                           spec.emit_tile(UOp.const(dtypes.int, tc)),
                           output_spec=OutputSpec((tile.Hq * tile.split_count * (tile.Hd + 2),), dtypes.float32))
    return KernelProgram("research.flash_live", spec.combine.kernel_name, KernelProgramProvenance.RESEARCH_ONLY,
                         spec.emit_combine(), output_spec=OutputSpec((tile.Hq * tile.Hd,), dtypes.float32))

  def _run_fused(self, spec, tc: int, q, cache):
    """Tile then combine, as the geometry search checks a candidate; returns the fp32 output."""
    import numpy as np
    from tinygrad import Tensor, dtypes
    from tinygrad.device import Device
    from tinygrad.llm.kernel_program import execute_research_program
    tile = spec.tile
    partial = execute_research_program(Tensor.empty(tile.Hq * tile.split_count * (tile.Hd + 2), dtype=dtypes.float32, device=self.device),
                                       q, cache, program=self._program(spec, tc, fused=False))
    out = execute_research_program(Tensor.empty(tile.Hq * tile.Hd, dtype=dtypes.float32, device=self.device),
                                   partial, program=self._program(spec, tc, fused=True))
    out.realize()
    Device[self.device].synchronize()
    return np.asarray(out.numpy()).astype(np.float32)

  def compiled_tile(self, payload: Mapping[str, Any], descriptor: Mapping[str, Any]):
    """The tile kernel alone (the measured body), compiled once: (call UOp for time_call, program UOp, spec)."""
    tc = context_tokens(payload)
    key = f"{payload['candidate_hash']}:{tc}"
    if key in self._tiles: return self._tiles[key]
    from tinygrad import Tensor, dtypes
    from tinygrad.codegen import to_program
    from tinygrad.device import Device
    from tinygrad.llm.kernel_program import execute_research_program
    try: spec = _spec(descriptor["tile"], descriptor["combine"], fused_combine=True); spec.validate()
    except ValueError as exc: raise ProtocolError("unsupported_plan", f"the emitter refuses this geometry: {exc}") from exc
    tile = spec.tile
    q, cache = self.inputs(descriptor["tile"])
    out = execute_research_program(Tensor.empty(tile.Hq * tile.split_count * (tile.Hd + 2), dtype=dtypes.float32, device=self.device),
                                   q, cache, program=self._program(spec, tc, fused=False))
    linear = out.schedule_linear()
    if len(linear.src) != 1: raise ProtocolError("provider_failure", "the flash tile must schedule exactly one program")
    call = linear.src[0]
    program = to_program(call.src[0], Device[self.device].renderer)
    self._tiles[key] = (call.replace(src=(program, *call.src[1:])), program, spec)
    return self._tiles[key]

  def compile(self, payload: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    from tinygrad.device import Device
    call, program, spec = self.compiled_tile(payload, descriptor)
    source, binary = program.src[3].arg, program.src[4].arg
    return {"compiler": type(Device[self.device].compiler).__name__, "device": self.device,
            "source_sha256": _sha256(source), "binary_sha256": _sha256(binary),
            "launch": {"global_size": list(program.arg.global_size or ()), "local_size": list(program.arg.local_size or ())},
            "kernel_name": spec.tile.kernel_name, "context_tokens": context_tokens(payload),
            "candidate_plan_hash": payload["candidate_hash"]}

  def check(self, payload: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """The candidate's fused output against the production tile (split 48) on the same inputs."""
    import numpy as np
    tc = context_tokens(payload)
    _call, _program, spec = self.compiled_tile(payload, descriptor)
    q, cache = self.inputs(descriptor["tile"])
    control = _control_spec(descriptor["tile"])
    evidence = {"oracle": f"production flash tile at split_count {CONTROL_SPLIT}, fused combine, same inputs",
                "control_kernel_name": control.tile.kernel_name, "context_tokens": tc, "tolerance": dict(CHECK_TOLERANCE)}
    expected = self._run_fused(control, tc, q, cache)
    got = self._run_fused(spec, tc, q, cache)
    if not np.isfinite(expected).all(): raise ProtocolError("provider_failure", "the control tile produced a non-finite output")
    finite = bool(np.isfinite(got).all())
    max_abs = float(np.max(np.abs(got - expected))) if got.shape == expected.shape else None
    close = finite and max_abs is not None and bool(np.allclose(got, expected, **CHECK_TOLERANCE))
    return {"correct": close, "max_abs_error": max_abs, "finite": finite, **evidence,
            "candidate_plan_hash": payload["candidate_hash"]}

  def measure(self, payload: Mapping[str, Any], descriptor: Mapping[str, Any]) -> dict[str, Any]:
    """Device-side time of the tile body, caches invalidated before every launch, no warmups."""
    from tinygrad.device import Device
    from tinygrad.engine.realize import time_call
    execution = payload.get("execution") if isinstance(payload.get("execution"), Mapping) else {}
    samples, warmups = payload.get("samples", execution.get("samples")), payload.get("warmups", execution.get("warmups", 0))
    if not isinstance(samples, int) or isinstance(samples, bool) or not 1 <= samples <= 100:
      raise ProtocolError("admission_rejected", "samples must be 1..100")
    if warmups != 0: raise ProtocolError("admission_rejected", "flash measure is cold: warmups must be 0")
    call, _program, spec = self.compiled_tile(payload, descriptor)
    device_flush = hasattr(Device[self.device], "invalidate_caches")
    samples_ns = [int(time_call(call, clear_l2=True) * 1e9) for _ in range(samples)]
    tile = spec.tile
    operations = 4 * tile.Hq * tile.Hd * context_tokens(payload)
    bytes_ = 2 * context_tokens(payload) * tile.Hd * 2 * (tile.Hq // tile.Hkv) + tile.Hq * (tile.Hd + 2) * 4
    return {"timing_mode": "device_timestamps_wait_true", "synchronized": True, "warmups": 0, "samples_ns": samples_ns,
            "l2_discipline": FLUSH_MODES[device_flush], "flush_between_launches": True,
            "summary_ns": {"min": min(samples_ns), "mean": sum(samples_ns) / len(samples_ns), "max": max(samples_ns),
                           "median": statistics.median(samples_ns), "range": max(samples_ns) - min(samples_ns)},
            "work_bytes": {"status": "estimated", "operations": operations, "bytes": bytes_,
                           "provenance": "modeled flash decode traffic at context_tokens (score+PV flops, KV fp16 reads); physical cache traffic unobserved"}}


__all__ = ["CHECK_TOLERANCE", "CONTROL_SPLIT", "FLUSH_MODES", "LiveFlash", "context_tokens"]
