"""The decode GEMV emitters as a search space: which families exist, their parameters spelled as a BoltBeam v2
schedule, and the program each one builds at a role's exact shape.

A family here is a row of extra/llm_research/semantic_kernel_lowering.py's registry (the campaign's one lowering
of a BoltBeam family + parameters to a tinygrad emitter). This module adds no kernel: it says which schedule fields
carry each emitter parameter, so BoltBeam's BubbleBeam and FutureSight can judge the rows with the chip's facts,
and it builds the exact-shape program the provider compiles, checks and times.

The rows are the emitters' own parameter values. They are offered on a device only when the model's decode would
route the role through these emitters there: the Q4_K/Q6_K primitive capability (qk_primitives.py, read from the
renderer facts, never a target name). Elsewhere the role runs a scheduler kernel and the space is Opt sequences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

EMITTER_PLAN_KIND = "tinygrad_decode_emitter.v1"
_POS_LANES = 16  # Q6K_POS_EXTENT: the Q6_K coop emitter's lanes per row


def _row(threads:int, tile_n:int, b_width:int, a_space:str = "global", stages:int = 1) -> dict[str, Any]:
  return {"schedule.launch.threads": threads, "schedule.tile.n": tile_n, "schedule.memory.b.vector_width": b_width,
          "schedule.memory.a.space": a_space, "schedule.pipeline.stage_count": stages}


@dataclass(frozen=True)
class EmitterFamily:
  family: str  # the registry family (semantic_kernel_lowering.build_registered_llm_emitter)
  quant: str
  storage: str  # the dtype the GGUF bytes are viewed as: uint32 words or uint16 halfs
  rows: tuple[dict[str, Any], ...]  # coupled schedule rows, one per emitter parameter combination
  one_subgroup: bool  # the emitter reduces inside one subgroup, so its threads may not exceed the subgroup width
  parameters: Callable[[Mapping[str, Any], int, int], dict[str, Any]]  # schedule, rows, k -> registry parameters

  def to_json(self) -> dict[str, Any]:
    return {"family": self.family, "quant": self.quant, "storage": self.storage, "coupled_rows": [dict(r) for r in self.rows],
            "threads_within_subgroup": self.one_subgroup}


def _q4_parameters(schedule:Mapping[str, Any], rows:int, k:int) -> dict[str, Any]:
  width, space = schedule["memory"]["b"]["vector_width"], schedule["memory"]["a"]["space"]
  style = {(4, "global"): "scalar", (16, "global"): "vector", (16, "shared"): "quad"}.get((width, space))
  if style is None: raise ValueError(f"no Q4_K G3 load style reads b {width} wide with a in {space} memory")
  tile, threads = schedule["tile"]["n"], schedule["launch"]["threads"]
  if style == "quad":
    if (tile, threads) != (16, 128): raise ValueError("the quad style runs 16 rows x 8 lanes per block")
    lanes = 32
  else:
    if tile != 1: raise ValueError(f"the {style} style runs one row per block")
    lanes = threads
  return {"rows": rows, "k": k, "lanes": lanes, "load_style": style, "epilogue": ""}


def _q6_parameters(schedule:Mapping[str, Any], rows:int, k:int) -> dict[str, Any]:
  tile, threads = schedule["tile"]["n"], schedule["launch"]["threads"]
  if threads != tile * _POS_LANES: raise ValueError(f"the Q6_K coop emitter runs {_POS_LANES} lanes per row")
  if schedule["memory"]["b"]["vector_width"] != 2 or schedule["memory"]["a"]["space"] != "global":
    raise ValueError("the Q6_K coop emitter reads 16-bit halfs and its activation from global memory")
  return {"rows": rows, "k": k, "row_tile": tile, "accumulators": schedule["pipeline"]["stage_count"]}


FAMILIES: tuple[EmitterFamily, ...] = (
  EmitterFamily("q4_g3_gemv.v1", "Q4_K", "uint32",
                (_row(32, 1, 4), _row(32, 1, 16), _row(128, 16, 16, "shared")), False,
                _q4_parameters),
  EmitterFamily("q6_coop_gemv.v1", "Q6_K", "uint16",
                tuple(_row(t * _POS_LANES, t, 2, stages=a) for t in (1, 2, 4) for a in (1, 2, 4)), True,
                _q6_parameters),
)
BY_FAMILY = {f.family: f for f in FAMILIES}


def emitters_apply(device:str) -> tuple[bool, str]:
  """Does the model's decode route Q4_K/Q6_K roles through these emitters on this device, and why."""
  from tinygrad.llm.device_facts import scan_device_facts
  from tinygrad.llm.qk_primitives import qk_primitive_capability_from_device_facts
  capability = qk_primitive_capability_from_device_facts(scan_device_facts(device))
  if capability.satisfied: return True, "the Q4_K/Q6_K primitive capability holds: 32-wide subgroups with shuffle"
  return False, (f"the Q4_K/Q6_K primitive capability does not hold here (wave_size {capability.wave_size}, "
                 f"shuffle {capability.supports_warp_shfl_xor}): the decode runs scheduler kernels")


def installed_row(family:EmitterFamily, backend:str | None, architecture:str | None) -> dict[str, Any]:
  """The row the model's decode installs for this family on this device, by the fork's own routing rules."""
  if family.family == "q4_g3_gemv.v1":
    from types import SimpleNamespace
    from tinygrad.llm.decode_routes import _q4k_single_projection_load_style
    linear = SimpleNamespace(route_admission=SimpleNamespace(capability=SimpleNamespace(backend=backend, architecture=architecture)))
    style = _q4k_single_projection_load_style(linear)
    return _row(32, 1, 16 if style == "vector" else 4)
  from tinygrad.llm.decode_kernels import q6k_coop_row_tile_for_target
  tile = q6k_coop_row_tile_for_target(backend, architecture)
  return _row(tile * _POS_LANES, tile, 2)


def build(candidate:Mapping[str, Any], device:str):
  """The exact-shape program for one emitter candidate: (output, words, x, call inputs) on `device`, with
  deterministic GGUF bytes. The caller schedules output to get the one program."""
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  schedule, workload = candidate["schedule"], candidate["workload"]
  family = BY_FAMILY.get(schedule["compute"]["family"])
  if family is None: raise ValueError(f"no decode emitter family {schedule['compute']['family']!r}")
  quant = str(workload["operands"]["b"]["quantization"]).upper()
  if quant.replace("_", "") != family.quant.replace("_", ""): raise ValueError(f"{family.family} reads {family.quant}, not {quant}")
  rows, k = int(workload["shape"]["n"]), int(workload["shape"]["k"])
  emitter = build_registered_llm_emitter(family.family, family.parameters(schedule, rows, k))
  block = 144 if family.quant == "Q4_K" else 210
  raw = fixture_bytes(family.quant, rows, k)
  storage = dtypes.uint32 if family.storage == "uint32" else dtypes.uint16
  words = Tensor(raw.view(np.uint32 if family.storage == "uint32" else np.uint16), dtype=storage, device=device).realize()
  x = Tensor(fixture_activation(k), dtype=dtypes.float16, device=device).realize()
  out = Tensor.empty(rows, dtype=dtypes.float32, device=device).uop_program(words, x, fxn=emitter)[0]
  return out, words, x, raw, block


def fixture_bytes(quant:str, rows:int, k:int):
  """Deterministic, finite GGUF blocks (the provider's fixture rule: small nonzero bytes, fixed fp16 scales)."""
  import numpy as np
  block = 144 if quant == "Q4_K" else 210
  raw = (1 + np.arange(rows * k // 256 * block, dtype=np.uint32) % 13).astype(np.uint8).reshape(-1, block)
  d, dmin = np.array([0.01], dtype=np.float16).view(np.uint8), np.array([0.005], dtype=np.float16).view(np.uint8)
  raw[:, 0:2] = d
  if quant == "Q4_K": raw[:, 2:4] = dmin
  else: raw[:, 208:210] = d
  return raw.reshape(-1)


def fixture_activation(k:int):
  import numpy as np
  return np.array([1.0 + (i % 7) / 16 for i in range(k)], dtype=np.float16)


__all__ = ["BY_FAMILY", "EMITTER_PLAN_KIND", "FAMILIES", "EmitterFamily", "build", "emitters_apply", "fixture_activation",
           "fixture_bytes", "installed_row"]
