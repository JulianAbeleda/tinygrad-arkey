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
  quants: tuple[str, ...]  # the weight formats the one emitter reads (a layout table, not one emitter per format)
  storage: str  # the dtype the GGUF bytes are viewed as: uint32 words or uint16 halfs
  rows: tuple[dict[str, Any], ...]  # coupled schedule rows, one per emitter parameter combination
  one_subgroup: bool  # the emitter reduces inside one subgroup, so its threads may not exceed the subgroup width
  parameters: Callable[[Mapping[str, Any], int, int], dict[str, Any]]  # schedule, rows, k -> registry parameters

  def to_json(self) -> dict[str, Any]:
    out = {"family": self.family, "quant": self.quants[0], "quants": list(self.quants), "storage": self.storage,
           "coupled_rows": [dict(r) for r in self.rows], "threads_within_subgroup": self.one_subgroup}
    if self.family == Q6_FOUR_WARP_FAMILY:  # the split a caller needs: 16-lane sub-groups per warp, 256-element blocks
      from tinygrad.llm.q6k_v_mmvq import POS, WARPS_PER_ROW
      out["lanes_per_block_slice"], out["block_elems"] = POS, 256
      out["installed_rule"] = {"warps_per_row": WARPS_PER_ROW}
    if self.family == BLOCK_QUANT_FAMILY:  # the layout facts a caller needs to propose its own rows: code bytes per block
      from tinygrad.llm.decode_kernels import BLOCK_QUANT_LAYOUTS
      out["code_unit_bytes"] = 2
      out["code_units_per_block"] = {q: BLOCK_QUANT_LAYOUTS[q].units for q in self.quants}
      out["installed_rule"] = block_quant_installed_rule()
    return out


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


BLOCK_QUANT_FAMILY = "block_quant_gemv.v1"
Q6_FOUR_WARP_FAMILY = "q6_four_warp_gemv.v1"


def _q6_four_warp_parameters(schedule:Mapping[str, Any], rows:int, k:int, quant:str, wave:int) -> dict[str, Any]:
  """warps_per_row = threads / wave (one row per CTA, tile.n 1); tile.k is one sweep of the CTA over k (256-element
  blocks x warps x sub-groups), so k a multiple of it is the emitter's own split rule."""
  threads, tile = schedule["launch"]["threads"], schedule["tile"]["n"]
  if tile != 1 or threads % wave: raise ValueError(f"the Q6_K four-warp emitter runs whole {wave}-lane warps on one row a CTA")
  if schedule["memory"]["b"]["vector_width"] != 2: raise ValueError("the Q6_K four-warp emitter reads 16-bit halfs")
  return {"rows": rows, "k": k, "warps_per_row": threads // wave}


def _block_quant_row(lane_bytes:int, warps_per_cta:int, wave:int, block_elems:int) -> dict[str, Any]:
  """The block-quant emitter's parameters as schedule fields: one row per warp (tile.n = warps per CTA), a CTA of
  warps x wave threads, tile.k one block (k must be whole blocks), and b's vector width the code bytes a lane loads per
  block (units_per_lane x 2). lanes_per_block = code bytes / lane_bytes, read back by _bq_parameters."""
  return {"schedule.launch.threads": warps_per_cta * wave, "schedule.tile.n": warps_per_cta, "schedule.tile.k": block_elems,
          "schedule.memory.b.vector_width": lane_bytes, "schedule.memory.a.space": "global", "schedule.pipeline.stage_count": 1}


def _bq_parameters(schedule:Mapping[str, Any], rows:int, k:int, quant:str, wave:int) -> dict[str, Any]:
  from tinygrad.llm.decode_kernels import BLOCK_QUANT_LAYOUTS
  units = BLOCK_QUANT_LAYOUTS[quant].units
  lane_bytes, threads, tile = schedule["memory"]["b"]["vector_width"], schedule["launch"]["threads"], schedule["tile"]["n"]
  if lane_bytes % 2 or (units * 2) % lane_bytes: raise ValueError(f"a {quant} lane loads whole 16-bit code units of the {units * 2} code bytes")
  if threads != tile * wave: raise ValueError(f"the block-quant emitter runs one {wave}-lane warp per row")
  return {"rows": rows, "k": k, "quant": quant, "wave_size": wave, "lanes_per_block": units * 2 // lane_bytes, "warps_per_cta": tile}


FAMILIES: tuple[EmitterFamily, ...] = (
  EmitterFamily("q4_g3_gemv.v1", ("Q4_K",), "uint32",
                (_row(32, 1, 4), _row(32, 1, 16), _row(128, 16, 16, "shared")), False,
                _q4_parameters),
  EmitterFamily("q6_coop_gemv.v1", ("Q6_K",), "uint16",
                tuple(_row(t * _POS_LANES, t, 2, stages=a) for t in (1, 2, 4) for a in (1, 2, 4)), True,
                _q6_parameters),
  # rows here are the route's own choice (block_quant_gemv_geometry); a caller proposes its own from the chip's facts
  EmitterFamily(BLOCK_QUANT_FAMILY, ("Q8_0", "Q5_0", "Q4_0", "Q5_K"), "uint16", (), False, _bq_parameters),
  # warps per row are the search dimension; a caller proposes them from the chip's thread limit
  EmitterFamily(Q6_FOUR_WARP_FAMILY, ("Q6_K",), "uint16", (), False, _q6_four_warp_parameters),
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


def installed_row(family:EmitterFamily, backend:str | None, architecture:str | None) -> dict[str, Any] | None:
  """The row the model's decode installs for this family on this device, by the fork's own routing rules."""
  if family.family == "q4_g3_gemv.v1":
    from types import SimpleNamespace
    from tinygrad.llm.decode_routes import _q4k_single_projection_load_style
    linear = SimpleNamespace(route_admission=SimpleNamespace(capability=SimpleNamespace(backend=backend, architecture=architecture)))
    style = _q4k_single_projection_load_style(linear)
    return _row(32, 1, 16 if style == "vector" else 4)
  if family.family in (BLOCK_QUANT_FAMILY, Q6_FOUR_WARP_FAMILY):
    return None  # shape- and device-dependent: the family's installed_rule (and installed_block_quant_row) say it
  from tinygrad.llm.decode_kernels import q6k_coop_row_tile_for_target
  tile = q6k_coop_row_tile_for_target(backend, architecture)
  return _row(tile * _POS_LANES, tile, 2)


def block_quant_installed_rule() -> dict[str, int]:
  """block_quant_gemv_geometry's measured constants, as data: a caller applies them to a role's shape and its chip."""
  from tinygrad.llm.decode_kernels import BLOCK_QUANT_MAX_WARPS_PER_CTA, BLOCK_QUANT_MIN_CTAS_PER_SM, BLOCK_QUANT_UNITS_PER_LANE
  return {"units_per_lane": BLOCK_QUANT_UNITS_PER_LANE, "max_warps_per_cta": BLOCK_QUANT_MAX_WARPS_PER_CTA,
          "min_ctas_per_sm": BLOCK_QUANT_MIN_CTAS_PER_SM}


def installed_block_quant_row(quant:str, rows:int, k:int, facts:Mapping[str, Any]) -> dict[str, Any]:
  """The row the decode route installs for one block-quant role: block_quant_gemv_geometry on the device's facts."""
  from tinygrad.llm.decode_kernels import BLOCK_QUANT_LAYOUTS, block_quant_gemv_geometry
  from tinygrad.llm.qk_layout import QUANT_FORMATS
  spec = block_quant_gemv_geometry(QUANT_FORMATS[quant], rows, k, wave_size=facts.get("wave_size"), sm_count=facts.get("sm_count"),
                                   max_threads=facts.get("max_threads"))
  return _block_quant_row(BLOCK_QUANT_LAYOUTS[quant].units * 2 // spec.lanes_per_block, spec.warps_per_cta, spec.wave_size,
                          QUANT_FORMATS[quant].block_elems)


def build(candidate:Mapping[str, Any], device:str, raw=None):
  """The exact-shape program for one emitter candidate: (output, words, x, call inputs) on `device`, with
  deterministic GGUF bytes. The caller schedules output to get the one program."""
  import numpy as np
  from tinygrad import Tensor, dtypes
  from extra.llm_research.semantic_kernel_lowering import build_registered_llm_emitter
  schedule, workload = candidate["schedule"], candidate["workload"]
  family = BY_FAMILY.get(schedule["compute"]["family"])
  if family is None: raise ValueError(f"no decode emitter family {schedule['compute']['family']!r}")
  from tinygrad.llm.qk_layout import QUANT_FORMATS, packed_nbytes
  quant = next((q for q in family.quants if q.replace("_", "") == str(workload["operands"]["b"]["quantization"]).upper().replace("_", "")), None)
  if quant is None: raise ValueError(f"{family.family} reads {list(family.quants)}, not {workload['operands']['b']['quantization']}")
  rows, k = int(workload["shape"]["n"]), int(workload["shape"]["k"])
  if family.family in (BLOCK_QUANT_FAMILY, Q6_FOUR_WARP_FAMILY):
    params = family.parameters(schedule, rows, k, quant, int(workload["target"]["subgroup_size"]))
  else: params = family.parameters(schedule, rows, k)
  emitter = build_registered_llm_emitter(family.family, params)
  raw = fixture_bytes(quant, rows, k) if raw is None else raw
  if raw.nbytes != packed_nbytes(QUANT_FORMATS[quant], rows * k): raise ValueError(f"{raw.nbytes} weight bytes are not {rows}x{k} {quant}")
  storage = dtypes.uint32 if family.storage == "uint32" else dtypes.uint16
  words = Tensor(raw.view(np.uint32 if family.storage == "uint32" else np.uint16), dtype=storage, device=device).realize()
  x = Tensor(fixture_activation(k), dtype=dtypes.float16, device=device).realize()
  out = Tensor.empty(rows, dtype=dtypes.float32, device=device).uop_program(words, x, fxn=emitter)[0]
  return out, words, x, raw, quant


def fixture_bytes(quant:str, rows:int, k:int):
  """Deterministic, finite GGUF blocks (the provider's fixture rule: small nonzero bytes, fixed fp16 scales)."""
  import numpy as np
  from tinygrad.llm.qk_layout import QUANT_FORMATS, packed_nbytes
  block = QUANT_FORMATS[quant].block_bytes
  raw = (1 + np.arange(packed_nbytes(QUANT_FORMATS[quant], rows * k), dtype=np.uint32) % 13).astype(np.uint8).reshape(-1, block)
  d, dmin = np.array([0.01], dtype=np.float16).view(np.uint8), np.array([0.005], dtype=np.float16).view(np.uint8)
  if quant == "Q6_K": raw[:, 208:210] = d  # Q6_K keeps d last
  else: raw[:, 0:2] = d                     # every other GGML block opens with d
  if quant == "Q4_K": raw[:, 2:4] = dmin
  return raw.reshape(-1)


def reference(quant:str, raw, elements:int):
  """The dequantized weights (float32 numpy) of `raw` packed bytes: the layout module's canonical Q4_K/Q6_K references,
  tinygrad's GGUF dequantizer for every other format."""
  from tinygrad import Tensor
  from tinygrad.llm.qk_layout import GGML_TYPE_IDS
  if quant in ("Q4_K", "Q6_K"):
    from extra.llm_research.layout import q4_k_reference, q6_k_reference
    return (q4_k_reference if quant == "Q4_K" else q6_k_reference)(Tensor(raw), elements).numpy().reshape(-1)
  from tinygrad.llm.gguf import ggml_data_to_tensor
  return ggml_data_to_tensor(Tensor(raw), elements, GGML_TYPE_IDS[quant]).numpy().reshape(-1)


def fixture_activation(k:int):
  import numpy as np
  return np.array([1.0 + (i % 7) / 16 for i in range(k)], dtype=np.float16)


__all__ = ["BLOCK_QUANT_FAMILY", "Q6_FOUR_WARP_FAMILY", "BY_FAMILY", "EMITTER_PLAN_KIND", "installed_block_quant_row", "reference", "FAMILIES", "EmitterFamily", "build", "emitters_apply", "fixture_activation",
           "fixture_bytes", "installed_row"]
