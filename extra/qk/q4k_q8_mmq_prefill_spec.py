"""Machine-search vocabulary for Q4_K x Q8_1 prefill (no kernel emitter)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
from extra.qk.prefill_primitive_spec import PrefillPrimitiveSpec, PrimitiveABI, LaunchMetadata, target_capabilities

_STAGING = ("register", "lds")
_WRITEBACK = ("owner", "partials")
_ACTIVATION_LAYOUT = "q8_1_ds4"
_TILE_LAYOUTS = ("tokens_k", "rows_k")

@dataclass(frozen=True)
class Q4KQ8MMQPrefillSpec(PrefillPrimitiveSpec):
  q4k_group_size: int = 32
  q8_block_size: int = 32
  activation_layout: str = "q8_1_ds4"
  tile_x_layout: str = "tokens_k"
  tile_y_layout: str = "rows_k"
  tile_m: int = 16
  tile_n: int = 16
  tile_k: int = 256
  wave_width: int = 32
  workgroup_size: int = 64
  accumulator_slots: int = 4
  staging_strategy: str = "register"
  writeback_strategy: str = "owner"
  lds_bytes: int = 0

  def validate(self) -> None:
    super().validate()
    if (self.quant_format, self.activation_format) != ("Q4_K", "Q8_1"): raise ValueError("MMQ requires Q4_K and Q8_1")
    if self.activation_layout != _ACTIVATION_LAYOUT: raise ValueError(f"MMQ requires activation_layout={_ACTIVATION_LAYOUT!r}")
    if self.tile_x_layout not in _TILE_LAYOUTS or self.tile_y_layout not in _TILE_LAYOUTS: raise ValueError("unsupported MMQ tile layout")
    if self.k % self.q4k_group_size or self.k % self.q8_block_size: raise ValueError("K violates Q4/Q8 alignment")
    if min(self.tile_m, self.tile_n, self.tile_k, self.accumulator_slots, self.wave_width, self.workgroup_size) <= 0: raise ValueError("tile/resource fields must be positive")
    caps = target_capabilities(self.target)
    if self.wave_width != caps["wave_width"]: raise ValueError(f"wave width {self.wave_width} is invalid for target {self.target}; expected {caps['wave_width']}")
    if self.workgroup_size > caps["max_workgroup_size"]: raise ValueError("workgroup size exceeds target capability")
    if self.workgroup_size != 64: raise ValueError("wave/workgroup size is not lowered; only workgroup 64 is supported")
    if self.workgroup_size % self.wave_width or self.workgroup_size // self.wave_width > 16: raise ValueError("invalid wave/workgroup mapping")
    if self.staging_strategy not in _STAGING or self.writeback_strategy not in _WRITEBACK: raise ValueError("unsupported staging/writeback strategy")
    # These are descriptor facts only until a corresponding lowering exists.
    if self.accumulator_slots != 4: raise ValueError("accumulator_slots is not lowered; only 4 is supported")
    if self.staging_strategy != "register": raise ValueError("staging_strategy is not lowered; only register is supported")
    if self.writeback_strategy != "owner": raise ValueError("writeback_strategy is not lowered; only owner is supported")
    if self.abi != PrimitiveABI(): raise ValueError("ABI variant is not lowered; only the canonical MMQ ABI is supported")
    if self.schedule_options: raise ValueError("schedule_options are not lowered; use the typed MMQ geometry fields")
    if self.tile_k % self.q4k_group_size or self.tile_k % self.q8_block_size: raise ValueError("tile_k violates quantization alignment")
    if self.lds_bytes < 0 or self.lds_bytes > 64 * 1024: raise ValueError("LDS budget exceeded")
    if self.writeback_strategy == "owner" and self.parts != 1: raise ValueError("owner writeback requires parts==1")

  def to_json(self) -> dict[str, Any]:
    d = super().to_json(); d["mmq"] = {k: getattr(self, k) for k in ("q4k_group_size","q8_block_size","activation_layout","tile_x_layout","tile_y_layout","tile_m","tile_n","tile_k","wave_width","workgroup_size","accumulator_slots","staging_strategy","writeback_strategy","lds_bytes")}; return d

def enumerate_q4k_q8_mmq_candidates(base: Q4KQ8MMQPrefillSpec, **axes: Iterable[Any]):
  """Yield only descriptors that validate; axes are the machine's decisions."""
  inert = {"accumulator_slots", "staging_strategy", "writeback_strategy", "wave_width", "workgroup_size", "lds_bytes"}
  requested_inert = sorted(inert.intersection(axes))
  if requested_inert: raise ValueError(f"inert search axes are not supported: {', '.join(requested_inert)}")
  keys = ("tile_m", "tile_n", "tile_k")
  import itertools
  for values in itertools.product(*(tuple(axes.get(k, (getattr(base, k),))) for k in keys)):
    yield Q4KQ8MMQPrefillSpec(**{**base.__dict__, **dict(zip(keys, values))})

def emit_q4k_q8_mmq_kernel(spec: Q4KQ8MMQPrefillSpec):
  raise NotImplementedError("MMQ lowering is intentionally not implemented by the primitive contract")
