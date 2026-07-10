from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from extra.qk.runtime_specs import ActivationQuantSpec, QuantizedTensorSpec


@dataclass(frozen=True)
class QuantFormatSpec:
  format: str
  block_size: int
  group_size: int
  scale_layout: str
  min_layout: str = ""
  signed: bool = False
  supported_activation_formats: tuple[str, ...] = ("fp16",)

  def tensor_spec(self) -> QuantizedTensorSpec:
    return QuantizedTensorSpec(self.format, block_size=self.block_size, group_size=self.group_size,
                               scale_layout=self.scale_layout, min_layout=self.min_layout, signed=self.signed)

  def to_json(self) -> dict[str, Any]:
    return {"format": self.format, "block_size": self.block_size, "group_size": self.group_size,
            "scale_layout": self.scale_layout, "min_layout": self.min_layout, "signed": self.signed,
            "supported_activation_formats": list(self.supported_activation_formats)}


@dataclass(frozen=True)
class ActivationFormatSpec:
  format: str
  block_size: int | None
  signed: bool | None
  scale_layout: str = ""

  def activation_spec(self) -> ActivationQuantSpec:
    return ActivationQuantSpec(self.format, block_size=self.block_size, signed=self.signed, scale_layout=self.scale_layout)

  def to_json(self) -> dict[str, Any]:
    return {"format": self.format, "block_size": self.block_size, "signed": self.signed,
            "scale_layout": self.scale_layout}


QUANT_FORMAT_SPECS: dict[str, QuantFormatSpec] = {
  "Q4_K": QuantFormatSpec("Q4_K", block_size=256, group_size=32, scale_layout="ggml_q4_k.scales",
                          min_layout="ggml_q4_k.mins", signed=False,
                          supported_activation_formats=("fp16", "Q8_1")),
  "Q6_K": QuantFormatSpec("Q6_K", block_size=256, group_size=16, scale_layout="ggml_q6_k.scales",
                          signed=True, supported_activation_formats=("fp16",)),
}

ACTIVATION_FORMAT_SPECS: dict[str, ActivationFormatSpec] = {
  "Q8_1": ActivationFormatSpec("Q8_1", block_size=32, signed=True, scale_layout="block_scale_sum"),
  "fp16": ActivationFormatSpec("fp16", block_size=None, signed=None),
  "fp32": ActivationFormatSpec("fp32", block_size=None, signed=None),
  "none": ActivationFormatSpec("none", block_size=None, signed=None),
}


def quant_spec(fmt:str) -> QuantFormatSpec:
  key = fmt.upper()
  if key not in QUANT_FORMAT_SPECS: raise KeyError(f"unknown quant format {fmt!r}")
  return QUANT_FORMAT_SPECS[key]


def activation_spec(fmt:str) -> ActivationFormatSpec:
  if fmt not in ACTIVATION_FORMAT_SPECS: raise KeyError(f"unknown activation format {fmt!r}")
  return ACTIVATION_FORMAT_SPECS[fmt]
