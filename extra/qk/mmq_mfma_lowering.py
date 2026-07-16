"""Narrow evidence adapter for the cooperative MMQ -> AMD CDNA MFMA path.

This module intentionally does not rewrite UOps, select routes, or emit code.  It
records the compiler boundary that must already have been crossed and rejects
claims which can only be inferred from a logical tile shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

MFMA_SCHEMA = "tinygrad.mmq.mfma_lowering.v1"


@dataclass(frozen=True)
class MFMAContract:
  arch: str
  shape: tuple[int, int, int]
  input_dtype: str
  accumulator_dtype: str
  wave_size: int
  a_fragment: tuple[int, ...]
  b_fragment: tuple[int, ...]
  accumulator: tuple[int, ...]

  def to_dict(self) -> dict[str, Any]:
    return {"schema": MFMA_SCHEMA, **self.__dict__}


def _require(payload: Mapping[str, Any], *keys: str) -> Any:
  value: Any = payload
  for key in keys:
    if not isinstance(value, Mapping) or value.get(key) is None:
      raise ValueError("incomplete MFMA lowering evidence: missing " + ".".join(keys))
    value = value[key]
  return value


def adapt_mfma_evidence(evidence: Mapping[str, Any]) -> MFMAContract:
  """Validate final compiler evidence for one CDNA ``16x16x16`` MFMA.

  The fragment vectors are the CDNA 64-lane contract: four half values per
  lane for A/B and four float values per lane for C/D.  No defaults are used.
  """
  backend = str(_require(evidence, "target", "backend"))
  arch = str(_require(evidence, "target", "arch"))
  if backend != "AMD" or arch not in {"gfx942", "gfx950"}:
    raise ValueError("MFMA adapter requires AMD CDNA gfx942/gfx950 evidence")
  if _require(evidence, "lowering", "op") != "WMMA":
    raise ValueError("MFMA adapter requires compiler lowering op WMMA")
  shape = tuple(_require(evidence, "lowering", "shape"))
  if shape != (16, 16, 16):
    raise ValueError("cooperative MMQ MFMA shape must be (16, 16, 16)")
  if _require(evidence, "lowering", "input_dtype") != "half" or _require(evidence, "lowering", "accumulator_dtype") != "float":
    raise ValueError("cooperative MMQ MFMA currently supports half inputs and float accumulators only")
  if int(_require(evidence, "target", "wave_size")) != 64:
    raise ValueError("CDNA MFMA requires wave size 64")
  if tuple(_require(evidence, "operands", "a_fragment")) != (4,) or tuple(_require(evidence, "operands", "b_fragment")) != (4,) or tuple(_require(evidence, "operands", "accumulator")) != (4,):
    raise ValueError("MFMA operand fragment vectors must be half4, half4, float4")
  instructions = str(_require(evidence, "instruction_evidence"))
  if "llvm.amdgcn.mfma.f32.16x16x16.f16" not in instructions and "v_mfma_f32_16x16x16_f16" not in instructions:
    raise ValueError("final instruction evidence has no f32 16x16x16 f16 MFMA")
  return MFMAContract(arch, shape, "half", "float", 64, (4,), (4,), (4,))


__all__ = ["MFMA_SCHEMA", "MFMAContract", "adapt_mfma_evidence"]
