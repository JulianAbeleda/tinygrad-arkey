"""Hardware facts and workload requests for research MMQ."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class MMQHardwareCapability:
  capability_id: str
  backend: str = "AMD"
  arch: str = "gfx1100"
  wave_width: int = 32
  max_lds_bytes: int = 64 * 1024
  def validate(self) -> None:
    if not self.capability_id or self.wave_width <= 0 or self.max_lds_bytes < 0: raise ValueError("invalid MMQ hardware capability")

GFX11_MMQ_CAPABILITY = MMQHardwareCapability("amd.gfx1100.q4k_q8_1_mmq.research.v1")

@dataclass(frozen=True)
class MMQRequest:
  role: str
  workload: str = ""
  route: str = "direct_packed"
  research_only: bool = True
  def validate(self) -> None:
    if not isinstance(self.role, str) or not self.role: raise ValueError("role is required request data")
    if self.route != "direct_packed": raise ValueError("only direct_packed is supported")
    if self.research_only is not True: raise ValueError("MMQ route is research-only")

__all__ = ["MMQHardwareCapability", "GFX11_MMQ_CAPABILITY", "MMQRequest"]
