"""Fail-closed research schedule descriptors for compiler-owned Q6_K."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib, json

@dataclass(frozen=True)
class CompilerQ6ScheduleConfig:
  tile_m: int = 64
  tile_n: int = 32
  tile_k: int = 64
  warp_m: int = 2
  warp_n: int = 2
  threads: int = 128
  tc_select: int = -1
  tc_opt: int = 2
  use_tc: int = 1
  tc_axis: int = 0

  def validate(self, m: int = 512, n: int = 4096, k: int = 12288) -> "CompilerQ6ScheduleConfig":
    if min(self.tile_m, self.tile_n, self.tile_k, self.warp_m, self.warp_n, self.threads) <= 0:
      raise ValueError("schedule dimensions must be positive")
    if self.warp_m * self.warp_n != 4 or self.threads != 128:
      raise ValueError("Q6 research schedule requires warp product 4 and 128 threads")
    if self.tc_select not in (-1, 8) or self.tc_opt not in (0, 1, 2) or self.use_tc != 1 or self.tc_axis != 0:
      raise ValueError("unsupported Q6 tensor-core selector")
    if any(x % y for x,y in ((m,self.tile_m),(n,self.tile_n),(k,self.tile_k))):
      raise ValueError("tiles must divide the admitted role shape")
    if self.tile_m * self.tile_n > 8192: raise ValueError("tile exceeds conservative register bound")
    return self

  def identity(self) -> str:
    self.validate()
    return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

  def lds_windows(self) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return A/B (offset, bytes, row-stride) with non-overlap validation."""
    self.validate()
    stride = self.tile_k + (self.tile_k // 16) * 4
    a_bytes, b_bytes = self.tile_m * stride, self.tile_n * stride
    a = (0, a_bytes, stride); b = (a_bytes, b_bytes, stride)
    if a[0] + a[1] > b[0] or b[0] + b[1] > 48 * 1024:
      raise ValueError("Q6 LDS windows overlap or exceed 48KiB")
    return a, b

  def to_json(self) -> dict[str, object]:
    self.validate(); return {**asdict(self), "identity": self.identity(), "default": self == CompilerQ6ScheduleConfig()}

VARIANTS = tuple(CompilerQ6ScheduleConfig(warp_m=a, warp_n=b) for a,b in ((2,2),(4,1),(1,4)))
