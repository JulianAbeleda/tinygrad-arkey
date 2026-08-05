#!/usr/bin/env python3
"""CPU-only construction oracle for the d512 single-stage flash-decode candidate.

This does not install a route.  It makes the proposed workgroup ownership and
resource/topology contract executable before an ordinary-UOp emitter is built.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse, json, math


@dataclass(frozen=True)
class SingleStageFlashPlan:
  query_heads: int = 32
  kv_heads: int = 8
  head_dim: int = 128
  splits: int = 4
  lanes: int = 32
  token_tile: int = 16
  output_fp16: bool = True

  @property
  def gqa(self): return self.query_heads // self.kv_heads
  @property
  def warps(self): return self.splits * self.gqa
  @property
  def threads(self): return self.warps * self.lanes
  @property
  def partial_floats(self): return self.splits * self.gqa * (self.head_dim + 2)
  @property
  def partial_bytes(self): return self.partial_floats * 4
  @property
  def tile_bytes(self): return self.splits * self.token_tile * self.head_dim * 2 * 2  # K + V, fp16
  @property
  def shared_bytes_upper_bound(self): return self.tile_bytes + self.partial_bytes
  @property
  def old_workgroups(self): return self.kv_heads * self.splits
  @property
  def new_workgroups(self): return self.kv_heads

  def owner(self, warp:int) -> tuple[int, int]:
    if not 0 <= warp < self.warps: raise ValueError(warp)
    return warp // self.gqa, warp % self.gqa

  def validate(self):
    if self.query_heads % self.kv_heads: raise ValueError("query_heads must divide kv_heads")
    if self.head_dim % self.lanes: raise ValueError("head_dim must divide lanes")
    if self.warps > 32 or self.threads > 1024: raise ValueError("ordinary workgroup thread limit exceeded")
    if self.shared_bytes_upper_bound > 96 * 1024: raise ValueError("candidate exceeds conservative 96 KiB shared-memory gate")
    owners = {self.owner(w) for w in range(self.warps)}
    expected = {(s, h) for s in range(self.splits) for h in range(self.gqa)}
    if owners != expected: raise ValueError("warp ownership is not bijective")

  def topology(self):
    self.validate()
    return {
      "schema": "tinygrad.flash_single_stage_plan.v1",
      "route_default": "closed",
      "grid": [self.kv_heads, 1, 1],
      "block": [self.lanes, self.warps, 1],
      "warp_owner": "split=warp//gqa, grouped_head=warp%gqa",
      "partial_exchange": {"address_space": "LOCAL", "bytes": self.partial_bytes,
                           "layout": "[split,gqa_head,head_dim+den+max]"},
      "tile_staging_upper_bound_bytes": self.tile_bytes,
      "shared_upper_bound_bytes": self.shared_bytes_upper_bound,
      "old_programs_per_layer": 2,
      "candidate_programs_per_layer": 1,
      "old_workgroups": self.old_workgroups,
      "candidate_workgroups": self.new_workgroups,
      "association_contract": "each warp emits the legacy split state; split-0 warp merges s=0..S-1 in legacy combine order",
      "output_contract": "direct [Hq,Hd] fp16" if self.output_fp16 else "direct [Hq,Hd] fp32",
    }


def ordered_split_merge(states:list[tuple[list[float], float, float]]) -> list[float]:
  """CPU oracle for flash_fused_gmax_combine_kernel's exact split association order."""
  if not states or any(len(s[0]) != len(states[0][0]) for s in states): raise ValueError("invalid split states")
  maximum = -1e30
  for _, _, mx in states: maximum = max(maximum, mx)
  acc, den = [0.0] * len(states[0][0]), 0.0
  for values, split_den, mx in states:
    weight = math.exp2((mx - maximum) * 1.4426950408889634)
    for i, value in enumerate(values): acc[i] += weight * value
    den += weight * split_den
  return [value / den for value in acc]


def merge_warp_owned_states(plan:SingleStageFlashPlan, warp_states:list[tuple[list[float], float, float]], grouped_head:int) -> list[float]:
  """Candidate LOCAL exchange: gather the same split order from the warp-owner layout."""
  plan.validate()
  if len(warp_states) != plan.warps or not 0 <= grouped_head < plan.gqa: raise ValueError("invalid warp states/head")
  return ordered_split_merge([warp_states[s * plan.gqa + grouped_head] for s in range(plan.splits)])


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--json", action="store_true")
  args = ap.parse_args()
  plan = SingleStageFlashPlan()
  payload = {"inputs": asdict(plan), **plan.topology()}
  print(json.dumps(payload, sort_keys=True, indent=2 if args.json else None))


if __name__ == "__main__": main()
