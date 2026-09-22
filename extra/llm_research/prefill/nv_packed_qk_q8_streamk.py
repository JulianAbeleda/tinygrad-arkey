"""Isolated Phase-1 Q4_K x on-device Q8 Stream-K gate/up primitive.

This module deliberately has no model attachment or default-on route.  It
reuses the repository's qualified native Stream-K body, whose ABI consumes
canonical packed Q4_K words and the Q8 record planes directly.
"""
from __future__ import annotations
from dataclasses import dataclass
from extra.llm_research.prefill.nv_q4_imma_provider import (
  M, N, K, MAIN_GRID, FIXUP_GRID, BLOCK, PARTIAL_SLOTS, compile_provider,
)
from extra.llm_research.prefill.nv_native_program_uop import native_nv_program

RECORD_BYTES = M*K + 2*M*(K//32)*4
WORKSPACE_BYTES = PARTIAL_SLOTS*128*128*4 + PARTIAL_SLOTS*4 + FIXUP_GRID[0]*2*4

@dataclass(frozen=True)
class LaunchMetadata:
  producer_grid: tuple[int, int, int] = (M, 8, 1)
  producer_block: tuple[int, int, int] = (128, 1, 1)
  main_grid: tuple[int, int, int] = MAIN_GRID
  main_block: tuple[int, int, int] = BLOCK
  fixup_grid: tuple[int, int, int] = FIXUP_GRID
  fixup_block: tuple[int, int, int] = BLOCK
  workspace_bytes: int = WORKSPACE_BYTES
  schedule: str = "stream-k split-K with deterministic slot-map fixup"

@dataclass
class Candidate:
  provider: object
  metadata: LaunchMetadata = LaunchMetadata()

  @classmethod
  def compile(cls, device):
    provider=compile_provider(device)
    metadata=LaunchMetadata(main_grid=(provider.geometry.owners,1,1),
      workspace_bytes=provider.geometry.partial_slots*128*128*4+provider.geometry.partial_slots*4+provider.geometry.fixup_grid*2*4)
    return cls(provider, metadata)

  def launch(self, out, partials, ids, words, q8, scales, sums, slotmap, *, wait=True):
    return self.provider.launch(out, partials, ids, words, q8, scales, sums, slotmap, wait=wait)

  def launch_main(self, out, partials, ids, words, q8, scales, sums, *, wait=True):
    return self.provider.main(out, partials, ids, words, q8, scales, sums,
                              vals=(M, N, K), global_size=(self.provider.geometry.owners,1,1),
                              local_size=BLOCK, wait=wait)

  def launch_fixup(self, out, partials, slotmap, *, wait=True):
    return self.provider.fixup(out, partials, slotmap, vals=(M, N),
                               global_size=FIXUP_GRID, local_size=BLOCK,
                               wait=wait)

  def programs(self):
    return self.provider.main, self.provider.fixup

def compile_candidate(device):
  return Candidate.compile(device)

def supports(m=M, n=N, k=K, role="ffn_gate"):
  return (m, n, k) == (M, N, K) and role in ("ffn_gate", "ffn_up")
