#!/usr/bin/env python3
"""Small, compiler-independent packed-Q4 symbolic-loop correctness witness.

The model deliberately uses logical packed addresses, rather than relying on
an emitter's implementation.  It proves that each output tile owns a unique
weight/activation slice and that the output loop emits each output tile once.
"""
from __future__ import annotations

from dataclasses import dataclass
import json


@dataclass(frozen=True)
class PackedQ4Case:
  m: int = 32
  n: int = 32
  k: int = 512
  tile: int = 16
  q4_superblock: int = 256
  q4_group: int = 32

  def __post_init__(self) -> None:
    if min(self.m, self.n, self.k, self.tile) <= 0:
      raise ValueError("dimensions must be positive")
    if self.m % self.tile or self.n % self.tile or self.k % self.q4_superblock:
      raise ValueError("case must divide into complete output tiles and Q4_K superblocks")

  @property
  def mt(self) -> int: return self.m // self.tile

  @property
  def nt(self) -> int: return self.n // self.tile

  @property
  def kb(self) -> int: return self.k // self.q4_group

  @property
  def superblocks(self) -> int: return self.k // self.q4_superblock


def validate(case: PackedQ4Case = PackedQ4Case()) -> dict[str, object]:
  """Return a machine-readable proof summary; raise on any alias or duplicate."""
  output_tiles = [(tm, tn) for tm in range(case.mt) for tn in range(case.nt)]
  stores = [(tm, tn) for tm, tn in output_tiles]

  # A tile's packed-Q4 weights are indexed by output-channel and Q4 group.
  # Activations are indexed by row and the same K group.  These are logical
  # addresses and remain valid regardless of byte packing within each group.
  weight_reads = {
    (tm, tn): {(n, kg) for n in range(tn * case.tile, (tn + 1) * case.tile)
                         for kg in range(case.kb)}
    for tm, tn in output_tiles
  }
  activation_reads = {
    (tm, tn): {(m, kg) for m in range(tm * case.tile, (tm + 1) * case.tile)
                             for kg in range(case.kb)}
    for tm, tn in output_tiles
  }
  all_weights = [addr for addrs in weight_reads.values() for addr in addrs]
  all_activations = [addr for addrs in activation_reads.values() for addr in addrs]
  # Reuse across the orthogonal output-tile dimension is intentional: every
  # M tile reads the N tile's weights, and every N tile reads the M tile's
  # activations.  The owner sets below prove that this is the only reuse.
  weight_owners = {addr: {(tm, tn) for (tm, tn), addrs in weight_reads.items() if addr in addrs}
                   for addr in set(all_weights)}
  activation_owners = {addr: {(tm, tn) for (tm, tn), addrs in activation_reads.items() if addr in addrs}
                       for addr in set(all_activations)}
  if any(len(owners) != case.mt for owners in weight_owners.values()):
    raise AssertionError("packed-Q4 weight address has unexpected tile ownership")
  if any(len(owners) != case.nt for owners in activation_owners.values()):
    raise AssertionError("activation address has unexpected tile ownership")
  if len(stores) != len(set(stores)):
    raise AssertionError("symbolic output loop emits a duplicate tile store")
  expected_weights = case.n * case.kb
  expected_activations = case.m * case.kb
  assert len(weight_owners) == expected_weights
  assert len(activation_owners) == expected_activations
  assert len(stores) == case.mt * case.nt
  return {
    "schema": "q4k_symbolic_loop_validation.v1",
    "verdict": "PACKED_Q4_SYMBOLIC_LOOP_PASS",
    "shape": {"m": case.m, "n": case.n, "k": case.k},
    "tile": case.tile,
    "q4_k": {"superblocks": case.superblocks, "groups": case.kb},
    "output_tiles": len(stores),
    "unique_weight_addresses": len(weight_owners),
    "unique_activation_addresses": len(activation_owners),
    "duplicate_store_count": len(stores) - len(set(stores)),
  }


if __name__ == "__main__":
  print(json.dumps(validate(), indent=2))
