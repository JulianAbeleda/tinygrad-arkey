"""Chain per-K32-group fp16 recurrences into a K256 epoch and across epochs.

Phase-1 shrank the hierarchical stage granularity to exactly one K32 group
(``mmq_llama_candidate_plan.py`` ``_geometry()``): both A (Q4 decode) and B
(plain fp16 activation) are re-decoded/re-staged every K32 group, so there is
no persistent/overwriteable multi-phase residency left to exploit within one
``HierarchicalPackedRecordStage`` -- the old "phase-major" machinery (2
Q8-phases x 4 K32-groups per K256 epoch, sharing one decode producer and
carrying a genuine internal WMMA-accumulator data recurrence across groups)
is retired.

Each K32-group ``LlamaOracleRecurrenceGraph`` here is instead independently
zero-seeded (``recurrence.initial``): its ``update`` is exactly this one
group's own partial dot-product contribution, with no algebraic dependency on
any other group's accumulator value.  Cross-group accumulation is therefore
plain addition -- what the groups DO still share is the single-buffered 16KB
LDS arena (``DBUF=0``, wmma.py I.3), so group g+1's LDS-overwriting producer
must be ordered after group g's LDS reads (its WMMA fragment loads) complete.
That ordering is carried only on the cheap producer/barrier graph
(``_producer_after_release``); the O(groups) SGPR/VGPR-sized arithmetic is
never re-walked/re-embedded across groups, unlike the retired phase-major
design -- doing so for a 4x deeper (8 vs. 2) per-epoch chain was measured to
blow up substitution cost superlinearly (see implementation notes).
"""
from __future__ import annotations

from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_llama_oracle_recurrence import LlamaOracleRecurrenceGraph


def _producer_after_release(producer: UOp, release: UOp) -> UOp:
  """Order the first LDS write after a collective release; producer-local store order carries the rest."""
  if producer.op is not Ops.GROUP or not producer.src or any(x.op is not Ops.STORE for x in producer.src):
    raise ValueError("group-chain staging requires a GROUP of LDS stores")
  first = producer.src[0]
  if first.src[0].op is not Ops.INDEX: raise ValueError("group-chain staging store lacks an INDEX address")
  address = first.src[0]
  guarded_address = address.replace(src=(address.src[0].after(release),)+address.src[1:])
  guarded_first = first.replace(src=(guarded_address,)+first.src[1:])
  return producer.substitute({first: guarded_first}, walk=True)


def _instantiate_group_subtiles(recurrence: LlamaOracleRecurrenceGraph, publish: UOp) -> tuple[tuple[UOp, ...], ...]:
  """Concretize the eight symbolic ``subtile_n`` values for one independently zero-seeded K32-group stage."""
  phase, subtile = recurrence.phases[0], recurrence.stage.subtile_n
  group = phase.groups[0]
  results = []
  for element in range(8):
    substitutions = {subtile: UOp.const(dtypes.weakint, element), phase.publish: publish}
    results.append(tuple(x.substitute(substitutions, walk=True) for x in group.update))
  return tuple(results)


def chain_group_stage(recurrence: LlamaOracleRecurrenceGraph, ordinal: int,
                      totals: tuple[tuple[UOp, ...], ...] | None,
                      prior_release: UOp | None) -> tuple[tuple[tuple[UOp, ...], ...], UOp]:
  """Order one K32-group recurrence behind the previous group's LDS-reuse release, then add its contribution."""
  persistent, producer = recurrence.stage.persistent_producer, recurrence.phases[0].producer
  if prior_release is not None:
    persistent = _producer_after_release(persistent, prior_release)
    producer = _producer_after_release(producer, prior_release)
  publish = UOp.barrier(UOp.group(persistent, producer)).replace(tag=("llama_group_chain_publish", ordinal))
  lanes = _instantiate_group_subtiles(recurrence, publish)
  release = UOp(Ops.BARRIER, dtypes.void, tuple(x for lane in lanes for x in lane)).replace(
    tag=("llama_group_chain_collective_release", ordinal))
  if totals is None:
    combined = lanes
  else:
    combined = tuple(tuple((prior+value).replace(tag=("llama_group_chain_fp32_join", ordinal, element, lane))
                           for lane, (prior, value) in enumerate(zip(prior_lanes, group_lanes)))
                     for element, (prior_lanes, group_lanes) in enumerate(zip(totals, lanes)))
  return combined, release


def group_major_accumulator_vectors(recurrences: tuple[LlamaOracleRecurrenceGraph, ...]) -> tuple[UOp, ...]:
  """Sum every independently zero-seeded K32-group recurrence's contribution, in LDS-reuse order."""
  if not recurrences: raise ValueError("group-major writeback requires at least one K32-group recurrence")
  totals: tuple[tuple[UOp, ...], ...] | None = None
  prior_release = None
  for ordinal, recurrence in enumerate(recurrences):
    totals, prior_release = chain_group_stage(recurrence, ordinal, totals, prior_release)
  assert totals is not None
  return tuple(UOp(Ops.STACK, dtypes.float.vec(8), lanes) for lanes in totals)


__all__ = ["chain_group_stage", "group_major_accumulator_vectors"]
