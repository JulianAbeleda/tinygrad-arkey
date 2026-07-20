"""Chain per-K32-group fp16 recurrences into a K256 epoch and across epochs.

Phase-1 shrank the hierarchical stage granularity to exactly one K32 group
(``mmq_llama_candidate_plan.py`` ``_geometry()``): both A (Q4 decode) and B
(plain fp16 activation) are re-decoded/re-staged every K32 group, so there is
no persistent/overwriteable multi-phase residency left to exploit within one
``HierarchicalPackedRecordStage`` -- the old "phase-major" machinery (2
Q8-phases x 4 K32-groups per K256 epoch, sharing one decode producer and
carrying a genuine internal WMMA-accumulator data recurrence across groups)
is retired.

Each K32-group ``LlamaOracleRecurrenceGraph`` is built independently
zero-seeded (``recurrence.initial``/``group.zero``): in isolation, its
``wmmas`` chain is exactly this one group's own partial dot-product
contribution. Cross-group accumulation used to be plain scalar addition of
each group's *already-decomposed* 8-lane result -- that gave every group its
own fresh, zero-seeded 8-VGPR WMMA accumulator chain (8 groups x 8 symbolic
``subtile_n`` elements == 64 independent WMMA chain heads: every "first" WMMA
of every (group, element) pair has a non-WMMA ``src[2]`` (a fresh CONST
zero), so ``AMDISARenderer._n_c_runs`` in ``tinygrad/renderer/isa/amd.py``
counts 64 separate physical C-accumulator runs and the low accumulator
window overflows the VGPR budget by 2x -- confirmed empirically: the
"vgpr lease exceeds virtual pool" gate fires while reserving exactly
``_n_c_runs(ctx)*8 == 512`` VGPRs).

The renderer's one escape hatch for many independent-looking WMMA chains
(``_progressive_c_assignment``/``_serialize_progressive_c_drains``, the
mechanism the retired int8 kernel relied on) turns out to be a dead end here:
its fragment-reuse proof (``extra/qk/amd_isa_renderer_policy.py``
``wmma_frag_buffer_proof_from_tag`` et al.) is hard-coded to the retired
int8 Q8_1 DS4 producer-witness schema (``"llama-q8-ds4-producer-instance.v1"``,
``hierarchical_record_store`` tag shapes) and returns ``None`` uniformly for
this fp16 per-K32-group stage's fragments (verified directly), so that path
can never engage for this kernel without also rewriting that int8-specific
proof code.

Instead, group g+1's WMMA chain is *seeded* with group g's own final WMMA
node (the raw ``dtypes.float.vec(8)`` UOp, not decomposed to scalars and
re-``STACK``ed): its symbolic zero-placeholder (``group.zero``) is swapped for
that concrete predecessor value. Because ``second.src[2] is first`` and
``first.src[2] is group.zero`` are exact UOp identities
(``prove_llama_oracle_recurrence``), the seed swap is a single O(1) node
reconstruction (re-pointing ``src[2]``) done AFTER the cheap, per-group-bounded
``subtile_n``-concretizing ``substitute`` walk -- never a further substitution
over the (growing) prior-chain ancestry. Substituting the seed in via another
``walk=True`` pass was tried and measured to blow up: each successive group's
walk would re-traverse the ever-growing embedded ancestor subtree from
scratch (no cross-call memoization), i.e. exactly the superlinear
substitution-embedding cost this module previously warned against. Plain
node reconstruction has no such cost: it only ever touches the two top nodes
being re-pointed, and referencing a prior UOp as an ``src`` is an O(1) DAG
edge, not a copy.

The result: only the very first group's WMMA (per ``subtile_n`` element, 8
total) has a non-WMMA ``src[2]``, so ``AMDISARenderer._n_c_runs`` counts
exactly 8 chain heads (one persistent fp32 accumulator per output subtile)
instead of 64 -- WITHOUT ever engaging ``_progressive_c_assignment`` or its
int8-specific fragment-reuse proof -- and every subsequent group's WMMA is
recognised as a plain chain continuation sharing that one physical C lease.
This mirrors the hand kernel's persistent-accumulator ``v_wmma``
(``src2==vdst`` across groups, ``extra/qk/prefill/wmma.py:600-631`` in the
hand-authored reference).

What the groups DO still share is the single-buffered 16KB LDS arena
(``DBUF=0``, wmma.py I.3), so group g+1's LDS-overwriting producer must be
ordered after group g's LDS reads (its WMMA fragment loads) complete. That
ordering is carried only on the cheap producer/barrier graph
(``_producer_after_release``).
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


def _instantiate_group_wmma_vectors(recurrence: LlamaOracleRecurrenceGraph, publish: UOp,
                                    seeds: tuple[UOp, ...] | None) -> tuple[UOp, ...]:
  """Concretize the eight symbolic ``subtile_n`` values, seeding each element's WMMA chain with the prior
  group's own final ``dtypes.float.vec(8)`` accumulator for that element (or the group's own zero seed on
  the first call). Returns the eight concrete final-WMMA vec8 UOps directly -- no scalar decomposition.

  The ``substitute(..., walk=True)`` below concretizes only THIS group's own small, bounded symbolic
  subtree (fragment loads + two WMMAs + zero seed): the symbolic recurrence built for each K32 group is
  always independently zero-seeded (never references another group), so this walk's cost never grows with
  the chain depth.  The seed swap that actually wires one group's accumulator into the next is instead a
  single O(1) node reconstruction (re-pointing ``src[2]``), not a further substitution -- substituting the
  seed in would force ``walk=True`` to re-traverse the ever-growing prior-chain ancestry on every group,
  which is exactly the superlinear substitution-embedding cost this module's docstring warns against.

  Cross-element WAR guard (group 0 / ``seeds is None`` only): AMDISARenderer's WMMA isel keys the A/B
  fragment's physical VGPR window on a *constant* ``"wmma_ab"`` id (register pressure), reusing the SAME
  registers for every one of the 8 ``subtile_n`` elements' chains. Isel recognizes each element's own chain
  by walking backward through consecutive ``Ops.WMMA`` ``src[2]`` links (the seed swap above) all the way to
  its group-0 zero seed -- this correctly threads a WAR-guard dependency *within* one element's own
  cross-group lineage (each later group's fragment reload waits on the previous group's WMMA read via the
  existing ``dep=(prev.src[0],)`` accumulate-tile guard elsewhere in ``isel_wmma``) -- but the 8 elements are
  otherwise-independent chains (never linked to each other via ``src[2]``), so isel visits each one as a
  separate top-level match with no ordering between them. Without an explicit edge, element e's chain-HEAD
  (group 0) fragment reload can be scheduled before element e-1's LAST WMMA has consumed the same physical
  window: REGALLOC_DEBUG_PRESSURE on build_llama_five_buffer_full_kernel(128,128,256) found several
  simultaneously-live DS_LOAD_B128 virtuals all constrained to the SAME single fixed VGPR, a hard (not just
  soft-pressure) allocation conflict. Every element's chain walk terminates at ITS OWN group-0 zero seed
  regardless of how many later groups it's chained through, so gating group 0's zero once (here) is
  sufficient -- no need to also touch ``seeds[element]`` for groups 1-7 (which is itself the recognized
  ``Ops.WMMA`` chain link and unnecessary/riskier to wrap).

  Chain ``.after()`` (a scheduling-only pseudo-op, no extra hardware barrier) onto the zero seed, ordering
  element e's chain head behind element e-1's own final WMMA. The seed is a plain zero ``Ops.CONST``, which
  is NOT spec-legal directly under ``Ops.AFTER`` (type_verify's spec_tensor requires a buffer/movement/WMMA/
  etc-typed first operand) -- wrap it in an identity ``Ops.BITCAST`` first (spec-legal under AFTER, and
  already unwrapped by the isel helpers that read this seed, e.g. ``_wmma_elems``).
  """
  phase, subtile = recurrence.phases[0], recurrence.stage.subtile_n
  group = phase.groups[0]
  second_sym = group.wmmas[1]
  results = []
  prior_element_result: UOp | None = None
  for element in range(8):
    substitutions = {subtile: UOp.const(dtypes.weakint, element), phase.publish: publish}
    second_c = second_sym.substitute(substitutions, walk=True)
    first_c = second_c.src[2]
    if seeds is not None:
      seeded_first = UOp(first_c.op, first_c.dtype, (first_c.src[0], first_c.src[1], seeds[element]), first_c.arg, tag=first_c.tag)
      second_c = UOp(second_c.op, second_c.dtype, (second_c.src[0], second_c.src[1], seeded_first), second_c.arg, tag=second_c.tag)
    elif prior_element_result is not None:
      zero = first_c.src[2]
      gated_zero = UOp(Ops.BITCAST, zero.dtype, (zero,)).after(prior_element_result)
      gated_first = UOp(first_c.op, first_c.dtype, (first_c.src[0], first_c.src[1], gated_zero), first_c.arg, tag=first_c.tag)
      second_c = UOp(second_c.op, second_c.dtype, (second_c.src[0], second_c.src[1], gated_first), second_c.arg, tag=second_c.tag)
    results.append(second_c)
    prior_element_result = second_c
  return tuple(results)


def chain_group_stage(recurrence: LlamaOracleRecurrenceGraph, ordinal: int,
                      totals: tuple[UOp, ...] | None,
                      prior_release: UOp | None) -> tuple[tuple[UOp, ...], UOp]:
  """Order one K32-group recurrence behind the previous group's LDS-reuse release, then chain its WMMA
  accumulation directly off the previous group's own final accumulator (per ``subtile_n`` element)."""
  persistent, producer = recurrence.stage.persistent_producer, recurrence.phases[0].producer
  if prior_release is not None:
    persistent = _producer_after_release(persistent, prior_release)
    producer = _producer_after_release(producer, prior_release)
  publish = UOp.barrier(UOp.group(persistent, producer)).replace(tag=("llama_group_chain_publish", ordinal))
  lanes = _instantiate_group_wmma_vectors(recurrence, publish, totals)
  release = UOp(Ops.BARRIER, dtypes.void, lanes).replace(tag=("llama_group_chain_collective_release", ordinal))
  return lanes, release


def group_major_accumulator_vectors(recurrences: tuple[LlamaOracleRecurrenceGraph, ...]) -> tuple[UOp, ...]:
  """Chain every K32-group recurrence's WMMA accumulation into one persistent fp32 accumulator per output
  subtile (per ``subtile_n`` element), in LDS-reuse order."""
  if not recurrences: raise ValueError("group-major writeback requires at least one K32-group recurrence")
  totals: tuple[UOp, ...] | None = None
  prior_release = None
  for ordinal, recurrence in enumerate(recurrences):
    totals, prior_release = chain_group_stage(recurrence, ordinal, totals, prior_release)
  assert totals is not None
  return totals


__all__ = ["chain_group_stage", "group_major_accumulator_vectors"]
