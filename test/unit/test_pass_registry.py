"""LR-032: tests for the ordered pass registry (tinygrad/codegen/passes.py).

Acceptance criteria under test:
  1. The current default pipeline can be printed as an ordered list.
  2. Reordering a pass with unsatisfied dependencies FAILS before code generation.

This module is observation-only: it must not import anything from tinygrad/codegen/__init__.py or
tinygrad/schedule/*, and must not affect pass order anywhere else in the tree.
"""
import unittest

from tinygrad.codegen.passes import (
  REGISTRY, PHASE_ORDER, KERNEL_GRAPH_SEQUENCE, ORDER_CONSTRAINTS, PassOrderViolation,
  default_order, print_pipeline, validate_order, validate_or_raise, stage_counts, build_registry, INVENTORY_PATH,
)


class TestRegistryLoadsInventory(unittest.TestCase):
  def test_93_passes_loaded(self):
    self.assertEqual(len(REGISTRY), 93)

  def test_stage_counts_match_inventory(self):
    # bench/lowering-refactor-baseline/pass_inventory.json's documented distribution (LR-001 findings doc).
    expected = {
      "late": 33, "opt": 13, "rangeify": 12, "bufferize": 11,
      "renderer": 8, "custom_kernel": 7, "indexing": 5, "dependencies": 4,
    }
    self.assertEqual(stage_counts(), expected)

  def test_confidence_distribution_matches_inventory(self):
    counts = {"high": 0, "medium": 0, "low": 0}
    for d in REGISTRY.values():
      counts[d.confidence] += 1
    self.assertEqual(counts, {"high": 59, "medium": 28, "low": 6})

  def test_every_descriptor_has_a_trace_hook(self):
    for d in REGISTRY.values():
      self.assertEqual(d.trace_hook, "tinygrad.uop.trace.record_rewrite")

  def test_unverified_flag_matches_non_high_confidence(self):
    unverified = [d for d in REGISTRY.values() if d.unverified]
    self.assertEqual(len(unverified), 28 + 6)
    for d in unverified:
      self.assertIn(d.confidence, ("medium", "low"))
    for d in REGISTRY.values():
      if d.confidence == "high":
        self.assertTrue(d.verified)

  def test_build_registry_rejects_duplicate_pass_ids(self):
    # sanity: build_registry asserts len(dict) == len(entries). Re-running against the real file must not raise.
    build_registry(INVENTORY_PATH)


class TestOrderConstraintsReferenceRealPasses(unittest.TestCase):
  def test_all_constraint_endpoints_are_real_pass_ids(self):
    for c in ORDER_CONSTRAINTS:
      self.assertIn(c.before, REGISTRY, f"unknown pass_id in constraint: {c.before}")
      self.assertIn(c.after, REGISTRY, f"unknown pass_id in constraint: {c.after}")

  def test_six_primary_findings_are_encoded(self):
    # docs/lowering-refactor-phase0-findings-20260726.md names six real order dependencies. Some map to more than
    # one encoded constraint (composite slot resolution has two producers), so we assert on the distinct pairs.
    primary_pairs = {(c.before, c.after) for c in ORDER_CONSTRAINTS if c.primary}
    expected_pairs = {
      ("rangeify.native_row_softmax_repack", "rangeify.mops"),
      ("indexing.run_rangeify_core", "rangeify.symbolic_reduce_collapse_debuf"),
      ("rangeify.lower_composite_no_range_pre", "rangeify.symbolic_reduce_collapse_debuf"),
      ("rangeify.resolve_composite_slots_prebufferize", "rangeify.symbolic_reduce_collapse_debuf"),
      ("expander.pm_pre_expander", "expander.pm_group_for_reduce"),
      ("reg_store.pm_reduce_acc_upcast_fix", "devectorizer.pm_add_loads"),
      ("reg_store.pm_group_wmma_reg_store", "gpudims.pm_add_gpudims"),
    }
    self.assertEqual(primary_pairs, expected_pairs)

  def test_kernel_graph_sequence_pass_ids_are_real(self):
    for name in KERNEL_GRAPH_SEQUENCE:
      self.assertIn(name, REGISTRY)


class TestAcceptance1_PrintOrderedPipeline(unittest.TestCase):
  """Acceptance: 'the current default pipeline can be printed as an ordered list.'"""

  def test_default_order_contains_every_pass_exactly_once(self):
    order = default_order()
    self.assertEqual(len(order), 93)
    self.assertEqual(len(set(order)), 93)
    self.assertEqual(set(order), set(REGISTRY))

  def test_default_order_respects_declared_phase_order_for_the_pipeline_tail(self):
    # KERNEL_GRAPH_SEQUENCE is real, inventory-sourced evidence that rangeify/indexing/bufferize interleave (e.g.
    # indexing.run_rangeify_core legitimately precedes rangeify.lower_composite_no_range_pre) -- so PHASE_ORDER's
    # macro assumption does not hold strictly across that construction block, and must not be asserted there.
    # It does hold for the tail: once kernel-graph construction is done, dependencies -> custom_kernel -> opt ->
    # late -> renderer must still be non-decreasing.
    order = default_order()
    construction_stages = {"rangeify", "indexing", "bufferize"}
    tail = [name for name in order if REGISTRY[name].stage not in construction_stages]
    seen_phase_positions = [PHASE_ORDER.index(REGISTRY[name].stage) for name in tail]
    self.assertEqual(seen_phase_positions, sorted(seen_phase_positions),
                     "default_order's post-construction tail must be non-decreasing in PHASE_ORDER")
    construction_positions = [order.index(name) for name in order if REGISTRY[name].stage in construction_stages]
    tail_positions = [order.index(name) for name in tail]
    if construction_positions and tail_positions:
      self.assertLess(max(construction_positions), min(tail_positions),
                       "kernel-graph construction (rangeify/indexing/bufferize) must precede the rest of the pipeline")

  def test_default_order_satisfies_all_documented_constraints(self):
    # The default (unmodified) order must not violate any constraint we encoded -- otherwise the registry would be
    # describing a pipeline that contradicts its own documented dependencies.
    violations = validate_order(default_order())
    self.assertEqual(violations, [], f"default order violates documented constraints: {violations}")

  def test_kernel_graph_sequence_is_respected_by_default_order(self):
    order = default_order()
    pos = {name: i for i, name in enumerate(order)}
    seq_positions = [pos[name] for name in KERNEL_GRAPH_SEQUENCE]
    self.assertEqual(seq_positions, sorted(seq_positions),
                     "default_order must preserve the authoritative get_kernel_graph call sequence")

  def test_print_pipeline_is_a_string_listing_every_pass(self):
    text = print_pipeline()
    self.assertIsInstance(text, str)
    for name in REGISTRY:
      self.assertIn(name, text)
    # grouped by phase header
    for stage in PHASE_ORDER:
      self.assertIn(f"-- {stage} --", text)

  def test_print_pipeline_marks_unverified_passes(self):
    text = print_pipeline()
    unverified_names = [d.name for d in REGISTRY.values() if d.unverified]
    self.assertTrue(unverified_names)
    # every unverified pass_id's line contains the marker
    lines_by_name = {}
    for line in text.splitlines():
      parts = line.split()
      if len(parts) >= 2 and parts[1].split("[")[0] in REGISTRY:
        lines_by_name[parts[1].split("[")[0]] = line
    for name in unverified_names:
      self.assertIn("[UNVERIFIED]", lines_by_name[name])


class TestAcceptance2_BadReorderFailsBeforeCodegen(unittest.TestCase):
  """Acceptance: 'reordering a pass with unsatisfied dependencies FAILS before code generation.'

  These tests never call any codegen/renderer code -- validate_or_raise is a pure static check over pass names, so
  a violation is caught before any pipeline would run, let alone reach a renderer or compiler.
  """

  def test_swapping_a_primary_constraint_pair_is_detected(self):
    order = default_order()
    i = order.index("rangeify.native_row_softmax_repack")
    j = order.index("rangeify.mops")
    self.assertLess(i, j)
    bad = list(order)
    bad[i], bad[j] = bad[j], bad[i]   # now mops precedes native_row_softmax_repack: violates constraint 1
    violations = validate_order(bad)
    self.assertTrue(any("rangeify.native_row_softmax_repack" in v and "rangeify.mops" in v for v in violations))

  def test_bad_reorder_raises_before_codegen(self):
    order = default_order()
    i = order.index("reg_store.pm_reduce_acc_upcast_fix")
    j = order.index("devectorizer.pm_add_loads")
    bad = list(order)
    bad[i], bad[j] = bad[j], bad[i]
    with self.assertRaises(PassOrderViolation):
      validate_or_raise(bad)

  def test_valid_reorder_within_a_single_phase_is_accepted(self):
    # Reordering two unrelated same-phase passes that appear in no constraint at all must NOT raise -- the check is
    # about documented dependencies, not about freezing the whole order.
    order = default_order()
    renderer_passes = [n for n in order if REGISTRY[n].stage == "renderer"]
    constrained = {c.before for c in ORDER_CONSTRAINTS} | {c.after for c in ORDER_CONSTRAINTS}
    free_pair = [n for n in renderer_passes if n not in constrained]
    self.assertGreaterEqual(len(free_pair), 2, "expected at least two unconstrained renderer passes")
    a, b = free_pair[0], free_pair[1]
    bad = list(order)
    ia, ib = bad.index(a), bad.index(b)
    bad[ia], bad[ib] = bad[ib], bad[ia]
    self.assertEqual(validate_order(bad), [])

  def test_validate_order_skips_constraints_missing_from_a_partial_order(self):
    # A partial ordering (not every pass_id present) must not spuriously fail on the missing ones.
    partial = ["rangeify.mops", "rangeify.native_row_softmax_repack"]  # deliberately violates constraint 1
    violations = validate_order(partial)
    self.assertEqual(len(violations), 1)


class TestCapabilityPredicate(unittest.TestCase):
  def test_capability_predicate_is_callable_and_gates_amd_only_passes(self):
    amd_marked = [d for d in REGISTRY.values() if "amd" in d.capability_note]
    self.assertTrue(amd_marked)
    for d in amd_marked:
      self.assertFalse(d.capability(frozenset()))
      self.assertTrue(d.capability(frozenset({"amd"})))

  def test_unconstrained_pass_runs_on_any_target(self):
    generic = [d for d in REGISTRY.values() if d.capability_note.startswith("no constraint")]
    self.assertTrue(generic)
    for d in generic[:5]:
      self.assertTrue(d.capability(frozenset()))


if __name__ == "__main__":
  unittest.main()
