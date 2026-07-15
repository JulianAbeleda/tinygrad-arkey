from extra.qk.mmq_index_base_bisect import bisect_small_mmq_graph, offending_helper


def test_smallest_mmq_bisection_keeps_the_four_requested_boundaries_separate():
  stages = bisect_small_mmq_graph()
  assert [s.name for s in stages] == ["q4_words_decode", "q8_activation_slice", "scale_sum_indexing", "intdot_matmul", "concatenation_reshape"]
  assert [s.value.shape for s in stages] == [(4, 32), (4, 32), (4,), (4, 4), (4, 4)]


def test_vector_pointer_base_fix_is_owned_by_scale_view_helper():
  assert "xscales.reshape" in offending_helper()
  assert "xsc2[:, group_idx]" in offending_helper()
