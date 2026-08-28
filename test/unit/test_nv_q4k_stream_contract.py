import numpy as np

from extra.llm_research.prefill.nv_q4k_imma_fragment_microgate import production_slotmap


def test_production_slotmap_is_bounded_and_complete_for_boundaries():
  slotmap = production_slotmap()
  assert slotmap.shape == (384, 2)
  used = slotmap[slotmap >= 0]
  assert used.size == 316
  assert np.unique(used).size == used.size
  assert used.min() == 0 and used.max() == 338
  # 226 tiles are wholly owned by one CTA; the remaining 158 have exactly
  # two contiguous-Stream-K boundary contributions.
  assert np.sum(slotmap[:, 0] < 0) == 226
  assert np.all(np.sum(slotmap >= 0, axis=1)[slotmap[:, 0] >= 0] == 2)
