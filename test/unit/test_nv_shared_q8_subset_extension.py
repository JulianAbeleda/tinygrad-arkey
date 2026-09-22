import numpy as np

from extra.llm_research.decode.nv_shared_q8_subset_extension import additive_subset_scores


def test_additive_subset_scores_enumerates_signed_cancellation():
  base=np.asarray([2.0,0.0])
  extras=np.asarray([[-1.0,1.0],[-1.0,-1.0]])
  cards,scores=additive_subset_scores(base,extras,denominator=2.0)
  assert cards.tolist() == [0,1,1,2]
  np.testing.assert_allclose(scores,[1.0,np.sqrt(2)/2,np.sqrt(2)/2,0.0],rtol=0,atol=1e-12)


def test_additive_subset_scores_rejects_shape_mismatch():
  import pytest
  with pytest.raises(ValueError): additive_subset_scores(np.zeros(3),np.zeros((2,4)),1.0)
