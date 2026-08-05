import numpy as np

from extra.llm_research.decode.q6k_q8_warp_direct_microgate import SEMANTIC_CONTRACT, _q8_reference


def test_q8_reference_is_deterministic_and_group_local():
  x=np.linspace(-1,1,64,dtype=np.float32).astype(np.float16)
  got=_q8_reference(x)
  assert got.dtype == np.float32 and got.shape == x.shape
  np.testing.assert_array_equal(got,_q8_reference(x.copy()))
  changed=x.copy(); changed[32:]*=np.float16(0.5)
  np.testing.assert_array_equal(got[:32],_q8_reference(changed)[:32])


def test_declared_model_semantic_contract_is_fail_closed():
  assert SEMANTIC_CONTRACT == {"top_k":10,"relative_l2_max":1e-3,"requires_exact_tokens":True,
    "requires_argmax_equal":True,"requires_top_k_sets_equal":True,"requires_perturbation_below_min_margin":True}
