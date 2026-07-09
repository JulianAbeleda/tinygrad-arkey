from extra.qk.prefill.lds2_s9_lifecycle_search import _candidate_space, _lower_kwargs
from extra.qk.prefill.wmma import default_lds2_lifecycle_template, lower_lds2_gemm_kernel


class Args:
  m = 512
  n = 12288
  k = 4096
  wm = 2
  wn = 4
  waves_m = 4
  waves_n = 2
  bk = 32
  pad = 16
  dbuf = 1
  plrab = 1


def test_lifecycle_search_candidates_are_bounded_and_explicit():
  candidates = _candidate_space(1, include_wait2=True)

  assert [c.name for c in candidates] == [
    "baseline",
    "prologue_init_counter_before_adv_k",
    "baseline_coop_store_wait2",
    "body_store_before_compute",
    "tail_compute_before_store",
  ]
  assert candidates[0].lifecycle_template == default_lds2_lifecycle_template(1)
  assert [s.op for s in candidates[1].lifecycle_template.prologue[-3:]] == ["init_counter", "adv_k", "label_loop"]
  assert candidates[2].wait_policy.lgkm_after_coop_store == 2
  assert all(c.lifecycle_template is None for c in candidates[3:])
  assert all(c.status == "skipped" for c in candidates[3:])


def test_lifecycle_search_runnable_candidates_lower():
  for candidate in _candidate_space(1, include_wait2=True):
    if candidate.lifecycle_template is None:
      continue
    insts = lower_lds2_gemm_kernel(**_lower_kwargs(Args), wait_policy=candidate.wait_policy,
                                   lifecycle_template=candidate.lifecycle_template)
    assert insts
