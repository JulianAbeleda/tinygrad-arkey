from extra.qk.q6k_14b_wmma_bench import M, N, K

def test_q6k_bench_is_8b_like_experimental_fixture_not_14b_authority():
  assert (M, N, K) == (512, 4096, 12288)
