from extra.llm_research.prefill.nv_direct_copy_2x2_microgate import ARMS, source_contract, source_for


def test_direct_copy_2x2_sources_change_only_declared_axes():
  sources = {name: source_for(arm) for name, arm in ARMS.items()}
  for name, arm in ARMS.items():
    assert source_contract(sources[name], arm)["pass"], name
  assert sources["candidate_writable"].replace("unsigned int *src", "const unsigned int *__restrict__ src") == \
         sources["candidate_const_restrict"]
  assert sources["llama_writable"].replace("unsigned int *src", "const unsigned int *__restrict__ src") == \
         sources["llama_const_restrict"]
  assert sources["candidate_writable"].replace("__launch_bounds__(256)", "__launch_bounds__(256, 1)") == \
         sources["llama_writable"]
  assert sources["candidate_const_restrict"].replace("__launch_bounds__(256)", "__launch_bounds__(256, 1)") == \
         sources["llama_const_restrict"]


def test_direct_copy_2x2_has_fixed_large_live_body_and_publication_shape():
  source = source_for(ARMS["candidate_writable"])
  assert source.count("float acc") == 64
  assert source.count(" = __fmaf_rn(acc") == 64*8
  assert source.count("sum = __fadd_rn(sum, acc") == 63
  assert source.count("scratch[tid +") == 18 and source.count(" = src[") == 18
  assert source.count("__syncthreads();") == 2
  assert "out[tid] = __float_as_uint(sum) ^ scratch[read_index];" in source
