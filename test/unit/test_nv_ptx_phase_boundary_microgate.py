from extra.llm_research.prefill.nv_ptx_phase_boundary_microgate import ARMS, source_contract, source_for


def test_ptx_phase_boundary_sources_have_fixed_large_live_body_and_consumers():
  for arm in ARMS:
    source=source_for(arm)
    contract=source_contract(source,arm)
    assert contract["pass"], (arm,contract)
    assert source.count("float acc") == 64
    assert source.count(" = __fmaf_rn(acc") == 64*8
    assert source.count("sum = __fadd_rn(sum, acc") == 63
    assert source.count("scratch[read_lane") == 18
    assert "const unsigned int *" not in source and "__restrict__" not in source


def test_split_bridge_has_eighteen_register_outputs_and_inputs_around_c_barrier():
  source=source_for("split_register_bridge")
  first,barrier,second=source.index("ld.global.u32"),source.index("__syncthreads();"),source.index("st.shared.u32")
  assert first < barrier < second
  assert source.count('"=r"(copy') == 18 and source.count('"r"(copy') == 18
  assert source.count("asm volatile(") == 2 and source.count("bar.sync 0;") == 0


def test_fused_region_owns_load_barrier_store_and_keeps_publication_barrier_in_c():
  source=source_for("fused_ptx_region")
  assert source.index("ld.global.u32") < source.index("bar.sync 0;") < source.index("st.shared.u32")
  assert source.count("asm volatile(") == 1 and source.count("bar.sync 0;") == 1
  assert source.count("__syncthreads();") == 1


def test_regionload_control_is_ordinary_direct_copy_without_ptx_or_qualifiers():
  source=source_for("regionload_control")
  assert source.count("scratch[tid +") == 18 and source.count(" = src[") == 18
  assert "asm" not in source and source.count("__syncthreads();") == 2
