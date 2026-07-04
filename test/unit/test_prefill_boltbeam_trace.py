from types import SimpleNamespace

from extra.qk.prefill_boltbeam_trace import _classify_kernel, _normalize_pmc_stats, _pmc_stats


def test_direct_packed_kernel_quant_overrides_shape_inventory_collision():
  role_by_shape = {(5120, 17408): {"role": "ffn_down", "quant": "Q6_K"}}
  info = _classify_kernel("prefill_q4k_direct_packed_load_gemm_5120_17408_512_4", role_by_shape)
  assert info["role"] == "ffn_down"
  assert info["quant"] == "Q4_K"
  assert info["shape"] == [512, 5120, 17408]


def test_direct_packed_direct_out_kernel_classifies():
  role_by_shape = {(5120, 17408): {"role": "ffn_down", "quant": "Q6_K"}}
  info = _classify_kernel("prefill_q6k_direct_packed_load_direct_out_gemm_5120_17408_512_1", role_by_shape)
  assert info["role"] == "ffn_down"
  assert info["quant"] == "Q6_K"
  assert info["shape"] == [512, 5120, 17408]


def test_q4_q8_prefill_kernel_classifies_as_q4_direct_packed():
  role_by_shape = {(17408, 5120): {"role": "ffn_gate_up", "quant": "Q4_K"}}
  info = _classify_kernel("prefill_q4k_q8_1_direct_packed_gemm_17408_5120_512_1", role_by_shape)
  assert info["role"] == "ffn_gate_up"
  assert info["quant"] == "Q4_K"
  assert info["shape"] == [512, 17408, 5120]


def test_q4_q8_sdot4_prefill_kernel_classifies_as_q4_direct_packed():
  role_by_shape = {(17408, 5120): {"role": "ffn_gate_up", "quant": "Q4_K"}}
  info = _classify_kernel("prefill_q4k_q8_1_sdot4_direct_packed_gemm_17408_5120_512_1", role_by_shape)
  assert info["role"] == "ffn_gate_up"
  assert info["quant"] == "Q4_K"
  assert info["shape"] == [512, 17408, 5120]


def test_q4_q8_mmq_prefill_kernel_classifies_as_q4_direct_packed():
  role_by_shape = {(17408, 5120): {"role": "ffn_gate_up", "quant": "Q4_K"}}
  info = _classify_kernel("prefill_q4k_q8_1_mmq_direct_packed_gemm_17408_5120_512_8", role_by_shape)
  assert info["role"] == "ffn_gate_up"
  assert info["quant"] == "Q4_K"
  assert info["shape"] == [512, 17408, 5120]


def test_generated_packed_tile_kernel_classifies():
  role_by_shape = {(17408, 5120): {"role": "ffn_gate_up", "quant": "Q4_K"}}
  info = _classify_kernel("prefill_q4_k_generated_tile_ffn_gate_up_512_17408_5120", role_by_shape)
  assert info["role"] == "ffn_gate_up"
  assert info["quant"] == "Q4_K"
  assert info["shape"] == [512, 17408, 5120]


def test_pmc_stats_normalize_to_boltbeam_vocab():
  sched = [
    SimpleNamespace(name="GL2C_HIT", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="GL2C_MISS", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="SQ_INSTS_VALU", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="SQ_BUSY_CYCLES", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="GRBM_GUI_ACTIVE", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="SQC_LDS_IDX_ACTIVE", xcc=1, inst=1, se=1, sa=1, wgp=1),
    SimpleNamespace(name="SQC_LDS_BANK_CONFLICT", xcc=1, inst=1, se=1, sa=1, wgp=1),
  ]
  blob = b"".join(int(v).to_bytes(8, "little") for v in (90, 10, 200, 50, 100, 40, 4))
  counters = _normalize_pmc_stats(_pmc_stats(SimpleNamespace(blob=blob, sched=sched)))
  assert counters["l2_hit_pct"] == 90.0
  assert counters["valu_busy_pct"] == 50.0
  assert counters["occupancy_pct"] == 50.0
  assert counters["lds_conflict_pct"] == 10.0
