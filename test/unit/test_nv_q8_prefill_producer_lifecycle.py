from extra.llm_research.prefill.nv_q8_producer_lifecycle import lifecycle, packet_bytes

def test_q8_mmq_packet_size_and_metadata():
  assert packet_bytes() == 2_359_296
  row=lifecycle()
  assert row["abi"]["record_bytes"] == 144
  assert row["packet"]["compression_vs_fp32"] == 32/9

def test_abi_preserving_reuse_scope():
  row=lifecycle()
  assert row["ordinary"] == {"producer_calls":249,"consumer_calls":249}
  assert row["abi_preserving_shared"]["producer_calls"] == 160
  assert row["abi_preserving_shared"]["avoided_calls"] == 89
  assert row["abi_preserving_shared"]["groups"]["attention_qkv"] == 54

def test_pinned_trace_accounting():
  row=lifecycle()
  assert row["observed"]["DS4"]["calls"] == 214
  assert row["observed"]["D4"]["calls"] == 35
  assert sum(x["total_ns"] for x in row["observed"].values()) == 858606
