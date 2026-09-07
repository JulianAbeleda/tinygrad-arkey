from extra.llm_research.prefill.nv_compiler_q4k_gkqo_model_arm import _append_unique_stage_pair


def test_streamk_main_and_fixup_share_one_stage_owner():
  stages={"native_o_outputs":[],"native_o_records":[]}
  out,record=object(),object()
  assert _append_unique_stage_pair(stages,"native_o",out,record)
  assert not _append_unique_stage_pair(stages,"native_o",out,record)
  assert stages=={"native_o_outputs":[out],"native_o_records":[record]}


def test_streamk_distinct_layer_buffers_remain_distinct():
  stages={"native_o_outputs":[],"native_o_records":[]}
  pairs=[(object(),object()) for _ in range(36)]
  assert all(_append_unique_stage_pair(stages,"native_o",out,record) for out,record in pairs)
  assert len(stages["native_o_outputs"])==len(stages["native_o_records"])==36
  assert all(stages["native_o_outputs"][i] is pairs[i][0] for i in range(36))
