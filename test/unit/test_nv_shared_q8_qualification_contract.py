from types import SimpleNamespace

import numpy as np

from extra.llm_research.decode.nv_shared_q8_progressive_qualification import (
  FUSED_GROUP_STEPS, SEMANTIC_CONTRACT, _child_command, _primitive_oracle, _semantic_comparison,
  _settled_context_required, _settled_continuous_windows, _settled_high_side_filter, _timing_hash_authority,
  _validate_run_extent)


def _row(tokens=(3,4)): return {"tokens":list(tokens)}


def test_fused_progression_excludes_block_zero_and_reaches_every_remaining_block():
  assert FUSED_GROUP_STEPS == (1,2,4,18,35)
  assert list(range(1,FUSED_GROUP_STEPS[-1]+1))[0] == 1
  assert list(range(1,FUSED_GROUP_STEPS[-1]+1))[-1] == 35


def test_semantic_contract_keeps_historical_atol_non_authoritative():
  baseline=np.array([[[100.,90.,30.,20.,10.]],[[110.,90.,40.,20.,0.]]],dtype=np.float32)
  candidate=baseline+np.array([[[0.011,-0.011,0,0,0]],[[0.011,-0.011,0,0,0]]],dtype=np.float32)
  result=_semantic_comparison(baseline,candidate,_row(),_row())
  assert result["semantic_pass"]
  assert result["historical_max_abs_gate"] == {"atol":0.01,"pass":False,"status":"reported_non_authoritative"}
  assert result["top_k_sets_equal"] and result["argmax_equal"]
  assert result["relative_l2"] <= SEMANTIC_CONTRACT["relative_l2_max"]
  assert result["perturbation_vs_min_margin"] < 1.0


def test_checked_in_primitive_oracle_is_bitwise_exact():
  result=_primitive_oracle("docs/task_workflow/output/nv-fused-rmsnorm-q8-provider-gate-20260805.json")
  assert result["bitwise_exact"] and result["case_count"] == 3


def test_timing_child_command_carries_explicit_closed_lease_count(tmp_path):
  args=SimpleNamespace(model="model.gguf",depth=512,count=8,max_context=1024,reps=3)
  cmd=_child_command(args,0,tmp_path/"row.json",18,mode="timing-child")
  assert cmd[cmd.index("--mode")+1] == "timing-child"
  assert cmd[cmd.index("--fused-groups")+1] == "18"

def test_timing_child_command_propagates_settled_composed_cooperative_flags(tmp_path):
  args=SimpleNamespace(model="model.gguf",depth=512,count=32,max_context=1024,reps=5,
    cooperative_q4=True,composed=True,settled_continuous=True)
  cmd=_child_command(args,0,tmp_path/"row.json",4,mode="timing-child")
  assert all(flag in cmd for flag in ("--cooperative-q4","--composed","--settled-continuous"))

class _Sync:
  def __init__(self): self.calls=0
  def synchronize(self): self.calls+=1

class _ClosableTokens:
  def __init__(self): self.value=0; self.closed=False
  def __iter__(self): return self
  def __next__(self): value=self.value; self.value+=1; return value
  def close(self): self.closed=True

def test_settled_continuous_warms_six_then_times_uninterrupted_windows():
  gen,dev=_ClosableTokens(),_Sync(); ticks=iter((0,8_000_000,8_000_000,16_000_000))
  row=_settled_continuous_windows(gen,dev,count=2,reps=2,clock=lambda:next(ticks))
  assert row["prelude_token"] == 0 and row["warmup_tokens"] == [1,2,3,4,5,6]
  assert row["timed_token_count"] == 4 and gen.value == 11
  assert row["samples_ms_per_token"] == [4.0,4.0] and dev.calls == 3
  assert row["token_stream_hash"] == __import__("hashlib").sha256(b"7,8,9,10").hexdigest()

def test_settled_filter_rejects_only_high_stall_and_retains_three_rows():
  limit,accepted,rejected=_settled_high_side_filter([5.0,5.01,4.99,5.02,6.0])
  assert limit == 5.26 and accepted == [5.0,5.01,4.99,5.02] and rejected == [6.0]
  _,accepted,rejected=_settled_high_side_filter([5.0,6.0,7.0])
  assert accepted == [5.0,6.0,7.0] and rejected == []

def test_settled_hash_authority_uses_whole_stream_not_rep_hashes():
  rows=[{"token_stream_hash":"same","token_hashes":["a","b"]},{"token_stream_hash":"same","token_hashes":["c","d"]}]
  assert _timing_hash_authority(rows,True) == {"same"}
  assert _timing_hash_authority(rows,False) == {"a","b","c","d"}

def test_settled_context_validation_fails_before_generator_exhaustion():
  assert _settled_context_required(512,32,5) == 679
  _validate_run_extent(512,32,679,5,True)
  import pytest
  with pytest.raises(ValueError,match="max_context >= 679"): _validate_run_extent(512,32,678,5,True)
