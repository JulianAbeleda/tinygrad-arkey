from extra.qk.mmq_residual_probe import ResidualCase,run_case,run_exact_isa_case
def test_real_residual_probe_binds_false_sites_without_candidate_identity():
  r=run_case(ResidualCase(2,False,1),warmups=1,rounds=3,system_snapshot_id="s")
  assert r["dynamic_contract"]["false_sites_execute"] is False
  assert r["isa_summary"]["global_store_sites"]>=2
  assert "candidate_id" not in r and len(r["binary_sha256"])==64
  assert r["protocol"]["compiled_global_size"][0]==r["protocol"]["timed_global_size"][0]+1
def test_exact_isa_formulation_admits_scalar_series():
  r=run_exact_isa_case(64,system_snapshot_id="s",warmups=1,rounds=3)
  assert r["admitted"] and r["actual"]=={"global_store_sites":65,"branch_sites":127,"predicate_sites":254}
  assert r["store_mnemonics"]==["global_store_b32"]
