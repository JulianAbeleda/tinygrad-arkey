import importlib.util, pathlib

PATH = pathlib.Path(__file__).parents[2] / "scratchpad" / "nv_decode_predispatch_breakdown.py"
SPEC = importlib.util.spec_from_file_location("nv_decode_predispatch_breakdown", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_reconcile_is_additive_and_does_not_include_hcq_work():
  row = {"pre_first_graph_us": 100.0, "before_tinyjit_us": 10.0, "prepare_inputs_us": 20.0,
         "reuse_checks_us": 5.0, "captured_before_concrete_us": 0.0, "concrete_copy_rebind_us": 15.0,
         "captured_to_run_linear_us": 10.0, "run_linear_to_graph_lookup_us": 10.0,
         "graph_runtime_lookup_us": 10.0, "graph_lookup_to_hcq_entry_us": 10.0, "hcq_cpu_us": 999.0}
  got = MOD.reconcile(row)
  assert got["explained_pre_first_us"] == 90.0
  assert got["closure_us"] == 10.0 and got["closure_pct"] == 10.0
  assert "hcq_cpu_us" not in got["parts"]


def test_summary_uses_true_median():
  got = MOD.summarize([{"token":1,"graph_calls":5,"wall_us":1.0}, {"token":1,"graph_calls":5,"wall_us":100.0},
                       {"token":1,"graph_calls":5,"wall_us":2.0}])
  assert got == {"wall_us": 2.0}


def test_rowwise_reconciliation_is_required_for_correlated_rows():
  rows = [{"pre_first_graph_us": 100.0, "before_tinyjit_us": 99.0},
          {"pre_first_graph_us": 200.0, "before_tinyjit_us": 199.0}]
  for row in rows:
    row.update({k:0.0 for k in ("prepare_inputs_us", "reuse_checks_us", "captured_before_concrete_us",
      "concrete_copy_rebind_us", "captured_to_run_linear_us", "run_linear_to_graph_lookup_us",
      "graph_runtime_lookup_us", "graph_lookup_to_hcq_entry_us")})
  assert MOD.med([MOD.reconcile(r)["closure_us"] for r in rows]) == 1.0
