import importlib.util, pathlib
P=pathlib.Path(__file__).parents[2]/"scratchpad/cuda_decode_group_span_ledger.py"
S=importlib.util.spec_from_file_location("span",P); M=importlib.util.module_from_spec(S); S.loader.exec_module(M)
def test_event_span_ledger_reconciles():
  x=M.summarize([{"start_us":0.,"end_us":3.,"span_us":3.},{"start_us":5.,"end_us":9.,"span_us":4.}])
  assert x["device_window_us"] == 9 and x["inter_group_gap_sum_us"] == 2 and x["event_reconciliation_error_us"] == 0
