import importlib.util, pathlib

PATH = pathlib.Path(__file__).parents[2] / "scratchpad" / "nv_decode_group_window_ledger.py"
SPEC = importlib.util.spec_from_file_location("nv_decode_group_window_ledger", PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


def test_summarize_keeps_wall_and_device_boundaries_separate():
  got = MOD.summarize(
    [{"wall_us": 10.0}, {"wall_us": 12.0}, {"wall_us": 11.0}],
    [{"wall_us": 12.0, "device_window_us": 8.0, "outside_window_us": 4.0},
     {"wall_us": 14.0, "device_window_us": 9.0, "outside_window_us": 5.0},
     {"wall_us": 13.0, "device_window_us": 8.5, "outside_window_us": 4.5}],
  )
  assert got == {"marker_off_wall_us": 11.0, "marker_on_wall_us": 13.0,
                 "device_window_us": 8.5, "outside_window_us": 4.5, "marker_overhead_us": 2.0}


def test_host_partition_reports_overlappable_cpu_separately():
  got = MOD.host_partition({
    "host_t0_ns": 0, "host_t1_ns": 20_000, "next_return_ns": 19_000,
    "host_calls_ns": [
      {"enter_ns": 2_000, "exit_ns": 4_000},
      {"enter_ns": 5_000, "exit_ns": 7_000},
    ],
    "item_calls_ns": [{"enter_ns": 10_000, "after_drain_ns": 15_000,
                        "drain_ns": 5_000, "exit_ns": 18_000}],
  })
  assert got == {"pre_first_graph_us": 2.0, "graph_call_cpu_sum_us": 4.0,
                 "inter_graph_host_sum_us": 1.0, "last_graph_return_to_item_us": 3.0,
                 "item_total_us": 8.0, "item_pre_drain_us": 5.0,
                 "item_after_drain_us": 3.0, "python_yield_tail_us": 1.0,
                 "post_next_synchronize_us": 1.0, "host_wall_us": 20.0}
