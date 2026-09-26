"""CPU-only unit tests for the shared ncu collector/parser used by the Nemotron-H vs vLLM/cuBLAS
kernel audit (docs/nemotron-vllm-parity/bench/audit/ncu_kernel_counters.py). No GPU, no ncu binary,
no CUDA driver: these only exercise command-line construction and CSV parsing.
"""
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "nemotron-vllm-parity" / "bench" / "audit"))

from ncu_kernel_counters import (  # noqa: E402
  NCU_SECTIONS, SCHEMA, bank_conflicts, build_evidence, build_ncu_command, default_provenance,
  parse_kernel_row, read_ncu_csv_rows, read_ncu_csv_units, sudo_memcap_scope, top_stall_reasons,
)

# A sanitized, 2-kernel sample of an `ncu --page raw --csv` export (row 0 = header, row 1 = units,
# rows 2+ = one kernel launch each): a GEMM kernel and a splitKreduce follow-up, mirroring the real
# audit run's shape. Kept as a string (not a fixtures/*.csv file) because the repo gitignores *.csv.
_NCU_RAW_CSV = '''\
"Kernel Name","Grid Size","Block Size","launch__block_size","launch__registers_per_thread","launch__shared_mem_per_block","launch__shared_mem_per_block_static","launch__shared_mem_per_block_dynamic","gpu__time_duration.sum","sm__warps_active.avg.pct_of_peak_sustained_active","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed","dram__bytes.sum.per_second","sm__throughput.avg.pct_of_peak_sustained_elapsed","smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio","smsp__average_warps_issue_stalled_wait_per_issue_active.ratio","smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio","l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"
"","","","","","Kbyte/block","Kbyte/block","byte/block","ms","%","%","Gbyte/s","%","ratio","ratio","ratio","conflict"
"gemm_kernel_a","(64, 18, 1)","(128, 1, 1)","128","158","5.12","4.096","0","0.244544","25.868285","55.515539","477.246","55.515539","0.20","0.35","0.10","0"
"splitKreduce_kernel","(8, 5, 1)","(64, 1, 1)","64","32","1.024","1.024","0","0.003200","58.554311","0.0","0.5028","8.368816","0.10","0.05","0.63","12"
'''


class TestBuildNcuCommand(unittest.TestCase):
  def test_default_sections_and_flags(self):
    cmd = build_ncu_command(out="/tmp/out", target_cmd=["python3", "script.py", "role", "128"])
    self.assertEqual(cmd[0], "ncu")
    for section in NCU_SECTIONS:
      self.assertIn("--section", cmd)
      self.assertIn(section, cmd)
    self.assertIn("--cache-control", cmd); self.assertEqual(cmd[cmd.index("--cache-control") + 1], "all")
    self.assertIn("--clock-control", cmd); self.assertEqual(cmd[cmd.index("--clock-control") + 1], "none")
    self.assertEqual(cmd[-4:], ["python3", "script.py", "role", "128"])
    self.assertNotIn("--nvtx", cmd)

  def test_nvtx_and_kernel_name_and_launch_count(self):
    cmd = build_ncu_command(out="/tmp/out", target_cmd=["a"], nvtx=True, kernel_name="regex:foo",
                             launch_count=5)
    self.assertIn("--nvtx", cmd)
    self.assertEqual(cmd[cmd.index("--kernel-name") + 1], "regex:foo")
    self.assertEqual(cmd[cmd.index("--launch-count") + 1], "5")

  def test_sudo_memcap_scope_wraps_and_orders_env(self):
    inner = build_ncu_command(out="/tmp/out", target_cmd=["a"])
    wrapped = sudo_memcap_scope(inner, mem_max="20G", env={"DEV": "CUDA"})
    self.assertEqual(wrapped[:5], ["sudo", "systemd-run", "--scope", "-q", "-p"])
    self.assertIn("MemoryMax=20G", wrapped)
    self.assertIn("env", wrapped)
    self.assertIn("DEV=CUDA", wrapped)
    # the ncu invocation itself is untouched, just relocated after the scope prefix
    self.assertEqual(wrapped[wrapped.index("DEV=CUDA") + 1:], inner)


class TestCsvParsing(unittest.TestCase):
  def setUp(self):
    self._tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    self._tmp.write(_NCU_RAW_CSV)
    self._tmp.close()
    self.fixture = pathlib.Path(self._tmp.name)
    self.rows = read_ncu_csv_rows(self.fixture)
    self.units = read_ncu_csv_units(self.fixture)

  def tearDown(self):
    self.fixture.unlink(missing_ok=True)

  def test_row_count_and_header_units(self):
    self.assertEqual(len(self.rows), 2)
    self.assertEqual(self.units["gpu__time_duration.sum"], "ms")
    self.assertEqual(self.units["dram__bytes.sum.per_second"], "Gbyte/s")

  def test_parse_kernel_row_units_aware_conversion(self):
    row = parse_kernel_row(self.rows[0], side="tinygrad", role="ssm_in", shape=128, units=self.units)
    self.assertEqual(row["kernel"], "gemm_kernel_a")
    self.assertEqual(row["grid"], "(64, 18, 1)")
    self.assertEqual(row["registers_per_thread"], 158)
    self.assertAlmostEqual(row["static_shared_bytes"], 4.096 * 1024.0)
    self.assertAlmostEqual(row["dynamic_shared_bytes"], 0.0)
    self.assertAlmostEqual(row["duration_us"], 0.244544 * 1000.0)  # ms -> us
    self.assertAlmostEqual(row["dram_throughput_bytes_per_sec"], 477.246 * 1e9)  # Gbyte/s -> bytes/s
    self.assertAlmostEqual(row["achieved_occupancy_pct"], 25.868285)
    self.assertAlmostEqual(row["tensor_pipe_util_pct"], 55.515539)

  def test_parse_kernel_row_without_units_falls_back_to_heuristic(self):
    # No units dict: same magnitude heuristic the pre-existing audit_table.py summarize() used.
    row = parse_kernel_row(self.rows[0], side="tinygrad")
    self.assertAlmostEqual(row["duration_us"], 0.244544 * 1000.0)
    self.assertAlmostEqual(row["dram_throughput_bytes_per_sec"], 477.246)  # no unit -> scale 1.0

  def test_top_stall_reasons_ranked_and_relative(self):
    stalls = top_stall_reasons(self.rows[0])
    self.assertEqual([s["reason"] for s in stalls], ["wait", "barrier", "long_scoreboard"])
    self.assertAlmostEqual(sum(s["pct"] for s in stalls), 100.0, places=3)

  def test_bank_conflicts_present_when_column_exists(self):
    # the column is present for both rows in this fixture; only the value differs
    self.assertEqual(bank_conflicts(self.rows[0]), {"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": 0.0})
    self.assertEqual(bank_conflicts(self.rows[1]), {"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": 12.0})

  def test_aux_flag_defaults_false_and_is_settable(self):
    row = parse_kernel_row(self.rows[1], side="reference", aux=True)
    self.assertTrue(row["aux"])
    self.assertEqual(row["side"], "reference")


class TestEvidenceBuilder(unittest.TestCase):
  def test_build_evidence_schema_and_shape(self):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
      f.write(_NCU_RAW_CSV)
      fixture = pathlib.Path(f.name)
    self.addCleanup(fixture.unlink, missing_ok=True)
    rows = [parse_kernel_row(r, side="tinygrad") for r in read_ncu_csv_rows(fixture)]
    prov = default_provenance(ncu_version="2026.2.1.0", device_name="NVIDIA GeForce RTX 5090",
                               source_report=str(fixture))
    evidence = build_evidence(rows, provenance=prov)
    self.assertEqual(evidence["schema"], SCHEMA)
    self.assertEqual(evidence["schema"], "tinygrad.nv_ncu_kernel_counters.v1")
    self.assertEqual(len(evidence["rows"]), 2)
    self.assertEqual(evidence["provenance"]["cache_control"], "all")
    self.assertEqual(evidence["provenance"]["clock_control"], "none")


if __name__ == "__main__":
  unittest.main()
