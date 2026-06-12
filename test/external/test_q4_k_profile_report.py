import pathlib, tempfile, unittest

from extra.q4_k_profile_report import Kernel, _label, classify_token, parse_log

AMD_LINE = "*** AMD        1 q4k_gemv_partial_4096_4096_1                   arg  3 mem   9.50 GB tm     55.08us/    60.41ms (   1955 GFLOPS  175|1101   GB/s) "
TOKEN_LINE = "226.08 ms,   4.42 tok/s,   21.10 GB/s, 4770/9504 MB  -- sample"

class TestQ4KProfileReport(unittest.TestCase):
  def test_label_strict(self):
    self.assertEqual(_label(pathlib.Path("8b-q4k-primitive-debug2-jitbs1.log")), ("8B", "Q4K_PRIMITIVE=1 named"))
    self.assertEqual(_label(pathlib.Path("8b-q4q6-primitive-debug2-jitbs1.log")), ("8B", "Q4K+Q6K_PRIMITIVE=1 named"))
    self.assertEqual(_label(pathlib.Path("14b-baseline-debug2-batched.log")), ("14B", "baseline batched"))
    with self.assertRaisesRegex(ValueError, "8b or 14b"):
      _label(pathlib.Path("q4k-primitive-debug2-jitbs1.log"))
    with self.assertRaisesRegex(ValueError, "exactly one of baseline or primitive"):
      _label(pathlib.Path("8b-debug2-jitbs1.log"))
    with self.assertRaisesRegex(ValueError, "exactly one of batched or jitbs1"):
      _label(pathlib.Path("8b-baseline-debug2.log"))

  def test_parse_log_counts_and_strictness(self):
    with tempfile.TemporaryDirectory() as td:
      p = pathlib.Path(td) / "8b-baseline-debug2-jitbs1.log"
      p.write_text("ignored setup line\n*** DISK:/h    2 view    9.00 MB @ 1 arg 2 mem 1 GB tm 1.0us/ 1.0us\n" + AMD_LINE + "\n" + TOKEN_LINE + "\n")
      parsed = parse_log(p)
      self.assertEqual(len(parsed.tokens), 1)
      self.assertEqual(parsed.stats.amd_lines, 1)
      self.assertEqual(parsed.stats.token_lines, 1)
      self.assertEqual(parsed.stats.non_amd_debug_lines, 1)
      self.assertEqual(parsed.stats.ignored_lines, 1)

      bad = pathlib.Path(td) / "bad.log"
      bad.write_text("*** AMD malformed line\n" + TOKEN_LINE + "\n")
      with self.assertRaisesRegex(ValueError, "malformed AMD DEBUG line"):
        parse_log(bad)

      empty = pathlib.Path(td) / "empty.log"
      empty.write_text("no useful profile data\n")
      with self.assertRaisesRegex(ValueError, "parsed zero token summaries"):
        parse_log(empty)

  def test_classification_boundary(self):
    kernels = [
      Kernel("copy        4 B,     AMD <- AMD", 1.0),
      Kernel("q4k_gemv_partial_4096_4096_1", 1.0),
      Kernel("q6k_gemv_partial_4096_12288_1", 1.0),
      Kernel("r_32_32_4_48_2_2_2_32", 1.0),
      Kernel("r_1024_16_4_2_32", 1.0),
      Kernel("r_4_2_8_16_4_(start_pos+1)", 1.0),
      Kernel("E_(start_pos+1)_8_4", 1.0),
      Kernel("E_2_8_16_4_4", 1.0),
      Kernel("r_32_4_1187", 1.0),
      Kernel("r_unrecognized", 1.0),
    ]
    buckets = [bucket for _, bucket in classify_token(kernels)]
    self.assertEqual(buckets, [
      "copy",
      "q4k_primitive_gemv",
      "q6k_primitive_gemv",
      "fallback_quant_fused",
      "fallback_quant_fused",
      "attention_misc",
      "attention_misc",
      "norm_sampling_misc",
      "norm_sampling_misc",
      "other_amd",
    ])

  def test_primitive_splitk_reduction_followups(self):
    buckets = [bucket for _, bucket in classify_token([
      Kernel("q4k_gemv_partial_4096_12288_4", 1.0),
      Kernel("r_reduce_a", 1.0),
      Kernel("r_reduce_b", 1.0),
      Kernel("r_reduce_c", 1.0),
      Kernel("r_32_32_4_48_2_2_2_32", 1.0),
    ])]
    self.assertEqual(buckets, [
      "q4k_primitive_gemv",
      "q4k_primitive_reduction",
      "q4k_primitive_reduction",
      "q4k_primitive_reduction",
      "fallback_quant_fused",
    ])

if __name__ == "__main__":
  unittest.main()
