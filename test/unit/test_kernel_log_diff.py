import pathlib

from extra.llm_research.decode.kernel_log_diff import canonical_name, diff, last_launches, parse, parse_text

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "decode_kernel_log_excerpt.txt"


def test_copy_lines_are_excluded_and_ansi_is_stripped():
  # The fixture's first two lines are real captured host<->device `copy` launches; they must
  # not appear as kernels, and none of the surviving keys may retain an ANSI escape byte.
  agg, total = parse(FIXTURE)
  assert not any(k.startswith("copy") for k in agg)
  assert all("\x1b" not in k for k in agg)
  assert sum(a.count for a in agg.values()) == 30
  assert len(agg) == 20
  assert total == 68143.75


def test_kernel_name_time_and_bandwidth_are_parsed_correctly():
  # r_1187_32_4_16_2_2_2_4_8 is the lm_head kernel from section 2.4: the FIRST GB/s column is
  # achieved traffic on packed bytes (510,504,960 bytes / 63.64ms = 8.02 GB/s, reported as 8.0
  # here); the SECOND column includes cache hits and is not DRAM bandwidth.
  agg, _ = parse(FIXTURE)
  lm_head = agg["r_1187_32_4_16_2_2_2_4_8"]
  assert lm_head.count == 1
  assert lm_head.us == 63520.0
  assert lm_head.achieved_gbs == 8.0
  assert lm_head.cache_gbs == 31.0
  # E_32_32_4 repeats 8 times in the excerpt (with two distinct trailing content hashes) and
  # must aggregate as a single canonical kernel once the hash is stripped.
  repeated = agg["E_32_32_4"]
  assert repeated.count == 8
  assert repeated.us == 19.86


def test_trailing_content_hash_is_normalized_across_trees():
  a = "E_32_32_4_fab82d40f922cf5f6decf5a3a82d6b4c2c4b20acb8161ea4de44b3da581fa65b"
  b = "E_32_32_4_b3dc75d0f2d8069aac270c44f8a857db71362caf208b9e6c174cd328420ade5e"
  assert a != b
  assert canonical_name(a) == canonical_name(b) == "E_32_32_4"


def test_us_and_ms_time_units_both_parse():
  # The lm_head line uses `ms`; every other launch line in the fixture uses `us`.
  agg, _ = parse(FIXTURE)
  assert agg["r_1187_32_4_16_2_2_2_4_8"].us == 63520.0     # ms line, normalized to us
  assert agg["r_32_4_1187"].us == 295.04                   # us line


def test_decode_token_slice_takes_the_last_n_launch_lines():
  # decode_runtime_overhead.py's graph-admission census establishes that the LAST N kernel
  # launches in a JIT=0 DEBUG=2 log are exactly one decode token (N=803 in the pinned TG0
  # baseline). This fixture is a 30-launch excerpt; slicing the last 3 must match a plain
  # parse of those same 3 raw lines, independent of how many launches precede them.
  text = FIXTURE.read_text()
  sliced = last_launches(text, 3)
  agg, total = parse_text(sliced)
  assert sum(a.count for a in agg.values()) == 3
  assert set(agg) == {"r_32_4_1187", "r_128_16_8_1187", "r_16_8"}
  assert agg["r_32_4_1187"].us == 295.04
  assert agg["r_128_16_8_1187"].us == 72.88
  assert agg["r_16_8"].us == 2.08
  assert total == 295.04 + 72.88 + 2.08
  # last_launches(n) for n >= the full launch count returns every launch, unchanged.
  agg_all, _ = parse_text(last_launches(text, 1000))
  assert sum(a.count for a in agg_all.values()) == 30


def test_diff_against_itself_is_zero_and_matches_launch_counts():
  rows, post_total, pre_total, post, pre = diff(FIXTURE, FIXTURE)
  assert post_total == pre_total
  assert post == pre
  assert all(delta == 0 for delta, *_ in rows)
  assert {k for _, k, *_ in rows} == set(post)
