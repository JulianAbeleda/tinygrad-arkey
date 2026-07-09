from extra.qk.prefill.lds2_s9_memory_search import candidate_proposals
from extra.qk.prefill.wmma import LDS2MemoryLayout, lower_lds2_gemm_kernel


class Args:
  m = 512
  n = 12288
  k = 4096
  wm = 2
  wn = 4
  waves_m = 4
  waves_n = 2
  bk = 32
  dbuf = 1
  plrab = 1


def test_memory_search_candidates_are_pad_bounded_and_report_lds_size():
  candidates = candidate_proposals(Args.wm, Args.wn, Args.waves_m, Args.waves_n, Args.bk, Args.dbuf)
  by_pad = {c["pad"]: c for c in candidates}

  assert list(by_pad) == [0, 8, 16, 24, 32]
  assert by_pad[16]["valid"] is True
  assert by_pad[16]["memory_layout"] == {"SA": 80, "SB": 80, "LDS_A": 10240, "BUFSZ": 20480, "NBUF": 2}
  assert by_pad[16]["lds_bytes"] == 40960
  assert by_pad[32]["valid"] is True
  assert by_pad[32]["lds_bytes"] == 49152


def test_memory_search_reports_invalid_lds_candidates_for_larger_tiles():
  candidates = candidate_proposals(wm=4, wn=4, waves_m=4, waves_n=2, bk=32, dbuf=1)
  by_pad = {c["pad"]: c for c in candidates}

  assert by_pad[0]["valid"] is True
  assert by_pad[16]["valid"] is True
  assert by_pad[16]["lds_bytes"] == 61440
  assert by_pad[24]["valid"] is False
  assert "LDS overflow" in by_pad[24]["invalid_reason"]


def test_memory_search_valid_candidates_lower_with_memory_layout():
  for candidate in candidate_proposals(Args.wm, Args.wn, Args.waves_m, Args.waves_n, Args.bk, Args.dbuf):
    if not candidate["valid"]:
      continue
    insts = lower_lds2_gemm_kernel(
      Args.m, Args.n, Args.k, Args.waves_m, Args.waves_n, Args.wm, Args.wn, Args.bk, candidate["pad"], Args.dbuf,
      PLRAB=Args.plrab, memory_layout=LDS2MemoryLayout(**candidate["memory_layout"]))
    assert insts
