import pytest

from extra.qk.prefill.wmma import (
  AMDRegisterLeaseAllocator, LDS2Cadence, LDS2LifecycleTemplate, LDS2MemoryLayout, LDS2RegLayout, LDS2WaitPolicy, build_gemm_lds2,
  default_lds2_cadence, default_lds2_lifecycle_template, default_lds2_memory_layout, default_lds2_reg_layout,
  default_lds2_wait_policy, env_lds2_lifecycle_template, env_lds2_reg_layout, env_lds2_wait_policy,
  lower_lds2_gemm_kernel)


def _raw(insts):
  return b"".join(inst.to_bytes() for inst in insts)


def test_default_lds2_reg_layout_matches_legacy_formula():
  layout = default_lds2_reg_layout(WM=2, WN=4, loadsA=2, loadsB=2)

  assert layout == LDS2RegLayout(
    FA=10,
    FB=10 + 2*8,
    ACCb=10 + 2*8 + 4*8,
    CTA=10 + 2*8 + 4*8 + 2*4*8,
    CTB=10 + 2*8 + 4*8 + 2*4*8 + 2*4,
    SCR=10 + 2*8 + 4*8 + 2*4*8 + 2*4 + 2*4,
    FB2=10 + 2*8 + 4*8 + 2*4*8 + 2*4 + 2*4 + 2,
  )


def test_register_lease_allocator_fixed_abi_and_virtual_pool():
  alloc = AMDRegisterLeaseAllocator.with_fixed_abi()
  assert alloc.virtual_vgpr_pool == 10 and alloc.virtual_sgpr_pool == 17
  frag = alloc.allocate("fragment", 16, bank="vgpr", align=8)
  assert frag.start == 16 and frag.end == 32
  with pytest.raises(ValueError, match="overlaps"):
    alloc.reserve("overlap", 20, 2, bank="vgpr")


def test_default_layout_leases_match_legacy_register_windows():
  alloc = AMDRegisterLeaseAllocator.with_fixed_abi()
  leases = [alloc.allocate(name, count, bank="vgpr") for name, count in (
    ("wmma_fragment_a", 16), ("wmma_fragment_b", 32), ("wmma_accumulator", 64),
    ("lds_pack_a", 8), ("lds_pack_b", 8), ("address_scratch", 2))]
  assert [(x.name, x.start, x.end) for x in leases] == [
    ("wmma_fragment_a", 10, 26), ("wmma_fragment_b", 26, 58), ("wmma_accumulator", 58, 122),
    ("lds_pack_a", 122, 130), ("lds_pack_b", 130, 138), ("address_scratch", 138, 140)]


def test_env_lds2_reg_layout_defaults_to_legacy_layout(monkeypatch):
  monkeypatch.delenv("PREFILL_LDS2_REG_BLOCK_SHIFT", raising=False)

  assert env_lds2_reg_layout(WM=2, WN=4, loadsA=2, loadsB=2) == default_lds2_reg_layout(2, 4, 2, 2)


def test_env_lds2_reg_layout_applies_opt_in_block_shift(monkeypatch):
  monkeypatch.setenv("PREFILL_LDS2_REG_BLOCK_SHIFT", "1")
  base = default_lds2_reg_layout(2, 4, 2, 2)

  assert env_lds2_reg_layout(WM=2, WN=4, loadsA=2, loadsB=2) == LDS2RegLayout(
    FA=base.FA + 1,
    FB=base.FB + 1,
    ACCb=base.ACCb + 1,
    CTA=base.CTA + 1,
    CTB=base.CTB + 1,
    SCR=base.SCR + 1,
    FB2=base.FB2 + 1,
  )


def test_default_lds2_memory_layout_matches_legacy_formula():
  layout = default_lds2_memory_layout(BM=128, BN=128, BK=32, PAD=16, DBUF=1)

  assert layout == LDS2MemoryLayout(
    SA=32*2 + 16,
    SB=32*2 + 16,
    LDS_A=(32*2 + 16) * 128,
    BUFSZ=(32*2 + 16) * 128 + (32*2 + 16) * 128,
    NBUF=2,
  )


def test_default_lds2_wait_policy_matches_legacy_counts():
  assert default_lds2_wait_policy() == LDS2WaitPolicy(
    vm_after_coop_load=0,
    lgkm_after_coop_store=0,
    lgkm_after_frag_load=0,
  )


def test_env_lds2_wait_policy_defaults_to_legacy_counts(monkeypatch):
  for key in ("PREFILL_LDS2_WAIT_VM_COOP_LOAD", "PREFILL_LDS2_WAIT_LGKM_COOP_STORE", "PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD"):
    monkeypatch.delenv(key, raising=False)

  assert env_lds2_wait_policy() == default_lds2_wait_policy()


def test_env_lds2_wait_policy_reads_opt_in_counts(monkeypatch):
  monkeypatch.setenv("PREFILL_LDS2_WAIT_VM_COOP_LOAD", "0")
  monkeypatch.setenv("PREFILL_LDS2_WAIT_LGKM_COOP_STORE", "2")
  monkeypatch.setenv("PREFILL_LDS2_WAIT_LGKM_FRAG_LOAD", "1")

  assert env_lds2_wait_policy() == LDS2WaitPolicy(
    vm_after_coop_load=0,
    lgkm_after_coop_store=2,
    lgkm_after_frag_load=1,
  )


def test_default_lds2_cadence_matches_dbuf_flag():
  assert default_lds2_cadence(0) == LDS2Cadence(double_buffer=False)
  assert default_lds2_cadence(1) == LDS2Cadence(double_buffer=True)


def test_default_lds2_lifecycle_template_captures_dbuf_body():
  template = default_lds2_lifecycle_template(1)

  assert template.double_buffer is True
  assert [(s.op, s.slot) for s in template.prologue] == [
    ("coop_load", 0), ("wait_coop_load", None), ("coop_store", 0), ("wait_coop_store", None), ("barrier", None),
    ("adv_k", None), ("init_counter", None), ("label_loop", None),
  ]
  assert [(s.op, s.slot) for s in template.body] == [
    ("coop_load", 1), ("compute", 0), ("wait_coop_load", None), ("coop_store", 1), ("wait_coop_store", None), ("barrier", None), ("adv_k", None),
    ("coop_load", 0), ("compute", 1), ("wait_coop_load", None), ("coop_store", 0), ("wait_coop_store", None), ("barrier", None), ("adv_k", None),
    ("branch_nl", None),
  ]
  assert [(s.op, s.slot) for s in template.tail] == [
    ("coop_load", 1), ("compute", 0), ("wait_coop_load", None), ("coop_store", 1), ("wait_coop_store", None), ("barrier", None),
    ("compute", 1),
  ]


def test_env_lds2_lifecycle_template_defaults_to_legacy_template(monkeypatch):
  monkeypatch.delenv("PREFILL_LDS2_LIFECYCLE_PROLOGUE_INIT_BEFORE_ADV_K", raising=False)

  assert env_lds2_lifecycle_template(1) == default_lds2_lifecycle_template(1)


def test_env_lds2_lifecycle_template_applies_opt_in_prologue_reorder(monkeypatch):
  monkeypatch.setenv("PREFILL_LDS2_LIFECYCLE_PROLOGUE_INIT_BEFORE_ADV_K", "1")
  template = env_lds2_lifecycle_template(1)

  assert [(s.op, s.slot) for s in template.prologue] == [
    ("coop_load", 0), ("wait_coop_load", None), ("coop_store", 0), ("wait_coop_store", None), ("barrier", None),
    ("init_counter", None), ("adv_k", None), ("label_loop", None),
  ]
  assert template.body == default_lds2_lifecycle_template(1).body
  assert template.tail == default_lds2_lifecycle_template(1).tail


@pytest.mark.parametrize("kwargs", [
  dict(M=512, N=12288, K=4096, WAVES_M=4, WAVES_N=2, WM=2, WN=4, BK=32, PAD=16, DBUF=1, PLRAB=1),
  dict(M=512, N=12288, K=4096, WAVES_M=4, WAVES_N=2, WM=2, WN=4, BK=32, PAD=16, DBUF=0, PLRA=1),
  dict(M=512, N=4096, K=12288, WAVES_M=4, WAVES_N=2, WM=2, WN=2, BK=64, PAD=16, DBUF=1),
])
def test_build_gemm_lds2_default_layout_is_byte_identical_when_explicit(kwargs):
  threads = kwargs["WAVES_M"] * kwargs["WAVES_N"] * 32
  cpr = kwargs["BK"] // 8
  rstride = threads // cpr
  bm = kwargs["WAVES_M"] * kwargs["WM"] * 16
  bn = kwargs["WAVES_N"] * kwargs["WN"] * 16
  reg_layout = default_lds2_reg_layout(kwargs["WM"], kwargs["WN"], bm // rstride, bn // rstride)
  memory_layout = default_lds2_memory_layout(bm, bn, kwargs["BK"], kwargs["PAD"], kwargs["DBUF"])

  wait_policy = default_lds2_wait_policy()
  cadence = default_lds2_cadence(kwargs["DBUF"])
  lifecycle_template = default_lds2_lifecycle_template(kwargs["DBUF"])

  implicit = _raw(build_gemm_lds2(**kwargs))
  explicit = _raw(build_gemm_lds2(
    **kwargs, reg_layout=reg_layout, memory_layout=memory_layout, wait_policy=wait_policy, cadence=cadence,
    lifecycle_template=lifecycle_template))

  assert explicit == implicit


def test_build_gemm_lds2_is_compatibility_wrapper_for_lowerer():
  kwargs = dict(M=512, N=12288, K=4096, WAVES_M=4, WAVES_N=2, WM=2, WN=4, BK=32, PAD=16, DBUF=1, PLRAB=1)

  assert _raw(build_gemm_lds2(**kwargs)) == _raw(lower_lds2_gemm_kernel(**kwargs))


def test_build_gemm_lds2_rejects_overlapping_layout():
  bad = LDS2RegLayout(FA=10, FB=11, ACCb=44, CTA=108, CTB=116, SCR=124, FB2=126)

  with pytest.raises(AssertionError, match="overlaps A/B fragments"):
    build_gemm_lds2(512, 12288, 4096, 4, 2, 2, 4, 32, 16, 1, PLRAB=1, reg_layout=bad)


def test_build_gemm_lds2_rejects_oversized_memory_layout():
  bad = LDS2MemoryLayout(SA=512, SB=512, LDS_A=65536, BUFSZ=65536, NBUF=2)

  with pytest.raises(AssertionError, match="LDS overflow"):
    build_gemm_lds2(512, 12288, 4096, 4, 2, 2, 4, 32, 16, 1, PLRAB=1, memory_layout=bad)


def test_build_gemm_lds2_rejects_invalid_wait_policy():
  bad = LDS2WaitPolicy(vm_after_coop_load=64)

  with pytest.raises(AssertionError, match="invalid LDS2 wait policy"):
    build_gemm_lds2(512, 12288, 4096, 4, 2, 2, 4, 32, 16, 1, PLRAB=1, wait_policy=bad)


def test_build_gemm_lds2_rejects_cadence_that_disagrees_with_dbuf():
  bad = LDS2Cadence(double_buffer=False)

  with pytest.raises(AssertionError, match="disagrees with DBUF"):
    build_gemm_lds2(512, 12288, 4096, 4, 2, 2, 4, 32, 16, 1, PLRAB=1, cadence=bad)


def test_build_gemm_lds2_rejects_lifecycle_that_disagrees_with_dbuf():
  bad = LDS2LifecycleTemplate(double_buffer=False, prologue=(), body=(), tail=())

  with pytest.raises(AssertionError, match="lifecycle double_buffer=False disagrees with DBUF=1"):
    build_gemm_lds2(512, 12288, 4096, 4, 2, 2, 4, 32, 16, 1, PLRAB=1, lifecycle_template=bad)
