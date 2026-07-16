import pytest

from extra.qk.amd_resource_artifact import extract_amd_physical_intervals


class Reg:
  def __init__(self, offset, sz=1): self.offset, self.sz = offset, sz


class Ins:
  _fields = (("dst", object()), ("src", object()))
  def __init__(self, dst, src): self.dst, self.src = dst, src


class SpillIns(Ins):
  def __str__(self): return "scratch spill"


def test_extracts_only_explicit_final_role_intervals():
  rows = extract_amd_physical_intervals(
    (Ins(Reg(456, 8), Reg(0)),), post_regalloc=True,
    role_evidence={"A": {"bank": "vgpr", "start": 200, "end": 208, "purpose": "fragment"}},
    fixed_register_ownership={"sgpr": (0,)},
  )
  assert rows[0].logical_role == "A"
  assert (rows[0].start, rows[0].end) == (200, 208)


@pytest.mark.parametrize("kwargs, message", [
  ({"role_evidence": {}}, "post_regalloc"),
  ({"post_regalloc": True, "role_evidence": {}}, "explicit"),
  ({"post_regalloc": True, "role_evidence": {"A": {"bank": "vgpr", "start": 200, "end": 208}}}, "unique"),
])
def test_extractor_rejects_missing_or_ambiguous_authority(kwargs, message):
  with pytest.raises((ValueError, TypeError), match=message):
    extract_amd_physical_intervals((Ins(Reg(456, 8), Reg(0)),), **kwargs)


def test_extractor_rejects_unowned_registers_and_spills():
  evidence = {"A": {"bank": "vgpr", "start": 200, "end": 208}}
  with pytest.raises(ValueError, match="unique explicit"):
    extract_amd_physical_intervals((Ins(Reg(456, 8), Reg(40)),), role_evidence=evidence, post_regalloc=True)
  with pytest.raises(ValueError, match="spill/scratch"):
    extract_amd_physical_intervals((SpillIns(Reg(456, 8), Reg(0)),), role_evidence=evidence, post_regalloc=True)
