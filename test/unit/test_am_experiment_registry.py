import pytest

from tinygrad.runtime.support.am.ip import common


def test_experiment_registry_preserves_keys_defaults_and_optional_reads(monkeypatch):
  monkeypatch.setattr(common, "_env_int", lambda key, default=0: (key, default))
  monkeypatch.setattr(common, "_env_optional_int", lambda key: (key, None))
  assert all(callable(getattr(common.AM_Experiment, name)) for name in common._AM_EXPERIMENT_NAMES)
  assert common.AM_Experiment.audit_pre_kdb() == ("AM_PSP_AUDIT_PRE_KDB", 0)
  assert common.AM_Experiment.gart_snooped() == ("AM_PSP_GART_SNOOPED", 1)
  assert common.AM_Experiment.gart_table_addr(123) == ("AM_PSP_GART_TABLE_ADDR", 123)
  assert common.AM_Experiment.exact_bootloader_wait() == ("AM_PSP_EXACT_BL_WAIT", 0)
  assert common.AM_Experiment.vram_msg1_paddr() == ("AM_PSP_SYSMSG1_VRAM_PADDR", None)
  assert common.AM_Experiment.mailbox_visibility_reads() == ("AM_PSP_MAILBOX_VIS_READS", 8)
  assert common.AM_Experiment.kdb_header_audit_bytes() == ("AM_PSP_KDB_HEADER_AUDIT_BYTES", 0x200)


def test_experiment_registry_rejects_unknown_name():
  with pytest.raises(AttributeError): getattr(common.AM_Experiment, "not_a_real_experiment")
