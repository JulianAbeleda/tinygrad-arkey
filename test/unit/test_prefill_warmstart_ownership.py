from types import SimpleNamespace

import tinygrad.llm.model as model_module
from tinygrad.llm.model import Transformer


def test_dense_prefill_warmstarts_require_resident_fp16_owner(monkeypatch):
  monkeypatch.setattr(model_module, "_prefill_v2_opts", lambda out_f, in_f: ((out_f, in_f),))
  monkeypatch.setattr(model_module, "_prefill_v2_without_parked_4x4", lambda opts: opts)
  resident = SimpleNamespace(_pf16_w=object())
  packed_only = SimpleNamespace()
  model = object.__new__(Transformer)
  model.config = SimpleNamespace(prefill_ubatch=512)
  model._prefill_v2_covered = lambda: ((resident, 1024, 512), (packed_only, 2048, 512))

  table = model._build_prefill_v2_warmstart()

  assert len(table) == 1
  assert next(iter(table.values())) == ((1024, 512),)


def test_packed_only_model_has_no_dense_prefill_warmstarts(monkeypatch):
  monkeypatch.setattr(model_module, "_prefill_v2_opts", lambda out_f, in_f: ((out_f, in_f),))
  packed_only = SimpleNamespace()
  model = object.__new__(Transformer)
  model.config = SimpleNamespace(prefill_ubatch=512)
  model._prefill_v2_covered = lambda: ((packed_only, 1024, 512),)
  assert model._build_prefill_v2_warmstart() == {}
