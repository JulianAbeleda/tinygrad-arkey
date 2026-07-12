from tinygrad.codegen.opt.kernel_pipeline import Stage1StorageAdapter
from extra.qk.compiler_policies import StoragePolicy

class C:
  def producer(self, e, s): return (e, s, "p")
  def fragments(self, e, s): return (e, s, "f")

def test_stage1_storage_adapter_delegates_without_changing_callbacks():
  a = Stage1StorageAdapter(C(), StoragePolicy("lds", 2, 16))
  assert a.producer(1, 2) == (1, 2, "p") and a.fragments(1, 2) == (1, 2, "f")
