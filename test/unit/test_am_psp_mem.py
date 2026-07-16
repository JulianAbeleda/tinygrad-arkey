import pytest

from tinygrad.runtime.support.am.ip.psp_mem import PAGE_SIZE, alloc_aligned_sysmem_window, map_sysmem_view
from tinygrad.runtime.support.memory import AddrSpace

class FakeView:
  def __init__(self, nbytes:int, offset:int=0): self.nbytes, self.offset = nbytes, offset
  def view(self, offset:int, size:int): return FakeView(size, self.offset + offset)

class FakePCI:
  def __init__(self, base:int, pages:int): self.base, self.pages = base, pages
  def alloc_sysmem(self, size:int, contiguous=False):
    assert contiguous and size == self.pages * PAGE_SIZE
    return FakeView(size), [self.base + i * PAGE_SIZE for i in range(self.pages)]

class FakeMM:
  def __init__(self): self.calls = []
  def alloc_vaddr(self, **kwargs): self.calls.append(("alloc", kwargs)); return 0x800000
  def map_range(self, *args, **kwargs): self.calls.append(("map", args, kwargs))

def test_aligned_sysmem_window_selects_matching_view_and_pages():
  size = alignment = 0x100000
  view, paddrs, raw_paddr, view_off, raw_pages = alloc_aligned_sysmem_window(FakePCI(0x101000, 512), size, alignment, "DMA")
  assert (view.nbytes, view.offset, raw_paddr, view_off, raw_pages) == (size, 0xff000, 0x101000, 0xff000, 512)
  assert paddrs == [0x200000 + i * PAGE_SIZE for i in range(256)]

def test_aligned_sysmem_window_rejects_noncontiguous_pages():
  pci = FakePCI(0x100000, 512)
  original = pci.alloc_sysmem
  def alloc(size, contiguous=False):
    view, paddrs = original(size, contiguous)
    paddrs[7] += PAGE_SIZE
    return view, paddrs
  pci.alloc_sysmem = alloc
  with pytest.raises(ValueError, match="GTT buffer is not contiguous"):
    alloc_aligned_sysmem_window(pci, 0x100000, 0x100000, "GTT")

def test_map_sysmem_view_preserves_mapping_flags_and_page_order():
  mm, view, paddrs = FakeMM(), FakeView(2 * PAGE_SIZE), [0x3000, 0x9000]
  assert map_sysmem_view(mm, view, paddrs, 0x100000, snooped=True) == 0x800000
  assert mm.calls[0] == ("alloc", {"size": 2 * PAGE_SIZE, "align": 0x100000})
  _, args, kwargs = mm.calls[1]
  assert args == (0x800000, 2 * PAGE_SIZE, [(0x3000, PAGE_SIZE), (0x9000, PAGE_SIZE)], AddrSpace.SYS)
  assert kwargs == {"uncached": True, "snooped": True, "boot": True}
