PAGE_SIZE = 0x1000

def alloc_aligned_sysmem_window(pci_dev, size:int, alignment:int, label:str):
  allocation_size = size + alignment
  raw_view, paddrs = pci_dev.alloc_sysmem(allocation_size, contiguous=True)
  expected_pages = allocation_size // PAGE_SIZE
  if len(paddrs) != expected_pages: raise ValueError(f"expected {allocation_size >> 20}MB sysmem pages, got {len(paddrs)}")
  if not all(paddr == paddrs[0] + i * PAGE_SIZE for i, paddr in enumerate(paddrs)):
    raise ValueError(f"PSP sysmem {label} buffer is not contiguous")
  view_off = (-paddrs[0]) % alignment
  if view_off + size > raw_view.nbytes:
    raise ValueError(f"failed to find aligned {size >> 20}MB PSP window in {allocation_size >> 20}MB {label} buffer: {paddrs[0]:#x}")
  first_page = view_off // PAGE_SIZE
  return raw_view.view(view_off, size), paddrs[first_page:first_page + size // PAGE_SIZE], paddrs[0], view_off, len(paddrs)

def map_sysmem_view(mm, view, paddrs:list[int], alignment:int, *, snooped:bool=False) -> int:
  from tinygrad.runtime.support.memory import AddrSpace
  addr = mm.alloc_vaddr(size=view.nbytes, align=alignment)
  mm.map_range(addr, view.nbytes, [(paddr, PAGE_SIZE) for paddr in paddrs], AddrSpace.SYS,
               uncached=True, snooped=snooped, boot=True)
  return addr
