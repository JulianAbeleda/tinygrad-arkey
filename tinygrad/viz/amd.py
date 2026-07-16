import contextlib, io, itertools, traceback
from tinygrad.renderer.amd import detect_format
from tinygrad.renderer.amd.dsl import Inst
# ** Assembly static analyzers

def get_stdout(f) -> str:
  buf = io.StringIO()
  try:
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf): f()
  except Exception: traceback.print_exc(file=buf)
  return buf.getvalue()

def get_elf_section(lib:bytes, name:str):
  from tinygrad.runtime.support.elf import elf_loader
  return next((sh for sh in elf_loader(lib)[1] if sh.name == name))

def amd_decode(lib:bytes, target:str) -> dict[int, Inst]:
  text = get_elf_section(lib, ".text")
  off, buf = text.header.sh_addr, text.content
  arch = "rdna3" if target.startswith("gfx11") else "rdna4" if target.startswith("gfx12") else "cdna"
  addr_table:dict[int, Inst] = {}
  offset = 0
  while offset < len(buf):
    remaining = buf[offset:]
    fmt = detect_format(remaining, arch)
    decoded = fmt.from_bytes(remaining)
    addr_table[off+offset] = decoded
    offset += decoded.size()
  return addr_table

def parse_branch(inst) -> int|None:
  x = inst.simm16 & 0xffff if "branch" in getattr(inst, "op_name", "").lower() else None
  return None if x is None else (x - 0x10000 if x & 0x8000 else x)*4

COND_TAKEN, COND_NOT_TAKEN, UNCOND = range(3)
def amdgpu_cfg(lib:bytes, target:str) -> dict:
  # decode
  pc_table = amd_decode(lib, target)
  # get leaders
  leaders:set[int] = {next(iter(pc_table))}
  for pc, inst in pc_table.items():
    if (offset:=parse_branch(inst)) is not None: leaders.update((pc+inst.size()+offset, pc+inst.size()))
  # build the cfg
  curr:int|None = None
  blocks:dict[int, list[int]] = {}
  paths:dict[int, dict[int, int]] = {}
  for pc, inst in pc_table.items():
    if pc in leaders:
      paths[curr:=pc] = {}
      blocks[pc] = []
    else: assert curr is not None, f"no basic block found for {pc}"
    blocks[curr].append(pc)
    # otherwise a basic block can have exactly one or two paths
    nx = pc+inst.size()
    if (offset:=parse_branch(inst)) is not None:
      if inst.op_name == "S_BRANCH": paths[curr][nx+offset] = UNCOND
      else: paths[curr].update([(nx+offset, COND_TAKEN), (nx, COND_NOT_TAKEN)])
    elif nx in leaders: paths[curr][nx] = UNCOND
  pc_tokens:dict[int, list[dict]] = {}
  from tinygrad.renderer.amd.dsl import Reg
  for pc, inst in pc_table.items():
    pc_tokens[pc] = tokens = []
    for name, f in inst._fields:
      if isinstance(val:=getattr(inst, name), Reg): tokens.append({"st":val.fmt(), "keys":[f"r{val.offset+i}" for i in range(val.sz)], "kind":1})
      elif name in {"op","opx","opy"}: tokens.append({"st":(op_name:=val.name.lower()), "keys":[op_name], "kind":0})
      elif name != "encoding" and val != f.default: tokens.append({"st":(s:=repr(val)), "keys":[s], "kind":1})
  # show a smaller view for repeated instructions in the graph
  lines:list[str] = []
  disasm = {pc:str(inst) for pc,inst in pc_table.items()}
  asm_width = max(len(asm) for asm in disasm.values())
  for pcs in blocks.values():
    new_pcs:list[int] = []
    for _,group in itertools.groupby(pcs, key=pc_table.get):
      group = list(group)
      new_pcs.append(pc:=group[0])
      if len(group)>1:
        pc_tokens[pc].append({"st":f"({len(group)}x)", "keys":[], "kind":0})
        for repeated_pc in group[1:]: del pc_tokens[repeated_pc]
      lines.append(f"{disasm[pc]:<{asm_width}}  # {pc:012X}"+(f"...{group[-1]:012X} ({len(group)}x)" if len(group)>1 else ""))
    pcs[:] = new_pcs
  from tinygrad.runtime.autogen import amdgpu_kd
  kd = amdgpu_kd.llvm_amdhsa_kernel_descriptor_t.from_buffer_copy(bytearray(get_elf_section(lib, ".rodata").content))
  vgpr_gran = kd.compute_pgm_rsrc1 & amdgpu_kd.COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT
  return {"data":{"blocks":blocks, "paths":paths, "pc_tokens":pc_tokens}, "src":"\n".join(lines), "lang":"python",
          "metadata":[[{"label":f"{r} Alloc", "value":v} for r,v in [("VGPR", (vgpr_gran+1)*8-7), ("LDS", kd.group_segment_fixed_size),
                                                                     ("Scratch", kd.private_segment_fixed_size)] if v>0]]}
