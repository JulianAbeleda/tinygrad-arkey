import struct
from typing import cast
from tinygrad.dtype import dtypes, PtrDType
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.isa import Register
from tinygrad.helpers import unwrap

def encode_instruction(x:UOp, opc:int, reg:int|None=None, pp:int=0, sel:int=0, we:int=0, *, write_mem:set, rm1st:set, rm2nd:set,
                       read_flags:set, lea:object) -> bytes|None:
  def _encode(reg_uop:UOp|None, rm_uop:UOp, idx_uop:UOp|None=None, disp_uop:UOp|None=None, vvvv_uop:UOp|None=None, imm_uop:UOp|None=None) -> bytes:
    nonlocal reg, opc
    reg = cast(int, cast(Register, reg_uop.reg).index if reg_uop is not None else reg)
    rm = cast(Register, rm_uop.reg).index
    idx = cast(Register, idx_uop.reg).index if idx_uop is not None and idx_uop.reg is not None else 4
    rm_sz = 8 if isinstance(rm_uop.dtype, PtrDType) and disp_uop is None else rm_uop.dtype.itemsize
    reg_sz = (reg_uop.dtype.itemsize if not isinstance(reg_uop.dtype, PtrDType) else 8) if reg_uop is not None else 0
    sz = reg_sz or rm_sz
    inst = bytes([])
    assert 0 <= reg <= 15 and 0 <= idx <= 15 and 0 <= rm <= 15
    r, _x, b = reg >> 3, idx >> 3, rm >> 3
    if sel: # VEX bytes
      vvvv = cast(Register, vvvv_uop.reg).index if vvvv_uop is not None else 0
      l = (max(reg_sz, rm_sz) > 16) & 0b1
      if sel == 1 and _x == b == we == 0: inst += bytes([0xC5, (~r & 0b1) << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
      else: inst += bytes([0xC4, (~r & 0b1) << 7 | (~_x & 0b1) << 6 | (~b & 0b1) << 5 | sel, we << 7 | (~vvvv & 0b1111) << 3 | l << 2 | pp])
    else: # optional PREFIX and REX bytes
      if sz == 2: inst += bytes([0x66])
      w = sz == 8
      # REX byte is required when 64 bit or an extended reg is used (index 8 - 15) or lower 8 bits of (rsp, rbp, rsi, rdi) are accessed
      if w | r | _x | b | (reg_sz == 1 & reg >> 2) | (rm_sz == 1 & rm >> 2): inst += bytes([0b0100 << 4 | w << 3 | r << 2 | _x << 1 | b])
      if (rm_sz == 1 or reg_sz == 1) and x.arg not in read_flags | {lea}: opc -= 1
    inst += opc.to_bytes((opc.bit_length() + 7) // 8, 'big')
    idx, rm, reg = idx & 0b111, rm & 0b111, reg & 0b111
    # 0b00 -- signals memory access with no displacement
    # 0b01 -- signals memory access with 8bit displacement
    # 0b10 -- signals memory access with 32bit displacement
    # 0b11 -- signals no memory access
    if disp_uop is not None:
      assert disp_uop.op is Ops.CONST, "displacement must be a constant"
      assert disp_uop.dtype in (dtypes.int8, dtypes.int32), "displacement can only be 1 or 4 byte signed int"
      # rbp/r13 always require a displacement
      if disp_uop.arg != 0 or rm == 0b101: mod = 0b01 if disp_uop.dtype.itemsize == 1 else 0b10
      else: mod = 0b00
    else: mod = 0b11
    # x 0b0 and idx 0b100 means rsp which means no index exists
    # rm 0b100 (rsp/r12) signals a sib byte is required, rm then is encoded in the base field of SIB
    _rm = rm if idx == 0b100 and _x == 0b0 else 0b100
    inst += bytes([mod << 6 | reg << 3 | _rm])
    if _rm == 0b100 and mod != 0b11:
      scale = {1: 0b00, 2: 0b01, 4: 0b10, 8: 0b11}[1 if idx == 0b100 and _x == 0b0 else rm_sz]
      inst += bytes([scale << 6 | idx << 3 | rm])
    if mod == 0b01 or mod == 0b10:
      assert disp_uop is not None
      inst += struct.pack(unwrap(disp_uop.dtype.fmt), disp_uop.arg)
    if imm_uop is not None:
      if imm_uop.op is Ops.CONST: inst += struct.pack(unwrap(imm_uop.dtype.fmt), imm_uop.arg)
      elif isinstance(imm_uop.reg, Register): inst += bytes([(imm_uop.reg.index & 0b1111) << 4 | 0b0000])
    return inst
  # when a uop writes to memory it takes the form of a store, dtype is void, no definition
  address:tuple[UOp|None, ...]
  if x.arg in write_mem:
    if len(x.src) > 3: address, rest = x.src[:3], x.src[3:]
    else: address, rest = (x, None, None), x.src
    return _encode(rest[0], *address, *(None, *rest[1:])) if reg is None else _encode(None, *address, *(None, *rest[:1]))

  if x.arg in rm1st:
    if len(x.src) > 2: address, rest = x.src[:3], x.src[3:]
    else: address, rest = (x.src[0], None, None), x.src[1:]
    imm_uop = rest[:1] if rest and rest[0].op is Ops.CONST else (None,)
    return _encode(x, *address, *(None, *imm_uop)) if reg is None else _encode(None, *address, *(x if sel else None, *imm_uop))

  if x.arg in rm2nd:
    if len(x.src) > 3: address, rest = x.src[1:4], x.src[:1] + x.src[4:]
    else: address, rest = (x.src[1], None, None), x.src[:1] + x.src[2:]
    # cmp/vucomiss reg, rm don't define a new register
    return _encode(x, *address, *rest) if x.dtype is not dtypes.void else _encode(rest[0], *address)

  return None
