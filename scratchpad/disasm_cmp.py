import os, sys
os.environ["ALLOW_DEVICE_USAGE"]="1"
sys.path.insert(0, os.getcwd())
from tinygrad import Tensor, Device
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad.codegen import to_program
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.renderer.amd import decode_inst
from test.amd.disasm import disasm

M = int(os.environ.get("GM", 64)); N = int(os.environ.get("GN", 64)); K = int(os.environ.get("GK", 64))

def ast_for(M,N,K):
  a=Tensor.empty(M,K,dtype="half"); b=Tensor.empty(K,N,dtype="half")
  return [u for u in (a@b).schedule_linear().toposort() if u.op is Ops.SINK][0]

def disasm_stream(raw, arch="rdna3"):
  out=[]; i=0
  while i < len(raw):
    chunk = raw[i:i+8]
    try:
      inst = decode_inst(chunk if len(chunk)>=8 else chunk+b"\0"*(8-len(chunk)), arch)
      sz = inst.size()
      s = disasm(inst)
    except Exception as e:
      sz=4; s=f"<decode-fail {raw[i:i+4].hex()}: {e}>"
    out.append((i, raw[i:i+sz].hex(), s))
    i += sz
  return out

# --- ISA renderer bytes ---
def isa_bytes(M,N,K):
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  cap={}
  orig=ren._resolve_labels
  def wrap(insts):
    r=orig(insts); cap['final']=list(r); return r
  ren._resolve_labels=wrap
  to_program(ast_for(M,N,K), ren)
  return b"".join(u.arg.to_bytes() for u in cap['final'])

# --- HIP compiled .text ---
def hip_text(M,N,K):
  ren = Device["AMD"].renderer
  prg = to_program(ast_for(M,N,K), ren)
  elf = next(u.arg for u in prg.src if u.op is Ops.BINARY)
  src = next(u.arg for u in prg.src if u.op is Ops.SOURCE)
  from tinygrad.runtime.support.elf import elf_loader
  image, sections, relocs = elf_loader(elf)
  for s in sections:
    if s.name == ".text":
      return bytes(s.content), src
  raise RuntimeError("no .text")

mode = os.environ.get("MODE","both")
if mode in ("isa","both"):
  raw = isa_bytes(M,N,K)
  print(f"===== ISA renderer {M}x{N}x{K}  ({len(raw)} bytes) =====")
  for off,hx,s in disasm_stream(raw):
    print(f"{off:5d}: {hx:<20} {s}")
if mode in ("hip","both"):
  raw, src = hip_text(M,N,K)
  print(f"\n===== HIP/LLVM {M}x{N}x{K}  ({len(raw)} bytes) =====")
  for off,hx,s in disasm_stream(raw):
    print(f"{off:5d}: {hx:<20} {s}")
