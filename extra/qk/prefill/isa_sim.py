"""Functional RDNA3 wave32 interpreter for the AMDISARenderer INS stream (DEV=PYTHON, no GPU).

Purpose: single out and trace correctness bugs in the pure-codegen prefill WMMA GEMM by executing the emitted
instruction stream against symbolic buffers and checking address/coverage invariants -- without a GPU. Reuses
test/amd/disasm.py (INS->text) so there is one source of truth for decoding; parses the text operands.

Stage 1 (address/coverage, no WMMA needed): reconstruct, per lane, the byte address of every global_load / global_store,
map to a matrix element (A[m,k] / B[k,n] / C[m,n]) via known row-major strides, and check coverage:
  - stores must cover every C[m,n] EXACTLY once (a gap => that output keeps garbage => scattered NaN),
  - loads must read only in-bounds A/B elements.
"""
import os, sys, re
os.environ["ALLOW_DEVICE_USAGE"]="1"
import numpy as np
from collections import Counter
from tinygrad import Tensor
from tinygrad.helpers import Target
from tinygrad.uop.ops import Ops
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.codegen import to_program
sys.path.insert(0, os.getcwd())
from test.amd.disasm import disasm

NLANE = 32
# symbolic buffer bases (large, distinct, so byte_addr - base identifies the buffer & offset)
BASES = {"OUT": 1<<40, "A": 2<<40, "B": 3<<40}
KARG = {0x0: "OUT", 0x8: "A", 0x10: "B"}   # kernarg offset -> buffer (tinygrad order [out, A, B])

def capture(M,N,K):
  ren=AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  cap={}
  orig=ren._insert_waitcnt
  def wrap(uops): cap['insts']=list(uops); return orig(uops)
  ren._insert_waitcnt=wrap
  a=Tensor.empty(M,K,dtype="half"); b=Tensor.empty(K,N,dtype="half")
  lin=(a@b).schedule_linear(); ast=[u for u in lin.toposort() if u.op is Ops.SINK][0]
  to_program(ast, ren)
  return cap['insts']

def parse_operand(tok):
  # returns ('v',idx) | ('s',idx) | ('spair',idx) | ('imm',val)
  tok=tok.strip()
  if tok=="null": return ('imm',0)
  m=re.fullmatch(r"v(\d+)", tok)
  if m: return ('v', int(m.group(1)))
  m=re.fullmatch(r"v\[(\d+):(\d+)\]", tok)
  if m: return ('v', int(m.group(1)))
  m=re.fullmatch(r"s(\d+)", tok)
  if m: return ('s', int(m.group(1)))
  m=re.fullmatch(r"s\[(\d+):(\d+)\]", tok)
  if m: return ('spair', int(m.group(1)))
  m=re.fullmatch(r"0x([0-9a-fA-F]+)", tok)
  if m: return ('imm', int(m.group(1),16))
  m=re.fullmatch(r"-?\d+", tok)
  if m: return ('imm', int(tok))
  # e.g. "v200.l" high/low -> treat as vreg (address math never uses .h)
  m=re.fullmatch(r"v(\d+)\.[lh]", tok)
  if m: return ('v', int(m.group(1)))
  return ('other', tok)

class Sim:
  def __init__(self):
    self.v={}   # idx -> np.array(NLANE,) uint64
    self.s={}   # idx -> python int
    self.scc=0
    self.v[0]=np.arange(NLANE, dtype=np.uint64)         # v0 = lane id
    self.s[0]=0; self.s[1]=0                             # kernarg base = 0
    self.s[2]=self.s[3]=self.s[4]=0                      # workgroup id = (0,0,0)
    self.loads=[]   # (ins_idx, buffer, per-lane elem-byte-offset array)
    self.stores=[]
  def vget(self,idx): return self.v.get(idx, np.zeros(NLANE,dtype=np.uint64))
  def val(self, op):
    k=op[0]
    if k=='v': return self.vget(op[1])
    if k=='imm': return np.full(NLANE, op[1]&0xffffffffffffffff, dtype=np.uint64)
    if k=='s': return np.full(NLANE, self.s.get(op[1],0)&0xffffffffffffffff, dtype=np.uint64)
    if k=='spair':
      lo=self.s.get(op[1],0); hi=self.s.get(op[1]+1,0); return np.full(NLANE,(lo|(hi<<32)),dtype=np.uint64)
    return np.zeros(NLANE,dtype=np.uint64)

def run(M,N,K,label,insts=None):
  if insts is None: insts=capture(M,N,K)
  # decode to (mnemonic, [operand-tokens], offset_imm)
  decoded=[]
  for u in insts:
    a=u.arg
    if isinstance(a,tuple): decoded.append(('CTRL',a)); continue
    txt=disasm(a)
    # strip trailing "offset:0xNN"
    off=0
    mo=re.search(r"offset:(0x[0-9a-f]+|\d+)", txt)
    if mo: off=int(mo.group(1),16) if mo.group(1).startswith('0x') else int(mo.group(1)); txt=txt[:mo.start()].strip()
    parts=txt.split(None,1)
    mn=parts[0]
    ops=[p.strip() for p in parts[1].split(',')] if len(parts)>1 else []
    decoded.append((mn,[parse_operand(o) for o in ops], off))
  sim=Sim()
  # execute with a simple label/branch loop driver
  labels={}
  for i,d in enumerate(decoded):
    if d[0]=='CTRL' and d[1][0]=='label': labels[d[1][1]]=i
  pc=0; guard=0
  while pc < len(decoded):
    guard+=1
    if guard>200000: print("GUARD TRIP"); break
    d=decoded[pc]
    if d[0]=='CTRL':
      a=d[1]
      if a[0]=='branch':
        kind=a[1]
        if kind=='s_cbranch_scc0':
          # branch taken when SCC==0; target is the matching 'out' label (loop exit)
          if sim.scc==0: pc=labels.get(('out',0), pc+1); continue
        elif kind=='s_branch':
          pc=labels.get(('top',0), pc+1); continue
      pc+=1; continue
    mn,ops,off=d
    def dst(): return ops[0][1]
    if mn=='s_load_b64':
      # s[dst], s[base], null offset:X  -> load buffer base for kernarg offset X
      buf=KARG.get(off)
      if buf is not None:
        di=ops[0][1]; base=BASES[buf]; sim.s[di]=base&0xffffffff; sim.s[di+1]=(base>>32)&0xffffffff
      pc+=1; continue
    if mn=='s_mov_b32': sim.s[dst()]= (ops[1][1] if ops[1][0]=='imm' else sim.s.get(ops[1][1],0)); pc+=1; continue
    if mn=='s_add_i32': sim.s[dst()]=(sim.s.get(ops[1][1],0) if ops[1][0]=='s' else ops[1][1])+(ops[2][1] if ops[2][0]=='imm' else sim.s.get(ops[2][1],0)); pc+=1; continue
    if mn=='s_cmp_lt_i32':
      A=sim.s.get(ops[0][1],0) if ops[0][0]=='s' else ops[0][1]; B=ops[1][1] if ops[1][0]=='imm' else sim.s.get(ops[1][1],0)
      sim.scc=1 if A<B else 0; pc+=1; continue
    if mn=='v_mov_b32': sim.v[dst()]=sim.val(ops[1]).copy(); pc+=1; continue
    if mn=='v_and_b32_e32': sim.v[dst()]=(sim.val(ops[1])&sim.val(ops[2]))&0xffffffff; pc+=1; continue
    if mn=='v_mul_lo_u32': sim.v[dst()]=(sim.val(ops[1])*sim.val(ops[2]))&0xffffffff; pc+=1; continue
    if mn=='v_add_nc_u32_e32': sim.v[dst()]=(sim.val(ops[1])+sim.val(ops[2]))&0xffffffff; pc+=1; continue
    if mn=='v_lshlrev_b32_e32': sim.v[dst()]=(sim.val(ops[2])<<(sim.val(ops[1])&31))&0xffffffff; pc+=1; continue
    if mn=='v_lshrrev_b32_e32': sim.v[dst()]=(sim.val(ops[2])>>(sim.val(ops[1])&31))&0xffffffff; pc+=1; continue
    if mn=='global_load_u16':
      # vdst, vaddr, saddr [offset]
      vaddr=sim.val(ops[1]); saddr=sim.val(ops[2]); addr=(saddr+vaddr+off)&0xffffffffffffffff
      # identify buffer
      for buf,base in BASES.items():
        rel=addr.astype(np.int64)-base
        if np.all((rel>=0)&(rel< (1<<38))): sim.loads.append((pc,buf,rel.copy())); break
      # set vdst to something (not needed for addr stage)
      sim.v[dst()]=np.zeros(NLANE,dtype=np.uint64); pc+=1; continue
    if mn=='global_store_b16':
      # addr, data, saddr [offset]
      vaddr=sim.val(ops[0]); saddr=sim.val(ops[2]) if len(ops)>2 else np.zeros(NLANE,dtype=np.uint64); addr=(saddr+vaddr+off)&0xffffffffffffffff
      for buf,base in BASES.items():
        rel=addr.astype(np.int64)-base
        if np.all((rel>=0)&(rel<(1<<38))): sim.stores.append((pc,buf,rel.copy())); break
      pc+=1; continue
    # ignore compute we don't model in stage 1 (v_pack, v_cvt, v_wmma) -- they don't affect addresses
    pc+=1
  # ---- coverage analysis ----
  print(f"\n{'='*64}\n{label}  {M}x{N}x{K}\n{'='*64}")
  # STORES: expect every C[m,n] (m<M,n<N) exactly once. elem = byte/2; row=elem//N, col=elem%N
  store_elems=Counter()
  for pc,buf,rel in sim.stores:
    if buf!="OUT": print(f"  store@{pc} -> WRONG buffer {buf}"); continue
    for r in rel: store_elems[int(r)//2]+=1
  expected=set(range(M*N))
  got=set(store_elems.keys())
  missing=sorted(expected-got); extra=sorted(got-expected); dup=sorted([e for e,c in store_elems.items() if c>1])
  print(f"STORE coverage: {len(got)}/{M*N} distinct output elems; missing={len(missing)} extra={len(extra)} dup={len(dup)}")
  if missing[:8]: print(f"  MISSING output elems (r,c): {[(e//N,e%N) for e in missing[:8]]}{' ...' if len(missing)>8 else ''}")
  if dup[:8]: print(f"  DUP output elems (r,c): {[(e//N,e%N) for e in dup[:8]]}{' ...' if len(dup)>8 else ''}")
  # LOADS: A elems r=elem//K,c=elem%K must be <M,<K ; B elems r=elem//N c=elem%N <K,<N
  for buf,(R,C) in {"A":(M,K),"B":(K,N)}.items():
    oob=0; elems=set()
    for pc,b,rel in sim.loads:
      if b!=buf: continue
      for r in rel:
        e=int(r)//2; elems.add(e)
        if e>=R*C: oob+=1
    print(f"LOAD {buf}: {len(elems)} distinct elems (of {R*C}); out-of-bounds reads={oob}")
  return missing, dup

def route_check(M,N,K,label,insts=None):
  # EPILOGUE ROUTING: does the store to output block (m,n) read the accumulator of subtile (m,n)?
  # Trace reg provenance: wmma tags its 8 vdst acc regs with subtile id + records acc_base->(A src0 base,B src1 base).
  # cvt/mov propagate; store reads data reg's tag. A bases sorted->m index, B bases sorted->n index.
  if insts is None: insts=capture(M,N,K)
  # first get per-store address (m,n) block via the address sim's decode+exec (reuse run's machinery minimally):
  # re-decode
  dec=[]
  for u in insts:
    a=u.arg
    if isinstance(a,tuple): dec.append(('CTRL',a)); continue
    txt=disasm(a); off=0
    mo=re.search(r"offset:(0x[0-9a-f]+|\d+)", txt)
    if mo: off=int(mo.group(1),16) if mo.group(1).startswith('0x') else int(mo.group(1)); txt=txt[:mo.start()].strip()
    parts=txt.split(None,1); mn=parts[0]
    ops=[parse_operand(o) for o in parts[1].split(',')] if len(parts)>1 else []
    dec.append((mn,ops,off,txt))
  # collect wmma acc->(A,B) bases
  wmma_acc={}   # acc_base -> (Abase, Bbase)
  for d in dec:
    if d[0]=='CTRL': continue
    if d[0].startswith('v_wmma'):
      vd=d[1][0][1]; a0=d[1][1][1]; b1=d[1][2][1]; wmma_acc[vd]=(a0,b1)
  Abases=sorted(set(v[0] for v in wmma_acc.values())); Bbases=sorted(set(v[1] for v in wmma_acc.values()))
  # element-provenance: fragment REG -> set of matrix elems it was packed from (real geometry, no base-order guess).
  ld_elem={}      # load-dest reg -> ('A'/'B', set(elems))
  frag_elems={}   # fragment reg -> ('A'/'B', set(elems))
  prov={}         # reg -> ('acc', acc_base)
  # also run address exec via a fresh Sim replicating run()'s loop (but capturing store addr + data reg tag)
  sim=Sim(); labels={}
  for i,d in enumerate(dec):
    if d[0]=='CTRL' and d[1][0]=='label': labels[d[1][1]]=i
  pc=0; guard=0; mism=[]; nstore=0; allpairs=[]
  while pc<len(dec):
    guard+=1
    if guard>500000: break
    d=dec[pc]
    if d[0]=='CTRL':
      a=d[1]
      if a[0]=='branch':
        if a[1]=='s_cbranch_scc0' and sim.scc==0: pc=labels.get(('out',0),pc+1); continue
        if a[1]=='s_branch': pc=labels.get(('top',0),pc+1); continue
      pc+=1; continue
    mn,ops,off=d[0],d[1],d[2]
    di=ops[0][1] if ops and ops[0][0] in ('v','s','spair') else None
    # address-affecting ops (same as run())
    if mn=='s_load_b64':
      buf=KARG.get(off)
      if buf is not None: base=BASES[buf]; sim.s[ops[0][1]]=base&0xffffffff; sim.s[ops[0][1]+1]=(base>>32)&0xffffffff
    elif mn=='s_mov_b32': sim.s[di]=ops[1][1] if ops[1][0]=='imm' else sim.s.get(ops[1][1],0)
    elif mn=='s_add_i32': sim.s[di]=(sim.s.get(ops[1][1],0) if ops[1][0]=='s' else ops[1][1])+(ops[2][1] if ops[2][0]=='imm' else sim.s.get(ops[2][1],0))
    elif mn=='s_cmp_lt_i32':
      A=sim.s.get(ops[0][1],0) if ops[0][0]=='s' else ops[0][1]; B=ops[1][1] if ops[1][0]=='imm' else sim.s.get(ops[1][1],0); sim.scc=1 if A<B else 0
    elif mn=='v_mov_b32': sim.v[di]=sim.val(ops[1]).copy(); prov[di]=prov.get(ops[1][1]) if ops[1][0]=='v' else None
    elif mn=='v_and_b32_e32': sim.v[di]=(sim.val(ops[1])&sim.val(ops[2]))&0xffffffff
    elif mn=='v_mul_lo_u32': sim.v[di]=(sim.val(ops[1])*sim.val(ops[2]))&0xffffffff
    elif mn=='v_add_nc_u32_e32': sim.v[di]=(sim.val(ops[1])+sim.val(ops[2]))&0xffffffff
    elif mn=='v_lshlrev_b32_e32': sim.v[di]=(sim.val(ops[2])<<(sim.val(ops[1])&31))&0xffffffff
    elif mn=='v_lshrrev_b32_e32': sim.v[di]=(sim.val(ops[2])>>(sim.val(ops[1])&31))&0xffffffff
    elif mn.startswith('v_wmma'):
      base=di
      for i in range(8): prov[base+i]=('acc',base)
    elif mn=='v_cvt_f16_f32': prov[di]=prov.get(ops[1][1]) if ops[1][0]=='v' else None
    elif mn=='v_pack_b32_f16':
      # accumulate real matrix elems from the two vgpr load-dest sources into this fragment reg
      s=set(); buf=None
      for o in ops[1:]:
        if o[0]=='v' and o[1] in ld_elem: b,es=ld_elem[o[1]]; s|=es; buf=b
      if buf is not None: frag_elems[di]=(buf, frag_elems.get(di,(buf,set()))[1]|s)
    elif mn=='global_load_u16':
      vaddr=sim.val(ops[1]); saddr=sim.val(ops[2]); addr=(saddr+vaddr+off)&0xffffffffffffffff
      for buf,base in BASES.items():
        rel=addr.astype(np.int64)-base
        if np.all((rel>=0)&(rel<(1<<38))): ld_elem[di]=(buf, set(int(x)//2 for x in rel)); break
      sim.v[di]=np.zeros(NLANE,dtype=np.uint64)
    elif mn=='global_store_b16':
      vaddr=sim.val(ops[0]); saddr=sim.val(ops[2]) if len(ops)>2 else np.zeros(NLANE,dtype=np.uint64)
      addr=(saddr+vaddr+off)&0xffffffffffffffff; rel=addr.astype(np.int64)-BASES["OUT"]
      elems=(rel//2).astype(np.int64); rows=elems//N; cols=elems%N
      mblk=set((int(r)//16) for r in rows); nblk=set((int(c)//16) for c in cols)
      dreg=ops[1][1] if ops[1][0]=='v' else None; tag=prov.get(dreg)
      nstore+=1
      if tag and tag[0]=='acc':
        acc=tag[1]; ab,bb=wmma_acc.get(acc,(None,None))
        # real rows this A-frag holds / cols this B-frag holds (from load addresses)
        arows=set(); bcols=set()
        for r in range(ab,ab+8):
          if r in frag_elems: arows|={e//K for e in frag_elems[r][1]}
        for r in range(bb,bb+8):
          if r in frag_elems: bcols|={e%N for e in frag_elems[r][1]}
        ambk=set(rr//16 for rr in arows); anbk=set(cc//16 for cc in bcols)   # the block this accumulator OWNS
        allpairs.append((next(iter(mblk)),next(iter(nblk)),sorted(ambk),sorted(anbk),acc))
        # store addr rows/cols must lie within the accumulator's owned rows/cols
        bad = not (set(int(r) for r in rows)<=arows and set(int(c) for c in cols)<=bcols)
        if bad: mism.append((pc,f"addr rows{sorted(set(int(r)//16 for r in rows))} cols{sorted(set(int(c)//16 for c in cols))}",
                             f"acc{acc} owns rowblk{sorted(ambk)} colblk{sorted(anbk)}"))
      else:
        mism.append((pc,'no-acc-tag',tag))
    pc+=1
  print(f"\n{'='*64}\nROUTING {label} {M}x{N}x{K}\n{'='*64}")
  print(f"stores={nstore} Abases={Abases} Bbases={Bbases}")
  print(f"routing mismatches: {len(mism)}")
  for x in mism[:16]: print("   ", x)
  print("addr-block <- accumulator-owned-block confusion (where differ):")
  diff=Counter((amb,anb,tuple(ownm),tuple(ownn)) for amb,anb,ownm,ownn,_ in allpairs if [amb]!=ownm or [anb]!=ownn)
  for k,c in sorted(diff.items()): print(f"   addr-block({k[0]},{k[1]}) <- acc-owns(row{list(k[2])},col{list(k[3])})  x{c}")
  return allpairs

if __name__=="__main__":
  for (M,N,K,lbl) in [(32,64,64,"2x4"),(64,32,64,"4x2"),(64,64,64,"4x4")]:
    insts=capture(M,N,K)
    run(M,N,K,lbl,insts)
    route_check(M,N,K,lbl,insts)
