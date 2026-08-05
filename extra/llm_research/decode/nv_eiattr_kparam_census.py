"""CPU-only raw EIATTR census; never guesses a native cbuf ABI."""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
from tinygrad.runtime.support.elf import elf_loader

def records(blob:bytes):
  _image, sections, _relocs=elf_loader(blob)
  out={}
  for sh in sections:
    if not sh.name.startswith('.nv.info.'): continue
    off=0; rows=[]; data=sh.content
    while off < sh.header.sh_size:
      typ,param,size=struct.unpack_from('BBH',data,off)
      payload=bytes(data[off+4:off+4+size]) if typ == 4 else b''
      row={'type':typ,'param':param,'size':size,'payload_hex':payload.hex()}
      # EIATTR_KPARAM_INFO / EIFMT_SVAL: index:u32, ordinal:u16,
      # cbuf offset:u16, packed:u32. Size is packed bits 18..31.
      if param == 0x17:
        if typ != 4 or size != 12: raise ValueError(f'{sh.name}: malformed KPARAM_INFO')
        index, ordinal, offset, packed=struct.unpack('<IHHI',payload)
        row['kparam']={'index':index,'ordinal':ordinal,'offset':offset,'size':packed>>18,'packed':packed}
      if param == 0x0a and typ == 4 and size >= 8:
        section_symbol_index, base_offset, param_size=struct.unpack_from('<IHH',payload)
        row['param_cbank']={'section_symbol_index':section_symbol_index,'base_offset':base_offset,'param_size':param_size}
      rows.append(row)
      off += 4+size if typ == 4 else 4
    out[sh.name[9:]]=rows
  return out

def parameter_layout(rows):
  cb=[r['param_cbank'] for r in rows if 'param_cbank' in r]
  ps=sorted((r['kparam'] for r in rows if 'kparam' in r),key=lambda x:x['ordinal'])
  if len(cb) != 1 or len(ps) != 19 or [p['ordinal'] for p in ps] != list(range(19)):
    raise ValueError('expected one PARAM_CBANK and exactly ordinals 0..18')
  base,size=cb[0]['base_offset'],cb[0]['param_size']
  if any(p['offset']+p['size'] > size for p in ps): raise ValueError('KPARAM outside PARAM_CBANK')
  return base,size,ps

def pack_cbuf(rows, values):
  """Pack pre-serialized ABI values by ordinal at declared cbuf offsets."""
  base,size,ps=parameter_layout(rows)
  if set(values) != set(range(19)): raise ValueError('need exactly ordinal values 0..18')
  out=bytearray(size)
  for p in ps:
    v=values[p['ordinal']]
    if not isinstance(v,(bytes,bytearray)) or len(v) != p['size']: raise ValueError(f"ordinal {p['ordinal']}: expected {p['size']} bytes")
    off=p['offset']; out[off:off+p['size']]=v
  return bytes(out)

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('cubin',type=Path); ap.add_argument('--out',type=Path); a=ap.parse_args()
  d={'schema':'tinygrad.nv.eiattr_kparam_census.v1','symbols':records(a.cubin.read_bytes()),
     'verdict':'RAW_ONLY_UNRESOLVED','reason':'record bytes are retained for differential inference; no KPARAM layout is asserted without known-signature corpus validation'}
  text=json.dumps(d,indent=2); print(text)
  if a.out: a.out.write_text(text+'\n')
if __name__=='__main__': main()
