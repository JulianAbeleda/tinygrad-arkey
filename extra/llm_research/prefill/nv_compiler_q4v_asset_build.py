"""Build an isolated finalized Q4-V cubin asset and manifest."""
import argparse, hashlib, json, pathlib
from tinygrad import Device
from tinygrad.codegen import to_program_cache
from extra.llm_research.prefill.nv_compiler_q4k_k_pp512_binding import CompilerKPP512Binding

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
  b=CompilerKPP512Binding.compile(Device['NV'], 'attn_v')
  p=b.main_program; binary=next(u.arg for u in p.src if getattr(u,'op',None).name=='BINARY')
  out=pathlib.Path(a.out); out.mkdir(parents=True,exist_ok=True); (out/'program.cubin').write_bytes(binary)
  i=p.arg; m={'schema':'tinygrad.nv.q4v.asset.v1','identity':b.candidate_identity,'binary_sha256':hashlib.sha256(binary).hexdigest(),
    'name':i.name,'global_size':i.global_size,'local_size':i.local_size,'globals':i.globals,'outs':i.outs,'ins':i.ins,'vals':[v.arg[1] for v in i.vars], 'aux':i.aux}
  (out/'manifest.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
