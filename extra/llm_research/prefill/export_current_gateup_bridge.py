"""Export the installed default gate/up PROGRAMs for the existing CUDA bridge."""
import argparse, hashlib, json, pathlib
from tinygrad import Device
from tinygrad.uop.ops import Ops
from extra.llm_research.prefill.nv_compiler_q4k_pp512_binding import binding_for


def main():
  ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("directory",type=pathlib.Path); args=ap.parse_args()
  capture=binding_for("NV",variant="streamk",producer_arithmetic="llama").new_capture()
  args.directory.mkdir(parents=True,exist_ok=True)
  manifest={"arch":Device["NV"].arch,"programs":{}}
  for label, program in (("main",capture.q_program),("fixup",capture.fixup_program)):
    source=[u.arg for u in program.src if u.op is Ops.SOURCE]
    binary=[u.arg for u in program.src if u.op is Ops.BINARY]
    if len(source)!=1 or len(binary)!=1: raise ValueError("PROGRAM must retain one SOURCE and BINARY")
    (args.directory/f"{label}.cu").write_text(source[0])
    (args.directory/f"{label}.cubin").write_bytes(binary[0])
    abi={key:getattr(program.arg,key) for key in ("outs","ins","globals")}
    abi["vals"]=program.arg.vals({})
    manifest["programs"][label]={"name":program.arg.name,"source_sha256":hashlib.sha256(source[0].encode()).hexdigest(),
      "binary_sha256":hashlib.sha256(binary[0]).hexdigest(),**{key:list(value) for key,value in abi.items()}}
  (args.directory/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
  print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
