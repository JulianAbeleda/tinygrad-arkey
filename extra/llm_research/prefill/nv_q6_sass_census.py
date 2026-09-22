"""Structured static census for generated and oracle CUDA cubins.

This is an experiment diagnostic. It parses nvdisasm/cuobjdump output and does
not add NVIDIA-specific semantics to tinygrad's UOp graph.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, pathlib, re, shutil, subprocess

DEFAULT_NVDISASM=pathlib.Path(".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm")
DEFAULT_CUOBJDUMP=pathlib.Path("/usr/local/cuda-13.2/bin/cuobjdump")
INSN_RE=re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?[A-Z0-9]+\s+)?([A-Z][A-Z0-9_.]*)\b",re.MULTILINE)
STACK_RE=re.compile(r"\[(?:R1|R1\.reuse)(?:\+0x([0-9a-fA-F]+))?\]")
RESOURCE_RE=re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")

def _tool(env_name:str, default:pathlib.Path, fallback:str) -> str:
  configured=os.getenv(env_name)
  if configured: return configured
  if default.is_file(): return str(default)
  if found:=shutil.which(fallback): return found
  raise FileNotFoundError(f"missing {fallback}; set {env_name}")

def _run(args:list[str]) -> str:
  return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT).stdout

def _instructions(disassembly:str) -> list[dict]:
  lines=disassembly.splitlines()
  out=[]
  for line in lines:
    if not (match:=INSN_RE.match(line)): continue
    opcode=match.group(2)
    stack=STACK_RE.search(line) if opcode.startswith(("LDL","STL")) else None
    out.append({"pc":int(match.group(1),16),"opcode":opcode,"family":opcode.split(".",1)[0],
      "stack_offset":int(stack.group(1),16) if stack and stack.group(1) else (0 if stack else None)})
  return out

def _spill_regions(instructions:list[dict]) -> dict:
  mma_pcs=[x["pc"] for x in instructions if x["family"] == "IMMA"]
  first_mma=min(mma_pcs) if mma_pcs else None
  last_mma=max(mma_pcs) if mma_pcs else None
  regions={name:{"LDL":0,"STL":0,"pcs":[],"stack_offsets":collections.Counter()}
           for name in ("pre_mma","mma_span","post_mma")}
  for ins in instructions:
    if ins["family"] not in ("LDL","STL"): continue
    region="pre_mma" if first_mma is None or ins["pc"] < first_mma else ("post_mma" if ins["pc"] > last_mma else "mma_span")
    regions[region][ins["family"]]+=1
    regions[region]["pcs"].append(f"0x{ins['pc']:x}")
    if ins["stack_offset"] is not None: regions[region]["stack_offsets"][f"0x{ins['stack_offset']:x}"]+=1
  return {name:{**data,"stack_offsets":dict(sorted(data["stack_offsets"].items(),key=lambda x:int(x[0],16)))} for name,data in regions.items()}

def analyze_cubin(cubin:pathlib.Path, out_dir:pathlib.Path, symbol:str|None=None) -> dict:
  cubin=cubin.resolve(); out_dir=out_dir.resolve(); out_dir.mkdir(parents=True,exist_ok=True)
  digest=hashlib.sha256(cubin.read_bytes()).hexdigest()
  disassembly=_run([_tool("NVDISASM",DEFAULT_NVDISASM,"nvdisasm"),"-c",str(cubin)])
  resources=_run([_tool("CUOBJDUMP",DEFAULT_CUOBJDUMP,"cuobjdump"),"--dump-resource-usage",str(cubin)])
  instructions=_instructions(disassembly)
  families=collections.Counter(x["family"] for x in instructions)
  exact=collections.Counter(x["opcode"] for x in instructions)
  resource_match=RESOURCE_RE.search(resources)
  result={"schema":"tinygrad.nv_q6_sass_census.v1","cubin":str(cubin),"cubin_sha256":digest,"symbol":symbol,
    "instruction_total":len(instructions),"families":dict(sorted(families.items())),"opcodes":dict(sorted(exact.items())),
    "resources":({"registers":int(resource_match.group(1)),"stack_bytes":int(resource_match.group(2)),
      "shared_static_bytes":int(resource_match.group(3)),"local_static_bytes":int(resource_match.group(4))} if resource_match else None),
    "spill_regions":_spill_regions(instructions),
    "notes":["Static instruction census only; these are not hardware performance counters.",
      "pre_mma/mma_span/post_mma are bounded by the first and last IMMA PCs and are not source-level attribution."]}
  stem=f"{cubin.stem}-{digest[:16]}"
  disasm_path=out_dir/f"{stem}.nvdisasm"; resource_path=out_dir/f"{stem}.resources.txt"; json_path=out_dir/f"{stem}.sass.json"
  disasm_path.write_text(disassembly); resource_path.write_text(resources); json_path.write_text(json.dumps(result,indent=2)+"\n")
  return {"cubin":str(cubin),"cubin_sha256":digest,"sass_json":str(json_path),"disassembly":str(disasm_path),
    "resources":str(resource_path),"summary":result}

def main() -> None:
  parser=argparse.ArgumentParser()
  parser.add_argument("--cubin",type=pathlib.Path,required=True)
  parser.add_argument("--out-dir",type=pathlib.Path,required=True)
  parser.add_argument("--symbol")
  args=parser.parse_args()
  print(json.dumps(analyze_cubin(args.cubin,args.out_dir,args.symbol),indent=2))

if __name__ == "__main__": main()
