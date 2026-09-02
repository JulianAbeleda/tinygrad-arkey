#!/usr/bin/env python3
"""Static sm120 microgate for two CUDA/PTX phase-boundary copy mechanisms.

No kernel is launched.  The control models the ordinary RegionLoad direct-copy
source shape.  Candidate A bridges eighteen values across the overwrite barrier
with C-visible registers; candidate B owns load/barrier/store in one PTX region.
"""
from __future__ import annotations

import argparse, collections, contextlib, ctypes, ctypes.util, hashlib, json, pathlib, re, signal, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tinygrad.runtime.support import compiler_cuda
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler


DEFAULT_OUT = ROOT / "docs/task_workflow/evidence/nv-ptx-phase-boundary-microgate-20260901"
NVDISASM = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
CUOBJDUMP = pathlib.Path("/usr/local/cuda-13.2/bin/cuobjdump")
INSN_RE = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b(.*)$", re.MULTILINE)
RESOURCE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
ARMS = ("regionload_control", "split_register_bridge", "fused_ptx_region")
EXPECTED_MEMORY = {"LDG": 18, "STS": 18, "LDS": 18, "STG": 1, "BAR": 2}


@contextlib.contextmanager
def hard_timeout(seconds:int):
  def expired(_signum, _frame): raise TimeoutError(f"phase-boundary microgate exceeded {seconds}s")
  old=signal.signal(signal.SIGALRM,expired); signal.setitimer(signal.ITIMER_REAL,seconds)
  try: yield
  finally: signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,old)


def _phase_body() -> list[str]:
  lines = ["  unsigned int tid = (unsigned int)threadIdx.x;", "  unsigned int mix = seed ^ (tid * 0x9e3779b9u);"]
  for i in range(64):
    salt=(0x45D9F3B*(i+1))&0xFFFFFFFF
    lines.append(f"  float acc{i} = __uint2float_rn((mix ^ 0x{salt:08x}u) & 0xffffu);")
  for rnd in range(8):
    for i in range(64):
      mul=1.0001+((i*7+rnd*3)%29)*0.000013
      add=0.125+((i*13+rnd*11)%127)*0.03125
      lines.append(f"  acc{i} = __fmaf_rn(acc{i}, {mul:.9f}f, {add:.9f}f);")
  return lines


def _split_copy() -> list[str]:
  names=[f"copy{i}" for i in range(18)]
  loads=[f'      "ld.global.u32 %{i}, [%18+{i*1024}];\\n\\t"' for i in range(18)]
  stores=[f'      "st.shared.u32 [%0+{i*1024}], %{i+1};\\n\\t"' for i in range(18)]
  return [
    f"  unsigned int {', '.join(names)};",
    "  unsigned int *src_lane = src + src_base + tid;",
    "  unsigned int shared_lane = (unsigned int)__cvta_generic_to_shared(scratch + tid);",
    "  asm volatile(", *loads,
    "      : " + ", ".join(f'"=r"({x})' for x in names),
    '      : "l"(src_lane)',
    '      : "memory");',
    "  __syncthreads();",
    "  asm volatile(", *stores,
    "      :",
    '      : "r"(shared_lane), ' + ", ".join(f'"r"({x})' for x in names),
    '      : "memory");',
  ]


def _fused_copy() -> list[str]:
  loads=[f'      "ld.global.u32 copy{i}, [%0+{i*1024}];\\n\\t"' for i in range(18)]
  stores=[f'      "st.shared.u32 [%1+{i*1024}], copy{i};\\n\\t"' for i in range(18)]
  return [
    "  unsigned int *src_lane = src + src_base + tid;",
    "  unsigned int shared_lane = (unsigned int)__cvta_generic_to_shared(scratch + tid);",
    "  asm volatile(",
    '      "{\\n\\t"',
    '      ".reg .u32 copy<18>;\\n\\t"',
    *loads,
    '      "bar.sync 0;\\n\\t"',
    *stores,
    '      "}\\n"',
    "      :",
    '      : "l"(src_lane), "r"(shared_lane)',
    '      : "memory");',
  ]


def source_for(arm:str) -> str:
  if arm not in ARMS: raise ValueError(arm)
  lines=[
    "#include <cuda_fp16.h>",
    'extern "C" __global__ void __launch_bounds__(256) phase_boundary_microgate(unsigned int *out, unsigned int *src, unsigned int seed, unsigned int src_base) {',
    "  extern __shared__ unsigned int scratch[];",
    *_phase_body(),
  ]
  if arm == "regionload_control":
    lines.append("  __syncthreads();")
    lines.extend(f"  scratch[tid + {i*256}u] = src[src_base + tid + {i*256}u];" for i in range(18))
  elif arm == "split_register_bridge": lines.extend(_split_copy())
  else: lines.extend(_fused_copy())
  lines += ["  __syncthreads();", "  float sum = acc0;"]
  lines.extend(f"  sum = __fadd_rn(sum, acc{i});" for i in range(1,64))
  lines += [
    "  unsigned int read_lane = (tid + ((seed >> 16) | 1u)) & 255u;",
    "  unsigned int shared_mix = scratch[read_lane];",
  ]
  lines.extend(f"  shared_mix ^= scratch[read_lane + {i*256}u];" for i in range(1,18))
  lines += ["  out[tid] = __float_as_uint(sum) ^ shared_mix;", "}"]
  return "\n".join(lines)+"\n"


def source_contract(source:str, arm:str) -> dict:
  asm_blocks=source.count("asm volatile(")
  ptx_loads=source.count("ld.global.u32")
  ptx_stores=source.count("st.shared.u32")
  direct_copies=sum(x.strip().startswith("scratch[tid +") and " = src[" in x for x in source.splitlines())
  expected=(direct_copies,ptx_loads,ptx_stores,asm_blocks,source.count("bar.sync 0;"),source.count("__syncthreads();"))
  wanted={"regionload_control":(18,0,0,0,0,2), "split_register_bridge":(0,18,18,2,0,2), "fused_ptx_region":(0,18,18,1,1,1)}[arm]
  forbidden={"qualified_pointer": "const unsigned int *" in source or "__restrict__" in source,
             "volatile_c_object": bool(re.search(r"(?:^|[;{}])\s*(?:volatile\s+unsigned|unsigned\s+volatile)",source)),
             "noinline": "noinline" in source, "function_split": source.count('__global__ void') != 1}
  contract={"direct_copies":direct_copies,"ptx_loads":ptx_loads,"ptx_stores":ptx_stores,"asm_blocks":asm_blocks,
            "inline_barriers":source.count("bar.sync 0;"),"c_barriers":source.count("__syncthreads();"),
            "accumulators":source.count("float acc"),"fmas":source.count(" = __fmaf_rn(acc"),
            "downstream_shared_consumers":source.count("scratch[read_lane"),"forbidden":forbidden}
  contract["pass"] = expected == wanted and contract["accumulators"] == 64 and contract["fmas"] == 64*8 and \
                     contract["downstream_shared_consumers"] == 18 and not any(forbidden.values())
  return contract


def run_tool(args:list[str]) -> str:
  return subprocess.run(args,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30).stdout


def tool_version(path:pathlib.Path) -> str: return run_tool([str(path),"--version"]).strip()


def nvrtc_version() -> str:
  major,minor=ctypes.c_int(),ctypes.c_int()
  compiler_cuda.nvrtc_check(compiler_cuda.nvrtc.nvrtcVersion(ctypes.byref(major),ctypes.byref(minor)))
  return f"{major.value}.{minor.value}"


def analyze_sass(sass:str, resources:str) -> dict:
  instructions=[]
  for ordinal,m in enumerate(INSN_RE.finditer(sass)):
    opcode,operands=m.group(2),m.group(3).strip()
    instructions.append({"ordinal":ordinal,"pc":int(m.group(1),16),"opcode":opcode,"family":opcode.split(".",1)[0],"operands":operands})
  families=collections.Counter(x["family"] for x in instructions)
  opcodes=collections.Counter(x["opcode"] for x in instructions)
  by_family={name:[x for x in instructions if x["family"] == name] for name in ("LDG","STS","LDS","STG","BAR","LDL","STL")}
  loads,stores,bars=by_family["LDG"],by_family["STS"],by_family["BAR"]
  match=RESOURCE_RE.search(resources)
  resource=({"registers":int(match.group(1)),"stack_bytes":int(match.group(2)),"shared_static_bytes":int(match.group(3)),
             "local_static_bytes":int(match.group(4))} if match else None)
  memory_exact=all(families.get(name,0) == count for name,count in EXPECTED_MEMORY.items())
  ordered=len(bars)==2 and len(loads)==18 and len(stores)==18 and max(x["ordinal"] for x in loads) < bars[0]["ordinal"] < \
          min(x["ordinal"] for x in stores) and max(x["ordinal"] for x in stores) < bars[1]["ordinal"]
  no_forbidden=not any(x["family"] in {"MEMBAR","ATOM","RED","LDL","STL"} for x in instructions)
  spill_free=resource is not None and resource["stack_bytes"] == resource["local_static_bytes"] == 0 and \
             families.get("LDL",0) == families.get("STL",0) == 0
  span=stores[0]["ordinal"]-loads[0]["ordinal"] if loads and stores else None
  return {"instruction_total":len(instructions),"families":dict(sorted(families.items())),"opcodes":dict(sorted(opcodes.items())),
          "resources":resource,"memory_exact":memory_exact,"ordered_across_overwrite_barrier":ordered,
          "no_forbidden_ops":no_forbidden,"spill_free":spill_free,"first_ldg_to_first_sts_instructions":span,
          "first_ldg_to_first_sts_pc_bytes":stores[0]["pc"]-loads[0]["pc"] if loads and stores else None,
          "ldg_opcodes":dict(sorted(collections.Counter(x["opcode"] for x in loads).items())),
          "sts_opcodes":dict(sorted(collections.Counter(x["opcode"] for x in stores).items())),
          "lds_opcodes":dict(sorted(collections.Counter(x["opcode"] for x in by_family["LDS"]).items())),
          "ldg_pcs":[f"0x{x['pc']:x}" for x in loads],"sts_pcs":[f"0x{x['pc']:x}" for x in stores],
          "lds_pcs":[f"0x{x['pc']:x}" for x in by_family["LDS"]],"barrier_pcs":[f"0x{x['pc']:x}" for x in bars],
          "one_physical_op_per_copy":len(loads)==len(stores)==18,
          "base_hard_pass":memory_exact and ordered and no_forbidden and spill_free and span is not None and span <= 160}


def compile_arm(arm:str,out_dir:pathlib.Path,repeats:int,timeout:int) -> dict:
  arm_dir=out_dir/arm; arm_dir.mkdir(parents=True,exist_ok=True)
  source=source_for(arm); (arm_dir/f"{arm}.cu").write_text(source)
  binaries=[]; hashes=[]; elapsed=[]; options=None
  for repeat in range(repeats):
    compiler=NVRTCCompiler("sm_120",ptx=False,cache_key=f"ptx_phase_boundary_{arm}_r{repeat}")
    current=list(compiler.compile_options)
    if options is None: options=current
    elif current != options: raise RuntimeError(f"NVRTC options changed across repeats for {arm}")
    start=time.perf_counter()
    with hard_timeout(timeout): binary=compiler.compile(source)
    elapsed.append(time.perf_counter()-start); binaries.append(binary); hashes.append(hashlib.sha256(binary).hexdigest())
  cubin=arm_dir/f"{arm}.cubin"; cubin.write_bytes(binaries[0])
  sass=run_tool([str(NVDISASM),"-c",str(cubin)]); resources=run_tool([str(CUOBJDUMP),"--dump-resource-usage",str(cubin)])
  (arm_dir/f"{arm}.nvdisasm").write_text(sass); (arm_dir/f"{arm}.resources.txt").write_text(resources)
  record={"arm":arm,"arch":"sm_120","launch_bounds":"256","source":str(arm_dir/f"{arm}.cu"),
          "source_sha256":hashlib.sha256(source.encode()).hexdigest(),"source_contract":source_contract(source,arm),
          "nvrtc_options":options,"compile_seconds":elapsed,"cubin":str(cubin),"cubin_bytes":len(binaries[0]),
          "cubin_sha256_repeats":hashes,"stable_cubin":len(set(hashes))==1,"sass":str(arm_dir/f"{arm}.nvdisasm"),
          "resource_report":str(arm_dir/f"{arm}.resources.txt"),"analysis":analyze_sass(sass,resources)}
  (arm_dir/f"{arm}.json").write_text(json.dumps(record,indent=2)+"\n")
  return record


def markdown(result:dict) -> str:
  rows=[]
  for name,arm in result["arms"].items():
    a,r=arm["analysis"],arm["analysis"]["resources"]
    rows.append(f"| `{name}` | {a['first_ldg_to_first_sts_instructions']} | {r['registers']} | {r['stack_bytes']} | "
                f"{a['instruction_total']} | `{a['ldg_opcodes']}` | {arm['stable_cubin']} | {arm['hard_pass']} |")
  return "\n".join([
    "# NV PTX phase-boundary synthetic microgate (2026-09-01)","",f"## Verdict: `{result['verdict']}`","",result["release_reason"],"",
    "| Arm | LDG->STS span | REG | Stack | Instructions | LDG opcode | Stable | Hard pass |","|---|---:|---:|---:|---:|---|---:|---:|",*rows,"",
    "## Contract","",
    "- All arms use sm120, launch_bounds(256), one 64-accumulator/8-FMA live phase body, 18 affine global-to-shared copies, and 18 downstream cross-thread shared consumers.",
    "- Exact physical census is 18 LDG, 18 STS, 18 LDS, 1 STG, and 2 BAR; no stack/local/LDL/STL/MEMBAR/ATOM traffic is admitted.",
    "- Candidate loads must precede overwrite BAR0 and candidate stores must lie strictly between overwrite BAR0 and publication BAR1.",
    "- Source contains no const/restrict pointer, volatile C memory object, noinline helper, or function split. Inline PTX asm is volatile by design.","",
    "## Toolchain","",f"- NVRTC: `{result['tools']['nvrtc_version']}`",f"- NVRTC options: `{result['nvrtc_options']}`",
    f"- nvdisasm: `{result['tools']['nvdisasm_version'].splitlines()[-2]}`",f"- cuobjdump: `{result['tools']['cuobjdump_version'].splitlines()[-2]}`","",
    "This is a CUDA/PTX compiler-boundary experiment only. No kernel was launched and no Q6 or production source was modified.","",
  ])


def main() -> int:
  parser=argparse.ArgumentParser(); parser.add_argument("--out-dir",type=pathlib.Path,default=DEFAULT_OUT)
  parser.add_argument("--repeats",type=int,default=3); parser.add_argument("--compile-timeout",type=int,default=45); args=parser.parse_args()
  if args.repeats < 2: raise ValueError("repeats must be at least two")
  if not NVDISASM.is_file() or not CUOBJDUMP.is_file(): raise FileNotFoundError("CUDA disassembly tools unavailable")
  args.out_dir.mkdir(parents=True,exist_ok=True)
  result={"schema":"tinygrad.nv_ptx_phase_boundary_microgate.v1","date":"2026-09-01",
          "contract":{"arch":"sm_120","launch_bounds":256,"accumulators":64,"fma_rounds":8,"logical_copies":18,
                      "expected_memory":EXPECTED_MEMORY,"span_limit":160,"gpu_launched":False},
          "tools":{"python":sys.version,"nvrtc_version":nvrtc_version(),"nvrtc_library":ctypes.util.find_library("nvrtc"),
                   "nvdisasm_path":str(NVDISASM),"nvdisasm_version":tool_version(NVDISASM),
                   "cuobjdump_path":str(CUOBJDUMP),"cuobjdump_version":tool_version(CUOBJDUMP)},"arms":{}}
  for arm in ARMS: result["arms"][arm]=compile_arm(arm,args.out_dir,args.repeats,args.compile_timeout)
  control_lds=result["arms"]["regionload_control"]["analysis"]["families"].get("LDS",0)
  passing=[]
  for name,arm in result["arms"].items():
    no_lds_increase=arm["analysis"]["families"].get("LDS",0) <= control_lds
    arm["no_lds_increase_vs_control"]=no_lds_increase
    arm["hard_pass"]=arm["stable_cubin"] and arm["source_contract"]["pass"] and arm["analysis"]["base_hard_pass"] and no_lds_increase
    if name != "regionload_control" and arm["hard_pass"]: passing.append(name)
  if passing:
    passing.sort(key=lambda n:(result["arms"][n]["analysis"]["first_ldg_to_first_sts_instructions"],
                               result["arms"][n]["analysis"]["resources"]["registers"]))
    result["verdict"]="SYNTHETIC_PASS_RELEASE_ONE_REAL_Q6_STATIC_RERUN"; result["release_arm"]=passing[0]; result["release_real_q6"]=True
    result["release_reason"]=(f"`{passing[0]}` satisfies deterministic cubin, exact traffic, barrier ordering, <=160 span, and spill gates. "
                              "It releases one CUDA-only real-Q6 static compile, not execution or timing.")
  else:
    result["verdict"]="CUDA_PTX_PHASE_BOUNDARY_LIMITATION_REJECT"; result["release_arm"]=None; result["release_real_q6"]=False
    result["release_reason"]=("Neither PTX mechanism simultaneously preserves the exact traffic/barrier contract and bounded spill-free schedule. "
                              "CUDA C/PTX inline regions therefore do not provide the required physical phase boundary in this toolchain.")
  result["nvrtc_options"]=next(iter(result["arms"].values()))["nvrtc_options"]
  result["commands"]={"test":"python3 -m pytest -q test/unit/test_nv_ptx_phase_boundary_microgate.py",
                      "microgate":f"python3 extra/llm_research/prefill/nv_ptx_phase_boundary_microgate.py --out-dir {args.out_dir} --repeats {args.repeats}"}
  (args.out_dir/"result.json").write_text(json.dumps(result,indent=2)+"\n"); (args.out_dir/"result.md").write_text(markdown(result))
  print(json.dumps({"verdict":result["verdict"],"release_arm":result["release_arm"],"arms":{
    name:{"span":arm["analysis"]["first_ldg_to_first_sts_instructions"],"regs":arm["analysis"]["resources"]["registers"],
          "stack":arm["analysis"]["resources"]["stack_bytes"],"instructions":arm["analysis"]["instruction_total"],
          "counts":{x:arm["analysis"]["families"].get(x,0) for x in EXPECTED_MEMORY},"barriers":arm["analysis"]["barrier_pcs"],
          "hard_pass":arm["hard_pass"]} for name,arm in result["arms"].items()}},indent=2))
  return 0 if result["release_real_q6"] else 2


if __name__ == "__main__": raise SystemExit(main())
