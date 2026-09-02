#!/usr/bin/env python3
"""NVRTC 2x2 compiler-scheduling microgate for a late 18-word direct copy.

This is a static compiler experiment. It does not launch a kernel or modify the
Q6 builder. The phase body keeps 64 independent accumulators live across the
copy region so register allocation and load-hoisting pressure are observable.
"""
from __future__ import annotations

import argparse, collections, contextlib, ctypes, ctypes.util, hashlib, json, pathlib, re, signal, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from tinygrad.runtime.support import compiler_cuda
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler


DEFAULT_OUT = ROOT / "docs/task_workflow/evidence/nv-direct-copy-2x2-microgate-20260901"
NVDISASM = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
CUOBJDUMP = pathlib.Path("/usr/local/cuda-13.2/bin/cuobjdump")
INSN_RE = re.compile(r"^\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)\b(.*)$", re.MULTILINE)
RESOURCE_RE = re.compile(r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)")
EXPECTED_MEMORY = {"LDG": 18, "STS": 18, "LDS": 1, "STG": 1, "BAR": 2}
ARMS = {
  "candidate_writable": {"qualified": False, "llama_frontend": False, "arch": "sm_120"},
  "candidate_const_restrict": {"qualified": True, "llama_frontend": False, "arch": "sm_120"},
  "llama_writable": {"qualified": False, "llama_frontend": True, "arch": "sm_120a"},
  "llama_const_restrict": {"qualified": True, "llama_frontend": True, "arch": "sm_120a"},
}


@contextlib.contextmanager
def hard_timeout(seconds: int):
  def expired(_signum, _frame): raise TimeoutError(f"microgate operation exceeded {seconds}s")
  old = signal.signal(signal.SIGALRM, expired)
  signal.setitimer(signal.ITIMER_REAL, seconds)
  try: yield
  finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, old)


def source_for(arm: dict) -> str:
  bounds = "__launch_bounds__(256, 1)" if arm["llama_frontend"] else "__launch_bounds__(256)"
  source_arg = "const unsigned int *__restrict__ src" if arm["qualified"] else "unsigned int *src"
  lines = [
    "#include <cuda_fp16.h>",
    f'extern "C" __global__ void {bounds} direct_copy_2x2(unsigned int *out, {source_arg}, unsigned int seed, unsigned int src_base) {{',
    "  extern __shared__ unsigned int scratch[];",
    "  const unsigned int tid = (unsigned int)threadIdx.x;",
    "  const unsigned int mix = seed ^ (tid * 0x9e3779b9u);",
  ]
  for i in range(64):
    salt = (0x45D9F3B * (i+1)) & 0xFFFFFFFF
    lines.append(f"  float acc{i} = __uint2float_rn((mix ^ 0x{salt:08x}u) & 0xffffu);")
  for rnd in range(8):
    for i in range(64):
      mul = 1.0001 + ((i*7+rnd*3) % 29) * 0.000013
      add = 0.125 + ((i*13+rnd*11) % 127) * 0.03125
      lines.append(f"  acc{i} = __fmaf_rn(acc{i}, {mul:.9f}f, {add:.9f}f);")
  lines.append("  __syncthreads();")
  for i in range(18):
    lines.append(f"  scratch[tid + {i*256}u] = src[src_base + tid + {i*256}u];")
  lines += ["  __syncthreads();", "  float sum = acc0;"]
  for i in range(1, 64): lines.append(f"  sum = __fadd_rn(sum, acc{i});")
  lines += [
    "  const unsigned int read_index = (((seed >> 8) % 18u) * 256u) + ((tid + (seed >> 16)) & 255u);",
    "  out[tid] = __float_as_uint(sum) ^ scratch[read_index];",
    "}",
  ]
  return "\n".join(lines) + "\n"


def source_contract(source: str, arm: dict) -> dict:
  direct = [x.strip() for x in source.splitlines() if x.strip().startswith("scratch[tid +") and " = src[" in x]
  forbidden = [x for x in ("asm", "volatile", "__noinline__", "membar", "atomic") if x in source.lower()]
  expected_arg = "const unsigned int *__restrict__ src" if arm["qualified"] else "unsigned int *src"
  expected_bounds = "__launch_bounds__(256, 1)" if arm["llama_frontend"] else "__launch_bounds__(256)"
  return {"direct_copies": len(direct), "barriers": source.count("__syncthreads();"), "forbidden_tokens": forbidden,
          "pointer_form_exact": expected_arg in source, "launch_bounds_exact": expected_bounds in source,
          "pass": len(direct) == 18 and source.count("src[") == 18 and source.count("__syncthreads();") == 2 and
                  not forbidden and expected_arg in source and expected_bounds in source}


def tool_version(path: pathlib.Path) -> str:
  cp = subprocess.run([str(path), "--version"], check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
  return cp.stdout.strip()


def nvrtc_version() -> str:
  major, minor = ctypes.c_int(), ctypes.c_int()
  compiler_cuda.nvrtc_check(compiler_cuda.nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
  return f"{major.value}.{minor.value}"


def run_tool(args: list[str]) -> str:
  return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30).stdout


def analyze_sass(sass: str, resources: str) -> dict:
  instructions = []
  for ordinal, match in enumerate(INSN_RE.finditer(sass)):
    opcode, operands = match.group(2), match.group(3).strip()
    instructions.append({"ordinal": ordinal, "pc": int(match.group(1), 16), "opcode": opcode,
                         "family": opcode.split(".", 1)[0], "operands": operands, "line": match.group(0).strip()})
  families = collections.Counter(x["family"] for x in instructions)
  opcodes = collections.Counter(x["opcode"] for x in instructions)
  loads, stores = [x for x in instructions if x["family"] == "LDG"], [x for x in instructions if x["family"] == "STS"]
  ldg_regs = []
  for ins in loads:
    match = re.match(r"(R\d+)(?:\.reuse)?\s*,", ins["operands"])
    ldg_regs.append(match.group(1) if match else None)
  sts_regs = []
  for ins in stores:
    regs = re.findall(r"\b(R\d+)(?:\.reuse)?\b", ins["operands"])
    sts_regs.append(regs[-1] if regs else None)
  resource_match = RESOURCE_RE.search(resources)
  resource = ({"registers": int(resource_match.group(1)), "stack_bytes": int(resource_match.group(2)),
               "shared_static_bytes": int(resource_match.group(3)), "local_static_bytes": int(resource_match.group(4))}
              if resource_match else None)
  first_span = stores[0]["ordinal"] - loads[0]["ordinal"] if loads and stores else None
  between = ([x for x in instructions if loads[0]["ordinal"] < x["ordinal"] < stores[0]["ordinal"]]
             if loads and stores else [])
  memory_exact = all(families.get(name, 0) == count for name, count in EXPECTED_MEMORY.items())
  no_forbidden = not any(x["family"].startswith(("MEMBAR", "ATOM", "RED", "LDL", "STL")) for x in instructions)
  spill_free = resource is not None and resource["stack_bytes"] == 0 and resource["local_static_bytes"] == 0 and \
               families.get("LDL", 0) == 0 and families.get("STL", 0) == 0
  return {
    "instruction_total": len(instructions), "families": dict(sorted(families.items())), "opcodes": dict(sorted(opcodes.items())),
    "resources": resource, "memory_exact": memory_exact, "no_forbidden_sync_or_spill_ops": no_forbidden,
    "spill_free": spill_free, "first_load_to_first_store_span_instructions": first_span,
    "first_ldg_pc": f"0x{loads[0]['pc']:x}" if loads else None, "first_sts_pc": f"0x{stores[0]['pc']:x}" if stores else None,
    "last_ldg_pc": f"0x{loads[-1]['pc']:x}" if loads else None, "last_sts_pc": f"0x{stores[-1]['pc']:x}" if stores else None,
    "ldg_opcodes": dict(sorted(collections.Counter(x["opcode"] for x in loads).items())),
    "ldg_registers": ldg_regs, "sts_registers": sts_regs,
    "same_position_register_reuse": sum(a is not None and a == b for a, b in zip(ldg_regs, sts_regs)),
    "register_overlap": sorted(set(x for x in ldg_regs if x) & set(x for x in sts_regs if x)),
    "phase_ops_between_first_ldg_and_first_sts": sum(x["family"] in {"FADD", "FFMA", "FMUL"} for x in between),
    "barrier_pcs": [f"0x{x['pc']:x}" for x in instructions if x["family"] == "BAR"],
    "hard_pass": memory_exact and no_forbidden and spill_free and first_span is not None and first_span <= 160,
  }


def compile_arm(name: str, arm: dict, out_dir: pathlib.Path, repeats: int, timeout: int) -> dict:
  arm_dir = out_dir / name
  arm_dir.mkdir(parents=True, exist_ok=True)
  source = source_for(arm)
  source_path = arm_dir / f"{name}.cu"
  source_path.write_text(source)
  binaries, hashes, elapsed, options = [], [], [], None
  for repeat in range(repeats):
    compiler = NVRTCCompiler(arm["arch"], ptx=False, cache_key=f"direct_copy_2x2_{name}_r{repeat}")
    if options is None: options = list(compiler.compile_options)
    elif options != list(compiler.compile_options): raise RuntimeError(f"NVRTC options changed across repeats for {name}")
    start = time.perf_counter()
    with hard_timeout(timeout): binary = compiler.compile(source)
    elapsed.append(time.perf_counter()-start)
    binaries.append(binary)
    hashes.append(hashlib.sha256(binary).hexdigest())
  cubin_path = arm_dir / f"{name}.cubin"
  cubin_path.write_bytes(binaries[0])
  sass = run_tool([str(NVDISASM), "-c", str(cubin_path)])
  resources = run_tool([str(CUOBJDUMP), "--dump-resource-usage", str(cubin_path)])
  sass_path, resources_path = arm_dir/f"{name}.nvdisasm", arm_dir/f"{name}.resources.txt"
  sass_path.write_text(sass)
  resources_path.write_text(resources)
  analysis = analyze_sass(sass, resources)
  record = {"factors": arm, "source": str(source_path), "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "cubin": str(cubin_path), "cubin_sha256_repeats": hashes, "stable_cubin": len(set(hashes)) == 1,
            "cubin_bytes": len(binaries[0]), "compile_seconds": elapsed, "nvrtc_options": options,
            "source_contract": source_contract(source, arm), "sass": str(sass_path), "resources": str(resources_path),
            "analysis": analysis}
  record["hard_pass"] = record["stable_cubin"] and record["source_contract"]["pass"] and analysis["hard_pass"]
  (arm_dir/f"{name}.json").write_text(json.dumps(record, indent=2)+"\n")
  return record


def contrast(arms: dict, left: str, right: str) -> dict:
  l, r = arms[left]["analysis"], arms[right]["analysis"]
  return {"left": left, "right": right,
          "span_improvement_left_minus_right": l["first_load_to_first_store_span_instructions"]-r["first_load_to_first_store_span_instructions"],
          "register_delta_left_minus_right": l["resources"]["registers"]-r["resources"]["registers"],
          "stack_delta_left_minus_right": l["resources"]["stack_bytes"]-r["resources"]["stack_bytes"]}


def markdown(result: dict) -> str:
  rows = []
  for name, arm in result["arms"].items():
    a, r = arm["analysis"], arm["analysis"]["resources"]
    rows.append(f"| `{name}` | `{arm['factors']['arch']}` | {arm['factors']['qualified']} | "
                f"{a['first_load_to_first_store_span_instructions']} | {r['registers']} | {r['stack_bytes']} | "
                f"`{a['ldg_opcodes']}` | {arm['stable_cubin']} | {arm['hard_pass']} |")
  return "\n".join([
    "# NV direct-copy 2x2 synthetic microgate (2026-09-01)", "", f"## Verdict: `{result['verdict']}`", "",
    result["release_reason"], "", "## Arms", "",
    "| Arm | Arch | const restrict | Span | REG | Stack | LDG opcode | Stable | Hard pass |",
    "|---|---|---:|---:|---:|---:|---|---:|---:|", *rows, "", "## Fixed contract", "",
    "- 64 independent FP32 accumulators remain live across an existing barrier, 18 paired global-u32-to-shared copies, and a publication barrier.",
    "- Exact memory census is 18 LDG, 18 STS, 1 LDS, 1 STG, and 2 BAR.",
    "- Inline PTX, volatile, noinline, function splitting, MEMBAR, and atomics are forbidden.",
    "- A hard arm requires span <=160, stable repeated cubin, and zero stack/LDL/STL/local spill traffic.", "",
    "## Contrasts", "", "```json", json.dumps(result["contrasts"], indent=2), "```", "",
    "## Toolchain", "", f"- NVRTC: `{result['tools']['nvrtc_version']}`",
    f"- nvdisasm: `{result['tools']['nvdisasm_version'].splitlines()[-2]}`",
    f"- cuobjdump: `{result['tools']['cuobjdump_version'].splitlines()[-2]}`", "",
    "No kernel was launched and no Q6 production or research builder was modified.", "",
  ])


def main() -> int:
  parser = argparse.ArgumentParser()
  parser.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT)
  parser.add_argument("--repeats", type=int, default=3)
  parser.add_argument("--compile-timeout", type=int, default=45)
  args = parser.parse_args()
  if args.repeats < 2: raise ValueError("repeats must be at least two")
  if not NVDISASM.is_file() or not CUOBJDUMP.is_file(): raise FileNotFoundError("required CUDA disassembly tools are unavailable")
  args.out_dir.mkdir(parents=True, exist_ok=True)
  result = {"schema": "tinygrad.nv_direct_copy_2x2_microgate.v1", "date": "2026-09-01",
            "contract": {"accumulators": 64, "fma_rounds": 8, "logical_copies": 18,
                         "expected_memory": EXPECTED_MEMORY, "span_limit": 160, "gpu_launched": False},
            "tools": {"python": sys.version, "nvrtc_version": nvrtc_version(), "nvrtc_library": ctypes.util.find_library("nvrtc"),
                      "nvdisasm_path": str(NVDISASM), "nvdisasm_version": tool_version(NVDISASM),
                      "cuobjdump_path": str(CUOBJDUMP), "cuobjdump_version": tool_version(CUOBJDUMP)}, "arms": {}}
  for name, arm in ARMS.items(): result["arms"][name] = compile_arm(name, arm, args.out_dir, args.repeats, args.compile_timeout)
  result["contrasts"] = {
    "Q_at_candidate": contrast(result["arms"], "candidate_writable", "candidate_const_restrict"),
    "Q_at_llama": contrast(result["arms"], "llama_writable", "llama_const_restrict"),
    "F_at_writable": contrast(result["arms"], "candidate_writable", "llama_writable"),
    "F_at_const_restrict": contrast(result["arms"], "candidate_const_restrict", "llama_const_restrict"),
  }
  baseline = result["arms"]["candidate_writable"]["analysis"]["first_load_to_first_store_span_instructions"]
  winners = [(name, arm) for name, arm in result["arms"].items() if arm["hard_pass"] and
             arm["analysis"]["first_load_to_first_store_span_instructions"] < baseline]
  winners.sort(key=lambda x: x[1]["analysis"]["first_load_to_first_store_span_instructions"])
  if winners:
    result["verdict"] = "SYNTHETIC_PASS_RELEASE_ONE_REAL_Q6_STATIC_RERUN"
    result["release_arm"] = winners[0][0]
    result["release_real_q6"] = True
    result["release_reason"] = (f"`{winners[0][0]}` passes the hard static gate and improves on candidate_writable span "
                                f"{baseline} -> {winners[0][1]['analysis']['first_load_to_first_store_span_instructions']}. "
                                "It releases one matching real-Q6 static compile only, not correctness or timing.")
  else:
    result["verdict"] = "SYNTHETIC_REJECT_NO_REAL_Q6"
    result["release_arm"] = None
    result["release_real_q6"] = False
    result["release_reason"] = "No arm both passed span/spill/census/stability gates and improved on candidate_writable; real Q6 remains blocked."
  result_path = args.out_dir / "result.json"
  result["commands"] = {"test": "python3 -m pytest -q test/unit/test_nv_direct_copy_2x2_microgate.py",
                        "microgate": f"python3 extra/llm_research/prefill/nv_direct_copy_2x2_microgate.py --out-dir {args.out_dir} --repeats {args.repeats}"}
  result_path.write_text(json.dumps(result, indent=2)+"\n")
  (args.out_dir/"result.md").write_text(markdown(result))
  print(json.dumps({"verdict": result["verdict"], "release_arm": result["release_arm"],
                    "arms": {name: {"span": arm["analysis"]["first_load_to_first_store_span_instructions"],
                                     "registers": arm["analysis"]["resources"]["registers"],
                                     "stack": arm["analysis"]["resources"]["stack_bytes"],
                                     "ldg_opcodes": arm["analysis"]["ldg_opcodes"], "hard_pass": arm["hard_pass"]}
                             for name, arm in result["arms"].items()}}, indent=2))
  return 0 if result["release_real_q6"] else 2


if __name__ == "__main__": raise SystemExit(main())
