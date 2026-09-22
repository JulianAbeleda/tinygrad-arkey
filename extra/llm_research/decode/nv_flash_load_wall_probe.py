#!/usr/bin/env python3
"""SASS-gated K/V load-wall probe for the matched S6 wide Flash score kernel.

This is deliberately source-level research scaffolding.  It keeps the current
wide-Q S6 kernel's arithmetic, grid, traffic, and partial ABI fixed while
forcing all sixteen 128-bit loads for K, V, or both phases to have distinct
live destinations before the first consumer.  A timing result is admissible
only when the emitted SASS actually retains that load wall without spills.
"""
from __future__ import annotations

import argparse, contextlib, json, pathlib, re, statistics, subprocess, sys, tempfile

ROOT_PATH = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PATH))

from extra.llm_research.decode.nv_flash_bounded_counter_probe import (
  HARNESS, METRICS, NCU, NVCC, ROOT, _ncu, _render, _sass_load_grammar, _wide_q_f32_source)


def _phase_wall(source:str, ridx:str, end_marker:str, prefix:str,
                loads:tuple[tuple[str, int], ...], phase_base_u32:int, *, split_asm:bool=False,
                movable_asm:bool=False) -> tuple[str, int]:
  marker = f"  for (int {ridx} = 0; {ridx} < 8; {ridx}++) {{\n"
  start = source.index(marker)
  end = source.index(end_marker, start)
  original = source[start:end]
  if not original.endswith("  }\n"): raise RuntimeError(f"could not isolate {ridx} loop")
  body = original[len(marker):-4]
  alu = "alu12" if ridx == "Ridx5" else "alu40"
  data = re.search(r"unsigned int\* (data2_\d+)", source)
  if data is None: raise RuntimeError("could not identify K/V cache argument")

  declarations:list[str] = []
  asm_lines:list[str] = []
  outputs:list[str] = []
  entries:list[tuple[str, int]] = []
  operand = 0
  for col in range(8):
    for pos, (_, offset) in enumerate(loads):
      name = f"{prefix}_{col}_{pos}"
      entries.append((name, col * 64 + offset))
      declarations.append(f"  uint4 {name};")
      asm_lines.append("ld.global.v4.u32 {" + ", ".join(f"%{operand+i}" for i in range(4)) +
                       f"}}, [%64+{(col * 64 + offset) * 4}];\\n\\t")
      outputs.extend(f'"=r"({name}.{component})' for component in "xyzw")
      operand += 4
  wall_base = f"{data.group(1)}+(alu9+alu10+alu11+{phase_base_u32})"
  if split_asm:
    for name, offset in entries:
      qualifier = "asm" if movable_asm else "asm volatile"
      declarations += [f'  {qualifier}("ld.global.v4.u32 {{%0, %1, %2, %3}}, [%4+{offset * 4}];"',
        f'    : "=r"({name}.x), "=r"({name}.y), "=r"({name}.z), "=r"({name}.w) : "l"({wall_base}));']
  else:
    declarations += ["  asm volatile(", *[f'    "{line}"' for line in asm_lines],
                     "    : " + ", ".join(outputs), f'    : "l"({wall_base}) : "memory");']

  consumers:list[str] = []
  for col in range(8):
    expanded = body.replace(f"int {alu} = (alu9+({ridx}<<6)+alu10+alu11);", f"int {alu} = (alu9+{col * 64}+alu10+alu11);")
    for pos, (name, _) in enumerate(loads):
      expanded, count = re.subn(rf"    uint4 {name} = .*?;\n", f"    uint4 {name} = {prefix}_{col}_{pos};\n", expanded, count=1)
      if count != 1: raise RuntimeError(f"could not replace {ridx}/{name} load")
    expanded = expanded.replace(ridx, str(col))
    consumers.append("  {\n" + expanded + "  }\n")
  return source[:start] + "\n".join(declarations) + "\n" + "".join(consumers) + source[end:], 16


def _load_wall_source(source:str, symbol:str, phases:str) -> tuple[str, str, dict[str, int]]:
  if phases not in ("lb", "k", "v", "vearly", "vsplit", "vmovable", "kv"): raise ValueError(phases)
  wall_symbol = symbol + f"_{phases}wall16"
  source = source.replace(symbol + "(", wall_symbol + "(", 1)
  rewrites = {"k": 0, "v": 0}
  if "k" in phases:
    source, rewrites["k"] = _phase_wall(source, "Ridx5", "  float buf5[1];", "kwall", (
      ("val1", 32), ("val2", 0)), 0)
  if "v" in phases:
    source, rewrites["v"] = _phase_wall(source, "Ridx9", "  float buf8[1];", "vwall", (
      ("val3", 0), ("val4", 32)), 524288, split_asm=phases in ("vsplit", "vmovable"), movable_asm=phases == "vmovable")
    if phases == "vearly":
      # The ordinary forced wall lives at the original V loop and therefore
      # discards the compiler's progressive prefetch runway.  Move the exact
      # same declarations/asm before K score work while leaving all consumers
      # in place.  This isolates lead time from load count and consumer math.
      wall_start = source.index("  uint4 vwall_0_0;")
      consumer_start = source.index("  {\n", wall_start)
      wall = source[wall_start:consumer_start]
      source = source[:wall_start] + source[consumer_start:]
      k_start = source.index("  for (int Ridx5 = 0; Ridx5 < 8; Ridx5++) {")
      source = source[:k_start] + wall + source[k_start:]
  return wall_symbol, source, rewrites


def _sass_text(binary:pathlib.Path, symbol:str) -> str:
  nvdisasm = ROOT / ".venv/lib/python3.12/site-packages/triton/backends/nvidia/bin/nvdisasm"
  env = dict(__import__("os").environ)
  env.update(NVDISASM_PATH=str(nvdisasm), PATH=f"{nvdisasm.parent}:{env.get('PATH', '')}")
  cp = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "--dump-sass", str(binary)],
                      capture_output=True, text=True, check=True, env=env)
  marker = f"Function : {symbol}\n"
  start = cp.stdout.index(marker) + len(marker)
  tail = cp.stdout[start:]
  return tail[:tail.index("Function :")] if "Function :" in tail else tail


def _ptxas_resources(stderr:str, symbol:str) -> dict:
  # nvcc reports one resource stanza per function. Preserve the raw stanza as
  # the authority because its order is compiler-controlled.
  blocks = re.split(r"ptxas info    : Compiling entry function '", stderr)
  block = next((x for x in blocks if x.startswith(symbol + "'")), "")
  regs = re.search(r"Used (\d+) registers", block)
  spill_store = re.search(r"(\d+) bytes spill stores", block)
  spill_load = re.search(r"(\d+) bytes spill loads", block)
  return {"registers": int(regs.group(1)) if regs else None,
          "spill_store_bytes": int(spill_store.group(1)) if spill_store else 0,
          "spill_load_bytes": int(spill_load.group(1)) if spill_load else 0,
          "raw": block[:1200]}


def main() -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--phase", choices=("lb", "k", "v", "vearly", "vsplit", "vmovable", "kv"), required=True)
  ap.add_argument("--passes", type=int, default=500)
  ap.add_argument("--reps", type=int, default=11)
  ap.add_argument("--splits", type=int, choices=(6, 8), default=6)
  ap.add_argument("--scalar-q", action="store_true", help="retain the installed scalar-f32 Q load grammar")
  ap.add_argument("--ncu", action="store_true")
  ap.add_argument("--artifacts-dir", type=pathlib.Path)
  ap.add_argument("--out", type=pathlib.Path, required=True)
  args = ap.parse_args()

  # Both arms use the already-proven bit-exact float4 Q spelling so Q request
  # grammar cannot contaminate the K/V cadence comparison.
  token_bound = args.splits * 128
  control, control_src = _render(args.splits, token_bound)
  candidate, candidate_src = _render(args.splits, token_bound)
  if not args.scalar_q:
    control, control_src, _ = _wide_q_f32_source(control_src, control)
    candidate, candidate_src, _ = _wide_q_f32_source(candidate_src, candidate)
  candidate, candidate_src, rewrites = _load_wall_source(candidate_src, candidate, args.phase)
  candidate_src, launch_bound_rewrites = candidate_src.replace(
    "__launch_bounds__(128)", "__launch_bounds__(128, 1)", 1), candidate_src.count("__launch_bounds__(128)")
  if launch_bound_rewrites != 1: raise RuntimeError(f"expected one candidate launch bound, found {launch_bound_rewrites}")
  candidate_start = candidate_src.index('extern "C"')
  candidate_src = candidate_src[candidate_start:]
  source = HARNESS.replace("__CONTROL_SOURCE__", control_src).replace("__CANDIDATE_SOURCE__", candidate_src)
  source = source.replace("__CONTROL__", control).replace("__CANDIDATE__", candidate)
  source = source.replace("__CONTROL_SPLITS__", str(args.splits)).replace("__CAND_SPLITS__", str(args.splits))
  source = source.replace("__CAND_OUT__", str(32 * args.splits * 130)).replace("__COMPARE__", "1")

  if args.artifacts_dir is not None: args.artifacts_dir.mkdir(parents=True, exist_ok=True)
  workdir = contextlib.nullcontext(str(args.artifacts_dir)) if args.artifacts_dir is not None else \
    tempfile.TemporaryDirectory(prefix="nv_flash_load_wall_")
  with workdir as td:
    cu, binary = pathlib.Path(td) / "probe.cu", pathlib.Path(td) / "probe"
    cu.write_text(source)
    build = subprocess.run([NVCC, "-arch=sm_120a", "-O3", "-lineinfo", "-std=c++17", "--ptxas-options=-v",
                            str(cu), "-o", str(binary)], capture_output=True, text=True)
    if build.returncode: raise RuntimeError(build.stderr[-12000:])
    run = subprocess.run([str(binary), str(args.passes), str(args.reps)], capture_output=True, text=True, check=True)
    sass = {"control": _sass_text(binary, control), "candidate": _sass_text(binary, candidate)}
    if args.artifacts_dir is not None:
      (args.artifacts_dir / "control.sass").write_text(sass["control"])
      (args.artifacts_dir / "candidate.sass").write_text(sass["candidate"])
    counters = ({arm: {state: _ncu(binary, symbol, "none" if state == "hot" else "all")
                       for state in ("hot", "cold")}
                 for arm, symbol in (("control", control), ("candidate", candidate))} if args.ncu else None)

  controls, candidates, exactness = [], [], None
  for line in run.stdout.splitlines():
    if m := re.match(r"exact_mismatches=(\d+) max_abs=([0-9.eE+-]+)", line):
      exactness = {"bit_mismatches": int(m.group(1)), "max_abs": float(m.group(2))}
    if m := re.match(r"rep=\d+ control=([0-9.]+) candidate=([0-9.]+)", line):
      controls.append(float(m.group(1))); candidates.append(float(m.group(2)))
  payload = {"schema": "tinygrad.nv_flash_load_wall_probe.v1", "phase": args.phase,
    "shape": {"Hq": 32, "Hkv": 8, "Hd": 128, "MAXC": 1024, "Tc": 512,
              "splits": args.splits, "token_bound": token_bound},
    "control": {"symbol": control, "samples_us": controls, "median_us": statistics.median(controls),
                "resources": _ptxas_resources(build.stderr, control), "sass": _sass_load_grammar(binary, control)},
    "candidate": {"symbol": candidate, "samples_us": candidates, "median_us": statistics.median(candidates),
                  "resources": _ptxas_resources(build.stderr, candidate), "sass": _sass_load_grammar(binary, candidate)},
    "ratio": statistics.median(candidates) / statistics.median(controls), "exactness": exactness,
    "source_load_rewrites": rewrites, "candidate_launch_bounds": [128, 1], "scalar_q": args.scalar_q,
    "ptxas": build.stderr.splitlines()}
  if counters is not None: payload["ncu"] = {"metrics": METRICS, "arms": counters}
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__": raise SystemExit(main())
