#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, math, pathlib, re, shutil, subprocess
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/amd-broad-backend-roadmap"
from extra.qk_probe_harness import probe_io
read_json, write_json = probe_io(OUT)
CAPTURE = OUT / "bb5a8_authority_kernel_capture"




def sha256(data: bytes | str) -> str:
  return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def objdump_path() -> str | None:
  for path in ("/opt/rocm/llvm/bin/llvm-objdump", "/opt/rocm-7.2.4/llvm/bin/llvm-objdump", "llvm-objdump-21", "llvm-objdump-20", "llvm-objdump"):
    if shutil.which(path): return path
  return None


def disassemble(lib: bytes) -> tuple[str, str | None]:
  objdump = objdump_path()
  if objdump is None: return "", "llvm-objdump not found"
  try:
    proc = subprocess.run([objdump, "-d", "-"], input=lib, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.stdout.decode(errors="ignore"), None
  except Exception as exc:
    return "", repr(exc)


def count(pattern: str, text: str) -> int:
  return len(re.findall(pattern, text))


def role_row(rows: list[dict[str, Any]], role: str) -> dict[str, Any]:
  return next((row for row in rows if row.get("role") == role), {})


def main() -> int:
  from tinygrad import Tensor, dtypes, Device
  from tinygrad.codegen import to_program
  from tinygrad.llm.model import _prefill_v2_opts
  from test.backend.test_linearizer import helper_realized_ast
  from test.helpers import replace_opts
  from tinygrad.codegen.opt.search import _time_program
  from tinygrad.renderer.amd.elf import group_segment_fixed_size_from_elf, kernel_descriptor_from_elf

  shape = read_json("bench/qk-tensile-extraction/shape_matrix.json", {})
  codegen = read_json("bench/qk-tensile-extraction/codegen_oracle.json", {})
  mapping = read_json("bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json", {})
  ffn_gate_up = role_row(shape.get("rows", []), "ffn_gate_up")

  m = int(ffn_gate_up.get("m", 512))
  n = int(ffn_gate_up.get("n", 12288))
  k = int(ffn_gate_up.get("k", 4096))
  flops = 2 * m * n * k
  opts = tuple(_prefill_v2_opts(n, k))

  CAPTURE.mkdir(parents=True, exist_ok=True)
  a = Tensor.randn(m, k, dtype=dtypes.float16, device="AMD").realize()
  b = Tensor.randn(k, n, dtype=dtypes.float16, device="AMD").realize()
  ast, bufs = helper_realized_ast(a @ b)
  prg = to_program(replace_opts(ast, opts), Device["AMD"].renderer)

  source = prg.src[3].arg
  lib = prg.src[4].arg
  assert isinstance(source, str) and isinstance(lib, bytes)
  timings = [float(x) for x in _time_program(prg, {}, bufs, cnt=7)]
  finite_timings = [x for x in timings if math.isfinite(x) and x > 0]
  best_s = min(finite_timings) if finite_timings else None
  median_s = sorted(finite_timings)[len(finite_timings)//2] if finite_timings else None
  best_tflops = flops / best_s / 1e12 if best_s else None
  median_tflops = flops / median_s / 1e12 if median_s else None
  disasm, disasm_error = disassemble(lib)

  source_path = CAPTURE / "tinygrad_ffn_gate_up_authority.hip"
  elf_path = CAPTURE / "tinygrad_ffn_gate_up_authority.hsaco"
  disasm_path = CAPTURE / "tinygrad_ffn_gate_up_authority.disasm"
  source_path.write_text(source)
  elf_path.write_bytes(lib)
  disasm_path.write_text(disasm)

  kd = kernel_descriptor_from_elf(lib)
  lds_bytes = group_segment_fixed_size_from_elf(lib)
  launch_global, launch_local = prg.arg.launch_dims({})
  disasm_mix = {
    "v_wmma": count(r"\bv_wmma", disasm),
    "ds_load_b128": count(r"\bds_load_b128\b", disasm),
    "ds_load_b64": count(r"\bds_load_b64\b", disasm),
    "ds_load_b32": count(r"\bds_load_b32\b", disasm),
    "ds_store_b128": count(r"\bds_store_b128\b", disasm),
    "ds_store_b64": count(r"\bds_store_b64\b", disasm),
    "ds_store_b32": count(r"\bds_store_b32\b", disasm),
    "global_load_b128": count(r"\bglobal_load.*b128\b", disasm),
    "global_load_b64": count(r"\bglobal_load.*b64\b", disasm),
    "global_load_b32": count(r"\bglobal_load.*b32\b", disasm),
    "s_waitcnt": count(r"\bs_waitcnt\b", disasm),
    "s_barrier": count(r"\bs_barrier\b", disasm),
    "scratch": count(r"\bscratch_", disasm),
    "disasm_lines": len(disasm.splitlines()),
  }
  source_mix = {
    "wmma_builtin": count(r"__builtin_amdgcn_wmma", source),
    "wmma_invocations": count(r"__WMMA_", source),
    "addrspace3_or_local": count(r"addrspace\\(3\\)|__local", source),
    "barrier": count(r"barrier", source),
    "source_lines": len(source.splitlines()),
  }
  timing_join_pass = best_tflops is not None and 25.0 <= best_tflops <= 55.0
  source_isa_capture_pass = bool(source and lib and disasm and disasm_mix["v_wmma"] > 0)
  exact_authority_row = abs((best_tflops or 0.0) - float(ffn_gate_up.get("tinygrad_tflops", 42.0))) <= 8.0
  causal_inputs_ready = source_isa_capture_pass and timing_join_pass and exact_authority_row

  result = {
    "date": "2026-06-19",
    "phase": "BB-5a.8_authority_kernel_capture",
    "schema": "amd_bb5a8_authority_kernel_capture_result_v1",
    "verdict": "PASS_AUTHORITY_KERNEL_CAPTURE_CAUSAL_INPUTS_READY" if causal_inputs_ready else
               "BLOCKED_AUTHORITY_KERNEL_CAPTURE_NOT_TIMING_EQUIVALENT" if source_isa_capture_pass else
               "BLOCKED_AUTHORITY_KERNEL_CAPTURE_FAILED",
    "gate_pass": causal_inputs_ready,
    "default_behavior_changed": False,
    "performance_claim": False,
    "authority_role": "ffn_gate_up",
    "shape": {"m": m, "n": n, "k": k, "flops": flops},
    "opts": [str(x) for x in opts],
    "timing": {
      "samples_s": timings,
      "best_s": best_s,
      "median_s": median_s,
      "best_tflops": round(best_tflops, 3) if best_tflops else None,
      "median_tflops": round(median_tflops, 3) if median_tflops else None,
      "reference_tinygrad_tflops": ffn_gate_up.get("tinygrad_tflops"),
      "reference_tensile_tflops": ffn_gate_up.get("median_tflops"),
      "timing_join_pass": timing_join_pass,
      "exact_authority_row_tolerance_tflops": 8.0,
      "within_reference_tolerance": exact_authority_row,
    },
    "program": {
      "name": prg.arg.name,
      "function_name": prg.arg.function_name,
      "global_size": launch_global,
      "local_size": launch_local,
      "globals": prg.arg.globals,
      "outs": prg.arg.outs,
      "ins": prg.arg.ins,
      "source_sha256": sha256(source),
      "elf_sha256": sha256(lib),
      "source_path": str(source_path.relative_to(ROOT)),
      "elf_path": str(elf_path.relative_to(ROOT)),
      "disasm_path": str(disasm_path.relative_to(ROOT)),
      "disasm_error": disasm_error,
    },
    "resource": {
      "lds_bytes": lds_bytes,
      "kernel_descriptor": {
        "group_segment_fixed_size": getattr(kd, "group_segment_fixed_size", None) if kd is not None else None,
        "private_segment_fixed_size": getattr(kd, "private_segment_fixed_size", None) if kd is not None else None,
        "kernarg_segment_size": getattr(kd, "kernarg_segment_size", None) if kd is not None else None,
        "compute_pgm_rsrc1": getattr(kd, "compute_pgm_rsrc1", None) if kd is not None else None,
        "compute_pgm_rsrc2": getattr(kd, "compute_pgm_rsrc2", None) if kd is not None else None,
      },
    },
    "mix": {"source": source_mix, "disasm": disasm_mix},
    "comparison_to_tensile": {
      "tensile_schedule": codegen.get("tensile_schedule"),
      "tensile_instruction_mix": codegen.get("tensile_instruction_mix"),
      "captured_has_wmma": disasm_mix["v_wmma"] > 0,
      "captured_has_lds": any(disasm_mix[k] > 0 for k in ("ds_load_b128", "ds_load_b64", "ds_load_b32", "ds_store_b128", "ds_store_b64", "ds_store_b32")),
      "captured_has_wide_lds_load_b128": disasm_mix["ds_load_b128"] > 0,
      "captured_has_scratch": disasm_mix["scratch"] > 0,
    },
    "scope": {
      "minimum_pass": [
        "Compile the ffn_gate_up authority shape under current pure-tinygrad prefill warmstart opts.",
        "Persist source, ELF, and objdump disassembly for that exact compiled program.",
        "Time the same compiled program and compare it with the 42.0 TFLOPS authority row.",
        "Extract LDS/resource and instruction mix rows for the same program.",
      ],
      "non_goals": [
        "Do not change default behavior.",
        "Do not claim a new performance result.",
        "Do not claim causal closure unless timing and same-kernel source/ISA/resource are joined.",
      ],
    },
    "input_artifacts": [
      "bench/qk-tensile-extraction/shape_matrix.json",
      "bench/qk-tensile-extraction/codegen_oracle.json",
      "bench/amd-broad-backend-roadmap/bb5a8_tensile_mapping_result.json",
    ],
    "mapping_input_verdict": mapping.get("verdict"),
    "decision": (
      "Authority kernel capture has the same compiled program joined to timing, source, ELF, disassembly, and resources. "
      "Use the captured mix against Tensile for the next causal-delta probe."
      if causal_inputs_ready else
      "Capture produced source/ELF/disassembly but did not satisfy timing-equivalence to the 42.0 TFLOPS authority row; "
      "do not use it for causal proof without resolving the timing mismatch."
      if source_isa_capture_pass else
      "Authority kernel capture failed before source/ISA evidence was complete."
    ),
    "next_action": (
      "Run a causal-delta probe over the captured tinygrad instruction mix versus Tensile: wide LDS reads, software-pipeline evidence, scratch/spill, and wait density."
      if causal_inputs_ready else
      "Repair authority capture until the captured program is timing-equivalent to the 42.0 TFLOPS row."
    ),
  }
  write_json("bb5a8_authority_kernel_capture_result.json", result)
  print(json.dumps({
    "out": "bench/amd-broad-backend-roadmap/bb5a8_authority_kernel_capture_result.json",
    "verdict": result["verdict"],
    "gate_pass": result["gate_pass"],
    "best_tflops": result["timing"]["best_tflops"],
    "v_wmma": disasm_mix["v_wmma"],
    "ds_load_b128": disasm_mix["ds_load_b128"],
    "lds_bytes": lds_bytes,
  }, indent=2))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
