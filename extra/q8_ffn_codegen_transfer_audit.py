#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, pathlib, re, subprocess, tempfile
from collections import Counter

from tinygrad.device import Device
from tinygrad.runtime.support.elf import elf_loader
from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked, hip_norm_source
from extra.q8_ffn_hcq_artifact import MMVQ_SOURCE, NORM_SOURCE

ROCM_LLVM = pathlib.Path("/opt/rocm/llvm/bin")

GROUPS = {
  "dot4": ("v_dot4",),
  "fma": ("v_fma", "v_mac", "v_mad"),
  "convert": ("v_cvt",),
  "valu": ("v_",),
  "salu": ("s_",),
  "ds": ("ds_",),
  "barrier": ("s_barrier",),
  "global_load": ("global_load", "flat_load"),
  "global_store": ("global_store", "flat_store"),
  "shuffle": ("v_permlane", "ds_bpermute", "ds_permute"),
  "branch": ("s_branch", "s_cbranch"),
  "waitcnt": ("s_waitcnt",),
}

INSN_RE = re.compile(r"^\s*(?:[0-9a-f]+:\s+(?:[0-9a-f]{2}\s+)*)?([a-zA-Z_][a-zA-Z0-9_\.]*)\b")

def run(cmd:list[str]) -> str:
  return subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout

def maybe_run(cmd:list[str]) -> tuple[bool, str]:
  try: return True, run(cmd)
  except Exception as e: return False, str(e)

def disassemble(blob:bytes, name:str) -> tuple[str, dict]:
  with tempfile.NamedTemporaryFile(suffix=f"_{name}.hsaco") as f:
    f.write(blob)
    f.flush()
    ok, text = maybe_run([str(ROCM_LLVM/"llvm-objdump"), "-d", "--no-show-raw-insn", f.name])
  if not ok: return text, {"error": text}
  mnems: Counter[str] = Counter()
  for line in text.splitlines():
    if line.lstrip().startswith(("/", "<")) or not line.strip(): continue
    if (m := INSN_RE.match(line)) is not None and not m.group(1).startswith("Disassembly"): mnems[m.group(1)] += 1
  grouped = {k: sum(v for m, v in mnems.items() if any(m.startswith(p) for p in prefs)) for k, prefs in GROUPS.items()}
  return text, {
    "instruction_count": sum(mnems.values()),
    "unique_mnemonics": len(mnems),
    "grouped_counts": grouped,
    "top_mnemonics": mnems.most_common(40),
  }

def readelf(blob:bytes, name:str) -> dict:
  with tempfile.NamedTemporaryFile(suffix=f"_{name}.hsaco") as f:
    f.write(blob)
    f.flush()
    sec_ok, sections = maybe_run([str(ROCM_LLVM/"llvm-readelf"), "-S", f.name])
    sym_ok, symbols = maybe_run([str(ROCM_LLVM/"llvm-readelf"), "-s", f.name])
    rel_ok, relocs = maybe_run([str(ROCM_LLVM/"llvm-readelf"), "-r", f.name])
  return {
    "sections_ok": sec_ok,
    "symbols_ok": sym_ok,
    "relocs_ok": rel_ok,
    "section_names": re.findall(r"\]\s+([.\w$]+)\s+", sections) if sec_ok else [],
    "kernel_symbols": sorted(set(re.findall(r"\s(\w*q8\w*)$", symbols, flags=re.MULTILINE))) if sym_ok else [],
    "readelf_relocations": [ln.strip() for ln in relocs.splitlines() if "R_AMDGPU_" in ln] if rel_ok else [],
  }

def elf_summary(blob:bytes) -> dict:
  image, sections, relocs = elf_loader(blob)
  return {
    "bytes": len(blob),
    "image_bytes": int(image.nbytes),
    "sections": [{"name": s.name, "size": int(s.header.sh_size), "addr": int(s.header.sh_addr),
                  "type": int(s.header.sh_type), "flags": int(s.header.sh_flags)} for s in sections],
    "tinygrad_relocations": [{"apply_image_offset": int(a), "rel_sym_offset": int(b), "type": int(t), "addend": int(add)}
                             for a, b, t, add in relocs],
  }

def inspect_runtime(blob:bytes, name:str) -> dict:
  dev = Device["AMD"]
  try:
    prg = dev.runtime(name, blob)
    return {
      "loads_in_amdprogram": True,
      "kernarg_size": int(getattr(prg, "kernargs_segment_size", -1)),
      "group_segment_size": int(getattr(prg, "group_segment_size", -1)),
      "private_segment_size": int(getattr(prg, "private_segment_size", -1)),
    }
  except Exception as e:
    return {"loads_in_amdprogram": False, "error": str(e)}

def inspect_blob(label:str, blob:bytes, runtime_name:str) -> dict:
  _, disasm = disassemble(blob, label)
  return {
    "label": label,
    "elf": elf_summary(blob),
    "readelf": readelf(blob, label),
    "disasm": disasm,
    "runtime": inspect_runtime(blob, runtime_name),
  }

def main() -> None:
  ap = argparse.ArgumentParser(description="B0/B1 audit for q8 FFN codegen/ASM transfer")
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("bench/q8-ffn-codegen-transfer/audit.json"))
  args = ap.parse_args()

  dev = Device["AMD"]
  blobs = {
    "fast_producer_hipcc_lld": compile_hipcc_linked(hip_norm_source(1024), args.arch),
    "fast_gateup_hipcc_lld": compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch),
    "comgr_producer_raw_c": dev.compiler.compile(NORM_SOURCE),
    "comgr_mmvq_raw_c": dev.compiler.compile(MMVQ_SOURCE),
  }
  results = {k: inspect_blob(k, v, f"q8_codegen_transfer_{k}") for k, v in blobs.items()}

  summary = {
    "fast_gateup_dot4": results["fast_gateup_hipcc_lld"]["disasm"].get("grouped_counts", {}).get("dot4"),
    "comgr_mmvq_dot4": results["comgr_mmvq_raw_c"]["disasm"].get("grouped_counts", {}).get("dot4"),
    "fast_gateup_loads": results["fast_gateup_hipcc_lld"]["runtime"]["loads_in_amdprogram"],
    "fast_producer_loads": results["fast_producer_hipcc_lld"]["runtime"]["loads_in_amdprogram"],
    "comgr_producer_loads": results["comgr_producer_raw_c"]["runtime"]["loads_in_amdprogram"],
    "comgr_mmvq_loads": results["comgr_mmvq_raw_c"]["runtime"]["loads_in_amdprogram"],
  }

  out = {
    "date": "2026-06-19",
    "phase": "B0_B1_codegen_transfer_audit",
    "arch": args.arch,
    "scope": "no execution; compile/disassemble/load-contract audit for transferring the passing q8 FFN lifecycle into tinygrad-owned codegen/ASM",
    "oracle": {
      "route": "Q8_FFN_HANDWRITTEN=1 hipcc/LLD artifact route",
      "a4_min_wd_speedup": 1.051,
      "a4_dnll": 0.0028866150416475556,
      "isolated_lifecycle_gate_us": 129.2,
    },
    "summary": summary,
    "objects": results,
    "verdict": "AUDIT_ONLY",
    "next": "Use this contract to attempt a tinygrad-owned q8 consumer first; keep the fast hipcc/LLD artifacts as oracle and regression target.",
  }
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(out, indent=2) + "\n")
  print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))

if __name__ == "__main__":
  main()
