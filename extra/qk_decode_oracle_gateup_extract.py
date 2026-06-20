#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
from typing import Any

from extra.q8_ffn_fast_artifact_probe import HIP_MMVQ_GATEUP_SOURCE, compile_hipcc_linked


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/qk-decode-primitive-transfer/decode_oracle_gateup_extract_result.json"
ARTDIR = ROOT / "bench/qk-decode-primitive-transfer/oracle"
LLVM = pathlib.Path("/opt/rocm/llvm/bin")


def read_json(rel: str) -> dict[str, Any]:
  with (ROOT / rel).open() as f:
    return json.load(f)


def run(cmd: list[str], timeout: int = 60) -> dict[str, Any]:
  p = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, check=False)
  return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout}


def sha256_bytes(x: bytes) -> str:
  return hashlib.sha256(x).hexdigest()


def parse_metadata(text: str) -> dict[str, Any]:
  keys = {
    "name": r"\.name:\s*([^\n]+)",
    "symbol": r"\.symbol:\s*([^\n]+)",
    "group_segment_fixed_size": r"\.group_segment_fixed_size:\s*(\d+)",
    "kernarg_segment_size": r"\.kernarg_segment_size:\s*(\d+)",
    "private_segment_fixed_size": r"\.private_segment_fixed_size:\s*(\d+)",
    "sgpr_count": r"\.sgpr_count:\s*(\d+)",
    "sgpr_spill_count": r"\.sgpr_spill_count:\s*(\d+)",
    "vgpr_count": r"\.vgpr_count:\s*(\d+)",
    "vgpr_spill_count": r"\.vgpr_spill_count:\s*(\d+)",
    "max_flat_workgroup_size": r"\.max_flat_workgroup_size:\s*(\d+)",
    "wavefront_size": r"\.wavefront_size:\s*(\d+)",
  }
  out: dict[str, Any] = {}
  for k, pat in keys.items():
    m = re.search(pat, text)
    if m is None: continue
    val = m.group(1).strip()
    out[k] = int(val) if val.isdigit() else val
  return out


def parse_symbols(text: str) -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for line in text.splitlines():
    if "q8_mmvq_gateup" not in line: continue
    # llvm-objdump --syms format: addr flags section size visibility name
    parts = line.split()
    rows.append({"raw": line.strip(), "parts": parts})
  return rows


def parse_disasm(text: str, symbol: str) -> dict[str, Any]:
  in_symbol = False
  insts: list[dict[str, Any]] = []
  counts: dict[str, int] = {}
  for line in text.splitlines():
    if re.search(rf"<{re.escape(symbol)}>:", line):
      in_symbol = True
      continue
    if in_symbol and re.search(r"^0*[0-9a-fA-F]+ <", line):
      break
    if not in_symbol: continue
    m = re.match(r"\s*([0-9a-fA-F]+):\s+([A-Za-z0-9_.$]+)", line)
    if m:
      pc, mnemonic = int(m.group(1), 16), m.group(2)
    else:
      m2 = re.match(r"\s*([A-Za-z_][A-Za-z0-9_.$]*)\b.*//\s*([0-9a-fA-F]+):", line)
      if not m2: continue
      mnemonic, pc = m2.group(1), int(m2.group(2), 16)
    insts.append({"pc": pc, "mnemonic": mnemonic, "text": line.strip()})
    counts[mnemonic] = counts.get(mnemonic, 0) + 1

  def count_prefix(prefix: str) -> int:
    return sum(v for k, v in counts.items() if k.startswith(prefix))

  grouped = {
    "dot4": sum(v for k, v in counts.items() if "dot4" in k),
    "fma": sum(v for k, v in counts.items() if k.startswith("v_fma") or k.startswith("v_mad") or "mad_mix" in k),
    "convert": count_prefix("v_cvt"),
    "valu": sum(v for k, v in counts.items() if k.startswith("v_")),
    "salu": sum(v for k, v in counts.items() if k.startswith("s_")),
    "ds": count_prefix("ds_"),
    "barrier": counts.get("s_barrier", 0),
    "global_load": sum(v for k, v in counts.items() if k.startswith("global_load")),
    "global_store": sum(v for k, v in counts.items() if k.startswith("global_store")),
    "shuffle": counts.get("ds_bpermute_b32", 0),
    "branch": sum(v for k, v in counts.items() if "branch" in k),
    "waitcnt": counts.get("s_waitcnt", 0),
    "s_clause": counts.get("s_clause", 0),
    "s_delay_alu": counts.get("s_delay_alu", 0),
  }
  return {
    "instruction_count": len(insts),
    "grouped": grouped,
    "top_mnemonics": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:40],
    "first_80": insts[:80],
  }


def main() -> int:
  ap = argparse.ArgumentParser(description="Extract q8_mmvq_gateup oracle HSACO, metadata, and ISA map seed")
  ap.add_argument("--arch", default="gfx1100")
  ap.add_argument("--out", type=pathlib.Path, default=OUT)
  ap.add_argument("--artdir", type=pathlib.Path, default=ARTDIR)
  args = ap.parse_args()

  manifest = read_json("bench/q8-ffn-amd-scheduler-project/artifact_build_manifest.json")
  loader = read_json("bench/q8-ffn-amd-scheduler-project/artifact_loader.json")
  oracle = read_json("bench/q8-ffn-amd-scheduler-project/oracle_contract.json")
  expected = manifest["artifacts"]["gateup"]

  blob = compile_hipcc_linked(HIP_MMVQ_GATEUP_SOURCE, args.arch)
  digest = sha256_bytes(blob)
  args.artdir.mkdir(parents=True, exist_ok=True)
  hsaco = args.artdir / "q8_mmvq_gateup.hsaco"
  hsaco.write_bytes(blob)

  readobj = run([str(LLVM / "llvm-readobj"), "--notes", "--symbols", "--sections", str(hsaco)])
  syms = run([str(LLVM / "llvm-objdump"), "--syms", str(hsaco)])
  disasm = run([str(LLVM / "llvm-objdump"), "--disassemble-all", "--no-show-raw-insn", str(hsaco)])
  metadata = parse_metadata(readobj["stdout"])
  symbols = parse_symbols(syms["stdout"])
  isa = parse_disasm(disasm["stdout"], "q8_mmvq_gateup")

  expected_grouped = oracle["instruction_contract"]["oracle_grouped"]
  grouped_matches = {k: isa["grouped"].get(k) == expected_grouped.get(k) for k in expected_grouped}
  result = {
    "date": "2026-06-20",
    "phase": "DECODE_ORACLE_GATEUP_EXTRACTION",
    "schema": "decode_oracle_gateup_extract_v1",
    "verdict": "PASS_DECODE_ORACLE_GATEUP_HSACO_METADATA_DISASM_EXTRACTED",
    "gate_pass": True,
    "default_behavior_changed": False,
    "performance_claim": False,
    "artifact": {
      "path": str(hsaco.relative_to(ROOT)),
      "sha256": digest,
      "bytes": len(blob),
      "expected_sha256": expected["hsaco_sha256"],
      "expected_bytes": expected["hsaco_bytes"],
      "source_sha256": hashlib.sha256(HIP_MMVQ_GATEUP_SOURCE.encode()).hexdigest(),
      "expected_source_sha256": expected["source_sha256"],
    },
    "metadata": metadata,
    "symbols": symbols,
    "isa": isa,
    "oracle_contract": {
      "launch": oracle["launch_contract"],
      "loader_gateup": loader["loader"]["gateup"],
      "expected_grouped": expected_grouped,
    },
    "tool_outputs": {
      "llvm_readobj_returncode": readobj["returncode"],
      "llvm_objdump_syms_returncode": syms["returncode"],
      "llvm_objdump_disasm_returncode": disasm["returncode"],
      "llvm_readobj_stdout_path": str((args.artdir / "q8_mmvq_gateup.readobj.txt").relative_to(ROOT)),
      "llvm_objdump_syms_path": str((args.artdir / "q8_mmvq_gateup.syms.txt").relative_to(ROOT)),
      "llvm_objdump_disasm_path": str((args.artdir / "q8_mmvq_gateup.disasm.txt").relative_to(ROOT)),
    },
    "gates": {
      "hash_matches_manifest": digest == expected["hsaco_sha256"],
      "bytes_match_manifest": len(blob) == expected["hsaco_bytes"],
      "metadata_symbol_matches": metadata.get("symbol") == "q8_mmvq_gateup.kd",
      "metadata_name_matches": metadata.get("name") == "q8_mmvq_gateup",
      "metadata_kernarg_matches_loader": metadata.get("kernarg_segment_size") == loader["loader"]["gateup"]["kernarg_size"],
      "metadata_lds_matches_loader": metadata.get("group_segment_fixed_size") == loader["loader"]["gateup"]["group_segment_size"],
      "metadata_private_matches_loader": metadata.get("private_segment_fixed_size") == loader["loader"]["gateup"]["private_segment_size"],
      "symbol_present": any("q8_mmvq_gateup" in row["raw"] for row in symbols),
      "disasm_has_16_dot4": isa["grouped"]["dot4"] == 16,
      "grouped_counts_match_contract": all(grouped_matches.values()),
    },
    "grouped_count_matches": grouped_matches,
    "next": {
      "phase": "OES-4 semantic ISA map",
      "work": "stage-label q8_mmvq_gateup.disasm.txt into load/unpack/dot4/scale/reduction/wait/store and compare to native/C7C",
    },
  }
  if not all(result["gates"].values()):
    result["verdict"] = "BLOCKED_DECODE_ORACLE_GATEUP_EXTRACTION_MISMATCH"
    result["gate_pass"] = False

  (args.artdir / "q8_mmvq_gateup.readobj.txt").write_text(readobj["stdout"])
  (args.artdir / "q8_mmvq_gateup.syms.txt").write_text(syms["stdout"])
  (args.artdir / "q8_mmvq_gateup.disasm.txt").write_text(disasm["stdout"])
  args.out.parent.mkdir(parents=True, exist_ok=True)
  args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps({
    "verdict": result["verdict"],
    "gates": result["gates"],
    "metadata": metadata,
    "artifact": result["artifact"],
    "out": str(args.out.relative_to(ROOT) if args.out.is_absolute() and args.out.is_relative_to(ROOT) else args.out),
  }, indent=2))
  return 0 if result["gate_pass"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
