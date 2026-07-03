#!/usr/bin/env python3
"""AMDGCN ISA primitive audit tool (docs/amd-gpu-holistic-primitive-model-20260623.md, Phase 8).
Formalizes the manual flow: code object (.co/.hsaco/.elf) -> clang-offload-bundler unbundle (if a HIP bundle) ->
llvm-objdump -d + llvm-readelf --notes -> per-kernel resources (VGPR/SGPR/LDS/scratch/kernarg) + ISA primitive flags
(has_v_dot2 / has_lds / has_cross_lane / has_vector_global_load / has_spill) + instruction counts.

  usage: PYTHONPATH=. .venv/bin/python extra/qk/amdgpu_isa_primitive_audit.py <co_or_elf> [<co_or_elf> ...]
         (no args: auto-discovers cached owned-route code objects /tmp/b4_tile_*.co + /tmp/b4_comb_*.co)
"""
import json, glob, pathlib, re, shutil, subprocess, sys
from extra.qk.isa_helpers import CROSS_LANE_RE

ROCM_LLVM = ["/opt/rocm/llvm/bin", "/opt/rocm-7.2.4/llvm/bin"]
def tool(name):
    for d in ROCM_LLVM:
        p = pathlib.Path(d)/name
        if p.exists(): return str(p)
    return shutil.which(name)

OBJDUMP, READELF, BUNDLER = tool("llvm-objdump"), tool("llvm-readelf"), tool("clang-offload-bundler")
FLAG_PATTERNS = {"has_v_dot2": r"\bv_dot2", "has_lds": r"\bds_(load|store|read|write)",
                 "has_cross_lane": CROSS_LANE_RE, 
                 "has_vector_global_load": r"\bglobal_load", "has_spill": r"\bscratch_(load|store)"}
COUNT_INS = ["v_dot2","ds_store","ds_load","ds_bpermute","global_load","v_fma_f32","v_exp","v_max","s_waitcnt","scratch_"]

def unbundle(path):
    p = pathlib.Path(path)
    if p.suffix == ".elf": return str(p)
    # try as a HIP fat binary bundle; fall back to using the .co directly
    out = f"/tmp/_isa_{p.stem}.elf"
    for tgt in ("hipv4-amdgcn-amd-amdhsa--gfx1100","hip-amdgcn-amd-amdhsa-gfx1100"):
        r = subprocess.run([BUNDLER,"--type=o","--unbundle",f"--input={path}",f"--output={out}",f"--targets={tgt}"], capture_output=True)
        if r.returncode == 0 and pathlib.Path(out).exists(): return out
    return str(path)

def audit(path):
    elf = unbundle(path); res = {"input": path, "elf": elf, "tooling_ok": bool(OBJDUMP and READELF)}
    if not res["tooling_ok"]: res["tooling_gap"]="llvm-objdump/readelf not found"; return res
    disasm = subprocess.run([OBJDUMP,"-d",elf], capture_output=True, text=True).stdout
    notes  = subprocess.run([READELF,"--notes",elf], capture_output=True, text=True).stdout
    syms   = subprocess.run([READELF,"-s",elf], capture_output=True, text=True).stdout
    res["kernels"]=[]
    # AMDGPU metadata (msgpack rendered by readelf) -> one block per .name
    for blk in re.split(r"\.name:", notes)[1:]:
        nm = blk.strip().split()[0] if blk.strip() else ""
        if not nm or nm.endswith(".kd"): continue
        def g(k, cast=int):
            mm=re.search(rf"\.{k}:\s*([0-9]+)", blk); return cast(mm.group(1)) if mm else None
        res["kernels"].append({"symbol":nm,"vgpr_count":g("vgpr_count"),"sgpr_count":g("sgpr_count"),
          "group_segment_lds_bytes":g("group_segment_fixed_size"),"private_segment_scratch_bytes":g("private_segment_fixed_size"),
          "kernarg_segment_size":g("kernarg_segment_size"),"max_flat_workgroup_size":g("max_flat_workgroup_size"),"wavefront_size":g("wavefront_size")})
    res["gfx_target"] = (re.search(r"amdhsa\.target:\s*\S+(gfx[0-9]+)", notes) or [None,"?"])[1] if "gfx" in notes else "gfx1100?"
    res["flags"] = {k: bool(re.search(pat, disasm)) for k,pat in FLAG_PATTERNS.items()}
    res["instr_counts"] = {ins: len(re.findall(rf"\b{re.escape(ins)}", disasm)) for ins in COUNT_INS}
    return res

def main():
    args = sys.argv[1:] or (sorted(glob.glob("/tmp/b4_tile_*.co"))[-1:] + sorted(glob.glob("/tmp/b4_comb_*.co"))[-1:])
    out = {"date":"2026-06-23","tool":"qk_amdgpu_isa_primitive_audit","objdump":OBJDUMP,"readelf":READELF,"bundler":BUNDLER,
           "results":[audit(a) for a in args]}
    OUT = pathlib.Path("bench/qk-post-default-runtime-kv-course"); OUT.mkdir(parents=True, exist_ok=True)
    (OUT/"isa_primitive_audit_tool_result.json").write_text(json.dumps(out, indent=2))
    for r in out["results"]:
        ks = r.get("kernels",[]); k0 = ks[0] if ks else {}
        print(f"{pathlib.Path(r['input']).name}: {k0.get('symbol','?')} vgpr={k0.get('vgpr_count')} sgpr={k0.get('sgpr_count')} "
              f"lds={k0.get('group_segment_lds_bytes')} scratch={k0.get('private_segment_scratch_bytes')} flags={r.get('flags')}", file=sys.stderr)
    print("artifact: bench/qk-post-default-runtime-kv-course/isa_primitive_audit_tool_result.json", file=sys.stderr)

if __name__ == "__main__": main()
