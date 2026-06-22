#!/usr/bin/env python3
"""Vendor-neutral ISA primitive audit wrapper (Lane 3 of docs/runtime-kv-isa-native-codegen-three-lane-scope-20260623.md).
AMD backend active (delegates to extra/qk_amdgpu_isa_primitive_audit.py); NVIDIA/Intel are scoped-only -> reported
as tooling-unavailable. Emits the normalized cross-vendor contract under bench/qk-isa-primitive-audit/.

  usage: PYTHONPATH=. .venv/bin/python extra/qk_isa_primitive_audit.py --vendor amd \
           --candidate owned_decode_attention --code-object /tmp/b4_tile_s47_*.co \
           [--wd-artifact bench/qk-post-owned-attention-default-audit/wd.json] [--out bench/qk-isa-primitive-audit/<name>.json]
"""
import argparse, glob, json, pathlib, sys

def amd_record(candidate, code_object, wd_artifact):
    from extra.qk_amdgpu_isa_primitive_audit import audit
    a = audit(code_object); k = (a.get("kernels") or [{}])[0]
    fl = a.get("flags", {})
    rec = {"candidate": candidate, "vendor": "amd", "arch": k.get("gfx_target") or "gfx1100",
           "code_object": code_object, "symbols": [kk.get("symbol") for kk in a.get("kernels", []) if kk.get("symbol")],
           "resources": {"vgpr": k.get("vgpr_count"), "sgpr": k.get("sgpr_count"), "lds_bytes": k.get("group_segment_lds_bytes"),
                         "scratch_bytes": k.get("private_segment_scratch_bytes"), "spills": (0 if fl.get("has_spill") is False else None)},
           "instruction_flags": {"has_vector_dot": fl.get("has_v_dot2", False), "has_lds": fl.get("has_lds", False),
                                 "has_cross_lane": fl.get("has_cross_lane", False), "has_vector_global_load": fl.get("has_vector_global_load", False),
                                 "has_spill": fl.get("has_spill", False)},
           "instr_counts": a.get("instr_counts", {}),
           "graph_lifecycle": {"route_fires": None, "fallback": None, "runtime_vars": []},
           "wd": {"artifact": wd_artifact, "tokens_match": None, "delta_pct": None}}
    if wd_artifact and pathlib.Path(wd_artifact).exists():
        try:
            w = json.loads(pathlib.Path(wd_artifact).read_text())
            rec["wd"]["tokens_match"] = w.get("correctness", {}).get("byte_identical_short_ctx_sdpa", w.get("correctness"))
            rows = w.get("rows", []); rec["wd"]["delta_pct"] = {str(r.get("ctx")): r.get("delta_pct") for r in rows} or None
            rec["graph_lifecycle"]["route_fires"] = bool(w.get("route_firing", {}).get("owned_flash_nodes_default", 0))
        except Exception as e: rec["wd"]["artifact_error"] = str(e)[:80]
    miss = [f for f in ("has_vector_dot","has_lds","has_cross_lane") if not rec["instruction_flags"][f]]
    rec["verdict"] = ("AMD_ISA_PRIMITIVE_GAP_FOUND" if rec["instruction_flags"]["has_spill"] or not rec["symbols"]
                      else "AMD_ISA_PRIMITIVE_CONFIRMED")
    rec["tooling_ok"] = a.get("tooling_ok", True)
    return rec

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", default="amd")
    ap.add_argument("--candidate", default="candidate")
    ap.add_argument("--code-object", default=None)
    ap.add_argument("--wd-artifact", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.vendor != "amd":
        rec = {"candidate": a.candidate, "vendor": a.vendor, "verdict": "ISA_BACKEND_TOOLING_LIMITED",
               "note": f"{a.vendor} backend is SCOPED-ONLY (not implemented). AMD: .co/AMDGCN/llvm-objdump (ready). "
                       f"NVIDIA: .cubin/SASS via cuobjdump/nvdisasm. Intel: device-module/Xe via IGC/ocloc. See "
                       f"docs/cross-vendor-isa-primitive-audit-and-search-result-20260623.md"}
    else:
        co = a.code_object or (sorted(glob.glob("/tmp/b4_tile_*.co"))[-1:] or [None])[0]
        if not co: print("no AMD code object found/given (--code-object)", file=sys.stderr); sys.exit(2)
        rec = amd_record(a.candidate, co, a.wd_artifact)
    outp = pathlib.Path(a.out or f"bench/qk-isa-primitive-audit/{a.candidate}.json")
    outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({k: rec.get(k) for k in ("candidate","vendor","arch","symbols","resources","instruction_flags","verdict")}, default=str), file=sys.stderr)
    print(f"artifact: {outp}", file=sys.stderr)

if __name__ == "__main__": main()
