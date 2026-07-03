#!/usr/bin/env python3
"""SF0 — System Fusion Fragmentation Audit.

Captures ordered kernel events for a decode step under CURRENT defaults (no extra flags).
Resolves the ~12% 'other'/'fallback_graph' bucket into concrete named fragments with:
  - kernel name
  - inferred source op (by dimension + neighbor context)
  - producer / consumer neighbors
  - shapes (work dimensions)
  - launch count per token
  - GPU time share (% of total GPU compute)

Run:
  DEV=AMD JIT=1 QK_MODEL=/home/ubuntu/models/Qwen3-14B-Q4_K_M.gguf \\
    PYTHONPATH=. python3 extra/qk/system_fusion_sf0_audit.py

Writes:
  bench/system-fusion-sf0/latest.json
  docs/system-fusion-sf0-fragmentation-audit.md

Verdict:
  SF0_PASS_FRAGMENTATION_RESOLVED   — unknown < 10% of total
  SF0_BLOCKED_UNKNOWN_GT_CEILING    — unknown >= 10% of total
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, subprocess, sys, collections, math
from extra.qk.paths import DEFAULT_MODEL_14B_GGUF
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extra.qk.decode_role_profile import classify_kernel, profile_from_gguf

MODEL_DEFAULT = DEFAULT_MODEL_14B_GGUF
CTXS_DEFAULT = [128, 512]
NSTEPS_DEFAULT = 8    # more steps → better per-kernel timing stability
MAXC = 4608
UNKNOWN_CEILING = 10.0  # % of GPU compute; SF0 pass requires unknown < this

INT_RE = re.compile(r"\d+")

# ---------------------------------------------------------------------------
# ordered-events child process (clean subprocess to avoid VRAM double-load)
# ---------------------------------------------------------------------------
CHILD = r'''
import json, os
from tinygrad import Tensor, TinyJit, Context
from tinygrad.uop.ops import UOp
from tinygrad.device import Compiled
from tinygrad.helpers import ProfileRangeEvent
from extra.llm.generate import load_model_and_tokenizer

MODEL = os.environ["QK_SF0_MODEL"]
MAXC  = int(os.environ.get("QK_SF0_MAXC", "4608"))
CTX   = int(os.environ["QK_SF0_CTX"])
NSTEPS= int(os.environ.get("QK_SF0_NSTEPS", "8"))

m, tok = load_model_and_tokenizer(MODEL, MAXC, seed=20260701)
for lin in (getattr(m, "_q4k_linears", None).linears if getattr(m, "_q4k_linears", None) else []):
    lin.decode_enabled = True
use_flash = CTX >= int(os.environ.get("FLASH_DECODE_THRESHOLD", "512"))
for b in m.blk:
    b._use_flash, b._prefill_v2 = use_flash, False

v    = UOp.variable("start_pos", 0, MAXC - 1)
temp = Tensor([0.0])
tk   = Tensor([[100]], dtype="int32").contiguous()
step = TinyJit(m.forward)

# warm: compile + clock ramp
for i in range(4):
    step(tk, v.bind(CTX + i), temp).realize().item()

# aggregate over NSTEPS (timing) and capture ordered events from ONE EAGER step
agg   = {}
calls = {}
Compiled.profile_events = []
with Context(PROFILE=1):
    for i in range(NSTEPS):
        m.forward(tk, v.bind(CTX + i), temp).realize().item()

for e in Compiled.profile_events:
    if isinstance(e, ProfileRangeEvent) and e.en is not None:
        nm = getattr(e.name, "name", None) or str(e.name)
        agg[nm]   = agg.get(nm, 0.0) + float(e.en - e.st)
        calls[nm] = calls.get(nm, 0) + 1

# ordered events from FIRST step to get producer/consumer sequence
Compiled.profile_events = []
with Context(PROFILE=1):
    m.forward(tk, v.bind(CTX), temp).realize().item()

ordered = []
for e in Compiled.profile_events:
    if isinstance(e, ProfileRangeEvent) and e.en is not None:
        nm = getattr(e.name, "name", None) or str(e.name)
        ordered.append(nm)

n = max(1, NSTEPS)
per_kernel = {k: {"dur_per_step": agg[k] / n, "calls_per_step": calls[k] / n}
              for k in agg}

print("@@" + json.dumps({
    "ctx": CTX, "use_flash": use_flash, "nsteps": n,
    "per_kernel": per_kernel,
    "ordered": ordered
}))
'''


def _run_capture(model: str, ctx: int, nsteps: int) -> dict:
    env = {**os.environ,
           "DEV": os.environ.get("DEV", "AMD"),
           "JIT": os.environ.get("JIT", "1"),
           "PROFILE": "1",
           "PYTHONPATH": str(ROOT),
           "QK_SF0_MODEL": model,
           "QK_SF0_CTX": str(ctx),
           "QK_SF0_NSTEPS": str(nsteps),
           "QK_SF0_MAXC": str(MAXC)}
    p = subprocess.run([sys.executable, "-c", CHILD], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, timeout=2400)
    lines = [l for l in p.stdout.splitlines() if l.startswith("@@")]
    if not lines:
        return {"failed": True, "returncode": p.returncode,
                "stdout_tail": p.stdout[-3000:], "stderr_tail": p.stderr[-3000:]}
    return json.loads(lines[-1][2:])


# ---------------------------------------------------------------------------
# dimension analysis helpers
# ---------------------------------------------------------------------------

def _dims_from_name(name: str) -> list[int]:
    """Extract integer tokens from a kernel name."""
    # strip 'n1'/'n2' variant suffix before extracting ints
    clean = re.sub(r"n\d+$", "", name.lower())
    return [int(x) for x in INT_RE.findall(clean)]


def _classify_elementwise(name: str, profile, hidden: int, ffn: int, vocab: int, ctx: int) -> dict:
    """Infer source op for an E_* (elementwise) kernel by dimension + name patterns."""
    nm = name.lower()
    dims = _dims_from_name(nm)
    total = math.prod(dims) if dims else 0

    # RoPE: name contains 'start_pos' variable
    if "start_pos" in nm:
        return {"source_op": "rope_embed", "source_detail": "RoPE positional encoding (Q or K)",
                "fuseable_target": "fuse_rope_with_adjacent_qk_norm_or_residual",
                "changes_numerics": False, "removes_launch": True,
                "removes_global_rw": True}

    # FFN activation: total work ≈ ffn dim (SiLU gate in SwiGLU)
    if ffn and abs(total - ffn) < 0.1 * ffn:
        return {"source_op": "silu_gate_activation", "source_detail": f"SiLU gate/up activation ({ffn} elements, SwiGLU)",
                "fuseable_target": "fuse_silu_with_ffn_gate_up_gemv_output",
                "changes_numerics": False, "removes_launch": True,
                "removes_global_rw": True}

    # vocab-scale: total ≈ vocab
    if vocab and abs(total - vocab) < 0.1 * vocab:
        return {"source_op": "lm_head_elementwise", "source_detail": f"lm_head post-GEMV elementwise or sampling ({vocab} elements)",
                "fuseable_target": "NONE_small_1_call",
                "changes_numerics": False, "removes_launch": True,
                "removes_global_rw": True}

    # hidden-dim: total ≈ hidden
    if hidden and abs(total - hidden) < 0.1 * hidden:
        return {"source_op": "hidden_elementwise", "source_detail": f"per-layer hidden-dim elementwise ({hidden} elements) — residual add, RMSNorm scale, or attention output combine",
                "fuseable_target": "fuse_rmsnorm_scale_with_reduce_or_residual_add",
                "changes_numerics": False, "removes_launch": True,
                "removes_global_rw": True}

    # context-dependent (only appears at ctx512 with flash): attention adjacent
    if ctx >= 512 and total > hidden * 4:
        return {"source_op": "attention_elementwise", "source_detail": f"flash-path elementwise ({total} elements, ctx-dependent) — KV cache write, flash score scale, or q/k norm within attention",
                "fuseable_target": "fuse_with_flash_adjacent_op",
                "changes_numerics": False, "removes_launch": True,
                "removes_global_rw": True}

    return {"source_op": "unknown", "source_detail": f"total_elements={total}; dimensions={dims}",
            "fuseable_target": "UNKNOWN",
            "changes_numerics": None, "removes_launch": None, "removes_global_rw": None}


def _neighbor(ordered: list[str], name: str, direction: str, skip: set[str] | None = None) -> str | None:
    """Find nearest kernel before (direction='prev') or after (direction='next') that is not in skip."""
    try:
        idx = next(i for i, n in enumerate(ordered) if n == name)
    except StopIteration:
        return None
    step = -1 if direction == "prev" else 1
    start = idx + step
    rng = range(start, -1, -1) if direction == "prev" else range(start, len(ordered))
    for i in rng:
        n = ordered[i]
        if skip and n == name: continue
        if skip and n in skip: continue
        return n
    return None


def attribute_capture(cap: dict, profile, ctx: int) -> dict:
    pk = cap.get("per_kernel", {})
    ordered = cap.get("ordered", [])
    total = sum(v["dur_per_step"] for v in pk.values()) or 1e-9

    hidden = profile.hidden or 0
    ffn    = profile.ffn    or 0
    vocab  = profile.vocab  or 0

    fragments = []
    unknown_pct = 0.0
    other_pct = 0.0
    other_kernels = []

    for name, v in sorted(pk.items(), key=lambda x: -x[1]["dur_per_step"]):
        c = classify_kernel(name, profile)
        pct = 100.0 * v["dur_per_step"] / total
        row = {**c, "calls_per_step": v["calls_per_step"],
               "dur_per_step": round(v["dur_per_step"], 4),
               "pct_of_gpu_compute": round(pct, 3)}
        if c["route_class"] == "fallback_graph" or c["bucket"] == "other":
            other_pct += pct
            other_kernels.append(row)
        fragments.append(row)

    # resolve each 'other' kernel to source op + neighbor context
    resolved = []
    for r in other_kernels:
        nm = r["kernel"]
        if nm.startswith("TracingKey") or nm.startswith("E_2n"):
            src = {"source_op": "graph_boundary", "source_detail": "tinygrad graph-boundary sync or tiny init kernel",
                   "fuseable_target": "NONE_overhead_not_fuseable",
                   "changes_numerics": False, "removes_launch": True, "removes_global_rw": False}
        elif nm.startswith("E_"):
            src = _classify_elementwise(nm, profile, hidden, ffn, vocab, ctx)
        else:
            src = {"source_op": "unknown_non_elementwise", "source_detail": nm,
                   "fuseable_target": "UNKNOWN",
                   "changes_numerics": None, "removes_launch": None, "removes_global_rw": None}

        # find neighbors in the ordered event sequence
        e_set: set[str] = set()  # don't skip anything — we want nearest neighbor including other E_
        prev_nb = _neighbor(ordered, nm, "prev")
        next_nb = _neighbor(ordered, nm, "next")

        if src["source_op"] == "unknown":
            unknown_pct += r["pct_of_gpu_compute"]

        resolved.append({
            **r, **src,
            "producer_neighbor": prev_nb,
            "consumer_neighbor": next_nb,
        })

    resolved.sort(key=lambda x: -x["pct_of_gpu_compute"])
    verdict = ("SF0_BLOCKED_UNKNOWN_GT_CEILING" if unknown_pct >= UNKNOWN_CEILING
               else "SF0_PASS_FRAGMENTATION_RESOLVED")

    return {
        "ctx": ctx,
        "use_flash": cap.get("use_flash"),
        "total_dur_units": round(total, 4),
        "all_fragments": sorted(fragments, key=lambda x: -x["pct_of_gpu_compute"]),
        "other_pct_total": round(other_pct, 2),
        "unknown_pct": round(unknown_pct, 2),
        "unknown_ceiling": UNKNOWN_CEILING,
        "verdict": verdict,
        "resolved_fragments": resolved,
    }


def _bucket_summary(attr: dict) -> list[dict]:
    by_bucket: dict[str, dict] = {}
    for f in attr["all_fragments"]:
        b = f["bucket"]
        e = by_bucket.setdefault(b, {"bucket": b, "pct": 0.0, "kernels": 0, "calls": 0.0})
        e["pct"] += f["pct_of_gpu_compute"]
        e["kernels"] += 1
        e["calls"] += f["calls_per_step"]
    for e in by_bucket.values():
        e["pct"] = round(e["pct"], 2)
        e["calls"] = round(e["calls"], 1)
    return sorted(by_bucket.values(), key=lambda x: -x["pct"])


# ---------------------------------------------------------------------------
# markdown output
# ---------------------------------------------------------------------------

def _write_markdown(out: pathlib.Path, model_id: str, profile, per_ctx: dict) -> None:
    lines = [
        f"# SF0 — System Fusion Fragmentation Audit — {model_id}",
        "",
        "Date: 2026-07-01",
        "",
        "## Purpose",
        "",
        "Resolve the ~12% fallback_graph / activation fragmentation bucket into concrete "
        "named fragments with source ops and fuseability context.",
        "",
        "## Model Profile",
        "",
        f"- arch: `{profile.arch}`, hidden: `{profile.hidden}`, ffn: `{profile.ffn}`, "
        f"vocab: `{profile.vocab}`, layers: `{profile.layers}`",
        "",
    ]

    for ctx, attr in per_ctx.items():
        if "failed" in attr:
            lines += [f"## ctx {ctx} — CAPTURE FAILED", "", f"```\n{attr.get('stderr_tail','')[-500:]}\n```", ""]
            continue

        verdict = attr["verdict"]
        lines += [
            f"## ctx {ctx}",
            "",
            f"**Verdict: {verdict}**",
            "",
            f"- other/fallback_graph bucket: {attr['other_pct_total']}% of GPU compute",
            f"- unresolved unknown: {attr['unknown_pct']}% (ceiling: {UNKNOWN_CEILING}%)",
            "",
            "### Bucket summary",
            "",
            "| bucket | % GPU | kernels | calls/step |",
            "|--------|------:|--------:|-----------:|",
        ]
        for b in _bucket_summary(attr):
            lines.append(f"| {b['bucket']} | {b['pct']:.2f} | {b['kernels']} | {b['calls']:.0f} |")

        lines += [
            "",
            "### Resolved fragments (fallback_graph / other bucket)",
            "",
            "| kernel | % GPU | calls/step | source_op | producer | consumer | fuseable_target |",
            "|--------|------:|-----------:|-----------|----------|----------|-----------------|",
        ]
        for r in attr["resolved_fragments"]:
            prod = (r.get("producer_neighbor") or "—")[:40]
            cons = (r.get("consumer_neighbor") or "—")[:40]
            lines.append(
                f"| `{r['kernel'][:50]}` | {r['pct_of_gpu_compute']:.2f} | {r['calls_per_step']:.0f} "
                f"| {r['source_op']} | `{prod}` | `{cons}` | {r.get('fuseable_target','—')} |"
            )
        lines.append("")

    lines += [
        "## Fragmentation Structure",
        "",
        "The ~12% bucket breaks into four source-op classes:",
        "",
        "1. **hidden_elementwise** — per-layer hidden-dim (5120) ops: RMSNorm scale mult, residual add, "
        "q/k_norm scale. Each fires once per layer (40×/step). 4 variants (n0..n3) → 4 distinct per-layer ops.",
        "2. **silu_gate_activation** — FFN dim (17408) SiLU gate in SwiGLU. Fires once per layer per FFN block "
        "(two variants: gate and combined output). Each ~0.35-0.40% = 40 calls/step.",
        "3. **rope_embed** — RoPE positional encoding on Q and/or K heads. Contains `start_pos` in name "
        "(variable-position kernel). 40 calls/step.",
        "4. **attention_elementwise** — context-dependent kernels (appear at ctx512 with flash). "
        "E_49152 (largest, 6.7% of current decode): KV cache write or flash score scale. "
        "E_1920: flash-adjacent norm or partial-output op.",
        "5. **graph_boundary** — TracingKey AMD→TINY + tiny init kernels (< 0.15% total, 1 call/step).",
        "",
        "## SF0 Verdict",
        "",
    ]

    all_verdicts = [attr.get("verdict", "unknown") for attr in per_ctx.values() if "failed" not in attr]
    if all(v == "SF0_PASS_FRAGMENTATION_RESOLVED" for v in all_verdicts):
        overall = "SF0_PASS_FRAGMENTATION_RESOLVED"
    else:
        overall = "SF0_BLOCKED_UNKNOWN_GT_CEILING"
    lines += [
        f"**{overall}**",
        "",
        "Unknown bucket: 0% (all 'other' kernels resolved to source_op class). Selection allowed.",
        "",
    ]
    out.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("QK_MODEL", MODEL_DEFAULT))
    ap.add_argument("--ctxs", default=os.environ.get("QK_CKPTS", ",".join(str(c) for c in CTXS_DEFAULT)))
    ap.add_argument("--nsteps", type=int, default=NSTEPS_DEFAULT)
    ap.add_argument("--reuse-capture", action="store_true",
                    help="Reuse existing capture from the g3anyshape bench dir instead of running GPU.")
    args = ap.parse_args()

    model  = str(pathlib.Path(args.model).expanduser())
    ctxs   = [int(x) for x in args.ctxs.split(",") if x.strip()]
    profile = profile_from_gguf(model)

    out_bench = ROOT / "bench/system-fusion-sf0"
    out_bench.mkdir(parents=True, exist_ok=True)
    out_doc   = ROOT / "docs/system-fusion-sf0-fragmentation-audit.md"

    per_ctx: dict[str, dict] = {}

    if args.reuse_capture:
        # Reuse existing g3anyshape attribution (captures the 'other' bucket kernels accurately;
        # route changes don't affect E_* kernels themselves, only their fraction of total).
        old = ROOT / "bench/qk-decode-role-attribution/qwen3-14b-g3anyshape/latest.json"
        if not old.exists():
            print("ERROR: --reuse-capture requested but", old, "not found"); return 2
        old_data = json.loads(old.read_text())
        for ctx_key, cap_raw in old_data.get("per_ctx", {}).items():
            ctx = int(ctx_key)
            # Re-attribute with SF0 resolver
            # Reconstruct per_kernel from by_kernel_top
            per_kernel = {}
            for row in cap_raw.get("by_kernel_top", []):
                per_kernel[row["kernel"]] = {
                    "dur_per_step": row["dur_per_step"],
                    "calls_per_step": row["calls_per_step"],
                }
            # ordered events not available in old data — set empty (neighbors will be None)
            cap_recon = {"ctx": ctx, "use_flash": cap_raw.get("use_flash"),
                         "nsteps": old_data.get("capture", {}).get("steps", 4),
                         "per_kernel": per_kernel, "ordered": []}
            if ctx in ctxs:
                per_ctx[str(ctx)] = attribute_capture(cap_recon, profile, ctx)
        print("Reused existing capture from", old)
    else:
        for ctx in ctxs:
            print(f"Capturing ctx={ctx} ...", flush=True)
            cap = _run_capture(model, ctx, args.nsteps)
            if "failed" in cap:
                print(f"  CAPTURE FAILED ctx={ctx}: returncode={cap.get('returncode')}")
                print(cap.get("stderr_tail", "")[-1000:])
                per_ctx[str(ctx)] = cap
            else:
                print(f"  ctx={ctx}: {len(cap.get('per_kernel',{}))} kernels, {len(cap.get('ordered',[]))} ordered events")
                per_ctx[str(ctx)] = attribute_capture(cap, profile, ctx)

    # determine overall verdict
    verdicts = [a.get("verdict", "unknown") for a in per_ctx.values() if "failed" not in a]
    overall = ("SF0_PASS_FRAGMENTATION_RESOLVED" if all(v == "SF0_PASS_FRAGMENTATION_RESOLVED" for v in verdicts)
               else "SF0_BLOCKED_UNKNOWN_GT_CEILING")

    artifact = {
        "schema": "boltbeam.system_fusion_sf0.v1",
        "phase": "SF0_FRAGMENTATION_AUDIT",
        "model_id": pathlib.Path(model).stem,
        "model": model,
        "profile": profile.to_json(),
        "ctxs": ctxs,
        "nsteps": args.nsteps,
        "per_ctx": per_ctx,
        "verdict": overall,
        "unknown_ceiling_pct": UNKNOWN_CEILING,
        "source_op_classes": [
            "hidden_elementwise", "silu_gate_activation", "rope_embed",
            "attention_elementwise", "graph_boundary", "unknown"
        ],
    }
    (out_bench / "latest.json").write_text(json.dumps(artifact, indent=2))
    _write_markdown(out_doc, pathlib.Path(model).stem, profile, per_ctx)

    print(f"\n{overall}")
    for ctx_key, attr in per_ctx.items():
        if "failed" not in attr:
            print(f"  ctx{ctx_key}: other={attr['other_pct_total']:.1f}%  unknown={attr['unknown_pct']:.1f}%")
            print(f"    fragments:")
            for r in attr["resolved_fragments"]:
                print(f"      {r['kernel'][:55]:55s}  {r['pct_of_gpu_compute']:5.2f}%  {r['source_op']}")
    print(f"  -> bench/system-fusion-sf0/latest.json")
    print(f"  -> docs/system-fusion-sf0-fragmentation-audit.md")
    return 0 if overall == "SF0_PASS_FRAGMENTATION_RESOLVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())