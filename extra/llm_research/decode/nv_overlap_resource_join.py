"""CPU-only fail-closed resource join for the NV decode-overlap question.

The 2026-08-12 descriptor census gives every tinygrad occurrence a complete
launch tuple (binary SHA-256, grid, block, registers/thread, static/dynamic
shared bytes, local-memory bytes).  The join consumes the dependency-
independent (support, quant/flash host) pairs produced by
`nv_co_schedule_candidate_scan.py` on the same DAG and computes, per pair, the
complementary CTA residency bound: how many (1 support CTA + 1 host CTA) pairs
can co-reside on one SM under the per-SM register/thread/shared budgets.

The verdict is GPU_ELIGIBLE only if some pair has a positive complementary CTA
bound AND an exact critical-path recovery >= 50 us.  Otherwise the join fails
closed (INCONCLUSIVE_FAIL_CLOSED).  The llama side stays the pinned 08-04
manifest/timeline (its mmvq rows still lack a local-memory field), so the
llama-side complementarity row remains absent; the tinygrad side is now the
complete census.  This tool never books recovery and never changes defaults.

Per-SM budgets (host: RTX 5090, sm_120 / GB202):
- registers: 65536 per SM (tinygrad/runtime/ops_nv.py; also
  nv-p3b-ffn-q8-cooperative-producer-static-record-20260805.md).
- threads: 1536 per SM (48 warps; the 08-05 record's "six CTAs/SM, 1536/256").
- shared memory: 228 KB per SM (Blackwell architecture max); with max 10 KB
  static and 0 dynamic shared in this census it is non-binding.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

REQUIRED = ("grid", "block", "registers_per_thread", "static_smem_bytes",
            "dynamic_smem_bytes", "local_mem_bytes")
PROMOTION_GATE_US = 50.0
DEFAULT_BUDGET = {"registers_per_sm": 65536, "threads_per_sm": 1536,
                  "smem_bytes_per_sm": 228 * 1024}
BUDGET_SOURCES = [
  "registers_per_sm 65536: tinygrad/runtime/ops_nv.py register file comment",
  "threads_per_sm 1536 (48 warps): nv-p3b-ffn-q8-cooperative-producer-static-record-20260805.md (six CTAs/SM, 1536/256)",
  "smem_bytes_per_sm 228 KB: Blackwell architecture max; non-binding here (max 10 KB static, 0 dynamic)",
]


def resource_complete(row: dict) -> bool:
  return all(isinstance(row.get(k), int if k.endswith("bytes") or k == "registers_per_thread" else list) for k in REQUIRED)


def _prod(x: list[int]) -> int:
  return x[0] * x[1] * x[2]


def complementary_cta_bound(support: dict, host: dict, budget: dict | None = None) -> int:
  """Max (1 support CTA + 1 host CTA) pairs co-resident on one SM.

  Register and thread usage scale per CTA (block product * per-thread
  registers); shared usage is per CTA.  A non-positive result means the pair
  cannot overlap on an SM, so co-scheduling cannot hide the support behind the
  host at all.
  """
  b = budget or DEFAULT_BUDGET
  ts, th = _prod(support["block"]), _prod(host["block"])
  regs = ts * support["registers_per_thread"] + th * host["registers_per_thread"]
  smem = (support["static_smem_bytes"] + support["dynamic_smem_bytes"]
          + host["static_smem_bytes"] + host["dynamic_smem_bytes"])
  return min(b["registers_per_sm"] // max(1, regs),
             b["threads_per_sm"] // max(1, ts + th),
             b["smem_bytes_per_sm"] // max(1, smem))


def build(manifest: dict, timeline: dict, tinygrad_census: dict,
          pairs: list[dict] | None = None, budget: dict | None = None) -> dict:
  llama = [r for r in manifest["rows"] if r["model_role"] != "vocab"]
  llama_complete = sum(resource_complete({**r["llama"]["mmvq"], "local_mem_bytes": None}) for r in llama)
  # The pinned llama mmvq rows still have no local-memory field, so the llama
  # complementarity row stays absent; this is the fail-closed seam on that side.
  census_nodes = tinygrad_census.get("nodes", [])
  descriptors = {n["index"]: n["descriptor"] for n in census_nodes
                 if resource_complete(n.get("descriptor", {}))}
  tg_complete = len(descriptors)
  hidden = timeline["classes"]["quantize_q8_1"]["hidden_behind_mmq_us"]

  base = {"schema": "tinygrad.nv_overlap_resource_join.v1",
          "llama_overlap_mass_us": {"quantize_q8_1_behind_mmvq": hidden},
          "llama_mmvq_rows": len(llama), "llama_complete_resource_rows": llama_complete,
          "tinygrad_complete_resource_rows": tg_complete,
          "tinygrad_logical_nodes": tinygrad_census.get("summary", {}).get("node_count"),
          "budget": (budget or DEFAULT_BUDGET), "budget_sources": BUDGET_SOURCES,
          "promotion_gate_us": PROMOTION_GATE_US}

  if not pairs:
    return {**base, "status": "INCONCLUSIVE_FAIL_CLOSED",
            "dependency_independent_pairs": 0, "pairs_with_descriptors": 0,
            "positive_bound_pairs": 0, "best_pair": None,
            "finding": "No dependency-independent (support, quant/flash host) pair list was supplied, so no resource-complementarity claim can be ranked. The join fails closed until the co-schedule scan names pairs."}

  ranked = []
  for p in pairs:
    s = descriptors.get(p.get("support_id"))
    h = descriptors.get(p.get("host_id"))
    if s is None or h is None:
      continue
    bound = complementary_cta_bound(s, h, budget)
    ranked.append({"support_id": p.get("support_id"), "support_name": p.get("support_name"),
                   "host_id": p.get("host_id"), "host_name": p.get("host_name"),
                   "host_family": p.get("host_family"),
                   "hideable_us": p.get("hideable_us"),
                   "recovery_us": p.get("recovery_us"),
                   "complementary_cta_bound": bound,
                   "support_block": s["block"], "support_threads_per_cta": _prod(s["block"]),
                   "support_regs": s["registers_per_thread"], "support_smem_bytes": s["static_smem_bytes"] + s["dynamic_smem_bytes"],
                   "host_block": h["block"], "host_threads_per_cta": _prod(h["block"]),
                   "host_regs": h["registers_per_thread"], "host_smem_bytes": h["static_smem_bytes"] + h["dynamic_smem_bytes"]})

  positive = [r for r in ranked if r["complementary_cta_bound"] > 0]
  best = max(positive, key=lambda r: (r.get("recovery_us") or 0.0)) if positive else None
  gate_pass = bool(best is not None and (best.get("recovery_us") or 0.0) >= PROMOTION_GATE_US)

  if gate_pass and best is not None:
    status = "GPU_ELIGIBLE"
    finding = (f"Pair ({best['support_name']} @ {best['support_id']} -> {best['host_name']} @ {best['host_id']}) "
               f"has a positive complementary CTA bound of {best['complementary_cta_bound']} and exact recovery "
               f"{best['recovery_us']} us >= {PROMOTION_GATE_US} us; the native two-queue span A/B arm is authorized.")
  else:
    status = "INCONCLUSIVE_FAIL_CLOSED"
    reason = (f"best dependency-independent pair recovery {(best['recovery_us'] if best else 'n/a')} us < "
              f"{PROMOTION_GATE_US} us" if best else "no dependency-independent pair has a positive complementary CTA bound")
    finding = (f"The tinygrad side is now a complete compiled-descriptor census ({tg_complete} complete tuples), and "
               f"{len(positive)} of {len(ranked)} dependency-independent pairs are SM-co-resident, but {reason}. "
               f"No GPU arm is authorized by this join.")

  return {**base, "status": status,
          "dependency_independent_pairs": len(pairs), "pairs_with_descriptors": len(ranked),
          "positive_bound_pairs": len(positive), "best_pair": best, "finding": finding}


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--census", default="docs/task_workflow/evidence/nv-descriptor-census-head-20260812.json")
  ap.add_argument("--scan", default="docs/task_workflow/evidence/nv-co-schedule-scan-head-20260812.json")
  ap.add_argument("--out")
  args = ap.parse_args()
  root = Path(__file__).parents[3]
  manifest = json.loads((root / "docs/task_workflow/output/nv-decode-llama-tinygrad-semantic-call-manifest-20260804.json").read_text())
  timeline = json.loads((root / "docs/task_workflow/output/nv-decode-llama-d512-timeline-ledger-20260804.json").read_text())
  census = json.loads((root / args.census).read_text())
  scan = json.loads((root / args.scan).read_text())
  out = build(manifest, timeline, census, scan.get("pairs", []))
  text = json.dumps(out, indent=2, sort_keys=True) + "\n"
  if args.out:
    path = Path(args.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(text)
  print(text, end="")


if __name__ == "__main__":
  main()
