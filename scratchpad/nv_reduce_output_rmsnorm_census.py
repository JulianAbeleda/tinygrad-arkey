#!/usr/bin/env python3
"""Per-site reduce_output_rmsnorm absorption census (CPU-only).

Captures the forced decode-only production graph for one or both arms:

  --arm baseline   the closed production graph (ordinary norms route, no
                   callify Context): the outer "before" census
  --arm baseline-context  the ordinary norms route under the same callify
                   Context flags the promoted route runs under, so the
                   reduce-output diff holds the callify substrate constant
  --arm ffn-before  the fp32 q/k site promoted (keeps the live-split flash
                   route alive on CPU) with the FFN-norm site closed, under
                   the same callify Context: the per-site "before" arm for
                   the M1 ffn-norm (1_4096) body-free removal
  --arm promoted   the fp32 q/k reduce-output route with the production
                   callify Context: the "after" census

``--arm both`` captures baseline then promoted and ``--arm ffn`` captures
ffn-before then promoted; each writes one merged evidence document carrying
the per-site before/after program counts, the exact-name added/removed diff,
and the materialization check (zero ``*_weight_store``-style additions), so
the P1 body-free 1:1 swap contract
(docs/task_workflow/input/nv-reduce-output-site-absorption-scope-20260812.md
section 5 gate 3) can be verified per site on DEV=CPU.  The arm split mirrors
the A/B harness (extra/llm_research/decode/nv_reduce_output_fp32_qk_ab.py):
the candidate arm decodes under both callify Context flags, the baseline arm
stays on the closed graph with no flags.
"""
import argparse, collections, json, pathlib

from extra.llm_research.decode.decode_harness import DEFAULT_MODEL
from extra.llm_research.decode.decode_runtime_overhead import _make_prompt, capture_decode_graph

# The production reduce-output body families, per site.
BODY_PREFIXES = ("reduce_output_rmsnorm_32_128", "reduce_output_rmsnorm_8_128", "reduce_output_rmsnorm_1_4096")
SITES = (
  # (site, body prefixes, ordinary tiling families observed on CPU for the
  # site's norms route: the q/k 128-dim reduce+epilogue families and the
  # 4096-dim block-norm reduce+epilogue families).
  ("qk_32_128", ("reduce_output_rmsnorm_32_128",),
   ("r_2_32_16_4_16_4_2_4_8", "E_8_128_4", "E_8_32_4_4", "E_8_(start_pos+1)_4")),
  ("qk_8_128", ("reduce_output_rmsnorm_8_128",),
   ("r_8_2_16_4_16_4_2_4_8", "E_2_128_4", "r_8_32_4")),
  ("ffn_down_1_4096", ("reduce_output_rmsnorm_1_4096",), ("r_1024_4", "E_1024_4")),
)


def _program_counts(census: dict) -> dict[str, int]:
  names = [record.get("program_name", "") for record in census["records"]]
  return dict(collections.Counter(names))


def _marker_row(x) -> dict:
  """Record the pre-callify marker input spelling observed at a model call site."""
  return {"op": x.uop.op.name, "base_op": x.uop.base.op.name, "shape": list(x.shape),
          "dtype": str(x.dtype), "buffer_identity": bool(x.uop.has_buffer_identity()),
          "precompiled_output_identity": bool(x.uop.has_precompiled_output_identity())}


def _capture(args, promoted: bool, callify: bool = True, ffn_promoted: bool|None = None) -> dict:
  from dataclasses import replace
  from tinygrad.llm.generate import load_model_and_tokenizer
  model, tokenizer = load_model_and_tokenizer(args.model, args.max_context, seed=20260617)
  model.config = replace(model.config, prefill_tc_attn=False, prefill_custom_kernel_attn=False)
  # The production-qualified direct greedy capture route, exactly like the A/B
  # harness.  This census is defined over the flash decode graph on CPU too;
  # the reduce-output kernels appear in both the flash and SDPA graphs, so the
  # program-count contract does not depend on which one the local CPU capture
  # selects.
  model._decode_direct_greedy_promoted = True
  model._decode_reduce_output_rmsnorm_promoted = promoted
  # The FFN-norm site is independently gateable: a per-site arm keeps the
  # fp32 q/k site promoted (the live-split flash route needs it) while the
  # ffn-norm markers close, so the ffn removal can be measured body-free.
  model._decode_reduce_output_ffn_rmsnorm_promoted = promoted if ffn_promoted is None else ffn_promoted
  for block in model.blk:
    block.config = replace(block.config, prefill_tc_attn=False, prefill_custom_kernel_attn=False)
    block._decode_reduce_output_rmsnorm_promoted = promoted
    block._decode_reduce_output_ffn_rmsnorm_promoted = promoted if ffn_promoted is None else ffn_promoted
  # Observe the exact pre-callify values passed by production model call sites.
  # Census-local; leaves model semantics unchanged.
  marker_inputs: list[dict] = []
  if promoted:
    import tinygrad.llm.model as model_module
    original_marker = model_module._decode_reduce_output_rmsnorm
    def observed_marker(norm, x, promoted):
      if promoted:
        marker_inputs.append(_marker_row(x))
      return original_marker(norm, x, promoted)
    original_fp16_marker = model_module._decode_reduce_output_rmsnorm_fp16_consumer
    def observed_fp16_marker(norm, x, promoted):
      marked = original_fp16_marker(norm, x, promoted)
      if promoted:
        row = _marker_row(x)
        row["marker_op"] = marked.uop.op.name
        row["input_identity_at_marker"] = bool(marked.uop.arg.input_identity_at_marker)
        row["owned_contiguous_candidate"] = bool(marked.uop.arg.owned_contiguous_candidate)
        row["reduce_input_at_marker"] = bool(marked.uop.arg.reduce_input_at_marker)
        row["residual_sum_at_marker"] = bool(marked.uop.arg.residual_sum_at_marker)
        marker_inputs.append(row)
      return marked
    model_module._decode_reduce_output_rmsnorm = observed_marker
    model_module._decode_reduce_output_rmsnorm_fp16_consumer = observed_fp16_marker
  ids = (tokenizer.prefix() if hasattr(tokenizer, "prefix") else []) + tokenizer.encode("the quick brown fox jumps. "*800)
  from tinygrad.llm.reduce_output_trace import REDUCE_OUTPUT_TRACE, reset_reduce_output_trace, reduce_output_trace_snapshot
  reset_reduce_output_trace()
  from tinygrad.callify import CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT, CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER
  from tinygrad.helpers import Context
  with Context(REDUCE_OUTPUT_TRACE=1,
               **({"CALLIFY_OWNED_PRECOMPILED_OUTPUT_REDIRECT": int(args.typed_semantic_producer),
                   "CALLIFY_TYPED_SEMANTIC_INPUT_PRODUCER": int(args.typed_semantic_producer)} if callify else {})):
    census = capture_decode_graph(model, _make_prompt(ids, args.depth), args.chunk_size, 3).to_dict()
  names = [record.get("program_name", "") for record in census["records"]]
  counts = _program_counts(census)
  bodies = {prefix: counts.get(prefix, 0) for prefix in BODY_PREFIXES}
  census["capture"] = {"phase": "decode", "fixed_depth": args.depth, "jit": "rollout_greedy_jit_flash",
                       "callify_owned_redirect": int(args.typed_semantic_producer) if callify else 0,
                       "typed_semantic_input_producer": int(args.typed_semantic_producer) if callify else 0}
  census["reduce_output"] = {"count": sum(bodies.values()),
                             "program_names": sorted(name for name in counts if "reduce_output_rmsnorm" in name),
                             "total_calls": len(names), "marker_input_count": len(marker_inputs),
                             "marker_input_histogram": dict(collections.Counter(json.dumps(row, sort_keys=True) for row in marker_inputs)),
                             "marker_input_examples": marker_inputs[:16],
                             "stage_trace": reduce_output_trace_snapshot()}
  return {"capture": census["capture"],
          "total_programs": len(names), "program_counts": counts, "bodies": bodies,
          "weight_store_names": {name: count for name, count in counts.items() if "weight_store" in name},
          "reduce_output": census["reduce_output"]}


def _diff(before: dict, after: dict) -> dict:
  before_counts, after_counts = before["program_counts"], after["program_counts"]
  before_names, after_names = set(before_counts), set(after_counts)
  added = {name: after_counts[name] for name in sorted(after_names - before_names)}
  removed = {name: before_counts[name] for name in sorted(before_names - after_names)}
  return {"before_total": before["total_programs"], "after_total": after["total_programs"],
          "net_program_delta": after["total_programs"] - before["total_programs"],
          "added": added, "removed": removed,
          "weight_materialization_delta": len(after["weight_store_names"]) - len(before["weight_store_names"]),
          "weight_store_before": before["weight_store_names"], "weight_store_after": after["weight_store_names"]}


def _sites(before: dict, after: dict, diff: dict) -> dict:
  import re
  _hash64 = re.compile(r"_([0-9a-f]{64})$")
  def families(record: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, count in record["program_counts"].items():
      stem = _hash64.sub("", name)
      counts[stem] = counts.get(stem, 0) + count
    return counts
  before_families, after_families = families(before), families(after)
  report = {}
  for site, prefixes, ordinary in SITES:
    before_bodies = sum(before["bodies"].get(p, 0) for p in prefixes)
    after_bodies = sum(after["bodies"].get(p, 0) for p in prefixes)
    removed_ordinary = {name: count for name, count in diff["removed"].items()
                        if not any(name.startswith(p) for p in BODY_PREFIXES)}
    added_bodies = {name: count for name, count in diff["added"].items()
                    if any(name.startswith(p) for p in prefixes)}
    report[site] = {"before_bodies": before_bodies, "after_bodies": after_bodies,
                    "body_delta": after_bodies - before_bodies,
                    "removed_ordinary_count": sum(removed_ordinary.values()),
                    "added_bodies": added_bodies,
                    "ordinary_families": {family: {"before": before_families.get(family, 0),
                                                   "after": after_families.get(family, 0)}
                                          for family in ordinary if before_families.get(family) or after_families.get(family)}}
  return report


def _gemv_shift_explanation(diff: dict) -> dict:
  """Account for the CPU-only o-proj -> ffn_down GEMV reduce fusion.

  Binding the shared residual ``h`` instead of re-materializing a fresh ADD
  lets the CPU scheduler fuse the attention-output (o-proj) GEMV reduce
  (``r_64_16_4_16_4_2_4_8``, k=4096) into the ffn_down GEMV reduce
  (``r_64_16_4_48_*``, k=12288): both write a 4096-wide output over the same
  grid geometry.  The ffn_down kernel name grows the o-proj reduce ranges and
  the graph-admission role stays ``ffn_down`` (the primary output).  This is a
  correctness-preserving CPU-scheduler multi-output fusion, not a q/k geometry
  change, and does not appear on the GPU graph where the o-proj/ffn_down use
  custom Q4K/Q6K GEMV kernels.
  """
  attn_qo = "r_64_16_4_16_4_2_4_8"
  ffn_down_before = "r_64_16_4_48_"
  ffn_down_after = "r_64_16_4_16_4_2_32_48_"
  return {
    "summary": "CPU scheduler fuses the attn_qo o-proj GEMV reduce into the ffn_down GEMV reduce",
    "removed_attn_qo_reduce": {n: c for n, c in sorted(diff["removed"].items()) if n.startswith(attn_qo)},
    "renamed_ffn_down_removed": {n: c for n, c in sorted(diff["removed"].items()) if n.startswith(ffn_down_before)},
    "renamed_ffn_down_added": {n: c for n, c in sorted(diff["added"].items()) if n.startswith(ffn_down_after)},
    "cpu_only": True,
    "correctness_authority": "full-logit GPU A/B (q/k families are unchanged in this diff)",
  }


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default=DEFAULT_MODEL)
  ap.add_argument("--depth", type=int, default=512)
  ap.add_argument("--max-context", type=int, default=4608)
  ap.add_argument("--chunk-size", type=int, default=32)
  ap.add_argument("--arm", choices=("baseline", "baseline-context", "ffn-before", "promoted", "both", "ffn"), default="promoted")
  ap.add_argument("--typed-semantic-producer", action="store_true",
                  help="enable only the closed typed CALL-input producer route with callify redirect (promoted arm)")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  if args.arm == "both": arms = ("baseline", "baseline-context", "promoted")
  elif args.arm == "ffn": arms = ("ffn-before", "promoted")
  else: arms = (args.arm,)
  def _arm_kind(arm: str) -> tuple[bool, bool|None, bool]:
    """Return (q/k promoted, ffn promoted or None to follow q/k, callify)."""
    if arm == "promoted": return True, None, True
    if arm == "ffn-before": return True, False, True
    if arm == "baseline-context": return False, None, True
    return False, None, False
  captured = {}
  for arm in arms:
    promoted, ffn_promoted, callify = _arm_kind(arm)
    record = _capture(args, promoted=promoted, callify=callify, ffn_promoted=ffn_promoted)
    record["arm"] = arm
    captured[arm] = record
  doc = {"schema": "tinygrad.nv_reduce_output_site_absorption_census.v1", "arms": captured}
  if "baseline" in captured and "promoted" in captured:
    before, after = captured["baseline"], captured["promoted"]
    diff = _diff(before, after)
    sites = _sites(before, after, diff)
    contract = {
      "program_count_identical_or_reduced": after["total_programs"] <= before["total_programs"],
      "net_program_delta": diff["net_program_delta"],
      "zero_weight_materializations": not diff["weight_store_before"] and not diff["weight_store_after"],
      "added_names_are_only_fused_bodies": bool(diff["added"]) and all(
        any(name.startswith(prefix) for prefix in BODY_PREFIXES) for name in diff["added"]),
      "per_site": {site: {"after_bodies": sites[site]["after_bodies"],
                          "body_delta": sites[site]["body_delta"]} for site in sites},
    }
    doc["diff"] = diff
    doc["sites"] = sites
    doc["contract"] = contract
  if "ffn-before" in captured and "promoted" in captured:
    before, after = captured["ffn-before"], captured["promoted"]
    diff = _diff(before, after)
    sites = _sites(before, after, diff)
    ffn_site = sites.get("ffn_down_1_4096", {})
    doc["ffn_diff"] = diff
    doc["ffn_site"] = ffn_site
    doc["ffn_contract"] = {
      "net_program_delta": diff["net_program_delta"],
      "zero_weight_materializations": not diff["weight_store_before"] and not diff["weight_store_after"],
      "after_bodies": ffn_site.get("after_bodies"),
      "body_delta": ffn_site.get("body_delta"),
      "explained_gemv_shift": _gemv_shift_explanation(diff),
    }
  out = pathlib.Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
  print(json.dumps({k: captured[k]["reduce_output"] for k in arms}, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
