#!/usr/bin/env python3
"""Exact full-shape qualifier for compiler-emitted wide Q6_K Stream-K."""
from __future__ import annotations

import argparse, json, pathlib, re, statistics
import numpy as np

from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.ops_nv import NVProgram
from extra.llm_research.layout import GGML_Q6_K, packed_u16_slice, read_metadata
from extra.llm_research.prefill.nv_compiler_q6k_imma_gate import _record, _run, _sass
from extra.llm_research.prefill.nv_compiler_q6k_streamk_transform import (
  transform_compiler_q6k_wide_live_publication, transform_compiler_q6k_wide_persistent_b,
  transform_compiler_q6k_wide_pair_reuse, transform_compiler_q6k_wide_straightline_k256,
  transform_compiler_q6k_wide_to_streamk, wide_active_fixup_source, wide_pair_alignment_proof)

M, N, K = 512, 4096, 12288
OUTPUT_TILES, TILE_ELEMENTS, MAX_SEGMENTS = 128, 16384, 3


def _buf(t: Tensor): return t.uop.buffer.get_buf("NV")
def _stats(xs: list[float]): return {"samples_us": xs, "min_us": min(xs), "median_us": statistics.median(xs), "max_us": max(xs)}


def _sass_phase_counts(record: dict[str, object]) -> dict[str, int]:
  text = pathlib.Path(str(record["sass"])).read_text()
  ops = re.findall(r"/\*[0-9a-f]+\*/\s+(?:@[!A-Z0-9.]+\s+)?([A-Z][A-Z0-9.]*)", text)
  families = [op.split(".", 1)[0] for op in ops]
  counts = {name: families.count(name) for name in ("LDG", "IMAD", "IADD", "LEA", "SHF", "LOP3", "MOV")}
  counts["address_ops"] = sum(counts[name] for name in ("IMAD", "IADD", "LEA", "SHF"))
  return counts


def main() -> int:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="/home/ubuntu/models/Qwen3-8B-Q4_K_M.gguf")
  ap.add_argument("--rounds", type=int, default=9)
  ap.add_argument("--owners", type=int, default=170)
  ap.add_argument("--force-partials", action="store_true")
  ap.add_argument("--persistent-q6-cache", action="store_true")
  ap.add_argument("--live-publication", action="store_true")
  ap.add_argument("--straightline-k256", action="store_true")
  ap.add_argument("--pair-reuse", choices=("control", "metadata", "high", "address", "word_pair"))
  ap.add_argument("--alternating-legacy-unroll2", action="store_true")
  ap.add_argument("--unroll", type=int, choices=(1, 2, 4, 8), default=2)
  ap.add_argument("--out", required=True)
  ap.add_argument("--artifacts", required=True)
  args = ap.parse_args()
  if args.rounds < 9: raise ValueError("qualification requires R9 or greater")
  if args.alternating_legacy_unroll2 and args.pair_reuse != "control":
    raise ValueError("alternating legacy comparison requires --pair-reuse control")
  artifacts = pathlib.Path(args.artifacts); artifacts.mkdir(parents=True, exist_ok=True)

  model = pathlib.Path(args.model); meta = read_metadata(model)
  info = next(i for i in meta.infos if i.name == "blk.0.ffn_down.weight")
  if info.typ != GGML_Q6_K: raise RuntimeError(f"illegal fixture {info}")
  halfs = packed_u16_slice(model, meta, info, device="NV").contiguous().realize()
  record = Tensor(_record(M, K)[0], device="NV").contiguous().realize()

  direct = _run("wide_direct", M, N, K, halfs, record, args.rounds, artifacts, (128, 128, 2, 4, 256))
  source = (artifacts / "wide_direct.cu").read_text()
  owners = args.owners
  main_source = transform_compiler_q6k_wide_to_streamk(source, owners=owners, force_partials=args.force_partials,
    unroll=None if args.straightline_k256 or args.pair_reuse else args.unroll)
  if args.straightline_k256: main_source = transform_compiler_q6k_wide_straightline_k256(main_source)
  if args.pair_reuse: main_source = transform_compiler_q6k_wide_pair_reuse(main_source, args.pair_reuse, owners=owners)
  if args.persistent_q6_cache: main_source = transform_compiler_q6k_wide_persistent_b(main_source)
  if args.live_publication: main_source = transform_compiler_q6k_wide_live_publication(main_source)
  fixup_source = wide_active_fixup_source()
  (artifacts / "main.cu").write_text(main_source); (artifacts / "fixup.cu").write_text(fixup_source)
  compiler = Device["NV"].compiler
  main_binary, fixup_binary = compiler.compile(main_source), compiler.compile(fixup_source)
  main_sass, fixup_sass = _sass(main_binary, artifacts / "main"), _sass(fixup_binary, artifacts / "fixup")
  main_sass["phase_counts"] = _sass_phase_counts(main_sass)
  fixup_sass["phase_counts"] = _sass_phase_counts(fixup_sass)

  output = Tensor.full((M, N), float("nan"), device="NV").contiguous().realize()
  partials = Tensor.full((2 * owners * TILE_ELEMENTS,), float("nan"), device="NV").contiguous().realize()
  partial_ids = Tensor.full((2 * owners,), -1, dtype=dtypes.int32, device="NV").contiguous().realize()
  main_program = NVProgram(Device["NV"], "q6k_imma_stream", main_binary)
  legacy_program = None; legacy_sass = None
  if args.alternating_legacy_unroll2:
    legacy_source = transform_compiler_q6k_wide_to_streamk(source, owners=owners, force_partials=args.force_partials,
      unroll=2, aligned_pair=False)
    (artifacts / "legacy_main.cu").write_text(legacy_source)
    legacy_binary = compiler.compile(legacy_source)
    legacy_sass = _sass(legacy_binary, artifacts / "legacy_main")
    legacy_sass["phase_counts"] = _sass_phase_counts(legacy_sass)
    legacy_program = NVProgram(Device["NV"], "q6k_imma_stream", legacy_binary)
  main_program(_buf(output), _buf(partials), _buf(partial_ids), _buf(halfs), _buf(record),
               global_size=(owners, 1, 1), local_size=(32, 2, 4), wait=True)

  slots: list[list[int]] = [[] for _ in range(OUTPUT_TILES)]
  for slot, tile in enumerate(partial_ids.numpy()):
    if tile >= 0: slots[int(tile)].append(slot)
  if not all(1 <= len(x) <= MAX_SEGMENTS for x in slots):
    raise RuntimeError(f"illegal segment census {[len(x) for x in slots]}")
  slot_map = np.full((OUTPUT_TILES, MAX_SEGMENTS), -1, np.int32)
  for tile, tile_slots in enumerate(slots): slot_map[tile, :len(tile_slots)] = tile_slots
  slot_map_t = Tensor(slot_map.reshape(-1), device="NV").contiguous().realize()
  active = Tensor(np.arange(OUTPUT_TILES, dtype=np.int32), device="NV").contiguous().realize()
  fixup_program = NVProgram(Device["NV"], "q6k_imma_fixup_active", fixup_binary)
  fixup_program(_buf(output), _buf(partials), _buf(slot_map_t), _buf(active),
                global_size=(OUTPUT_TILES, 1, 1), local_size=(256, 1, 1), vals=(M, N), wait=True)

  direct_binary = compiler.compile(source)
  direct_name = re.search(r'__global__ void __launch_bounds__\(256\) (\w+)\(', source)
  if direct_name is None: raise RuntimeError("wide direct symbol not found")
  direct_program = NVProgram(Device["NV"], direct_name.group(1), direct_binary)
  reference = Tensor.full((M, N), float("nan"), device="NV").contiguous().realize()
  direct_program(_buf(reference), _buf(record), _buf(halfs), global_size=(32, 4, 1), local_size=(32, 2, 4), wait=True)
  got, expected = output.numpy(), reference.numpy(); diff = np.abs(got - expected)
  raw_partials = partials.numpy().reshape(2 * owners, 128, 128)
  cpu_reduced = np.empty_like(expected)
  for tile, tile_slots in enumerate(slots):
    tm, tn = divmod(tile, 32)
    cpu_reduced[tm*128:(tm+1)*128, tn*128:(tn+1)*128] = sum((raw_partials[s] for s in tile_slots[1:]), raw_partials[tile_slots[0]].copy())
  cpu_fixup_diff, segmented_math_diff = np.abs(got - cpu_reduced), np.abs(cpu_reduced - expected)
  max_at = np.unravel_index(int(np.argmax(segmented_math_diff)), segmented_math_diff.shape)
  bad = segmented_math_diff > 1e-6

  legacy_correctness = None
  if legacy_program is not None:
    legacy_program(_buf(output), _buf(partials), _buf(partial_ids), _buf(halfs), _buf(record),
      global_size=(owners, 1, 1), local_size=(32, 2, 4), wait=True)
    fixup_program(_buf(output), _buf(partials), _buf(slot_map_t), _buf(active),
      global_size=(OUTPUT_TILES, 1, 1), local_size=(256, 1, 1), vals=(M, N), wait=True)
    legacy_got=output.numpy(); legacy_diff=np.abs(legacy_got-expected)
    legacy_correctness={"finite":bool(np.isfinite(legacy_got).all()),"max_abs":float(legacy_diff.max()),
      "allclose_rtol2e5_atol2e3":bool(np.allclose(legacy_got,expected,rtol=2e-5,atol=2e-3))}

  main_samples, legacy_samples, fixup_samples = [], [], []
  for round_idx in range(args.rounds):
    calls=(legacy_program,main_program) if round_idx%2 == 0 else (main_program,legacy_program)
    if legacy_program is None: calls=(main_program,)
    for program in calls:
      sample=program(_buf(output), _buf(partials), _buf(partial_ids), _buf(halfs), _buf(record),
        global_size=(owners, 1, 1), local_size=(32, 2, 4), wait=True) * 1e6
      (main_samples if program is main_program else legacy_samples).append(sample)
    fixup_samples.append(fixup_program(_buf(output), _buf(partials), _buf(slot_map_t), _buf(active),
      global_size=(OUTPUT_TILES, 1, 1), local_size=(256, 1, 1), vals=(M, N), wait=True) * 1e6)

  correctness = {"finite": bool(np.isfinite(got).all()), "max_abs": float(diff.max()),
    "mean_abs": float(diff.mean()), "allclose_rtol2e5_atol2e3": bool(np.allclose(got, expected, rtol=2e-5, atol=2e-3)),
    "cpu_fixup_max_abs": float(cpu_fixup_diff.max()), "segmented_math_max_abs": float(segmented_math_diff.max()),
    "segmented_max_at": list(map(int, max_at)), "segmented_max_values": [float(cpu_reduced[max_at]), float(expected[max_at])],
    "bad_values": int(bad.sum()), "bad_rows": np.flatnonzero(bad.any(axis=1)).tolist()[:32],
    "bad_columns": np.flatnonzero(bad.any(axis=0)).tolist()[:32],
    "nonzero_by_tile": [int(np.count_nonzero(raw_partials[x[0]])) for x in slots],
    "finite_by_tile": [int(np.isfinite(raw_partials[x[0]]).sum()) for x in slots],
    "sample_cpu_reduced": cpu_reduced[:2, :8].tolist(), "sample_direct": expected[:2, :8].tolist()}
  main_t, fixup_t = _stats(main_samples), _stats(fixup_samples)
  alternating_ab = None
  if legacy_program is not None:
    legacy_t=_stats(legacy_samples); recoveries=[control-candidate for control,candidate in zip(legacy_samples,main_samples)]
    alternating_ab={"legacy_unroll2":legacy_t,"candidate_aligned_pair":main_t,
      "paired_recovery_us":_stats(recoveries),
      "candidate_wins":sum(candidate<control for control,candidate in zip(legacy_samples,main_samples)),
      "rounds":args.rounds,"alternated_call_order":True,"legacy_correctness":legacy_correctness,"legacy_sass":legacy_sass}
  result = {"schema": "tinygrad.nv_compiler_q6k_wide_streamk.v1", "shape": {"M": M, "N": N, "K": K},
    "fixture": {"model": str(model), "weight": info.name, "format": "Q6_K"}, "owners": owners,
    "persistent_q6_cache": args.persistent_q6_cache,
    "live_publication": args.live_publication,
    "straightline_k256": args.straightline_k256,
    "pair_reuse": args.pair_reuse,
    "pair_alignment_proof": wide_pair_alignment_proof(owners) if args.pair_reuse else None,
    "alternating_ab": alternating_ab,
    "segment_census": {str(n): sum(len(x) == n for x in slots) for n in range(1, MAX_SEGMENTS + 1)},
    "correctness": correctness, "timing": {"main": main_t, "fixup": fixup_t,
      "pair_min_sum_us": main_t["min_us"] + fixup_t["min_us"],
      "pair_median_sum_us": main_t["median_us"] + fixup_t["median_us"]},
    "compiler": {"direct": direct, "main_sass": main_sass, "fixup_sass": fixup_sass,
      "llama_dependency": False, "canonical_packed_inputs": True}, "passed": False}
  result["passed"] = bool(correctness["finite"] and correctness["allclose_rtol2e5_atol2e3"] and
                          main_sass["imma"] >= 64 and main_sass["local_load"] == main_sass["local_store"] == 0)
  out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, sort_keys=True))
  return 0 if result["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
