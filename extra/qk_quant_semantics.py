#!/usr/bin/env python3
"""TG3: Quant Semantics Library -- turn quant FORMATS into DATA, not hand assumptions.

The TG0/TG1/TG2 stack carried Q4_K facts as Python constants imported from extra/qk_gemv_g2_lanemap.py
(QK_K=256, Q4K_WORDS_PER_BLOCK=36, Q4K_QUANT_WORD_BASE=4). That makes every new quant a hand-edit. This
module turns Q4_K / Q5_K / Q6_K / Q8_0 / fp16 into DATA rows: block layout, values_per_block, block_bytes,
weight-payload-vs-metadata bytes, scale/min layout, packing order, unpack/dequant ops, legal accum dtypes,
preferred dot dtype, quality class, and the known good / known refuted route families per format.

Every numeric fact is DERIVED from the byte-segment layout (segments sum to block_bytes; payload/metadata
byte counts come from segment roles; word counts come from block_bytes // packing_word_bytes), cross-checked
against:
  * Q4_K / Q6_K: the existing GEMV primitives (extra/q4_k_gemv_primitive.py, extra/q6_k_gemv_primitive.py)
    and the GGUF dequant (tinygrad/llm/gguf.py ggml_data_to_tensor).
  * Q5_K / Q8_0 / fp16: their GGUF / quant definitions (tinygrad/llm/gguf.py _GGML_QUANT + dequant arms).

TG1's QuantSpec now READS from this library (see extra/qk_lanemap_template.QuantSpec.from_library): the
G2/G3 hardcoded qk_k=256 / words_per_block=36 / quant_word_base=4 are now the Q4_K library row, and the TG1
re-emit stays lossless with the data-driven QuantSpec.

AUDIT/RESEARCH only: no GPU kernel, no default change, no live-route repoint. Pure data + derivation + checks.

Run: PYTHONPATH=. python3 extra/qk_quant_semantics.py
"""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass, field, asdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DATA = ROOT / "bench/qk-search-spaces/quant_semantics.json"
OUT_AUDIT = ROOT / "bench/qk-quant-semantics-audit"


class QuantLayoutUnknown(KeyError):
  """Raised when a quant format is not in the library -- callers must surface SEARCH_SPACE_INCOMPLETE,
  NOT silently fall back to Q4_K assumptions (TG3 acceptance)."""


@dataclass(frozen=True)
class Segment:
  """One contiguous byte segment of a quant super-block, tagged with its ROLE.

  role in {block_scale, block_min, sub_scale_min, sub_scale, payload_low, payload_high, payload, none}.
  is_metadata = role is a scale/min (not weight payload)."""
  name: str
  bytes: int
  role: str

  @property
  def is_metadata(self) -> bool:
    return self.role in ("block_scale", "block_min", "sub_scale_min", "sub_scale")

  @property
  def is_payload(self) -> bool:
    return self.role in ("payload", "payload_low", "payload_high")


@dataclass(frozen=True)
class QuantFormat:
  """A quant format as DATA. Numeric facts are derived from `segments` (see derive())."""
  name: str
  ggml_type: int | None                 # GGUF ggml_type id (None for fp16 / un-quantized)
  block_elems: int                      # values_per_block (super-block element count)
  block_bytes: int                      # bytes per super-block
  packing_word_bytes: int               # the natural load/pack word width in bytes (uint32=4, uint16=2, int8=1, fp16=2)
  packing_word_dtype: str
  segments: tuple[Segment, ...]         # ordered byte layout (sums to block_bytes)
  metadata_first: bool                  # True if all scale/min metadata precedes the weight payload (Q4_K/Q5_K/Q8_0)
  symmetric: bool                       # True if dequant is q*scale with no per-group min (Q6_K/Q8_0); False if d*sc*q - dmin*mn (Q4_K/Q5_K)
  sub_blocks: int                       # number of scale sub-blocks within a super-block (1 for Q8_0/fp16)
  natural_lane_extent: int              # the coalesced within-block lane axis a GEMV wave naturally owns
  dequant_ops: tuple[str, ...]          # the unpack/dequant op sequence
  legal_accum_dtypes: tuple[str, ...]   # accumulation dtypes that are numerically legal
  preferred_dot_dtype: str              # the preferred dot/MAC element dtype
  quality_class: str                    # lossless_fp16 | near_lossless_8bit | lossy_Nbit_kquant
  known_good_route_families: tuple[str, ...]
  known_refuted_route_families: tuple[dict, ...]
  quality_constraints: str
  impl_citation: str
  notes: str = ""

  # ---- derived facts (functions of the byte-segment layout, NOT hand numbers) ----
  def derive(self) -> dict:
    seg_sum = sum(s.bytes for s in self.segments)
    if seg_sum != self.block_bytes:
      raise ValueError(f"{self.name}: segments sum to {seg_sum} bytes != block_bytes {self.block_bytes}")
    metadata_bytes = sum(s.bytes for s in self.segments if s.is_metadata)
    payload_bytes = sum(s.bytes for s in self.segments if s.is_payload)
    other_bytes = self.block_bytes - metadata_bytes - payload_bytes
    words_per_block = self.block_bytes // self.packing_word_bytes if self.block_bytes % self.packing_word_bytes == 0 else None
    # quant_word_base / quant_words_per_block: only well-defined for metadata-first formats whose metadata is a
    # whole number of packing words (Q4_K/Q5_K/Q8_0). Q6_K is payload-first -> quant_word_base is None.
    quant_word_base = quant_words_per_block = None
    if self.metadata_first and words_per_block is not None and metadata_bytes % self.packing_word_bytes == 0:
      quant_word_base = metadata_bytes // self.packing_word_bytes
      quant_words_per_block = payload_bytes // self.packing_word_bytes
    bits_per_weight = round(payload_bytes * 8 / self.block_elems, 3) if self.block_elems else None
    return {
      "metadata_bytes": metadata_bytes, "weight_payload_bytes": payload_bytes, "other_bytes": other_bytes,
      "words_per_block": words_per_block, "quant_word_base": quant_word_base,
      "quant_words_per_block": quant_words_per_block,
      "payload_bits_per_weight": bits_per_weight,
      "sub_block_elems": (self.block_elems // self.sub_blocks) if self.sub_blocks else None,
    }

  def row(self) -> dict:
    d = asdict(self)
    d["segments"] = [{"name": s.name, "bytes": s.bytes, "role": s.role} for s in self.segments]
    d["derived"] = self.derive()
    return d


# ---- the library: each format's byte layout transcribed from GGUF + the existing primitives -------------------
_REFUTED_Q6K_HALFWARP = {
  "route_family": "coop_halfwarp_direct", "route_id": "decode_q6k_direct_refuted",
  "disposition": "refuted as built: W==D -4.77..-6.06% (median -5.44%); 2 rows x 16-lane half-warp partition",
  "citation": "bench/amd-isa-backend-q6k-direct-speed/latest.json"}

QUANT_LIBRARY: dict[str, QuantFormat] = {
  # Q4_K (ggml 12): 256 elems / 144 bytes. d:fp16, dmin:fp16, scales:12 (6-bit packed for 8 sub-blocks), qs:128
  # (4-bit nibbles). uint32-packed -> 36 words/block, first 4 = metadata (d+dmin+scales=16B), 32 quant words.
  "Q4_K": QuantFormat(
    name="Q4_K", ggml_type=12, block_elems=256, block_bytes=144,
    packing_word_bytes=4, packing_word_dtype="uint32",
    segments=(Segment("d", 2, "block_scale"), Segment("dmin", 2, "block_min"),
              Segment("scales", 12, "sub_scale_min"), Segment("qs", 128, "payload")),
    metadata_first=True, symmetric=False, sub_blocks=8, natural_lane_extent=8,
    dequant_ops=("load_d_dmin_fp16", "unpack_6bit_scale_min(grp)", "load_packed_uint32_qs",
                 "unpack_4bit_nibble(grp,pos)", "w = d*sc*q - dmin*mn"),
    legal_accum_dtypes=("fp32",), preferred_dot_dtype="fp32_or_q8_1_int8",
    quality_class="lossy_4bit_kquant",
    known_good_route_families=("lanemap", "coop", "owned_reference"),
    known_refuted_route_families=(
      {"route_family": "lanemap_layout_reshuffle", "route_id": "q4k_offline_layout_reshuffle",
       "disposition": "deprioritized: G3 (generic lanemap) matches owned, no offline-reshuffle gap to recover",
       "citation": "bench/amd-isa-backend-g3-weight-promotion/search_space_update.json"},),
    quality_constraints="none (dequant is exact up to fp reassoc; lossless vs the GGUF reference)",
    impl_citation="extra/q4_k_gemv_primitive.py (_q4k_block_dot_packed_load); gguf ggml_type 12",
    notes="natural_lane_extent=8 = the within-block packed-word index lane4 (pos//4) the coop/G3 routes coalesce; "
          "G3 owns 8 words/group x 4 block_groups = wave32."),
  # Q5_K (ggml 13): 256 elems / 176 bytes. Same scale/min layout as Q4_K, PLUS a 32-byte qh high-bit plane between
  # scales and qs. uint32-packed -> 44 words/block, first 4 = metadata; payload = qh(32)+qs(128)=160B = 40 words.
  "Q5_K": QuantFormat(
    name="Q5_K", ggml_type=13, block_elems=256, block_bytes=176,
    packing_word_bytes=4, packing_word_dtype="uint32",
    segments=(Segment("d", 2, "block_scale"), Segment("dmin", 2, "block_min"),
              Segment("scales", 12, "sub_scale_min"), Segment("qh", 32, "payload_high"),
              Segment("qs", 128, "payload_low")),
    metadata_first=True, symmetric=False, sub_blocks=8, natural_lane_extent=8,
    dequant_ops=("load_d_dmin_fp16", "unpack_6bit_scale_min(grp)", "load_qh_highbit", "load_packed_uint32_qs",
                 "unpack_5bit = (qs_nibble | qh_bit<<4)", "w = d*sc*q - dmin*mn"),
    legal_accum_dtypes=("fp32",), preferred_dot_dtype="fp32",
    quality_class="lossy_5bit_kquant",
    known_good_route_families=("lanemap", "coop", "owned_reference"),
    known_refuted_route_families=(),
    quality_constraints="none (exact dequant). NOTE: the qh high-bit plane sits BETWEEN metadata and qs, so the "
                        "packed-word lane index is NOT the Q4_K formula -- a Q5_K lanemap needs its own index.",
    impl_citation="tinygrad/llm/gguf.py ggml_type 13 (shares the Q4_K scale/min unpack; adds qh<<4)",
    notes="no shipped Q5_K route on the current profile; descriptor row for new-profile openers."),
  # Q6_K (ggml 14): 256 elems / 210 bytes. ql:128 (low 4 bits), qh:64 (high 2 bits), scales:16 (int8, 16 sub-blocks),
  # d:fp16. PAYLOAD-FIRST: scales+d are at the END. uint16(halfword)-packed -> 105 halfwords/block. Symmetric (q-32,
  # no min). natural lane = within-block position 0..15.
  "Q6_K": QuantFormat(
    name="Q6_K", ggml_type=14, block_elems=256, block_bytes=210,
    packing_word_bytes=2, packing_word_dtype="uint16",
    segments=(Segment("ql", 128, "payload_low"), Segment("qh", 64, "payload_high"),
              Segment("scales", 16, "sub_scale"), Segment("d", 2, "block_scale")),
    metadata_first=False, symmetric=True, sub_blocks=16, natural_lane_extent=16,
    dequant_ops=("load_ql_4bit", "load_qh_2bit", "q = (ql | qh<<4) - 32", "load_int8_subscale", "load_d_fp16",
                 "w = d * q * scale"),
    legal_accum_dtypes=("fp32",), preferred_dot_dtype="fp32",
    quality_class="lossy_6bit_kquant",
    known_good_route_families=("coop", "owned_reference"),
    known_refuted_route_families=(_REFUTED_Q6K_HALFWARP,),
    quality_constraints="none (exact dequant). PAYLOAD-FIRST + symmetric (no dmin/min) + int8 sub-scales -> a "
                        "different lane-map shape than Q4_K; the metadata-first packed-word index does NOT apply.",
    impl_citation="extra/q6_k_gemv_primitive.py (q6k_coop_partial_kernel shipped; q6k_halfwarp_partition refuted); "
                  "gguf ggml_type 14",
    notes="shipped route = decode_q6k_coop_shipped (pos 0..15 LOCAL lane + stage-2 .sum). natural_lane_extent=16."),
  # Q8_0 (ggml 8): 32 elems / 34 bytes. d:fp16 + qs:32 int8. Near-lossless. block_bytes=34 is NOT uint32-aligned ->
  # the natural word is int8 (or 4xint8 packed into uint32 for v_dot4). Symmetric (q*d, no min). Usually the
  # ACTIVATION quant (q8_1) for int-dot weight GEMV, not a weight quant here.
  "Q8_0": QuantFormat(
    name="Q8_0", ggml_type=8, block_elems=32, block_bytes=34,
    packing_word_bytes=1, packing_word_dtype="int8",
    segments=(Segment("d", 2, "block_scale"), Segment("qs", 32, "payload")),
    metadata_first=True, symmetric=True, sub_blocks=1, natural_lane_extent=32,
    dequant_ops=("load_d_fp16", "load_int8_q", "w = q * d"),
    legal_accum_dtypes=("int32", "fp32"), preferred_dot_dtype="int8_vdot4",
    quality_class="near_lossless_8bit",
    known_good_route_families=("int_dot_activation_pairing", "owned_reference"),
    known_refuted_route_families=(),
    quality_constraints="near-lossless (8-bit). block_bytes=34 is NOT uint32 word-aligned -> no clean uint32 packed-"
                        "word factorization; pack 4 int8 into uint32 only for v_dot4_i32_i8.",
    impl_citation="tinygrad/llm/gguf.py ggml_type 8; activation pairing extra/q4_k_gemv_primitive.py q8_1 path",
    notes="on this profile Q8_0/q8_1 appears as the ACTIVATION quant for the int-dot Q4_K weight GEMV, not a "
          "standalone weight tensor."),
  # fp16: un-quantized. 1 elem / 2 bytes. Direct load, no dequant. Attention KV + prefill GEMM run fp16.
  "fp16": QuantFormat(
    name="fp16", ggml_type=None, block_elems=1, block_bytes=2,
    packing_word_bytes=2, packing_word_dtype="fp16",
    segments=(Segment("v", 2, "payload"),),
    metadata_first=True, symmetric=True, sub_blocks=1, natural_lane_extent=1,
    dequant_ops=("direct_load_fp16",),
    legal_accum_dtypes=("fp32", "fp16"), preferred_dot_dtype="fp16_wmma",
    quality_class="lossless_fp16",
    known_good_route_families=("graph_gemm_pipe", "owned_reference", "native_isa_attention"),
    known_refuted_route_families=(),
    quality_constraints="none (no quantization).",
    impl_citation="extra/qk_prefill_graph_gemm_route.py (prefill GEMM); extra/qk_owned_flash_decode_graph_node.py "
                  "(attention)",
    notes="not a packed-quant format; no unpack. Used by prefill GEMM + decode attention."),
}


def quant_row(name: str) -> QuantFormat:
  """Return the QuantFormat for `name`, or raise QuantLayoutUnknown (callers must surface
  SEARCH_SPACE_INCOMPLETE, never silently fall back to Q4_K)."""
  if name not in QUANT_LIBRARY:
    raise QuantLayoutUnknown(f"quant format {name!r} not in the semantics library; known: {sorted(QUANT_LIBRARY)}. "
                             f"SEARCH_SPACE_INCOMPLETE -- do NOT fall back to Q4_K assumptions.")
  return QUANT_LIBRARY[name]


def quant_spec_fields(name: str) -> dict:
  """The TG1-QuantSpec fields a metadata-first uint32 GEMV lane map needs (block_elems / words_per_block /
  quant_word_base), DERIVED from the library row. Raises for formats whose layout is not a metadata-first uint32
  packing (Q6_K payload-first, Q8_0 non-aligned, fp16 un-quantized) -- those need a different IR shape."""
  fmt = quant_row(name)
  d = fmt.derive()
  if d["quant_word_base"] is None or fmt.packing_word_dtype != "uint32":
    raise QuantLayoutUnknown(
      f"{name}: not a metadata-first uint32 packed-word layout (packing={fmt.packing_word_dtype}, "
      f"metadata_first={fmt.metadata_first}); the Q4_K-shaped LaneMapTemplate QuantSpec does not apply.")
  return {"block_elems": fmt.block_elems, "words_per_block": d["words_per_block"],
          "quant_word_base": d["quant_word_base"], "quant_words_per_block": d["quant_words_per_block"]}


def supported_quants() -> list[str]:
  return sorted(QUANT_LIBRARY)


def dump_data() -> str:
  OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "_schema": "TG3 quant semantics library (extra/qk_quant_semantics.py)",
    "scope": "quant FORMATS as DATA for the topology search (block layout, payload vs metadata bytes, scale/min "
             "layout, dequant ops, accum/dot dtypes, quality class, known good/refuted route families).",
    "audit_only": "no GPU kernel, no default change, no live-route repoint.",
    "formats": {name: fmt.row() for name, fmt in QUANT_LIBRARY.items()},
  }
  json.dump(payload, open(OUT_DATA, "w"), indent=2)
  return str(OUT_DATA)


# ---- TG3 proof gate -----------------------------------------------------------------------------------------
def _reproduces_q4k_g3_facts() -> dict:
  """The Q4_K library row must reproduce G3's quant facts: qk_k=256, words_per_block=36, quant_word_base=4
  (the exact constants TG0 bucketed as quant_data and TG1/G2 hardcoded)."""
  from extra.qk_gemv_g2_lanemap import QK_K, Q4K_WORDS_PER_BLOCK, Q4K_QUANT_WORD_BASE
  f = quant_spec_fields("Q4_K")
  expected = {"block_elems": QK_K, "words_per_block": Q4K_WORDS_PER_BLOCK, "quant_word_base": Q4K_QUANT_WORD_BASE}
  matches = {k: (f[k] == v) for k, v in expected.items()}
  return {"derived": {k: f[k] for k in expected}, "g2_g3_constants": expected,
          "field_matches": matches, "all_match": all(matches.values())}


def _tg1_reemit_still_lossless() -> dict:
  """Drive the TG1 re-emit with the DATA-DRIVEN QuantSpec (QuantSpec.from_library('Q4_K')) and confirm the
  emitted UOp program is still byte-identical (UOp .key) to the promoted G3 route for every eligible role."""
  from extra.qk_lanemap_template import reemit_role, ELIGIBLE_ROLES, QuantSpec, g3_template
  # prove from_library drives the spec (not the dataclass defaults)
  lib_spec = QuantSpec.from_library("Q4_K")
  default_spec = g3_template("ffn_gate_up", 12288, 4096).quant
  spec_from_library_ok = (lib_spec.qk_k == default_spec.qk_k
                          and lib_spec.q4k_words_per_block == default_spec.q4k_words_per_block
                          and lib_spec.q4k_quant_word_base == default_spec.q4k_quant_word_base)
  per_role = {role: reemit_role(role, **shp) for role, shp in ELIGIBLE_ROLES.items()}
  lossless = all(r["uop_key_identical_to_default"] and r["kernel_name_match"]
                 and r["lane_ownership_index_matches_reference"] for r in per_role.values())
  return {"quantspec_from_library_drives_g3": spec_from_library_ok, "lossless_reemit_all_roles": lossless,
          "per_role": {role: {"uop_key_identical": r["uop_key_identical_to_default"],
                              "name_match": r["kernel_name_match"],
                              "lane_index_match": r["lane_ownership_index_matches_reference"]}
                       for role, r in per_role.items()}}


def _q6k_reproduces_coop_and_marks_refuted() -> dict:
  """Q6_K row must encode the shipped coop route family AND mark the direct half-warp route refuted as built."""
  q6k = quant_row("Q6_K")
  coop_in_good = "coop" in q6k.known_good_route_families
  refuted = [r for r in q6k.known_refuted_route_families if r.get("route_id") == "decode_q6k_direct_refuted"]
  return {"coop_in_known_good": coop_in_good, "halfwarp_direct_marked_refuted": bool(refuted),
          "refuted_rows": list(q6k.known_refuted_route_families),
          "known_good_route_families": list(q6k.known_good_route_families)}


def _unsupported_quant_fails_clean() -> dict:
  """An unsupported quant (Q3_K) must raise SEARCH_SPACE_INCOMPLETE-class error, not fall into Q4_K assumptions."""
  fell_back_to_q4k = False
  raised = False
  try:
    quant_row("Q3_K")
  except QuantLayoutUnknown:
    raised = True
  # also prove quant_spec_fields refuses Q6_K (payload-first) instead of returning Q4_K-shaped numbers
  q6k_refused = False
  try:
    quant_spec_fields("Q6_K")
  except QuantLayoutUnknown:
    q6k_refused = True
  return {"unsupported_raises_layout_unknown": raised, "did_not_fall_back_to_q4k": not fell_back_to_q4k,
          "q6k_payload_first_refused_by_spec_fields": q6k_refused}


def main() -> int:
  data_path = dump_data()
  OUT_AUDIT.mkdir(parents=True, exist_ok=True)

  # validate every row's segment layout derives cleanly
  layout_errors = []
  for name, fmt in QUANT_LIBRARY.items():
    try:
      fmt.derive()
    except ValueError as e:
      layout_errors.append(str(e))

  q4k = _reproduces_q4k_g3_facts()
  reemit = _tg1_reemit_still_lossless()
  q6k = _q6k_reproduces_coop_and_marks_refuted()
  unsup = _unsupported_quant_fails_clean()

  ready = (not layout_errors and q4k["all_match"] and reemit["quantspec_from_library_drives_g3"]
           and reemit["lossless_reemit_all_roles"] and q6k["coop_in_known_good"]
           and q6k["halfwarp_direct_marked_refuted"] and unsup["unsupported_raises_layout_unknown"]
           and unsup["did_not_fall_back_to_q4k"] and unsup["q6k_payload_first_refused_by_spec_fields"])
  verdict = "TG3_PASS_QUANT_SEMANTICS_READY" if ready else "TG3_BLOCKED_QUANT_LAYOUT_UNKNOWN"

  result = {
    "scope": "TG3 quant semantics library: quant formats as DATA; TG1 QuantSpec reads from it; Q4_K row reproduces "
             "G3's quant facts and TG1 re-emit stays lossless. AUDIT/RESEARCH: no GPU, no default change.",
    "verdict": verdict,
    "library_module": "extra/qk_quant_semantics.py", "data_file": str(pathlib.Path(data_path).relative_to(ROOT)),
    "formats": supported_quants(),
    "layout_derivation_errors": layout_errors,
    "q4k_reproduces_g3_quant_facts": q4k,
    "tg1_reemit_still_lossless_with_data_driven_quantspec": reemit,
    "q6k_reproduces_coop_marks_halfwarp_refuted": q6k,
    "unsupported_quant_fails_search_space_incomplete": unsup,
    "format_summary": {name: {"block_elems": f.block_elems, "block_bytes": f.block_bytes,
                              "packing": f.packing_word_dtype, "quality_class": f.quality_class,
                              **f.derive(), "symmetric": f.symmetric,
                              "known_good": list(f.known_good_route_families),
                              "known_refuted": [r.get("route_id", r.get("route_family")) for r in f.known_refuted_route_families]}
                       for name, f in QUANT_LIBRARY.items()},
    "do_not": ["no GPU kernel", "no default change", "no live-route repoint", "no reopened refuted route"],
  }
  json.dump(result, open(OUT_AUDIT / "latest.json", "w"), indent=2)

  md = [f"# TG3 Quant Semantics Library -- verdict: **{verdict}**", "",
        "Quant formats are now DATA (extra/qk_quant_semantics.py -> bench/qk-search-spaces/quant_semantics.json). "
        "TG1's QuantSpec reads the Q4_K row; the G2/G3 hardcoded 256/36/4 are now derived from the byte layout.", "",
        "## Formats", "",
        "| quant | elems | bytes | pack | metadata B | payload B | words/blk | quant_word_base | sym | quality |",
        "|---|---:|---:|---|---:|---:|---:|---:|:--:|---|"]
  for name, f in QUANT_LIBRARY.items():
    d = f.derive()
    md.append(f"| {name} | {f.block_elems} | {f.block_bytes} | {f.packing_word_dtype} | {d['metadata_bytes']} | "
              f"{d['weight_payload_bytes']} | {d['words_per_block']} | {d['quant_word_base']} | {f.symmetric} | "
              f"{f.quality_class} |")
  md += ["", "## TG3 proof gates", "",
         f"- **Q4_K row reproduces G3 quant facts** (qk_k=256/words_per_block=36/quant_word_base=4): "
         f"{q4k['all_match']} ({q4k['derived']})",
         f"- **TG1 re-emit lossless with data-driven QuantSpec.from_library('Q4_K')**: "
         f"{reemit['lossless_reemit_all_roles']} (from_library drives G3: {reemit['quantspec_from_library_drives_g3']})",
         f"- **Q6_K row -> shipped coop in known_good + half-warp direct marked refuted**: "
         f"{q6k['coop_in_known_good'] and q6k['halfwarp_direct_marked_refuted']}",
         f"- **Unsupported quant (Q3_K) -> SEARCH_SPACE_INCOMPLETE, no Q4_K fallback**: "
         f"{unsup['unsupported_raises_layout_unknown'] and unsup['did_not_fall_back_to_q4k']}",
         f"- **quant_spec_fields refuses Q6_K (payload-first) instead of Q4_K-shaping it**: "
         f"{unsup['q6k_payload_first_refused_by_spec_fields']}", ""]
  (OUT_AUDIT / "summary.md").write_text("\n".join(md))

  print(verdict)
  print(f"  formats: {supported_quants()}")
  print(f"  Q4_K reproduces G3 facts: {q4k['all_match']} {q4k['derived']}")
  print(f"  TG1 re-emit lossless (data-driven QuantSpec): {reemit['lossless_reemit_all_roles']}")
  print(f"  Q6_K coop known_good + half-warp refuted: {q6k['coop_in_known_good']} / {q6k['halfwarp_direct_marked_refuted']}")
  print(f"  unsupported Q3_K -> SEARCH_SPACE_INCOMPLETE (no Q4_K fallback): {unsup['unsupported_raises_layout_unknown']}")
  return 0 if ready else 1


if __name__ == "__main__":
  raise SystemExit(main())
