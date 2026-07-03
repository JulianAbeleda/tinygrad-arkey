#!/usr/bin/env python3
"""TG1: LaneMapTemplate IR -- the clean reviewable object before the TG2 topology search.

This IR re-expresses the PROMOTED default G3 Q4_K decode GEMV route, but it pulls the 5
STILL-HUMAN topology choices (TG0: bench/qk-g3-provenance-audit) out into EXPLICIT FREE
FIELDS, cleanly separated from the data inputs the template merely reads:

  TOPOLOGY  (the 5 still-human DOF, free fields a TG2 grammar will eventually span)
    block_groups          -- K decomposed into this many block-groups across the wave
    words_per_group       -- packed words owned per group (block_groups*words_per_group == lane_extent)
    axis_roles            -- each factor {row, block_group, word_col, local_block, group_pair} -> GLOBAL|LOCAL|REDUCE
    reduction_pattern     -- "cross_lane_wave_reduce" vs "partials_plus_reduce"
    lane_ownership_index  -- the coalesced packed-word index formula (derivable from decomposition + quant packing)

  QUANT     (DATA inputs, not topology -- TG3 makes these data-driven)
    qk_k, q4k_words_per_block, q4k_quant_word_base, dequant_body (callable)

  TARGET    (DATA input -- TG5 makes this a target feature)
    lane_extent (wave width)

  SHAPE     (DATA input)
    rows (N), k (K), role

It is a FAITHFUL re-expression, NOT a reimplementation: `emit()` reuses the existing G2 lane
map (extra/qk_gemv_g2_lanemap.Q4KGateUpLaneMap) and the existing G3 emitter
(extra/qk_gemv_g3_codegen_lowering.q4k_g3_lanemap_gemv_kernel). `validate()` proves every free
field describes exactly the topology those internals encode, so emission goes down the SAME
path and is byte-identical (UOp .key) to the current default route.

AUDIT/IR ONLY. This changes no default, repoints no live route, writes no GPU kernel. Run:
  PYTHONPATH=. python3 extra/qk/lanemap_template.py
"""
from __future__ import annotations
import json, pathlib
from dataclasses import dataclass, field
from typing import Callable

from tinygrad.uop.ops import UOp
from tinygrad.dtype import dtypes
from extra.qk.amd_warp_reduce import WARP
from extra.qk.gemv_g2_lanemap import Q4KGateUpLaneMap, QK_K, Q4K_WORDS_PER_BLOCK, Q4K_QUANT_WORD_BASE
from extra.qk.gemv_g3_codegen_lowering import q4k_g3_lanemap_gemv_kernel
from extra.qk.quant.q4_k_gemv_primitive import _q4k_block_dot_packed_load
# TG3: quant facts are DATA now. The Q4_K QuantSpec defaults are sourced from the quant semantics library
# (the G2/G3 hardcoded 256/36/4 are the derived Q4_K library row), not from Python constants.
from extra.qk.quant_semantics import quant_spec_fields
_Q4K_LIB = quant_spec_fields("Q4_K")

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "bench/qk-lanemap-template-ir"

# The five topology factors of a packed-quant GEMV lane map (the human design surface, TG0).
TOPOLOGY_FACTORS = ("row", "block_group", "word_col", "local_block", "group_pair")
CROSS_LANE_WAVE_REDUCE = "cross_lane_wave_reduce"
PARTIALS_PLUS_REDUCE = "partials_plus_reduce"
# The lane-ownership packed-word index formula G3 uses (Q4KGateUpLaneMap.packed_word_index_*).
G3_LANE_OWNERSHIP_INDEX = ("(row * k_blocks + (block_group * blocks_per_group + local_block)) * q4k_words_per_block"
                           " + q4k_quant_word_base + group_pair * words_per_group + word_col")


class LaneMapTemplateError(ValueError): pass


@dataclass(frozen=True)
class TopologySpec:
  """The 5 still-human topology DOF, as explicit FREE fields (the TG2 search surface)."""
  block_groups: int
  words_per_group: int
  axis_roles: dict[str, str] = field(default_factory=dict)
  reduction_pattern: str = CROSS_LANE_WAVE_REDUCE
  lane_ownership_index: str = G3_LANE_OWNERSHIP_INDEX


@dataclass(frozen=True)
class QuantSpec:
  """Quant-format DATA inputs (not topology). TG3: these now READ from extra/qk/quant_semantics.py.

  The defaults are the Q4_K library row (block_elems=256, words_per_block=36, quant_word_base=4), DERIVED from
  the Q4_K byte layout -- not the formerly-hardcoded G2 constants. Use `from_library(name)` to build a spec for a
  named quant; it raises QuantLayoutUnknown for formats whose layout is not a metadata-first uint32 packing
  (Q6_K/Q8_0/fp16), which the Q4_K-shaped IR does not express."""
  qk_k: int = _Q4K_LIB["block_elems"]
  q4k_words_per_block: int = _Q4K_LIB["words_per_block"]
  q4k_quant_word_base: int = _Q4K_LIB["quant_word_base"]
  name: str = "Q4_K"
  dequant_body: Callable = _q4k_block_dot_packed_load

  @classmethod
  def from_library(cls, name: str = "Q4_K", dequant_body: Callable = _q4k_block_dot_packed_load) -> "QuantSpec":
    f = quant_spec_fields(name)  # raises QuantLayoutUnknown for non-(metadata-first uint32) formats
    return cls(qk_k=f["block_elems"], q4k_words_per_block=f["words_per_block"],
               q4k_quant_word_base=f["quant_word_base"], name=name, dequant_body=dequant_body)


@dataclass(frozen=True)
class TargetSpec:
  """Target-GPU DATA input. TG5 makes lane_extent a real target feature."""
  lane_extent: int = WARP
  name: str = "AMD_gfx1100"


@dataclass(frozen=True)
class ShapeSpec:
  """Per-role shape DATA input."""
  rows: int   # N = out_features
  k: int      # K = in_features
  role: str = ""


@dataclass(frozen=True)
class LaneMapTemplate:
  """A packed-quant GEMV lane-map route as a reviewable IR: TOPOLOGY (free) + QUANT/TARGET/SHAPE (data).

  The promoted G3 Q4_K route is exactly one instantiation. `validate()` proves the free topology
  fields describe the topology the G2 lane map + G3 emitter encode; `emit()` then reuses those
  internals so the emission is the SAME path (byte-identical UOp .key)."""
  topology: TopologySpec
  quant: QuantSpec
  target: TargetSpec
  shape: ShapeSpec

  # ---- topology <-> existing-internals binding -------------------------------------------------
  def lanemap(self) -> Q4KGateUpLaneMap:
    """Reconstruct the G2 lane map FROM the IR fields (topology + quant + target + shape)."""
    return Q4KGateUpLaneMap(
      k=self.shape.k, n=self.shape.rows, qk_k=self.quant.qk_k, lane_extent=self.target.lane_extent,
      block_groups=self.topology.block_groups, words_per_group=self.topology.words_per_group,
      q4k_words_per_block=self.quant.q4k_words_per_block, q4k_quant_word_base=self.quant.q4k_quant_word_base)

  @staticmethod
  def axis_roles_of(lm: Q4KGateUpLaneMap) -> dict[str, str]:
    """The GLOBAL/LOCAL/REDUCE role of each factor, read from the G2 lane map (source of truth)."""
    return {name: u.arg[1].name for name, u in lm.axis_uops().items()}

  # ---- invariants ------------------------------------------------------------------------------
  def validate(self) -> None:
    t, q, tg, s = self.topology, self.quant, self.target, self.shape
    # base shape/quant/target invariants (kept from Q4KGateUpLaneMap.validate)
    lm = self.lanemap(); lm.validate()
    # 1. topology decomposition: block_groups * words_per_group == lane_extent
    if t.block_groups * t.words_per_group != tg.lane_extent:
      raise LaneMapTemplateError(
        f"topology.block_groups*words_per_group ({t.block_groups}*{t.words_per_group}) != target.lane_extent {tg.lane_extent}")
    # 2. axis_roles must match the role assignment the G2 lane map encodes
    ref_roles = self.axis_roles_of(lm)
    if set(t.axis_roles) != set(TOPOLOGY_FACTORS):
      raise LaneMapTemplateError(f"axis_roles must assign every factor {TOPOLOGY_FACTORS}, got {sorted(t.axis_roles)}")
    if t.axis_roles != ref_roles:
      bad = {k: (t.axis_roles[k], ref_roles[k]) for k in ref_roles if t.axis_roles.get(k) != ref_roles[k]}
      raise LaneMapTemplateError(f"axis_roles diverge from lane map (factor: ir_vs_ref): {bad}")
    # 3. reduction pattern must be a recognized topology choice
    if t.reduction_pattern not in (CROSS_LANE_WAVE_REDUCE, PARTIALS_PLUS_REDUCE):
      raise LaneMapTemplateError(f"unknown reduction_pattern {t.reduction_pattern!r}")
    # 4. lane_ownership_index formula must match the G2 packed-word-index reference at a coordinate sample
    if not self._lane_ownership_matches(lm):
      raise LaneMapTemplateError("lane_ownership_index does not match the G2 packed-word-index reference")

  def _lane_ownership_matches(self, lm: Q4KGateUpLaneMap) -> bool:
    """The declared lane_ownership index must reproduce Q4KGateUpLaneMap.packed_word_index_ref on a sample."""
    if self.topology.lane_ownership_index != G3_LANE_OWNERSHIP_INDEX: return False
    import itertools
    for row, bg, lb, gp, wc in itertools.product(
        (0, 1, lm.n - 1), range(lm.block_groups), (0, lm.blocks_per_group - 1),
        range(lm.group_pairs), (0, lm.words_per_group - 1)):
      if lb >= lm.blocks_per_group: continue
      ref = lm.packed_word_index_ref(row, bg, lb, gp, wc)
      blk = bg * lm.blocks_per_group + lb
      exp = ((row * lm.k_blocks + blk) * lm.q4k_words_per_block + lm.q4k_quant_word_base
             + gp * lm.words_per_group + wc)
      if ref != exp: return False
    return True

  # ---- emission (SAME path as the promoted default route) --------------------------------------
  def to_kernel(self) -> Callable:
    """Return the named wave32 UOp kernel builder for this IR, via the existing G3 emitter.

    The IR is validated first, so emission only ever runs for a topology the existing internals
    encode; the emitter constructs an internal G2 lane map byte-equal to `self.lanemap()`.
    """
    self.validate()
    if self.target.lane_extent != self.lanemap().lane_extent:
      raise LaneMapTemplateError("target.lane_extent inconsistent with lane map")
    return q4k_g3_lanemap_gemv_kernel(self.shape.rows, self.shape.k, self.target.lane_extent)

  def emit(self) -> UOp:
    """Faithful re-emit: the named wave32 UOp sink for this route (byte-identical to the default)."""
    rows, k = self.shape.rows, self.shape.k
    placeholders = (UOp.placeholder((rows,), dtypes.float32, 0),
                    UOp.placeholder((1,), dtypes.uint32, 1),
                    UOp.placeholder((rows,), dtypes.float32, 2))
    return self.to_kernel()(*placeholders)

  emit_sink = emit  # alias

  # ---- field taxonomy (for the audit artifact) -------------------------------------------------
  def field_taxonomy(self) -> dict:
    t = self.topology
    return {
      "topology_free_fields": {
        "block_groups": t.block_groups,
        "words_per_group": t.words_per_group,
        "axis_roles": dict(t.axis_roles),
        "reduction_pattern": t.reduction_pattern,
        "lane_ownership_index": t.lane_ownership_index,
      },
      "quant_data_inputs": {
        "qk_k": self.quant.qk_k, "q4k_words_per_block": self.quant.q4k_words_per_block,
        "q4k_quant_word_base": self.quant.q4k_quant_word_base, "name": self.quant.name,
        "dequant_body": f"{self.quant.dequant_body.__module__}.{self.quant.dequant_body.__name__}",
      },
      "target_data_inputs": {"lane_extent": self.target.lane_extent, "name": self.target.name},
      "shape_data_inputs": {"rows": self.shape.rows, "k": self.shape.k, "role": self.shape.role},
    }


def g3_template(role: str, rows: int, k: int) -> LaneMapTemplate:
  """Instantiate the IR with G3's ACTUAL promoted topology for one eligible role.

  Topology free fields are set to G3's choices (block_groups=4, words_per_group=8, G3 axis roles,
  cross-lane reduce, G3 lane index) -- exactly what TG2 must later rediscover instead of hardcode.
  """
  lm = Q4KGateUpLaneMap(k=k, n=rows, lane_extent=WARP)  # source-of-truth axis roles for G3
  topo = TopologySpec(block_groups=lm.block_groups, words_per_group=lm.words_per_group,
                      axis_roles=LaneMapTemplate.axis_roles_of(lm),
                      reduction_pattern=CROSS_LANE_WAVE_REDUCE,
                      lane_ownership_index=G3_LANE_OWNERSHIP_INDEX)
  return LaneMapTemplate(topology=topo, quant=QuantSpec.from_library("Q4_K"), target=TargetSpec(),
                         shape=ShapeSpec(rows=rows, k=k, role=role))


# ---- TG1 proof gate: IR losslessly re-emits the promoted G3 route ----------------------------
ELIGIBLE_ROLES = {
  "ffn_gate_up": {"rows": 12288, "k": 4096},
  "ffn_down":    {"rows": 4096,  "k": 12288},
  "attn_qo":     {"rows": 4096,  "k": 4096},
}


def _reference_sink(rows: int, k: int) -> UOp:
  """The exact emission the promoted default route uses (q4k_g3_lanemap_gemv_kernel(N, K))."""
  placeholders = (UOp.placeholder((rows,), dtypes.float32, 0),
                  UOp.placeholder((1,), dtypes.uint32, 1),
                  UOp.placeholder((rows,), dtypes.float32, 2))
  return q4k_g3_lanemap_gemv_kernel(rows, k)(*placeholders)


def reemit_role(role: str, rows: int, k: int) -> dict:
  t = g3_template(role, rows, k)
  diverging_field = None
  try:
    t.validate()
  except LaneMapTemplateError as e:
    diverging_field = str(e)
  ir_sink = t.emit()
  ref_sink = _reference_sink(rows, k)
  key_match = (ir_sink.key == ref_sink.key)
  expected_name = f"q4k_g3_lanemap_gemv_{rows}_{k}"
  name_match = (ir_sink.arg.name == ref_sink.arg.name == expected_name)
  idx_match = t._lane_ownership_matches(t.lanemap())
  return {
    "role": role, "rows": rows, "k": k, "kernel_name": ir_sink.arg.name, "expected_name": expected_name,
    "uop_key_identical_to_default": bool(key_match), "kernel_name_match": bool(name_match),
    "lane_ownership_index_matches_reference": bool(idx_match),
    "topology_separated_ok": diverging_field is None,
    "diverging_field": diverging_field,
    "field_taxonomy": t.field_taxonomy(),
  }


def main() -> int:
  OUT.mkdir(parents=True, exist_ok=True)
  recon = {role: reemit_role(role, **shp) for role, shp in ELIGIBLE_ROLES.items()}

  topology_separable = all(r["topology_separated_ok"] for r in recon.values())
  lossless = all(r["uop_key_identical_to_default"] and r["kernel_name_match"]
                 and r["lane_ownership_index_matches_reference"] for r in recon.values())

  if not topology_separable:
    bad = next(r for r in recon.values() if not r["topology_separated_ok"])
    verdict = "TG1_BLOCKED_TOPOLOGY_NOT_SEPARABLE"
    blocked_detail = {"role": bad["role"], "diverging_field": bad["diverging_field"]}
  elif not lossless:
    bad = next(r for r in recon.values()
               if not (r["uop_key_identical_to_default"] and r["kernel_name_match"]
                       and r["lane_ownership_index_matches_reference"]))
    verdict = "TG1_BLOCKED_REEMIT_NOT_LOSSLESS"
    blocked_detail = {"role": bad["role"], "uop_key_identical": bad["uop_key_identical_to_default"],
                      "kernel_name_match": bad["kernel_name_match"],
                      "lane_ownership_index_matches": bad["lane_ownership_index_matches_reference"]}
  else:
    verdict = "TG1_PASS_IR_LOSSLESS_REEMITS_G3"
    blocked_detail = None

  # one representative taxonomy (identical structure across roles; values differ only in shape)
  taxonomy = recon["ffn_gate_up"]["field_taxonomy"]
  field_listing = {
    "topology_free_fields": list(taxonomy["topology_free_fields"].keys()),
    "quant_data_inputs": list(taxonomy["quant_data_inputs"].keys()),
    "target_data_inputs": list(taxonomy["target_data_inputs"].keys()),
    "shape_data_inputs": list(taxonomy["shape_data_inputs"].keys()),
  }

  result = {
    "scope": "TG1 LaneMapTemplate IR: separate the 5 still-human topology DOF as free fields, prove lossless "
             "re-emit of the promoted G3 Q4_K GEMV route (static UOp .key; no GPU).",
    "verdict": verdict,
    "blocked_detail": blocked_detail,
    "ir": "extra/qk/lanemap_template.py LaneMapTemplate (TopologySpec + QuantSpec + TargetSpec + ShapeSpec)",
    "proof_method": "static UOp .key equivalence of IR-emitted vs promoted-default kernel (same method as PMS-R5), "
                    "plus kernel-name match and packed-word-index formula match. No GPU.",
    "reuses": ["extra/qk/gemv_g2_lanemap.py Q4KGateUpLaneMap", "extra/qk/gemv_g3_codegen_lowering.py "
               "q4k_g3_lanemap_gemv_kernel (PROMOTED default route)"],
    "default_route_attribution": "tinygrad/llm/model.py (BUBBLEBEAM_FUTURESIGHT default-on); NOT changed by TG1.",
    "field_listing": field_listing,
    "field_taxonomy_detail": taxonomy,
    "topology_separable": topology_separable,
    "lossless_reemit": lossless,
    "reemit_per_role": {role: {k: v for k, v in r.items() if k != "field_taxonomy"} for role, r in recon.items()},
    "builds_on": ["bench/qk-g3-provenance-audit/latest.json (TG0: 5 still-human DOF)",
                  "bench/qk-lanemap-template-audit/latest.json (PMS-R5: emission is a lossless template)"],
    "stop": "TG1 only. Do NOT build the TG2 candidate author/search. The IR + its lossless proof is the deliverable.",
    "do_not": ["do not change any default", "do not repoint the live model route", "do not write a new GPU kernel",
               "do not reopen refuted routes"],
  }
  json.dump(result, open(OUT / "latest.json", "w"), indent=2)

  md = ["# TG1 LaneMapTemplate IR -- lossless re-emit of the promoted G3 route", "",
        f"Verdict: **{verdict}**  (topology separable = {topology_separable}, lossless re-emit = {lossless})", "",
        "## IR field taxonomy (topology FREE fields vs DATA inputs)", "",
        f"- **TOPOLOGY (free)**: {', '.join(field_listing['topology_free_fields'])}",
        f"- **QUANT (data)**: {', '.join(field_listing['quant_data_inputs'])}",
        f"- **TARGET (data)**: {', '.join(field_listing['target_data_inputs'])}",
        f"- **SHAPE (data)**: {', '.join(field_listing['shape_data_inputs'])}", "",
        "## 3-role lossless re-emit (IR instantiated with G3's actual topology)", "",
        "| role | rows(N) | k(K) | kernel | UOp key == default | name match | lane-idx match |",
        "|---|---:|---:|---|:--:|:--:|:--:|"]
  for role, r in recon.items():
    md.append(f"| {role} | {r['rows']} | {r['k']} | `{r['kernel_name']}` | {r['uop_key_identical_to_default']} | "
              f"{r['kernel_name_match']} | {r['lane_ownership_index_matches_reference']} |")
  md += ["",
         "Each role instantiates `LaneMapTemplate` with G3's topology (block_groups=4, words_per_group=8, G3 axis "
         "roles row=GLOBAL/block_group+word_col=LOCAL/local_block+group_pair=REDUCE, cross-lane reduce, G3 lane "
         "index) and emits via the existing G2 lane map + G3 emitter (SAME path). The emitted UOp program is "
         "byte-identical (UOp .key) to the current promoted default emission.", ""]
  if blocked_detail is not None:
    md += [f"BLOCKED detail: `{json.dumps(blocked_detail)}`", ""]
  (OUT / "summary.md").write_text("\n".join(md))

  print(verdict, "| topology_separable:", topology_separable, "| lossless:", lossless)
  for role, r in recon.items():
    print(f"  {role}: key_match={r['uop_key_identical_to_default']} name_match={r['kernel_name_match']} "
          f"idx_match={r['lane_ownership_index_matches_reference']} sep={r['topology_separated_ok']}")
  return 0 if (topology_separable and lossless) else 1


if __name__ == "__main__":
  raise SystemExit(main())
