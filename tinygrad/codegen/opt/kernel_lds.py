"""Pure cooperative LDS ownership math for compiler-bound kernel geometry."""
from __future__ import annotations

import functools, math
from dataclasses import dataclass
from typing import Callable, TypeAlias, TYPE_CHECKING

from tinygrad.codegen.opt.packed_weight import (PackedWeightTransform, Q4KInt8FragmentProvider, Q6KInt8FragmentProvider,
                                                Q8ActivationRecordTransform, Q8Int8FragmentProvider)
from tinygrad.codegen.opt.tc import LaneMap
from tinygrad.codegen.late.native_fragment import PackedFragmentSpec
from tinygrad.dtype import AddrSpace, PtrDType, dtypes
from tinygrad.uop.ops import AxisType, Ops, UOp
if TYPE_CHECKING: from tinygrad.uop.ops import KernelLDSWindow, KernelTileGeometry

# The precontract fold/fragment/cooperative-store math below elects one lane-quad's worth of work per
# store and folds exactly one warp's lanes into the fragment layout. That is a real limitation of this
# implementation, not a per-target snapshot: AMD wave32 and Metal's 32-wide SIMD group both satisfy it,
# while e.g. AMD CDNA's wave64 genuinely does not (see test_wave64_cdna_descriptor_is_self_consistent_but_unsupported).
_PRECONTRACT_WARP_THREADS = 32


def validate_wmma_descriptor(tc) -> None:
  """Admit a WMMA descriptor this precontract mapping can prove, from its own declared facts.

  This asks what ``tc`` itself claims, never what a specific target's numbers are: (1) is it
  self-consistent -- do its own dims/threads/elements_per_thread/opts re-derive a valid thread/value ->
  fragment-coordinate map from its own swizzle (``LaneMap.validate``'s contract), and do the resulting
  lane-coordinate remaps form an honest permutation of that coordinate space, not a partial or colliding
  map; (2) is it a dtype pairing and warp width this precontract fold/fragment/cooperative-store math is
  actually proven for. Neither question compares ``tc`` against another target's frozen numbers, so any
  descriptor -- AMD's, Metal's, or a future target's -- that answers both honestly is admitted.
  """
  dims, threads, elements_per_thread, opts, swizzle, dtype_in, dtype_out = (getattr(tc, name, None) for name in
    ("dims", "threads", "elements_per_thread", "opts", "swizzle", "dtype_in", "dtype_out"))
  if not (isinstance(dims, tuple) and len(dims) == 3 and all(isinstance(d, int) and d > 0 for d in dims) and not dims[2] & (dims[2]-1)):
    raise ValueError("WMMA descriptor dims must be three positive ints with a power-of-two K extent")
  if not isinstance(opts, tuple): raise ValueError("WMMA descriptor opts must be a tuple")
  local_axes = sum(1 for o in opts if isinstance(o, str) and o[:1] == "l")
  upcast_axes = sum(1 for o in opts if isinstance(o, str) and o[:1] == "u")
  reduce_axes = dims[2].bit_length() - 1
  try:
    lane_map = LaneMap(swizzle, local_axes, upcast_axes, reduce_axes, opts, dims, threads, elements_per_thread)
    lane_map.validate()
    remaps = lane_map.remaps()
  except (AssertionError, AttributeError, IndexError, TypeError, ValueError) as exc:
    raise ValueError("WMMA descriptor is not self-consistent") from exc
  universe = {f"l{i}" for i in range(local_axes)} | {f"u{i}" for i in range(upcast_axes)} | {f"r{i}" for i in range(reduce_axes)}
  if any(set(remap) != universe or set(remap.values()) != universe for remap in remaps):
    raise ValueError("WMMA descriptor remaps are not a permutation of their own lane/value coordinate space")
  if threads != _PRECONTRACT_WARP_THREADS:
    raise ValueError(f"WMMA descriptor threads must be {_PRECONTRACT_WARP_THREADS} for this precontract "
                      "path's warp-wide fragment and cooperative-store math")
  if dtype_in not in (dtypes.half, dtypes.char):
    raise ValueError("WMMA descriptor dtype_in is not a pairing this precontract path expresses")
  if dtype_out != (dtypes.float if dtype_in == dtypes.half else dtypes.int):
    raise ValueError("WMMA descriptor dtype_out is not a pairing this precontract path expresses")


def binary_axis_count(tc, operand_idx:int) -> int:
  """How many binary (size-2) upcast axes ``tc`` itself says operand ``operand_idx`` folds.

  ``tc.elements_per_thread[operand_idx]`` is the descriptor's own per-thread element count for
  that operand; a Horner fold over binary axes reproduces it exactly when there are
  ``log2(elements_per_thread[operand_idx])`` of them (postrange.py derives ``tc_upcast_axes`` the
  same way). Not a per-target constant: RDNA3's 16/16/8 gives 4/4/3, Metal's 2/2/2 gives 1/1/1.
  """
  return int(math.log2(tc.elements_per_thread[operand_idx]))


def fold_binary_axes(axes:tuple) -> UOp:
  """Horner-fold ``axes`` (MSB first) into one element index: ``axes[0]*2**(n-1) + ... + axes[n-1]``.

  Reduces with the first axis as the seed (not a ``0`` starting accumulator) so the emitted UOp
  tree is exactly the old hand-unrolled ``((axes[0]*2+axes[1])*2+axes[2])*2+axes[3]`` for RDNA3's
  four axes -- no redundant ``+0`` node relying on a later simplification pass to disappear -- and
  the same Horner shape for any other axis count, including Metal's one.
  """
  if not axes: raise ValueError("cannot fold zero binary axes into an element index")
  return functools.reduce(lambda acc, a: acc*2 + a, axes[1:], axes[0])


def _tc_opt_bit_trace(tc) -> tuple[list[int], list[int], list[str], list[int]]:
  """Replay `tc.opts`'s local/upcast-axis creation order plus `tc.get_reduce_axes()`'s reduce
  splits -- exactly the order ``postrange.py::_apply_generic_tensor_core_opt``'s ``shift_to`` loop
  (lines ~460-475) creates them in -- to give each canonical ``tc.base_shape_str()`` position its
  target dim (0=N, 1=M, 2=K), its bit-significance ordinal within that dim (0 = that dim's own
  LSB), its kind ('l': physically bound to `lane` bit `bit_index`; 'u': a per-thread upcast index;
  'r': a K-tile reduce index), and `bit_index` (the lane-bit index for 'l', the ordinal for 'u'/'r').
  """
  own_dim, own_ordinal, kind, bit_index = [], [], [], []
  per_dim_count = [0, 0]; l_count = u_count = 0
  for opt in tc.opts:
    d = int(opt[1])
    own_dim.append(d); own_ordinal.append(per_dim_count[d]); per_dim_count[d] += 1
    if opt[0] == "l": kind.append("l"); bit_index.append(l_count); l_count += 1
    else: kind.append("u"); bit_index.append(u_count); u_count += 1
  for r in range(len(tc.get_reduce_axes())):
    own_dim.append(2); own_ordinal.append(r); kind.append("r"); bit_index.append(r)
  return own_dim, own_ordinal, kind, bit_index


@dataclass(frozen=True)
class WmmaOperandLaneLayout:
  """One WMMA operand's (A or B) derived within-tile addressing.

  Each of the operand's two within-tile axes (row and K) is described by two ordered term tuples,
  and every bit position of the axis index is covered by exactly one of them:

  * ``{row,k}_contract_terms`` -- ``(element_bit, axis_bit)`` pairs: the operand's own
    CONTRACT/binary-axis element (the ``fold_binary_axes`` value over ``tc.base_upcast_axes()``,
    MSB first -- PG0/PG1a's existing, unchanged derivation) contributes element bit
    ``element_bit`` at axis position ``axis_bit``. The element may be split across row and K with
    each fragment at an arbitrary axis position (e.g. NVIDIA's m16n8k16 A: element bit 1 at row
    bit 3, element bits 0 and 2 at K bits 0 and 3).
  * ``{row,k}_lane_terms`` -- ``(lane_bit, axis_bit)`` pairs: physical ``lane`` bit ``lane_bit``
    contributes at axis position ``axis_bit``.

  ``element_bits`` is the operand's full contract-element width
  (``log2(tc.elements_per_thread[operand])``); :func:`_fold_operand_axis` collapses a whole
  identity run ``((i, i), ...)`` back to the folded element UOp unchanged only when this axis owns
  every element bit.
  """
  row_contract_terms: tuple[tuple[int, int], ...]
  row_lane_terms: tuple[tuple[int, int], ...]
  k_contract_terms: tuple[tuple[int, int], ...]
  k_lane_terms: tuple[tuple[int, int], ...]
  element_bits: int


def derive_wmma_operand_lane_layout(tc) -> tuple[WmmaOperandLaneLayout, WmmaOperandLaneLayout]:
  """Derive each operand's (A, B) row/K lane-bit layout from ``tc.opts`` + ``tc.lane_map`` -- the
  exact swizzle-substitution machinery ``postrange.py::_apply_generic_tensor_core_opt`` (lines
  ~494-508) already threads through ordinary UOp substitution on the register-resident/generic
  TC-opt route, the one route T4 GPU-validated correct on Metal (max_abs_error 0.0, 100% coverage,
  deterministic, ``scratchpad/t4_fused_generic_tc_execute.py``). This precontract/LDS staging path
  hand-builds its own fragment-load addresses instead of letting the ordinary UOp expander apply
  that substitution, so it must reproduce the same lane -> coordinate correspondence by other
  means; this function is that means, derived from ``tc``'s own declared descriptor fields, never a
  per-backend branch and never a memorized/hand-guessed bit permutation.

  Cross-checked (compile-only, no GPU, before being wired into any address-building code) against
  two independent grounds truth: (1) AMD's shipped-correct precontract formula -- ``row = lane %
  tc_dim`` with the operand's whole contract axis folded into K -- which this function reproduces
  bit-for-bit for ``tc.amd_rdna3`` (the descriptor family the AMD non-regression six routes actually
  use); PACKED_WMMA_ROUTES is untouched by this function's introduction. (2) the literal address
  arithmetic Metal's compiler-verified generic-TC-opt kernel emits
  (``scratchpad/t1_generic_tc_dequant_probe.py``'s ``rung1_dense_fp16`` render on
  ``METAL:METAL:Apple9``), read directly out of the compiled C source, which this function's output
  matches term-for-term for ``tc.metal``.

  NVIDIA's m16n8k16 splits each operand's contract element across row and K (and places the row
  term at the top of the row axis), so the general form here is per-axis term tuples rather than
  "one LSB-aligned contiguous contract run on one side" -- the old shape is exactly the AMD/Metal
  special case of these tuples. The structural checks below (every row/K axis bit position used
  exactly once, every element bit used exactly once) still fail closed: a ``tc`` descriptor whose
  substitution does not resolve into lane bits and contract element bits at every position raises
  ``ValueError`` (an untested TC descriptor, e.g. AMD's CDNA wave64 families, raises here) rather
  than emit an address this derivation was never shown correct for.
  """
  validate_wmma_descriptor(tc)
  own_dim, own_ordinal, kind, bit_index = _tc_opt_bit_trace(tc)
  base = tc.base_shape_str()
  if len(base) != len(own_dim): raise ValueError("tc base_shape_str length does not match the opts/reduce trace")
  perms = tc.permutes_for_shape_str(base)
  bua = tc.base_upcast_axes()

  layouts = []
  for operand_idx, row_dim in ((0, 1), (1, 0)):
    perm = perms[operand_idx]
    inv = [0] * len(perm)
    for i, v in enumerate(perm): inv[v] = i

    ept = tc.elements_per_thread[operand_idx]
    n_contract = int(math.log2(ept))
    if 2 ** n_contract != ept: raise ValueError(f"operand {operand_idx} elements_per_thread is not a power of two")
    # `element` is the Horner fold of bua[:n_contract] (MSB first), so bua[i] is element bit
    # n_contract-1-i -- the same fold `PrecontractCandidateContract.assemble` feeds
    # `PrecontractContractSpec.element`.
    element_bit = {name: n_contract - 1 - i for i, name in enumerate(bua[:n_contract])} if n_contract else {}
    contract_positions = {base.index(name) for name in element_bit}

    def _classify(positions:list[int], operand_idx=operand_idx, inv=inv, element_bit=element_bit,
                  contract_positions=contract_positions) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
      contract_terms:list[tuple[int, int]] = []
      lane_terms:list[tuple[int, int]] = []
      for pos in positions:
        target = inv[pos]
        if target in contract_positions: contract_terms.append((element_bit[base[target]], own_ordinal[pos]))
        elif kind[target] == "l": lane_terms.append((bit_index[target], own_ordinal[pos]))
        else: raise ValueError(f"operand {operand_idx} lane layout does not resolve cleanly at canonical position {pos}")
      return tuple(contract_terms), tuple(lane_terms)

    row_positions = sorted([i for i in range(len(own_dim)) if own_dim[i] == row_dim and kind[i] != "r"], key=lambda i: own_ordinal[i])
    k_positions = sorted([i for i in range(len(own_dim)) if own_dim[i] == 2], key=lambda i: own_ordinal[i])
    row_contract, row_lane = _classify(row_positions)
    k_contract, k_lane = _classify(k_positions)
    # Structural fail-closed checks: every axis bit position and every element bit used exactly once.
    row_bits = int(math.log2(tc.dims[1] if row_dim == 1 else tc.dims[0]))
    k_bits = int(math.log2(tc.dims[2]))
    if sorted(p for _, p in row_contract + row_lane) != list(range(row_bits)):
      raise ValueError(f"operand {operand_idx} row axis positions do not exactly cover its {row_bits} bits")
    if sorted(p for _, p in k_contract + k_lane) != list(range(k_bits)):
      raise ValueError(f"operand {operand_idx} K axis positions do not exactly cover its {k_bits} bits")
    used_element_bits = {eb for eb, _ in row_contract + k_contract}
    if used_element_bits != set(range(n_contract)):
      raise ValueError(f"operand {operand_idx} contract element bits are not used exactly once across row/K "
                       f"({sorted(used_element_bits)} != {list(range(n_contract))})")
    layouts.append(WmmaOperandLaneLayout(row_contract, row_lane, k_contract, k_lane, n_contract))
  return layouts[0], layouts[1]


def _fold_operand_axis(contract_terms:tuple[tuple[int, int], ...], lane_terms:tuple[tuple[int, int], ...],
                       lane:UOp, contract_element:UOp, element_bits:int):
  """Build the within-tile index UOp from :func:`derive_wmma_operand_lane_layout`'s term tuples.

  The contract part is ``sum(((contract_element >> element_bit) & 1) << axis_bit)`` -- or, when
  this axis owns every element bit as an identity run ``((i, i), ...)``, the folded element UOp
  itself, unchanged. The lane part is ``sum(((lane >> lane_bit) & 1) << axis_bit)``.

  Two collapses keep the UOp trees (and therefore the rendered source) identical to the
  pre-derivation idioms the established families are pinned on: a contiguous ascending lane-bit
  run at contiguous ascending axis positions collapses to the same ``lane % span`` /
  ``(lane // 2**b) % span * 2**p`` two-op form the pre-derivation code emitted (AMD's rdna3 row,
  whose terms are ``((0, 0), (1, 1), (2, 2), (3, 3))``, renders as ``lane % 16`` -- verified:
  `scratchpad/pg2_amd_all_routes_rendered_source_equality.py` six hashes unmoved), and a whole
  identity contract run collapses to ``contract_element`` (AMD's rdna3 K, Metal's single-element
  contract axes). A non-contiguous run (Metal's real, swizzle-scrambled bit sets; NVIDIA's split
  contracts) falls back to the explicit per-bit sum -- there is no shorter equivalent, and no
  existing rendered source depends on its exact shape.
  """
  n_contract = len(contract_terms)
  if n_contract and n_contract == element_bits and contract_terms == tuple((i, i) for i in range(n_contract)):
    expr: UOp | None = contract_element
  elif n_contract:
    expr = None
    for element_bit, axis_bit in contract_terms:
      contribution = ((contract_element // (1 << element_bit)) % 2) * (1 << axis_bit)
      expr = contribution if expr is None else expr + contribution
  else:
    expr = None
  n_lane = len(lane_terms)
  first_lane_bit, first_axis_bit = lane_terms[0] if n_lane else (0, 0)
  if n_lane and lane_terms == tuple((first_lane_bit + i, first_axis_bit + i) for i in range(n_lane)):
    span = 1 << n_lane
    lane_part = lane % span if first_lane_bit == 0 else (lane // (1 << first_lane_bit)) % span
    lane_term = lane_part if first_axis_bit == 0 else lane_part * (1 << first_axis_bit)
    return lane_term if expr is None else expr + lane_term
  for lane_bit, axis_bit in lane_terms:
    contribution = ((lane // (1 << lane_bit)) % 2) * (1 << axis_bit)
    expr = contribution if expr is None else expr + contribution
  if expr is None: raise ValueError("operand axis has neither a contract-axis nor a lane-bit contribution")
  return expr


def contract_symbolic_upcast(value:UOp, axis:UOp) -> UOp:
  """Materialize one scalar value over its owned UPCAST axis as a legal vector carrier."""
  if axis.op is not Ops.RANGE or axis.arg[-1] is not AxisType.UPCAST or axis.vmin != 0:
    raise ValueError("symbolic contraction requires a zero-based UPCAST range")
  if value.dtype is dtypes.void or value.dtype.count != 1: raise ValueError("symbolic contraction requires a non-void scalar value")
  if axis not in value.backward_slice_with_self: raise ValueError("symbolic contraction value does not own the requested axis")
  width = axis.vmax+1
  return UOp(Ops.CONTRACT, value.dtype.vec(width), (value,), ((axis.arg[0], width),))


def lower_symbolic_barrier_dependencies(root:UOp, axis:UOp) -> UOp:
  """Contract scalar UPCAST values before they cross an effect barrier.

  Effect barriers preserve ordering, but must not retain scalar-shaped values over
  an upcast axis: those otherwise survive expansion as illegal program UNROLLs.
  """
  if axis.op is not Ops.RANGE or axis.arg[-1] is not AxisType.UPCAST or axis.vmin != 0:
    raise ValueError("symbolic barrier lowering requires a zero-based UPCAST range")
  lowered: dict[UOp, UOp] = {}
  for node in root.toposort():
    src = tuple(lowered[x] for x in node.src)
    if node.op is Ops.BARRIER:
      src = tuple(contract_symbolic_upcast(x, axis) if x.dtype is not dtypes.void and x.dtype.count == 1 and
                  axis in x.backward_slice_with_self else x for x in src)
    lowered[node] = node if src == node.src else node.replace(src=src)
  return lowered[root]


@dataclass(frozen=True)
class PrecontractOperandTemplate:
  role: str
  source: UOp
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp


@dataclass(frozen=True)
class PackedPrecontractOperandTemplate:
  """Packed B source decoded at logical cooperative tile-production coordinates."""
  role: str
  source: UOp
  transform: PackedWeightTransform|Q8ActivationRecordTransform
  row_axis: UOp
  k_axis: UOp
  row_tile_base: UOp
  fragment_provider: Q4KInt8FragmentProvider|Q6KInt8FragmentProvider|Q8Int8FragmentProvider|None = None
  fragment_spec: PackedFragmentSpec|None = None


PrecontractOperand: TypeAlias = PrecontractOperandTemplate | PackedPrecontractOperandTemplate

@dataclass(frozen=True)
class PrecontractThreadAxes:
  wave_m: UOp
  wave_n: UOp
  lane: UOp

@dataclass(frozen=True)
class PrecontractKAxis:
  tile_owner: UOp
  substep_owner: UOp
  tile_base: UOp
  substep: UOp

@dataclass(frozen=True)
class PrecontractContractSpec:
  role: str
  axes: tuple[UOp, ...]
  arg: tuple[tuple[int, int], ...]
  element: UOp
  descriptor_remap: tuple[tuple[str, str], ...]

@dataclass(frozen=True)
class PrecontractLDSStage:
  allocation: UOp
  producer: UOp
  barrier: UOp
  fragment_a: UOp
  fragment_b: UOp
  fragment_b_k16: tuple[UOp, UOp]|None = None
  fragment_b_spec: PackedFragmentSpec|None = None






@dataclass(frozen=True)
class PrecontractProducerInstance:
  epoch: UOp
  slot: UOp
  role_nodes: tuple[UOp, UOp]

@dataclass(frozen=True)
class PrecontractFragmentInstance:
  epoch: UOp
  slot: UOp
  ready: UOp
  fragments: tuple[UOp, UOp]

@dataclass(frozen=True)
class PrecontractFactors:
  subtiles_m: int
  subtiles_n: int
  waves_m: int
  waves_n: int
  k_substeps: int
  vectors_per_row: int
  loads_a: int
  loads_b: int


@dataclass(frozen=True)
class PrecontractCandidateContract:
  """Single owner of tensor-core candidate geometry, storage, operand, and CONTRACT assembly."""
  context: object; tc: object
  factors: PrecontractFactors; register_mode: bool

  @property
  def pipeline(self): return getattr(self.context, "pipeline", None)

  @classmethod
  def create(cls, context:object, tc) -> PrecontractCandidateContract:
    geometry = getattr(context, "geometry", None)
    if geometry is None: raise ValueError("precontract candidate requires explicit geometry")
    pipeline = getattr(context, "pipeline", None)
    register_hint = getattr(getattr(pipeline, "storage", None), "kind", None) == "global_register_resident"
    factors = derive_precontract_shape_factors(geometry, tc) if register_hint else derive_precontract_factors(geometry, tc)
    register_mode = False
    if pipeline is not None:
      from tinygrad.codegen.opt.kernel_pipeline import pipeline_policy_from_candidate
      policy = pipeline_policy_from_candidate(pipeline)
      if policy.storage_kind != "lds":
        coverage = getattr(pipeline, "wait_coverage", None)
        if coverage is None or not coverage.passed: raise ValueError("register-resident candidate lacks proven wait dependency coverage")
      register_mode = policy.storage_kind == "global_register_resident"
      if register_mode != register_hint: raise ValueError("candidate storage policy disagrees with geometry contract")
    return cls(context, tc, factors, register_mode)

  def assemble(self, *, in0:UOp, in1:UOp, original_axes:tuple[UOp, UOp, UOp], outer_n:UOp, outer_m:UOp,
               logical_outer_n:UOp|None = None,
               wave_m:UOp, wave_n:UOp, lane:UOp, tc_upcast_axes:tuple[tuple[tuple[int, int], ...], ...],
               range_by_id:dict[int, UOp], allocation_id:Callable[[], int]|None
               ) -> tuple[tuple[PrecontractOperand, ...], PrecontractThreadAxes, tuple[PrecontractContractSpec, ...], UOp|None]:
    geometry, tc = self.context.geometry, self.tc
    packed_outer_n = outer_n if logical_outer_n is None else logical_outer_n
    contracts = []
    for operand_idx, role in enumerate(("A", "B")):
      axes = tuple(range_by_id[a] for a, size in tc_upcast_axes[operand_idx] if size == 2)
      expected = binary_axis_count(tc, operand_idx)
      if len(axes) != expected: raise ValueError(f"candidate {role} contract does not have {expected} binary axes")
      element = fold_binary_axes(axes)
      contracts.append(PrecontractContractSpec(role, axes, tc_upcast_axes[operand_idx], element,
                                               tuple(tc.lane_map.remaps()[operand_idx].items())))
    contracts = tuple(contracts)
    validate_precontract_contracts(tc, contracts, context="candidate", mismatch="does not match actual descriptor operand mapping")

    # Preserve descriptor, allocation, then operand UOp creation order: linearization tie-breaks follow graph insertion order.
    allocation = None
    if not self.register_mode:
      total_bytes = geometry.lds_windows[-1].end if self.pipeline is None else self.pipeline.active_lds_bytes
      if allocation_id is None: raise ValueError("LDS candidate requires an allocation ID owner")
      tag = ("kernel_tile_lds", geometry) if self.pipeline is None else ("kernel_tile_lds", geometry, self.pipeline)
      allocation = UOp.placeholder((total_bytes//tc.dtype_in.itemsize,), tc.dtype_in, allocation_id(), addrspace=AddrSpace.LOCAL).replace(tag=tag)

    packed_activation = getattr(self.context, "packed_activation", None)
    activation_provider = getattr(self.context, "packed_activation_provider", None)
    if packed_activation is None:
      operand_a:PrecontractOperand = PrecontractOperandTemplate("A", in0, original_axes[1], original_axes[2], outer_m*geometry.tile[0])
    else:
      if (not isinstance(packed_activation, Q8ActivationRecordTransform) or
          not isinstance(activation_provider, Q8Int8FragmentProvider) or activation_provider.transform != packed_activation):
        raise ValueError("packed activation provider does not own the admitted Q8 record transform")
      if (original_axes[1].vmax+1, original_axes[2].vmax+1) != packed_activation.logical_shape:
        raise ValueError("packed activation row/K ownership does not match admitted transform")
      packed_a_params = [u for u in in0.toposort() if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and
                         u.ptrdtype.base == packed_activation.storage_dtype]
      if len(packed_a_params) != 1: raise ValueError(f"packed A carrier must reach exactly one canonical Q8 PARAM, found {len(packed_a_params)}")
      if getattr(packed_a_params[0].arg, "slot", packed_a_params[0].arg) != 1:
        raise ValueError(f"packed A carrier must own ABI slot 1, got PARAM {packed_a_params[0].arg!r}")
      operand_a = PackedPrecontractOperandTemplate("A", packed_a_params[0], packed_activation, original_axes[1], original_axes[2],
                                                   outer_m*geometry.tile[0], activation_provider)
    packed_weight = getattr(self.context, "packed_weight", None)
    if packed_weight is None:
      operand_b:PrecontractOperand = PrecontractOperandTemplate("B", in1, original_axes[0], original_axes[2], outer_n*geometry.tile[1])
    else:
      if self.register_mode: raise ValueError("packed-weight candidate requires LDS tile storage")
      if (original_axes[0].vmax+1, original_axes[2].vmax+1) != (packed_weight.rows, packed_weight.k): raise ValueError(
        "packed-weight candidate row/K ownership does not match admitted transform")
      packed_params = [u for u in in1.toposort() if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and
                       u.ptrdtype.base == packed_weight.storage_dtype]
      if len(packed_params) != 1: raise ValueError(f"packed-weight B carrier must reach exactly one canonical packed PARAM, found {len(packed_params)}")
      if getattr(packed_params[0].arg, "slot", packed_params[0].arg) != 2: raise ValueError(
        f"packed-weight B carrier must own ABI slot 2, got PARAM {packed_params[0].arg!r}")
      if any(u.op is Ops.PARAM and isinstance(u.dtype, PtrDType) and u.ptrdtype.base == dtypes.half for u in in1.toposort()): raise ValueError(
        "packed-weight B carrier unexpectedly reaches a dense fp16 PARAM")
      fragment_provider = getattr(self.context, "packed_fragment_provider", None)
      if fragment_provider is not None and (not isinstance(fragment_provider, (Q4KInt8FragmentProvider, Q6KInt8FragmentProvider)) or
                                             fragment_provider.transform != packed_weight):
        raise ValueError("packed fragment provider does not own the admitted packed-weight transform")
      fragment_spec = PackedFragmentSpec.q6k_k64() if isinstance(fragment_provider, Q6KInt8FragmentProvider) else None
      operand_b = PackedPrecontractOperandTemplate("B", packed_params[0], packed_weight, original_axes[0], original_axes[2],
                                                   packed_outer_n*geometry.tile[1], fragment_provider, fragment_spec)
    operands:tuple[PrecontractOperand, ...] = (operand_a, operand_b)
    validate_precontract_operand_templates(operands, dtype_in=tc.dtype_in, context="candidate")
    return operands, PrecontractThreadAxes(wave_m, wave_n, lane), contracts, allocation


def derive_precontract_shape_factors(geometry:KernelTileGeometry, tc) -> PrecontractFactors:
  """Derive WMMA tile factors without consulting any storage allocation.

  This is the shared geometry contract for LDS and register-resident producers.
  ``derive_precontract_factors`` below adds the LDS-window checks needed by the
  legacy staged implementation.
  """
  validate_wmma_descriptor(tc)
  tm, tn, tk = geometry.tile
  if (tm % (geometry.waves[0] * tc.dims[1]) or tn % (geometry.waves[1] * tc.dims[0]) or
      tk % tc.dims[2]):
    raise ValueError("tile must divide into whole per-wave tensor-core subtiles and K steps")
  sm = tm // (geometry.waves[0] * tc.dims[1])
  sn = tn // (geometry.waves[1] * tc.dims[0])
  ks = tk // tc.dims[2]
  if ks < 2:
    raise ValueError("current atomic staging requires at least two tensor-core K steps")
  vectors_per_row = tk * tc.dtype_in.itemsize // 16
  if vectors_per_row <= 0 or tk * tc.dtype_in.itemsize % 16:
    raise ValueError("K row must contain whole b128 vectors")
  rows = (tm, tn)
  loads = tuple(row * vectors_per_row // geometry.threads for row in rows)
  if any(row * vectors_per_row % geometry.threads for row in rows) or any(x <= 0 for x in loads):
    raise ValueError("operand vectors must divide evenly across cooperative threads")
  return PrecontractFactors(sm, sn, *geometry.waves, ks, vectors_per_row, *loads)


def validate_precontract_operand_templates(operands:tuple[PrecontractOperand, ...], *, dtype_in=dtypes.half,
                                           context:str="precontract") -> None:
  """Validate source dtype and live row/K ownership independent of storage."""
  if tuple(x.role for x in operands) != ("A", "B"):
    raise ValueError(f"{context} operands must be exactly ordered A and B")
  for operand in operands:
    if operand.row_axis.op is not Ops.RANGE or operand.k_axis.op is not Ops.RANGE:
      raise ValueError(f"{context} {operand.role} template does not retain row/K ownership")
    if isinstance(operand, PackedPrecontractOperandTemplate):
      if dtype_in == dtypes.half and operand.fragment_provider is not None:
        raise ValueError(f"{context} fp16 packed template cannot carry an int8 fragment provider")
      if dtype_in == dtypes.char and operand.fragment_provider is None:
        raise ValueError(f"{context} int8 packed template requires a typed logical fragment provider")
      if dtype_in not in (dtypes.half, dtypes.char):
        raise ValueError(f"{context} packed template cannot produce {dtype_in.name} values")
      if (operand.role not in ("A", "B") or not isinstance(operand.source.dtype, PtrDType) or
          operand.source.ptrdtype.base != operand.transform.storage_dtype):
        raise ValueError(f"{context} packed template must use canonical packed storage dtype")
      if isinstance(operand.transform, PackedWeightTransform) != (operand.role == "B"):
        raise ValueError(f"{context} packed weight transform must own B and Q8 record transform must own A")
      if operand.fragment_provider is not None and operand.fragment_provider.logical_shape != (operand.transform.rows, operand.transform.k):
        raise ValueError(f"{context} packed fragment provider logical ownership does not match the transform")
      # The packed carrier no longer contains the dense source expression, so
      # these two ranges are the only remaining proof of logical ownership.
      # Keep the transform and carrier bounds in the same contract as the
      # producer: accepting a detached/partial domain would silently decode a
      # different row or read past the packed allocation.
      if (operand.row_axis.vmax + 1 != operand.transform.rows or
          operand.k_axis.vmax + 1 != operand.transform.k):
        raise ValueError(f"{context} packed B row/K ownership does not match the transform")
      packed_units = operand.transform.packed_bytes // operand.transform.storage_width
      if operand.source.ptrdtype.size != packed_units:
        raise ValueError(f"{context} packed B carrier does not exactly cover the transform")
    elif (operand.row_axis not in operand.source.backward_slice_with_self or
          operand.k_axis not in operand.source.backward_slice_with_self or
          operand.source.dtype.scalar() != dtype_in):
      raise ValueError(f"{context} {operand.role} template does not retain scalar {dtype_in.name} row/K ownership")


def validate_precontract_contracts(tc, contracts:tuple[PrecontractContractSpec, ...], *,
                                   context:str="precontract", mismatch:str="does not match the descriptor") -> None:
  """Validate A/B CONTRACT axes, folded element identity, and descriptor remaps."""
  if tuple(c.role for c in contracts) != ("A", "B"):
    raise ValueError(f"{context} contracts must be exactly ordered A and B")
  descriptor_remaps = tuple(tuple(x.items()) for x in tc.lane_map.remaps())
  for operand_idx, contract in enumerate(contracts):
    expected = binary_axis_count(tc, operand_idx)
    folded = fold_binary_axes(contract.axes) if len(contract.axes) == expected else None
    if (len(contract.axes) != expected or any(a.op is not Ops.RANGE or a.vmax + 1 != 2 for a in contract.axes) or
        contract.arg != tuple((a.arg[0], 2) for a in contract.axes) or contract.element is not folded or
        contract.descriptor_remap != descriptor_remaps[operand_idx]):
      raise ValueError(f"{context} {contract.role} contract {mismatch}")


def validate_precontract_carriers(fragment_dtype, accumulator_dtype, *, tc, context:str="precontract") -> None:
  """Validate the stable WMMA fragment and accumulator carrier ABI against a specific descriptor."""
  validate_wmma_descriptor(tc)
  dtype_in, dtype_out, elements = tc.dtype_in, tc.dtype_out, tc.elements_per_thread
  expected_fragments = (dtype_in.vec(elements[0]), dtype_in.vec(elements[1]))
  if fragment_dtype not in expected_fragments:
    raise ValueError(f"{context} fragment carrier must match the tensor-core input carrier")
  if accumulator_dtype != dtype_out.vec(elements[2]):
    raise ValueError(f"{context} accumulator carrier must match the tensor-core output carrier")


def validate_precontract_wmma_abi(node: UOp, *, context: str = "precontract", tc: object|None = None) -> None:
  """Validate the WMMA node ABI before a backend/devectorizer sees it.

  The tensor-core matcher accepts two descriptor-sized input fragments and one
  descriptor-sized accumulator, producing the same output carrier.  The argument carries the corresponding four binary A/B axes and
  three binary C axes.  Keep this check storage-independent so LDS and
  register-resident templates cannot drift into different ABI rules.
  """
  if not isinstance(node, UOp) or node.op is not Ops.WMMA:
    raise ValueError(f"{context} WMMA ABI validator requires an Ops.WMMA node")
  if len(node.src) != 3:
    raise ValueError(f"{context} WMMA ABI requires A, B, and C inputs")
  arg = node.arg
  if not isinstance(arg, tuple) or len(arg) < 8:
    raise ValueError(f"{context} WMMA descriptor argument is incomplete")
  try: dims = tuple(arg[1])
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{context} WMMA descriptor dimensions are invalid") from exc
  if len(dims) != 3 or any(not isinstance(d, int) or d <= 0 for d in dims):
    raise ValueError(f"{context} WMMA descriptor dimensions are invalid")
  dtype_in, dtype_out = arg[2], arg[3]
  if dtype_in not in (dtypes.half, dtypes.char) or \
     dtype_out != (dtypes.float if dtype_in == dtypes.half else dtypes.int) or arg[5] != _PRECONTRACT_WARP_THREADS:
    raise ValueError(f"{context} WMMA descriptor carrier ABI drifted")
  axes = arg[6]
  if not isinstance(axes, tuple) or len(axes) != 3:
    raise ValueError(f"{context} WMMA descriptor requires A/B/C axis groups")
  # The per-operand binary-axis counts come from the descriptor itself
  # (log2 of elements_per_thread, the one existing derivation) whenever the
  # caller holds it.  The 4/4/3 fallback is the RDNA3-shaped legacy surface
  # (the RDNA3-only consumer adapter and its unit tests); production callers
  # that can reach more than one family must pass the descriptor.
  expected_counts = tuple(binary_axis_count(tc, i) for i in range(3)) if tc is not None else (4, 4, 3)
  for role, count, group in zip(("A", "B", "C"), expected_counts, axes):
    if not isinstance(group, tuple) or len(group) != count or any(not isinstance(x, tuple) or len(x) != 2 or x[1] != 2 for x in group):
      raise ValueError(f"{context} {role} WMMA contract requires {count} binary axes")
  # Fragment/accumulator widths are derived from the arg's own axis-group sizes (2**|group|), not a
  # per-target frozen elements_per_thread: the arg tuple never carries elements_per_thread directly.
  expected_a, expected_b = (dtype_in.vec(2**len(g)) for g in (axes[0], axes[1]))
  expected_out = dtype_out.vec(2**len(axes[2]))
  if node.src[0].dtype != expected_a: raise ValueError(f"{context} A fragment carrier does not match the descriptor")
  if node.src[1].dtype != expected_b: raise ValueError(f"{context} B fragment carrier does not match the descriptor")
  if node.src[2].dtype != expected_out: raise ValueError(f"{context} accumulator carrier does not match the descriptor")
  if node.dtype != expected_out: raise ValueError(f"{context} WMMA result carrier does not match the descriptor")


def validate_precontract_thread_axes(geometry:KernelTileGeometry, factors:PrecontractFactors,
                                     threads:PrecontractThreadAxes, subtile_m:UOp, subtile_n:UOp,
                                     *, context:str="precontract") -> None:
  """Validate live wave/lane and subtile RANGE ownership against tile factors."""
  # A size-1 wave axis has no live cross-wave choice to make, so the caller collapses it to a
  # CONST 0 (see postrange.py's _apply_tc_opt) instead of an unsupported size-one RANGE. Accept
  # that degenerate case here rather than requiring a live RANGE for a wave count of exactly 1.
  def _wave_ok(wave:UOp, count:int) -> bool:
    if count == 1:
      return wave.op is Ops.CONST and wave.arg == 0
    return (wave.op, wave.vmax + 1, wave.arg[-1]) == (Ops.RANGE, count, AxisType.LOCAL)
  if (not _wave_ok(threads.wave_m, factors.waves_m) or
      not _wave_ok(threads.wave_n, factors.waves_n) or
      (threads.lane.op, threads.lane.vmax + 1, threads.lane.arg[-1]) !=
      (Ops.RANGE, geometry.wave_size, AxisType.WARP)):
    raise ValueError(f"{context} thread axes do not match derived wave geometry")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax + 1 != factors.subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax + 1 != factors.subtiles_n):
    raise ValueError(f"{context} subtile axes do not match derived geometry")

@dataclass(frozen=True)
class PrecontractPipelineTemplate:
  """Validated immutable inputs for every epoch of a precontract LDS pipeline."""
  geometry: KernelTileGeometry
  tc: object
  allocation: UOp
  operands: tuple[PrecontractOperand, ...]
  threads: PrecontractThreadAxes
  subtile_m: UOp
  subtile_n: UOp
  contracts: tuple[PrecontractContractSpec, ...]
  pipeline_plan: object

  def __post_init__(self) -> None:
    factors = derive_precontract_factors(self.geometry, self.tc)
    validate_precontract_operand_templates(self.operands, dtype_in=self.tc.dtype_in, context="precontract pipeline")
    validate_precontract_thread_axes(self.geometry, factors, self.threads, self.subtile_m, self.subtile_n,
                                     context="precontract pipeline")
    validate_precontract_contracts(self.tc, self.contracts, context="precontract pipeline")
    slot_bytes = self.geometry.lds_windows[-1].end
    if (getattr(self.pipeline_plan, "slot_bytes", None) != slot_bytes or
        self.allocation.op is not Ops.DEFINE_LOCAL or self.allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
        self.allocation.ptrdtype.base != self.tc.dtype_in or
        self.allocation.ptrdtype.size*self.tc.dtype_in.itemsize != self.pipeline_plan.active_lds_bytes):
      raise ValueError("precontract pipeline allocation does not exactly cover its active LDS slots")

  @property
  def factors(self) -> PrecontractFactors: return derive_precontract_factors(self.geometry, self.tc)

  def producer(self, epoch:UOp, slot:UOp) -> PrecontractProducerInstance:
    return instantiate_precontract_producer(self.geometry, tc=self.tc, allocation=self.allocation,
      operands=self.operands, threads=self.threads, epoch=epoch, slot=slot)

  def fragments(self, epoch:UOp, slot:UOp, ready:UOp, k_substep:int) -> PrecontractFragmentInstance:
    if not 0 <= k_substep < self.factors.k_substeps: raise ValueError("precontract K substep is out of range")
    return instantiate_precontract_fragments(self.geometry, tc=self.tc, allocation=self.allocation, threads=self.threads,
      k_substep=UOp.const(dtypes.weakint,k_substep), subtile_m=self.subtile_m, subtile_n=self.subtile_n,
      contracts=self.contracts, epoch=epoch, slot=slot, ready=ready)

def derive_precontract_factors(geometry:KernelTileGeometry, tc) -> PrecontractFactors:
  factors = derive_precontract_shape_factors(geometry, tc)
  tm, tn, tk = geometry.tile
  rows = (tm, tn)
  for window,row in zip(geometry.lds_windows, rows):
    if window.stride_bytes < tk*tc.dtype_in.itemsize or window.end-window.base != row*window.stride_bytes:
      raise ValueError("LDS windows must exactly cover padded operand rows")
  return factors


def _window(geometry:KernelTileGeometry, role:str) -> KernelLDSWindow:
  if role not in ("A", "B"): raise ValueError(f"cooperative LDS role must be A or B, got {role!r}")
  return next(w for w in geometry.lds_windows if w.role == role)


def cooperative_store_octet_rows(vectors_per_row:int, *, bank_cycle_lanes:int|None) -> int:
  """Rows spanned by one LDS bank-cycle group of the cooperative store.

  ``bank_cycle_lanes`` is the target's declared lanes-serviced-per-b128-bank-cycle fact (e.g.
  ``Renderer.lds_bank_cycle_lanes``; AMD RDNA3: 8, from a wave32 b128 LDS access), not a restated
  constant. The cooperative store elects consecutive lanes onto consecutive ``(row, vector)``
  slots, so one such cycle group covers ``bank_cycle_lanes // vectors_per_row`` consecutive rows.

  Unreferenced anywhere in this tree today -- nothing calls this function, including the rest of
  this module (PG1a's audit already flagged the two ``cooperative_store_row*`` functions below as
  the actually-exercised path). Parameterized anyway, on the same declared fact as
  ``cooperative_store_row_rotation``, rather than left frozen to AMD's 8: its docstring's original
  "RDNA3 services..." claim was exactly the per-target-snapshot defect this campaign removes,
  whether or not the function currently has a caller.
  """
  if bank_cycle_lanes is None or vectors_per_row <= 0 or bank_cycle_lanes % vectors_per_row:
    raise ValueError("cooperative octet must cover whole rows")
  return bank_cycle_lanes // vectors_per_row


def cooperative_store_row_rotation(*, vectors_per_row:int, rows:int, stride_bytes:int, vector_bytes:int=16,
                                   bank_dwords:int|None, bank_cycle_lanes:int|None) -> bool:
  """Whether re-electing lane-quads onto rotated rows removes the store bank conflict.

  ``bank_dwords`` and ``bank_cycle_lanes`` are declared target facts (``Renderer.lds_bank_dwords`` /
  ``Renderer.lds_bank_cycle_lanes``), never restated constants here. They are real hardware
  structure, not something derivable from ``vectors_per_row``/``rows``/``stride_bytes`` alone, so a
  target that does not report them (Apple does not document its threadgroup memory banking the way
  AMD's ISA manuals document LDS) always returns ``False``. That is always the *safe* choice, never
  a correctness one: the rotation this function gates is an exact one-writer cover of the tile
  whichever way it comes out (see :func:`cooperative_store_row`'s docstring), so an unknown target
  simply forgoes the optimization rather than guessing at undocumented bank structure.

  Bank arithmetic.  A b128 slot ``(row, vector)`` starts at dword ``row*S + vector*V`` with
  ``S = stride_bytes//4`` and ``V = vector_bytes//4``; it occupies ``V`` consecutive dwords, so
  the only thing that matters for a b128 access is which of the ``bank_dwords//V`` aligned dword
  *quads* it lands in:  ``Q(row, vector) = (row*(stride_bytes//vector_bytes) + vector) mod (bank_dwords//V)``.
  For AMD's shipped geometry (32 banks of 4 B, stride 80, vector 16) that is ``Q = (5*row + vector) mod 8``.

  The rotation below (the octet-of-eight, rows-four-apart pairing) is proven only for an eight-lane
  bank cycle -- i.e. only when ``bank_dwords//V`` and the independently declared ``bank_cycle_lanes``
  both land on 8, the same scope-limiting shape as this file's ``_PRECONTRACT_WARP_THREADS`` guard.
  A target whose declared facts don't land there (a different cycle width, or one whose two facts
  disagree at this vector width) also skips: this is an optimization proven for one geometry, not a
  bank-count-agnostic algorithm.

  An octet is conflict-free iff its eight slots hit eight distinct ``Q``.  With
  ``vectors_per_row == 4`` an octet is two rows x four vectors:
    * rows ``r, r+1``  -> ``{c,c+1,c+2,c+3}`` u ``{c+5,c+6,c+7,c}``  -- ``c`` twice, ``c+4`` unused
      => 2 cycles instead of 1, i.e. a 2-way conflict on every cooperative store.
    * rows ``r, r+4``  -> ``{c,c+1,c+2,c+3}`` u ``{c+4,c+5,c+6,c+7}``  -- all eight, conflict-free.
  So the conflict is not a property of the LDS *layout* (which padding and XOR swizzles both
  attack, and which is already load-optimal at stride 80) but of the store's lane->row
  *election*.  Rotating the low three bits of the row index pairs rows four apart instead of
  adjacent while leaving every lane-quad on one contiguous ``vector_bytes*vectors_per_row``
  source row, so the wave still issues exactly the same global-memory segments.
  """
  if bank_dwords is None or bank_cycle_lanes is None: return False
  if vector_bytes <= 0 or bank_dwords <= 0 or bank_dwords % (vector_bytes//4) or stride_bytes % vector_bytes: return False
  quads = bank_dwords // (vector_bytes//4)
  if quads != 8 or bank_cycle_lanes != 8 or vectors_per_row != 4 or rows % 8: return False
  m = (stride_bytes//vector_bytes) % quads
  def conflicted(delta:int) -> bool:
    q = [(m*row + vector) % quads for row in (0, delta) for vector in range(vectors_per_row)]
    return len(set(q)) != len(q)
  return conflicted(1) and not conflicted(4)


def cooperative_store_row(raw_row, *, vectors_per_row:int, rows:int, stride_bytes:int, vector_bytes:int=16,
                          bank_dwords:int|None=None, bank_cycle_lanes:int|None=None):
  """Apply the lane->row re-election of :func:`cooperative_store_row_rotation`.

  ``raw_row = linear_vector // vectors_per_row`` is the row the naive election would store.
  The rotation ``q -> ((q & 1) << 2) | (q >> 1)`` on the low three bits maps lane-quads
  ``0..7`` onto rows ``0,4,1,5,2,6,3,7``, so every octet (quads ``2j, 2j+1``) holds rows four
  apart.  It is an involution-free permutation of each aligned eight-row block, hence still an
  exact one-writer cover of the tile, and it is loop-invariant so it hoists out of the K loop.

  ``bank_dwords``/``bank_cycle_lanes`` default to ``None`` (unknown) so a caller that has not yet
  threaded a target's declared bank facts through gets today's always-safe behaviour: the rotation
  is skipped, ``raw_row`` returned unchanged -- still an exact one-writer cover, just not necessarily
  bank-conflict-optimal.
  """
  if not cooperative_store_row_rotation(vectors_per_row=vectors_per_row, rows=rows, stride_bytes=stride_bytes,
                                        vector_bytes=vector_bytes, bank_dwords=bank_dwords,
                                        bank_cycle_lanes=bank_cycle_lanes): return raw_row
  return (raw_row//8)*8 + (raw_row % 2)*4 + (raw_row % 8)//2


def instantiate_precontract_producer(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                     operands:tuple[PrecontractOperand,...], threads:PrecontractThreadAxes,
                                     epoch:UOp, slot:UOp, logical_row_tile_bases:tuple[UOp|None,UOp|None]|dict[str,UOp|None]|None=None,
                                     logical_k_block:UOp|None=None) -> PrecontractProducerInstance:
  factors=derive_precontract_factors(geometry,tc)
  item_bytes, vector_bytes = tc.dtype_in.itemsize, 16
  vector_elements = vector_bytes // item_bytes
  slot_base=slot*(geometry.lds_windows[-1].end//item_bytes)
  thread=(threads.wave_m*geometry.waves[1]+threads.wave_n)*geometry.wave_size+threads.lane
  role_nodes=[]
  for operand in operands:
    stores=[]; window=_window(geometry,operand.role); loads=factors.loads_a if operand.role == "A" else factors.loads_b
    rows=geometry.tile[0] if operand.role == "A" else geometry.tile[1]
    for row_iteration in range(loads):
      linear_vector=thread+row_iteration*geometry.threads
      row,vector=linear_vector//factors.vectors_per_row,linear_vector%factors.vectors_per_row
      row=cooperative_store_row(row,vectors_per_row=factors.vectors_per_row,rows=rows,
                                stride_bytes=window.stride_bytes,vector_bytes=vector_bytes)
      logical_k=vector*vector_elements
      logical_base = (logical_row_tile_bases.get(operand.role) if isinstance(logical_row_tile_bases, dict) else logical_row_tile_bases[0 if operand.role == "A" else 1]) if logical_row_tile_bases is not None else None
      logical_row = (logical_base if logical_base is not None else operand.row_tile_base) + row
      if isinstance(operand, PackedPrecontractOperandTemplate):
        value = operand.fragment_provider.fragment(operand.source, logical_row, epoch*geometry.tile[2]+logical_k, vector_elements).value \
          if operand.fragment_provider is not None else \
          operand.transform.dequant_tile(operand.source, logical_row, epoch*geometry.tile[2]+logical_k, vector_elements).value
      else:
        value = UOp(Ops.STACK,tc.dtype_in.vec(vector_elements),tuple(operand.source.substitute({
          operand.row_axis:logical_row, operand.k_axis:epoch*geometry.tile[2]+logical_k+elem}) for elem in range(vector_elements)))
      tag=("kernel_tile_store",operand.role,row_iteration,epoch,slot)
      # Keep lane ownership explicit.  A vector pointer with a vectorized
      # logical index can be lowered as INDEX(LOAD(ptr), lane); that turns the
      # destination into a loaded temporary and, for repeated index lanes,
      # silently aliases distinct K elements.  Scalar addresses preserve the
      # producer's exact one-writer cover; the backend may still regroup the
      # adjacent stores after their addresses are proven.
      base=slot_base+(window.base+row*window.stride_bytes+logical_k*item_bytes)//item_bytes
      stores.append(UOp.group(*(allocation.index(base+elem).store(value.gep(elem)).replace(tag=tag).end()
                               for elem in range(vector_elements))))
    role_nodes.append(UOp.group(*stores))
  return PrecontractProducerInstance(epoch,slot,(role_nodes[0],role_nodes[1]))

def instantiate_precontract_fragments(geometry:KernelTileGeometry, *, tc, allocation:UOp, threads:PrecontractThreadAxes,
                                      k_substep:UOp, subtile_m:UOp, subtile_n:UOp,
                                      contracts:tuple[PrecontractContractSpec,...], epoch:UOp, slot:UOp,
                                      ready:UOp, logical_row_tile_bases:tuple[UOp|None,UOp|None]|dict[str,UOp|None]|None=None,
                                      logical_k_block:UOp|None=None) -> PrecontractFragmentInstance:
  factors=derive_precontract_factors(geometry,tc); item_bytes=tc.dtype_in.itemsize
  slot_base=slot*(geometry.lds_windows[-1].end//item_bytes)
  ordered=allocation.after(ready); lane=threads.lane
  # Same derivation the legacy stage's `_fragment` uses (see `build_precontract_lds_stage`): the
  # per-subtile row extent is the descriptor's own M dim (`tc.dims[1]`) for role A and N dim
  # (`tc.dims[0]`) for role B, and the within-tile lane/element bits come from
  # `derive_wmma_operand_lane_layout` -- not RDNA3's lane%16 ABI, which overflows the B window of
  # any descriptor whose B rows are narrower than 16 (e.g. NVIDIA's m16n8k16).
  operand_layouts = derive_wmma_operand_lane_layout(tc)
  def fragment(role,subtile,wave,subtiles,contract):
    window=_window(geometry,role)
    tc_dim = tc.dims[1] if role == "A" else tc.dims[0]
    operand_idx = 0 if role == "A" else 1
    layout = operand_layouts[operand_idx]
    row=(wave*subtiles+subtile)*tc_dim+_fold_operand_axis(layout.row_contract_terms, layout.row_lane_terms, lane, contract.element, layout.element_bits)
    logical_k=k_substep*tc.dims[2]+_fold_operand_axis(layout.k_contract_terms, layout.k_lane_terms, lane, contract.element, layout.element_bits)
    idx=slot_base+(window.base+row*window.stride_bytes+logical_k*item_bytes)//item_bytes
    semantic=(role,epoch,slot,k_substep,subtile)
    load=ordered.index(idx,dtype=tc.dtype_in).replace(tag=("kernel_tile_fragment_load",*semantic)).load()
    return UOp(Ops.CONTRACT,tc.dtype_in.vec(tc.elements_per_thread[operand_idx]),(load,),contract.arg,
               tag=("kernel_tile_fragment",*semantic))
  frags=(fragment("A",subtile_m,threads.wave_m,factors.subtiles_m,contracts[0]),
         fragment("B",subtile_n,threads.wave_n,factors.subtiles_n,contracts[1]))
  return PrecontractFragmentInstance(epoch,slot,ready,frags)

def build_precontract_lds_stage(geometry:KernelTileGeometry, *, tc, allocation:UOp,
                                operands:tuple[PrecontractOperand, ...], threads:PrecontractThreadAxes,
                                k_axis:PrecontractKAxis, subtile_m:UOp, subtile_n:UOp,
                                contracts:tuple[PrecontractContractSpec, ...], pipeline_plan=None,
                                lds_bank_dwords:int|None=None, lds_bank_cycle_lanes:int|None=None,
                                lds_read_before_next_write_ordered:bool|None=None,
                                logical_row_tile_bases:tuple[UOp|None,UOp|None]|dict[str,UOp|None]|None=None) -> PrecontractLDSStage:
  """Build an unwired vector cooperative stage while full operand index templates still exist.

  ``lds_bank_dwords``/``lds_bank_cycle_lanes`` are the calling renderer's declared bank facts
  (``Renderer.lds_bank_dwords``/``Renderer.lds_bank_cycle_lanes``), threaded straight through to
  :func:`cooperative_store_row`. Defaulting to ``None`` here matches that function's own default:
  a caller that does not pass a target's facts gets the same always-safe, rotation-skipped result
  as an unknown target, never AMD's arithmetic applied by accident.

  ``lds_read_before_next_write_ordered`` is the calling renderer's declared ordering fact
  (``Renderer.lds_read_before_next_write_ordered``, MB2). This stage reuses one physical LDS window
  across every K-tile iteration when ``pipeline_plan is None`` (the only shape any caller uses today);
  the barrier built below from ``producer`` orders this iteration's stores -> this iteration's reads,
  but nothing orders reads -> the *next* iteration's stores unless a second barrier is emitted at loop
  entry, before the stores. Polarity matches the field's own docstring: the barrier is the default,
  safe behaviour; only an explicit ``True`` skips it, so a caller that does not pass this (``None``)
  gets the barrier, same as an unrecognised target would.
  """
  factors = derive_precontract_factors(geometry, tc)
  validate_precontract_operand_templates(operands, dtype_in=tc.dtype_in, context="precontract")
  try:
    from extra.llm_research.prefill.nv_compiler_streamk_codegen import record_range_provenance
    record_range_provenance(operands)
  except ImportError:
    pass
  for operand in operands:
    if operand.row_tile_base.dtype.scalar() not in (dtypes.int, dtypes.weakint): raise ValueError("precontract row tile base must be integer")
  validate_precontract_thread_axes(geometry, factors, threads, subtile_m, subtile_n, context="precontract")
  if (k_axis.tile_owner.op is not Ops.RANGE or k_axis.tile_owner.arg[-1] is not AxisType.REDUCE or
      k_axis.tile_owner not in k_axis.tile_base.backward_slice_with_self):
    raise ValueError("precontract K tile owner must be a live REDUCE range in tile base")
  if (k_axis.substep_owner.op is not Ops.RANGE or k_axis.substep_owner.arg[-1] is not AxisType.UNROLL or
      k_axis.substep_owner.vmax+1 != factors.k_substeps or k_axis.substep_owner not in k_axis.substep.backward_slice_with_self):
    raise ValueError("precontract K substep owner must be a live derived-size UNROLL range in substep")
  if (subtile_m.op is not Ops.RANGE or subtile_m.vmax+1 != factors.subtiles_m or
      subtile_n.op is not Ops.RANGE or subtile_n.vmax+1 != factors.subtiles_n):
    raise ValueError("precontract K/subtile axes are invalid")
  validate_precontract_contracts(tc, contracts, context="precontract", mismatch="does not match actual descriptor operand mapping")
  slot_bytes = geometry.lds_windows[-1].end
  total_bytes = slot_bytes if pipeline_plan is None else pipeline_plan.active_lds_bytes
  if (allocation.op is not Ops.DEFINE_LOCAL or allocation.ptrdtype.addrspace is not AddrSpace.LOCAL or
      allocation.ptrdtype.base != tc.dtype_in or allocation.ptrdtype.size * tc.dtype_in.itemsize != total_bytes):
    raise ValueError("precontract caller allocation must be one exact dtype_in LDS window")
  item_bytes, vector_bytes = tc.dtype_in.itemsize, 16
  vector_elements = vector_bytes // item_bytes
  stores = []
  slot_base = UOp.const(dtypes.weakint, 0) if pipeline_plan is None else \
    (k_axis.tile_owner % pipeline_plan.buffer_count) * (slot_bytes // item_bytes)
  thread = (threads.wave_m * geometry.waves[1] + threads.wave_n) * geometry.wave_size + threads.lane
  # Entry barrier: closes the read(this iter) -> write(next iter) gap the single exit barrier below
  # (`barrier`, ordering stores -> reads) leaves open, whenever the target has not declared it needs
  # none. Only meaningful for the single-buffered (`pipeline_plan is None`) shape every caller uses
  # today -- the pipelined path expresses its loop-carried dependency through
  # `KernelStage1PipelinePlan` instead, not through this stage. Anchored on `k_axis.tile_owner` (the
  # live REDUCE range this whole stage is nested in, already validated above), not on anything this
  # call itself produces: a barrier built from this call's own fragment reads would make the
  # dependency graph cyclic (the reads depend on the exit barrier, which depends on the stores, which
  # would then depend on the entry barrier depending on the reads). Anchoring on the range instead
  # keeps the entry barrier inside the K-loop body -- not hoisted above it, since it depends on
  # loop-varying state -- while `allocation.after(...)` below (the same construction the exit barrier
  # already uses for fragment reads, just applied to the store side) sequences every store after it in
  # the rendered source, exactly the ordering llama.cpp's own loop-entry barrier provides.
  needs_entry_barrier = pipeline_plan is None and lds_read_before_next_write_ordered is not True
  store_allocation = allocation.after(k_axis.tile_owner.barrier()) if needs_entry_barrier else allocation
  staged_group_metadata = (tc.dtype_in == dtypes.char and len(operands) == 2 and
    isinstance(operands[0], PackedPrecontractOperandTemplate) and isinstance(operands[0].fragment_provider, Q8Int8FragmentProvider) and
    isinstance(operands[1], PackedPrecontractOperandTemplate) and
    isinstance(operands[1].fragment_provider, (Q4KInt8FragmentProvider, Q6KInt8FragmentProvider)))
  if staged_group_metadata:
    metadata_bytes_per_row = factors.vectors_per_row * 4
    for operand in operands:
      window = _window(geometry, operand.role)
      if window.stride_bytes < geometry.tile[2] + metadata_bytes_per_row:
        raise ValueError("typed K-quant/Q8_1 staging requires one half2 metadata packet per cooperative vector owner")
  for operand in operands:
    window = _window(geometry, operand.role)
    loads = factors.loads_a if operand.role == "A" else factors.loads_b
    rows = geometry.tile[0] if operand.role == "A" else geometry.tile[1]
    for row_iteration in range(loads):
      linear_vector = thread + row_iteration*geometry.threads
      row, vector = linear_vector//factors.vectors_per_row, linear_vector%factors.vectors_per_row
      row = cooperative_store_row(row, vectors_per_row=factors.vectors_per_row, rows=rows,
                                  stride_bytes=window.stride_bytes, vector_bytes=vector_bytes,
                                  bank_dwords=lds_bank_dwords, bank_cycle_lanes=lds_bank_cycle_lanes)
      logical_k = vector * vector_elements
      logical_base = (logical_row_tile_bases.get(operand.role) if isinstance(logical_row_tile_bases, dict) else logical_row_tile_bases[0 if operand.role == "A" else 1]) if logical_row_tile_bases is not None else None
      logical_row = (logical_base if logical_base is not None else operand.row_tile_base) + row
      if isinstance(operand, PackedPrecontractOperandTemplate):
        value = operand.fragment_provider.fragment(operand.source, logical_row, k_axis.tile_base + logical_k, vector_elements).value \
          if operand.fragment_provider is not None else \
          operand.transform.dequant_tile(operand.source, logical_row, k_axis.tile_base + logical_k, vector_elements).value
      else:
        value = UOp(Ops.STACK, tc.dtype_in.vec(vector_elements), tuple(operand.source.substitute({
          operand.row_axis: logical_row, operand.k_axis: k_axis.tile_base + logical_k + elem}) for elem in range(vector_elements)))
      index = slot_base + (window.base + row * window.stride_bytes + logical_k * item_bytes) // item_bytes
      store_tag = ("kernel_tile_store", operand.role, row_iteration)
      stores.append(UOp.group(*(store_allocation.index(index+elem).store(value.gep(elem)).replace(tag=store_tag).end()
                               for elem in range(vector_elements))))
      if staged_group_metadata:
        if operand.role == "A":
          scale, raw_sum, _ = operand.transform.metadata(operand.source, logical_row, k_axis.tile_base+logical_k)
          metadata = UOp(Ops.STACK, dtypes.half.vec(2), (scale.cast(dtypes.half), raw_sum.cast(dtypes.half)))
        elif isinstance(operand.fragment_provider, Q4KInt8FragmentProvider):
          d, dmin, scale, minimum, _ = operand.fragment_provider.metadata(operand.source, logical_row, k_axis.tile_base+logical_k)
          metadata = UOp(Ops.STACK, dtypes.half.vec(2),
            ((d*scale.cast(dtypes.float)).cast(dtypes.half), (-dmin*minimum.cast(dtypes.float)).cast(dtypes.half)))
        else:
          # Every cooperative vector owns one K16 Q6 fragment, but every K32
          # correction consumes the pair.  Duplicate the pair in both vector
          # packets so the accumulator can select the even owner exactly like
          # the Q4 K32 packet path while retaining both independent scales.
          k32_base = k_axis.tile_base+(logical_k//32)*32
          d, scale0, scale1, _ = operand.fragment_provider.k32_metadata(operand.source, logical_row, k32_base)
          metadata = UOp(Ops.STACK, dtypes.half.vec(2),
            ((d*scale0.cast(dtypes.float)).cast(dtypes.half), (d*scale1.cast(dtypes.float)).cast(dtypes.half)))
        metadata_index = slot_base + window.base + row*window.stride_bytes + geometry.tile[2] + vector*4
        metadata_tag = ("kernel_tile_group_metadata_store", operand.role, row_iteration)
        metadata_bits = tuple(metadata.gep(part).bitcast(dtypes.uint16) for part in range(2))
        metadata_bytes = tuple(metadata_bits[part//2].rshift((part%2)*8).bitwise_and(0xff).cast(dtypes.uint8).bitcast(dtypes.char)
                               for part in range(4))
        stores.append(UOp.group(*(store_allocation.index(metadata_index+elem).store(metadata_bytes[elem]).replace(tag=metadata_tag).end()
                                 for elem in range(4))))
  producer = UOp.group(*stores)
  barrier = UOp.barrier(producer)
  wave_m, wave_n, lane = threads.wave_m, threads.wave_n, threads.lane
  ordered = allocation.after(barrier)
  # Which physical `lane` bits (and which share of the operand's own contract/binary axis) supply
  # this WMMA operand's within-tile row and K-leftover index is Apple's undocumented
  # `simdgroup_half8x8` per-lane ABI for Metal, and RDNA3's documented-by-blog-post-only lane%16 ABI
  # for AMD -- derived once here from `tc` itself (never a per-backend branch, never a hand-guessed
  # bit permutation), see `derive_wmma_operand_lane_layout`'s docstring for the two independent
  # grounds truth this was checked against before being wired in.
  operand_layouts = derive_wmma_operand_lane_layout(tc)
  def _fragment(role:str, subtile:UOp, wave:UOp, subtiles:int, contract:PrecontractContractSpec,
                k16_half:int|None=None) -> UOp:
    window = _window(geometry, role)
    # The per-subtile row extent is the descriptor's own M dim (`tc.dims[1]`) for role A and N dim
    # (`tc.dims[0]`) for role B -- the exact same per-role dim `derive_precontract_shape_factors`
    # already divides tm/tn by to get `subtiles`/`sm`/`sn` above.
    tc_dim = tc.dims[1] if role == "A" else tc.dims[0]
    operand_idx = 0 if role == "A" else 1
    layout = operand_layouts[operand_idx]
    row = (wave * subtiles + subtile) * tc_dim + _fold_operand_axis(layout.row_contract_terms, layout.row_lane_terms, lane, contract.element, layout.element_bits)
    logical_k = k_axis.substep * tc.dims[2] + _fold_operand_axis(layout.k_contract_terms, layout.k_lane_terms, lane, contract.element, layout.element_bits)
    index = slot_base + (window.base + row * window.stride_bytes + logical_k * item_bytes) // item_bytes
    load = ordered.index(index, dtype=tc.dtype_in).replace(tag=("kernel_tile_fragment_load", role)).load()
    if k16_half is not None:
      if role != "B" or k16_half not in (0,1): raise ValueError("K16 fragment mask is only valid for Q6_K B")
      in_half = ((logical_k%32)<16) if k16_half == 0 else ((logical_k%32)>=16)
      load = in_half.where(load, UOp.const(tc.dtype_in,0))
    return UOp(Ops.CONTRACT, tc.dtype_in.vec(tc.elements_per_thread[operand_idx]), (load,), contract.arg,
               tag=("kernel_tile_fragment", role))
  fragment_a=_fragment("A",subtile_m,wave_m,factors.subtiles_m,contracts[0])
  fragment_b=_fragment("B",subtile_n,wave_n,factors.subtiles_n,contracts[1])
  q6_b = isinstance(operands[1], PackedPrecontractOperandTemplate) and \
    isinstance(operands[1].fragment_provider,Q6KInt8FragmentProvider)
  fragment_b_k16 = tuple(_fragment("B",subtile_n,wave_n,factors.subtiles_n,contracts[1],half) for half in (0,1)) if q6_b else None
  fragment_b_spec = operands[1].fragment_spec if isinstance(operands[1], PackedPrecontractOperandTemplate) else None
  return PrecontractLDSStage(allocation,producer,barrier,fragment_a,fragment_b,fragment_b_k16,fragment_b_spec)
