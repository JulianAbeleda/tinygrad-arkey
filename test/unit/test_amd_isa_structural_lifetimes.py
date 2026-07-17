from types import SimpleNamespace

from tinygrad import dtypes
from tinygrad.codegen import line_rewrite
from tinygrad.codegen.late.regalloc import pressure_schedule
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa import amd
from tinygrad.renderer.isa.amd import AMDOps, lower_inst, pre_regalloc_matcher
from tinygrad.uop.ops import Ops, UOp


def _vreg(name:str, index:int) -> Register:
  return Register(name, index)


def test_wide_fragment_release_order_survives_pre_regalloc_cleanup():
  """A later wide producer starts only after the prior fragment's FP32 update."""
  addr0 = UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg("v4", 4),))
  load0 = UOp(Ops.INS, dtypes.int32, (addr0, UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 0).rtag()),
              AMDOps.DS_LOAD_B128, tag=(_vreg("v200", 200),))
  wmma = UOp(Ops.INS, dtypes.int32, (load0,), AMDOps.V_WMMA_I8, tag=(_vreg("v8", 8),))
  updates = tuple(UOp(Ops.INS, dtypes.float32,
                      (UOp(Ops.INS, dtypes.float32, (wmma,), AMDOps.V_CVT_I2F, tag=(_vreg(f"v{32+i}", 32+i),)),),
                      AMDOps.V_ADD, tag=(_vreg(f"v{48+i}", 48+i),)) for i in range(8))

  # The unrelated integer dependency models an old address/order carrier.  It
  # must be discarded rather than becoming part of the retained boundary.
  stale = UOp(Ops.INS, dtypes.int32, (addr0,), AMDOps.V_IADD, tag=(_vreg("v7", 7),))
  addr1 = UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg("v5", 5),))
  load1 = UOp(Ops.INS, dtypes.int32,
              (addr1, UOp(Ops.NOOP, dtypes.void), UOp.const(dtypes.int32, 16).rtag(), stale) + updates,
              AMDOps.DS_LOAD_B128, tag=(_vreg("v200", 200),))
  linear = pressure_schedule(list(UOp.sink(load1).toposort()))
  cleaned = line_rewrite(linear, pre_regalloc_matcher)
  selected = next(x for x in cleaned if x.op is Ops.INS and x.arg is AMDOps.DS_LOAD_B128 and x.src[2].arg == 16)

  assert len(selected.src) == 3 and selected.src[0].op is Ops.AFTER
  assert selected.src[0].src[0] is addr1
  assert set(selected.src[0].src[1:]) == set(updates)
  assert stale not in selected.backward_slice_with_self
  assert max(cleaned.index(x) for x in updates) < cleaned.index(selected)

  # The canonical operand shape remains directly lowerable as one b128 load;
  # the AFTER is a zero-code register alias, not an extra ISA operand.
  inst, waits = lower_inst(selected)
  assert waits == [inst]
  assert "ds_load_b128(v[200:203], v[5]" in str(inst.arg)


def test_progressive_c_marked_carriers_serialize_all_lane_drains(monkeypatch):
  symbolic0 = UOp(Ops.WMMA, dtypes.float32.vec(8), src=())
  symbolic1 = UOp(Ops.WMMA, dtypes.float32.vec(8), src=(symbolic0,))
  operands0 = tuple(UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg(f"v{64+i}", 64+i),)) for i in range(24))
  operands1 = tuple(UOp(Ops.INS, dtypes.int32, arg=AMDOps.MOV, tag=(_vreg(f"v{128+i}", 128+i),)) for i in range(24))
  def marked(root, operands):
    machine = UOp(Ops.INS, dtypes.float32, operands, AMDOps.V_WMMA, tag=(_vreg("v8", 8),))
    marker = UOp(Ops.NOOP, dtypes.void, arg=("selected_wmma_root", root))
    carrier = UOp(Ops.NOOP, dtypes.float32.vec(8), src=(machine,) + tuple(
      UOp(Ops.INS, dtypes.float32, (machine,), AMDOps.MOV, tag=(_vreg(f"v{8+i}", 8+i),)) for i in range(1, 8)) + (marker,))
    drains = tuple(UOp(Ops.INS, dtypes.float32, (carrier,), AMDOps.V_CVT_I2F,
                       tag=(_vreg(f"v{96+i+(16 if root is symbolic1 else 0)}", 96+i+(16 if root is symbolic1 else 0)),)) for i in range(8))
    return machine, carrier, drains
  machine0, carrier0, drains0 = marked(symbolic0, operands0)
  machine1, carrier1, drains1 = marked(symbolic1, operands1)
  monkeypatch.setattr(amd, "_progressive_c_assignment", lambda ctx: ({symbolic0:0, symbolic1:0}, 1))
  serialized = amd._serialize_progressive_c_drains(SimpleNamespace(), UOp.sink(*drains0, *drains1))
  assert serialized is not None
  selected = [u for u in serialized.toposort() if u.op is Ops.INS and u.arg is AMDOps.V_WMMA]
  assert len(selected) == 2
  second = next(u for u in selected if set(drains0).issubset(u.src))
  assert second.src[:24] == operands1 and second.src[24:] == drains0
  linear = pressure_schedule(list(serialized.toposort()))
  assert max(linear.index(x) for x in drains0) < linear.index(second)
  cleaned = line_rewrite(linear, pre_regalloc_matcher)
  cleaned_second = next(u for u in cleaned if u is second)
  assert cleaned_second.src[:24] == operands1 and cleaned_second.src[24:] == drains0
  assert "v_wmma_f32_16x16x16_f16" in str(lower_inst(cleaned_second).arg)


def test_kmajor_physical_ownership_order_policy():
  # Unique ownership is the historical byte-stable phase-major order.
  assert amd._wmma_kmajor_order([8, 16, 24, 32], 2) == [
    (0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
  # Shared leases complete S0 -> S1 for one logical chain before the lease's next S0.
  order = amd._wmma_kmajor_order([8, 16, 8, 16, 8, 16, 8, 16], 2)
  assert len(order) == 16
  assert order == [(chain, phase) for chain in (0, 2, 4, 6) for phase in (0, 1)] + \
                  [(chain, phase) for chain in (1, 3, 5, 7) for phase in (0, 1)]


def test_kmajor_shared_lease_lowers_16_wmmas_chain_major_with_pack_reuse(monkeypatch):
  carriers = [UOp(Ops.NOOP, dtypes.float16.vec(16), arg=("frag", i)) for i in range(4)]
  seed = UOp.const(dtypes.float32.vec(8), 0.0)
  chains = []
  for chain_i in range(8):
    head = UOp(Ops.WMMA, dtypes.float32.vec(8), (carriers[chain_i % 2], carriers[2 + chain_i % 2], seed), (chain_i, 0))
    order = () if chain_i < 2 else (UOp(Ops.NOOP, dtypes.void, (chains[chain_i-2][1],)),)
    chains.append((head, UOp(Ops.WMMA, dtypes.float32.vec(8),
                            (carriers[chain_i % 2], carriers[2 + chain_i % 2], head) + order, (chain_i, 1))))
  roots = [chain[1] for chain in chains]
  uses = {u:[] for chain in chains for u in chain}
  for head, root in chains: uses[head].append(root)
  ctx = SimpleNamespace(uses=uses)
  monkeypatch.setattr(amd, "_c_low", lambda _ctx: True)
  monkeypatch.setattr(amd, "_wmma_chain_head_acc", lambda head: (head, 0, UOp(Ops.NOOP, dtypes.void, arg=("acc", head.arg))))
  head_ids = {id(head):chain_i for chain_i, (head, _root) in enumerate(chains)}
  monkeypatch.setattr(amd, "_acc_base", lambda _ctx, key: 8 + (head_ids[key[0]] % 2) * 8)
  monkeypatch.setattr(amd, "_wmma_frag_proof_reuse_key", lambda _ctx, role, carrier: (role, carrier.arg))
  monkeypatch.setattr(amd, "_wmma_operand_regs", lambda carrier: 8)
  monkeypatch.setattr(amd, "_ab_base", lambda _ctx, key, width: 64 + carriers.index(next(c for c in carriers if c.arg == key[1][1])) * 8)
  pack_calls = []
  def fake_pack(_ctx, carrier, base, dep, role):
    pack_calls.append((role, carrier, base))
    return tuple(UOp(Ops.INS, dtypes.int32, dep, AMDOps.MOV, tag=(_vreg(f"v{base+i}", base+i),)) for i in range(8))
  monkeypatch.setattr(amd, "_pack_frag_tile", fake_pack)

  assert amd._try_wmma_kmajor_phase(ctx, roots[-1]) is not None
  lowered = UOp.sink(*(ctx._wmma_memo[root] for root in roots))
  selected = [u for u in lowered.toposort() if u.op is Ops.INS and u.arg is AMDOps.V_WMMA]
  # The lowered graph contains all 16 WMMAs exactly once and topologically sorts without a cycle.
  assert len(selected) == 16 and len(set(selected)) == 16
  assert len(pack_calls) == 4  # two physical A fragments and two physical B fragments, each packed once
  symbolic_order = [next(tile.arg for tile, out in ctx._wmma_memo.items() if out.src[0] is machine) for machine in selected]
  assert all(symbolic_order.index((chain, 0)) < symbolic_order.index((chain, 1)) for chain in range(8))
  markers = [src.arg[1] for head, _root in chains for src in ctx._wmma_memo[head].src
             if src.op is Ops.NOOP and isinstance(src.arg, tuple) and src.arg[:1] == ("selected_wmma_root",)]
  assert markers == roots

  drains = [tuple(UOp(Ops.INS, dtypes.float32, (ctx._wmma_memo[root].src[lane],), AMDOps.V_CVT_I2F,
                         tag=(_vreg(f"v{160+chain_i*8+lane}", 160+chain_i*8+lane),)) for lane in range(8))
            for chain_i, root in enumerate(roots)]
  monkeypatch.setattr(amd, "_progressive_c_assignment", lambda _ctx: ({root:chain_i % 2 for chain_i, root in enumerate(roots)}, 2))
  serialized = amd._serialize_progressive_c_drains(ctx, UOp.sink(*(drain for chain in drains for drain in chain)))
  assert serialized is not None
  serialized_wmmas = [u for u in serialized.toposort() if u.op is Ops.INS and u.arg is AMDOps.V_WMMA]
  # Chain 2 is the next owner of v8..v15: its S0 cannot issue until every chain-0 tail conversion has drained.
  chain2_head = next(u for u in serialized_wmmas if set(drains[0]).issubset(u.src))
  assert all(drain in chain2_head.backward_slice for drain in drains[0])
  assert len(serialized_wmmas) == 16 and len(set(serialized_wmmas)) == 16
