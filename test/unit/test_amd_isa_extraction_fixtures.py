import hashlib, os, unittest

from test.unit.test_amd_isa_wmma import _tc_matmul_ast, _tc_matmul_ast_k64, _tc_matmul_ast_k64_rolled
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, lower_inst
from extra.llm_research.amd_isa_proof import amd_isa_operand_path_tag, amd_isa_proof_manifest, reset_amd_isa_proof_manifest
from tinygrad.uop.ops import Ops, UOp


FIXTURES = {
  "tc_16x16x16_unrolled": {
    "ast": _tc_matmul_ast,
    "binary_sha256": "4a558d215767eee1b292251f3251b7796748df2842ede22249918359d183e2b6",
    "mnemonic_sha256": "f415079ccd151e67c3bdad2faaa5fdf86963008f603472f5a3d050bab7842b8a",
    "instruction_bytes": 972,
    "instruction_count": 149,
    "wmma_count": 1,
  },
  "tc_16x16x64_unrolled": {
    "ast": _tc_matmul_ast_k64,
    "binary_sha256": "65215110cefc400829a37104b014e883bd630422b560308593e05effce76160d",
    "mnemonic_sha256": "4d3e8fec86693c0a722304e4cbbeb09ea8d738162ff34ccab8781b3da34bcb03",
    "instruction_bytes": 2952,
    "instruction_count": 452,
    "wmma_count": 4,
  },
  "tc_16x16x64_rolled": {
    "ast": _tc_matmul_ast_k64_rolled,
    "binary_sha256": "2edf8aae8e3d2945a7c64ce1f92dd8e248f1b3caf9809072a443439cfb96824e",
    "mnemonic_sha256": "834da500a2792a7531ed03c74ddd69f6e6c0cdfb00590b5f816e425fed537045",
    "instruction_bytes": 1380,
    "instruction_count": 214,
    "wmma_count": 1,
  },
}


def _emit_fixture(ast_fn):
  to_program_cache.clear()
  ren = AMDISARenderer(Target.parse("AMD:ISA:gfx1100"))
  captured = {}
  orig_resolve_labels = ren._resolve_labels
  def wrap_resolve_labels(insts):
    resolved = orig_resolve_labels(insts)
    captured["final"] = list(resolved)
    return resolved
  ren._resolve_labels = wrap_resolve_labels
  prg = to_program(ast_fn(), ren)
  lin = [u for u in prg.src if u.op is Ops.LINEAR][0]
  mnemonics = [str(u.arg) for u in lin.src if not isinstance(u.arg, tuple)]
  binary = b"".join(u.arg.to_bytes() for u in captured["final"])
  return {
    "binary_sha256": hashlib.sha256(binary).hexdigest(),
    "mnemonic_sha256": hashlib.sha256("\n".join(mnemonics).encode()).hexdigest(),
    "instruction_bytes": len(binary),
    "instruction_count": len(mnemonics),
    "wmma_count": sum(1 for line in mnemonics if line.startswith("v_wmma_f32_16x16x16_f16")),
  }


class TestAMDISAExtractionFixtures(unittest.TestCase):
  def test_wmma_emitted_code_fixtures_are_unchanged(self):
    for name, expected in FIXTURES.items():
      with self.subTest(name=name):
        got = _emit_fixture(expected["ast"])
        comparable = {k: v for k, v in expected.items() if k != "ast"}
        self.assertEqual(got, comparable)

  def test_opt_in_proof_manifest_records_wmma_load_store_and_waitcnt_rows(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      _emit_fixture(_tc_matmul_ast)
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    wmma = [r for r in rows if r["kind"] == "wmma"]
    loads = [r for r in rows if r["kind"] == "global_load"]
    b128_loads = [r for r in rows if r["kind"] == "global_load_b128"]
    stores = [r for r in rows if r["kind"] == "global_store"]
    waitcnts = [r for r in rows if r["kind"] == "waitcnt"]
    self.assertEqual(len(wmma), 1)
    self.assertGreaterEqual(len(loads), 1)
    self.assertGreaterEqual(len(b128_loads), 1)
    self.assertGreaterEqual(len(stores), 8)
    self.assertGreaterEqual(len(waitcnts), 1)
    self.assertEqual(wmma[0]["logical_op"], "V_WMMA")
    self.assertTrue(wmma[0]["accumulator_in_place"])
    self.assertEqual(wmma[0]["c_vgpr_range"][1] - wmma[0]["c_vgpr_range"][0], 7)
    self.assertEqual(loads[0]["logical_op"], "GLOBAL_LOAD")
    self.assertIn("dest_vgpr", loads[0])
    self.assertEqual(b128_loads[0]["logical_op"], "GLOBAL_LOAD_B128")
    self.assertIn("dest_vgpr_range", b128_loads[0])
    self.assertTrue(stores[0]["emitted"].startswith("global_store_"))
    self.assertIn("data_vgpr", stores[0])
    self.assertEqual(waitcnts[0]["logical_op"], "WAITCNT")
    self.assertIn("simm16", waitcnts[0])

  def test_opt_in_proof_manifest_preserves_global_store_owner_metadata(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      off = UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_CONST, tag=(Register("v10", 10),))
      ptr = UOp(Ops.INS, dtypes.ulong, arg=AMDOps.S_LOAD_PTR, tag=(Register("s6", 6),))
      val = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(Register("v11", 11),))
      owner = {"m": 3, "n": 5, "warp_id": 0, "lane_id": 53, "accumulator_slot": 245}
      owner_tag = tuple(sorted(owner.items()))
      lower_inst(UOp(
        Ops.INS, dtypes.void,
        src=(off, ptr, val, UOp.const(dtypes.int32, 4).rtag()),
        arg=AMDOps.GLOBAL_STORE,
        tag=("store_owner", owner_tag),
      ))
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    stores = [r for r in rows if r["kind"] == "global_store"]
    self.assertEqual(len(stores), 1)
    self.assertEqual(stores[0]["store_owner"], owner)
    self.assertEqual(stores[0]["addr_vgpr"], 10)
    self.assertEqual(stores[0]["data_vgpr"], 11)

  def test_opt_in_proof_manifest_preserves_gated_global_store_owner_metadata(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      gate = UOp(Ops.INS, dtypes.bool, arg=AMDOps.V_CONST, tag=(Register("v9", 9),))
      off = UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_CONST, tag=(Register("v10", 10),))
      val = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(Register("v11", 11),))
      ptr = UOp(Ops.INS, dtypes.ulong, arg=AMDOps.S_LOAD_PTR, tag=(Register("s6", 6),))
      owner = {"m": 3, "n": 5, "warp_id": 0, "lane_id": 21, "accumulator_slot": 42}
      owner_tag = tuple(sorted(owner.items()))
      lower_inst(UOp(
        Ops.INS, dtypes.void,
        src=(gate, off, val, UOp.const(dtypes.int32, 0).rtag(), ptr, UOp.const(dtypes.int32, 4).rtag()),
        arg=AMDOps.GATED_STORE,
        tag=("store_owner", owner_tag),
      ))
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    stores = [r for r in rows if r["kind"] == "global_store"]
    self.assertEqual(len(stores), 1)
    self.assertTrue(stores[0]["gated"])
    self.assertEqual(stores[0]["store_owner"], owner)
    self.assertEqual(stores[0]["gate_vgpr"], 9)
    self.assertEqual(stores[0]["addr_vgpr"], 10)
    self.assertEqual(stores[0]["data_vgpr"], 11)

  def test_opt_in_proof_manifest_records_ds_and_barrier_rows(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      addr = UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_CONST, tag=(Register("v10", 10),))
      data = UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_CONST, tag=(Register("v11", 11),))
      lower_inst(UOp(Ops.INS, dtypes.float32, src=(addr, UOp.const(dtypes.int32, 0).rtag()),
                     arg=AMDOps.DS_LOAD, tag=(Register("v12", 12),)))
      lower_inst(UOp(Ops.INS, dtypes.void, src=(addr, data, UOp.const(dtypes.int32, 0).rtag(), UOp.const(dtypes.int32, 4).rtag()),
                     arg=AMDOps.DS_STORE))
      lower_inst(UOp(Ops.INS, dtypes.int32, src=(addr, data, UOp.const(dtypes.int32, 16).rtag()),
                     arg=AMDOps.DS_LOAD_B128, tag=(Register("v20", 20),)))
      lower_inst(UOp(Ops.INS, dtypes.void, src=(addr, data, UOp.const(dtypes.int32, 32).rtag()),
                     arg=AMDOps.DS_STORE_B128))
      lower_inst(UOp(Ops.INS, dtypes.void, arg=AMDOps.BARRIER))
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    by_kind = {r["kind"]: r for r in rows}
    self.assertEqual(by_kind["ds_load"]["logical_op"], "DS_LOAD")
    self.assertEqual(by_kind["ds_store"]["logical_op"], "DS_STORE")
    self.assertEqual(by_kind["ds_load_b128"]["logical_op"], "DS_LOAD_B128")
    self.assertEqual(by_kind["ds_store_b128"]["logical_op"], "DS_STORE_B128")
    self.assertEqual(by_kind["barrier"]["logical_op"], "BARRIER")
    self.assertEqual(by_kind["ds_load_b128"]["dest_vgpr_range"], [20, 23])
    self.assertEqual(by_kind["ds_store_b128"]["data_vgpr_range"], [11, 14])

  def test_explicit_operand_path_tags_propagate_to_load_ds_and_wmma_rows(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      regs = [UOp(Ops.INS, dtypes.int32, arg=AMDOps.V_CONST, tag=(Register(f"v{i}", i),)) for i in range(1, 25)]
      ptr = UOp(Ops.INS, dtypes.ulong, arg=AMDOps.S_LOAD_PTR, tag=(Register("s6", 6),))
      load_meta = dict(operand_id="A", source_operand_id="arg0", fetch_group="k0", cache_policy="default",
                       width_bytes=4, vector_width_bytes=4, semantic_owner={"role": "lhs"})
      lower_inst(UOp(Ops.INS, dtypes.float32, src=(regs[0], ptr, UOp.const(dtypes.int32, 0).rtag()), arg=AMDOps.GLOBAL_LOAD,
                     tag=amd_isa_operand_path_tag((Register("v30", 30),), **load_meta)))
      lower_inst(UOp(Ops.INS, dtypes.float32, src=(regs[1], UOp.const(dtypes.int32, 0).rtag()), arg=AMDOps.DS_LOAD,
                     tag=amd_isa_operand_path_tag((Register("v31", 31),), operand_id="B", source_operand_id="arg1",
                                                  fetch_group=2, width_bytes=2, retained_fragment={"bytes": 16})))
      lower_inst(UOp(Ops.INS, dtypes.float32, src=tuple(regs), arg=AMDOps.V_WMMA,
                     tag=amd_isa_operand_path_tag((Register("v17", 17),), semantic_ownership={"A": "lhs", "B": "rhs"},
                                                  vector_width_bytes=32, retained_fragment=True)))
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    by_kind = {row["kind"]: row for row in rows}
    for key, value in load_meta.items(): self.assertEqual(by_kind["global_load"][key], value)
    self.assertEqual(by_kind["ds_load"]["operand_id"], "B")
    self.assertEqual(by_kind["ds_load"]["retained_fragment"], {"bytes": 16})
    self.assertEqual(by_kind["wmma"]["semantic_ownership"], {"A": "lhs", "B": "rhs"})
    self.assertNotIn("operand_id", by_kind["wmma"])
