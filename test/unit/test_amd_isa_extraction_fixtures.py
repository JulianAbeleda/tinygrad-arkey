import hashlib, os, unittest

from test.unit.test_amd_isa_wmma import _tc_matmul_ast, _tc_matmul_ast_k64, _tc_matmul_ast_k64_rolled
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.dtype import dtypes
from tinygrad.helpers import Target, getenv
from tinygrad.renderer.isa import Register
from tinygrad.renderer.isa.amd import AMDISARenderer, AMDOps, amd_isa_proof_manifest, lower_inst, reset_amd_isa_proof_manifest
from tinygrad.uop.ops import Ops, UOp


FIXTURES = {
  "tc_16x16x16_unrolled": {
    "ast": _tc_matmul_ast,
    "binary_sha256": "e27a9438da59750ba37f16d92e5f4e3ca7390ba0698766946016f013b2b12d50",
    "mnemonic_sha256": "f3e42f67ea11e74e916247aa56e6662325d8adabfb9768fdc422b3d087460e5c",
    "instruction_bytes": 680,
    "instruction_count": 104,
    "wmma_count": 1,
  },
  "tc_16x16x64_unrolled": {
    "ast": _tc_matmul_ast_k64,
    "binary_sha256": "7dd01b56cc51c3a45351be2950c803d2ab575d5036f5aea4aaae49b3923e8cc2",
    "mnemonic_sha256": "6744a3d0c15d84cc92623a1b47cfc717fded491f06638aa4373efd009872d6ad",
    "instruction_bytes": 1940,
    "instruction_count": 287,
    "wmma_count": 4,
  },
  "tc_16x16x64_rolled": {
    "ast": _tc_matmul_ast_k64_rolled,
    "binary_sha256": "bb745fa92c7a20e12fe6c22783fced323725a31acfefcb299c56e1e30fc8770f",
    "mnemonic_sha256": "cfdb6ccae473bbdfeec06cab9028f2f94415acb52d9b0c4e773e101c93b8c073",
    "instruction_bytes": 772,
    "instruction_count": 121,
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

  def test_opt_in_proof_manifest_preserves_accumulator_source_identity(self):
    old = os.environ.get("AMD_ISA_PROOF_MANIFEST")
    os.environ["AMD_ISA_PROOF_MANIFEST"] = "1"
    getenv.cache_clear()
    try:
      reset_amd_isa_proof_manifest()
      carrier = UOp(Ops.NOOP, dtypes.float32, arg=("accum", 12345, 6, 9))
      val = UOp(Ops.INS, dtypes.float32, arg=AMDOps.V_CONST, tag=(Register("v30", 30),))
      lower_inst(UOp(Ops.INS, dtypes.float32, src=(carrier, UOp.const(dtypes.int32, 9).rtag()),
                     arg=AMDOps.ACCUM_READ, tag=(Register("v31", 31),)))
      lower_inst(UOp(Ops.INS, dtypes.void, src=(val, carrier, UOp.const(dtypes.int32, 9).rtag()),
                     arg=AMDOps.ACCUM_WRITE))
      rows = amd_isa_proof_manifest()
    finally:
      if old is None: os.environ.pop("AMD_ISA_PROOF_MANIFEST", None)
      else: os.environ["AMD_ISA_PROOF_MANIFEST"] = old
      getenv.cache_clear()

    read = next(r for r in rows if r["kind"] == "accum_read")
    write = next(r for r in rows if r["kind"] == "accum_write")
    self.assertEqual(read["define_reg_id"], 12345)
    self.assertEqual(read["element"], 6)
    self.assertEqual(read["source_pin_vgpr"], 9)
    self.assertEqual(read["dest_vgpr"], 31)
    self.assertEqual(write["define_reg_id"], 12345)
    self.assertEqual(write["element"], 6)
    self.assertEqual(write["dest_pin_vgpr"], 9)
    self.assertEqual(write["source_vgpr"], 30)
