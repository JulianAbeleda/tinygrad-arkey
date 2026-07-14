"""WP1: ABI-rooted operand attribution over final-ISA rows. Synthetic rows exercise the dataflow
without a GPU; the real gfx1100 attn_qo capture is verified by the adapter's own path."""
import unittest

from extra.qk.operand_attribution import attribute_operands, operand_paths_for_manifest

ABI = {"outs": [0], "ins": [1, 2]}  # arg0=out, arg1=a, arg2=b (a @ b.T contract)


def _row(index, mnemonic, operands, cls, reads, writes):
  return {"index": index, "mnemonic": mnemonic, "operands": operands, "instruction_class": cls,
          "reads": reads, "writes": writes, "pc": index * 4}


def _positive_flow():
  # kernarg pointer loads: B at offset 0x10 (arg2), out+A at offset 0 (arg0,arg1).
  # Each global-load address is a same-operand 64-bit pointer pair (low+high dword).
  return [
    _row(0, "s_load_b64", "s[4:5], s[0:1], 0x10", "salu", ["s0", "s1"], ["s4", "s5"]),
    _row(1, "s_load_b128", "s[0:3], s[0:1], null", "salu", ["s0", "s1"], ["s0", "s1", "s2", "s3"]),
    _row(2, "v_add_co_u32", "v10, vcc_lo, s2, v0", "valu_int", ["s2", "v0"], ["v10", "vcc_lo"]),       # A addr lo
    _row(3, "v_add_co_ci_u32_e64", "v11, null, s3, v0, vcc_lo", "valu_int", ["s3", "v0", "vcc_lo"], ["v11"]),  # A addr hi
    _row(4, "v_add_co_u32", "v12, vcc_lo, s4, v0", "valu_int", ["s4", "v0"], ["v12", "vcc_lo"]),       # B addr lo
    _row(5, "v_add_co_ci_u32_e64", "v13, null, s5, v0, vcc_lo", "valu_int", ["s5", "v0", "vcc_lo"], ["v13"]),  # B addr hi
    _row(6, "global_load_b128", "v[20:23], v[10:11], off", "global_load", ["v10", "v11"], ["v20", "v21", "v22", "v23"]),
    _row(7, "global_load_b128", "v[24:27], v[12:13], off", "global_load", ["v12", "v13"], ["v24", "v25", "v26", "v27"]),
    _row(8, "ds_store_b128", "v40, v[20:23] offset:0", "lds_store", ["v40", "v20", "v21", "v22", "v23"], []),
    _row(9, "ds_store_b128", "v40, v[24:27] offset:5120", "lds_store", ["v40", "v24", "v25", "v26", "v27"], []),
    _row(10, "ds_load_b128", "v[30:33], v41 offset:0", "lds_load", ["v41"], ["v30", "v31", "v32", "v33"]),
    _row(11, "ds_load_b128", "v[34:37], v41 offset:5120", "lds_load", ["v41"], ["v34", "v35", "v36", "v37"]),
    _row(12, "v_wmma_f32_16x16x16_f16", "v[50:57], v[30:33], v[34:37], v[50:57]", "dot_mfma",
         ["v30", "v31", "v32", "v33", "v34", "v35", "v36", "v37"], ["v50", "v51"]),
  ]


class TestOperandAttribution(unittest.TestCase):
  def test_positive_ab_flow(self):
    res = attribute_operands(_positive_flow(), ABI)
    rows = res["rows"]
    self.assertEqual(res["sgpr_pointer_roots"], {"s4": "b", "s5": "b", "s0": "out", "s1": "out", "s2": "a", "s3": "a"})
    self.assertEqual(rows[6]["operand_id"], "a")   # A global load
    self.assertEqual(rows[7]["operand_id"], "b")   # B global load
    self.assertEqual(rows[8]["operand_id"], "a")   # A ds_store
    self.assertEqual(rows[9]["operand_id"], "b")   # B ds_store
    self.assertEqual(rows[10]["operand_id"], "a")  # A ds_load (exact offset match)
    self.assertEqual(rows[11]["operand_id"], "b")  # B ds_load
    self.assertEqual(rows[12]["source_operands"], ["a", "b"])  # wmma srcA=a, srcB=b

  def test_ambiguous_address_merge_is_unknown(self):
    # an address computed from BOTH A and B pointers cannot own a single operand -> load is unknown
    rows = [
      _row(0, "s_load_b64", "s[4:5], s[0:1], 0x10", "salu", ["s0", "s1"], ["s4", "s5"]),
      _row(1, "s_load_b128", "s[0:3], s[0:1], null", "salu", ["s0", "s1"], ["s0", "s1", "s2", "s3"]),
      _row(2, "v_add_co_u32", "v10, vcc_lo, s2, v0", "valu_int", ["s2"], ["v10"]),        # A
      _row(3, "v_add_co_u32", "v10, vcc_lo, s4, v10", "valu_int", ["s4", "v10"], ["v10"]),  # A + B -> merge
      _row(4, "global_load_b128", "v[20:23], v[10:11], off", "global_load", ["v10", "v11"], ["v20"]),
    ]
    res = attribute_operands(rows, ABI)
    self.assertEqual(res["rows"][4]["operand_id"], "unknown")
    self.assertEqual(res["rows"][4]["missing"], "global_load_address_provenance")

  def test_missing_kernarg_leaves_loads_unknown(self):
    # no kernarg pointer load -> no roots -> global load address untraceable -> unknown, never guessed
    rows = [
      _row(0, "v_add_co_u32", "v10, vcc_lo, s2, v0", "valu_int", ["s2"], ["v10"]),
      _row(1, "global_load_b128", "v[20:23], v[10:11], off", "global_load", ["v10", "v11"], ["v20"]),
    ]
    res = attribute_operands(rows, ABI)
    self.assertEqual(res["rows"][1]["operand_id"], "unknown")

  def test_double_buffered_ds_load_stays_unknown(self):
    # a ds_load at an offset with no exact ds_store region match -> double-buffer discriminator, not a guess
    rows = _positive_flow() + [
      _row(13, "ds_load_b128", "v[38:41], v41 offset:10240", "lds_load", ["v41"], ["v38", "v39", "v40", "v41"]),
    ]
    res = attribute_operands(rows, ABI)
    self.assertEqual(res["rows"][13]["operand_id"], "unknown")
    self.assertEqual(res["rows"][13]["missing"], "double_buffered_lds_window_binding")

  def test_manifest_projection_maps_ids_and_joins_binary(self):
    res = attribute_operands(_positive_flow(), ABI)
    paths = operand_paths_for_manifest(res, _positive_flow(), binary_sha256="ab" * 32)
    self.assertTrue(all(p["binary_sha256"] == "ab" * 32 for p in paths))
    a_load = next(p for p in paths if p["kind"] == "global_load" and p["operand_id"] == "A")
    self.assertEqual(a_load["source_operand_id"], "A")
    wmma = next(p for p in paths if p["kind"] == "wmma")
    self.assertEqual(wmma["operand_id"], "C")
    self.assertEqual(wmma["source_operands"], ["A", "B"])
    # any unknown row keeps its discriminator (no silent drop)
    self.assertTrue(all("missing" in p for p in paths if p["operand_id"] == "unknown"))


if __name__ == "__main__":
  unittest.main()
