import hashlib, unittest

from test.unit.test_amd_isa_wmma import _tc_matmul_ast, _tc_matmul_ast_k64, _tc_matmul_ast_k64_rolled
from tinygrad.codegen import to_program, to_program_cache
from tinygrad.helpers import Target
from tinygrad.renderer.isa.amd import AMDISARenderer
from tinygrad.uop.ops import Ops


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
