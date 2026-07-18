import struct

from tinygrad.renderer.amd.elf import assemble_linear
from tinygrad.runtime.autogen.amd.rdna3.ins import s_endpgm
from tinygrad import dtypes
from tinygrad.uop.ops import Ops, UOp

from extra.qk.mmq_hsaco_static_audit import audit_hsaco


def _binary():
  params = tuple(UOp.placeholder((1,), dtypes.float32, i) for i in range(5))
  return assemble_linear(UOp(Ops.PROGRAM, src=(UOp(Ops.SINK, src=params),)),
                         UOp(Ops.LINEAR, src=(UOp(Ops.INS, arg=s_endpgm()),)), "gfx1100")


def _listing(*, branch=False, trailing="s_code_end"):
  rows = []
  if branch: rows.append("s_branch 0 // 0000000000000100: BF820000")
  rows.append(f"s_endpgm // {0x104 if branch else 0x100:016x}: BFB00000")
  if trailing: rows.append(f"{trailing} // {0x108 if branch else 0x104:016x}: BF9F0000")
  return "\n".join(rows)


def _with_words(*words):
  binary = bytearray(_binary())
  for index, word in enumerate(words): struct.pack_into("<I", binary, 0x100 + index * 4, word)
  return bytes(binary)


def test_static_audit_reuses_saved_disassembly_and_binds_both_hashes():
  binary, disassembly = _binary(), _listing()
  result = audit_hsaco(binary, disassembly)
  assert result["schema"] == "tinygrad.amd.hsaco_static_audit.v1"
  assert result["binary_sha256"] and result["disassembly_sha256"]
  assert result["passed"] is True and result["verdict"] == "PASS"
  assert "descriptor" not in result and "resources" not in result
  assert result["termination"]["trailing_padding_dwords"] > 0


def test_static_audit_accepts_direct_branch_to_instruction_boundary():
  result = audit_hsaco(_with_words(0xBF820000, 0xBFB00000), _listing(branch=True))
  assert result["passed"] is True
  assert result["control_flow"]["direct"][0]["target"] == 0x104


def test_static_audit_rejects_branch_inside_image_but_not_at_instruction_boundary():
  listing = "s_branch 0 // 0000000000000100: BF820000\n" \
                "s_endpgm // 0000000000000108: BFB00000"
  result = audit_hsaco(_with_words(0xBF820000, 0xBF9F0000, 0xBFB00000), listing)
  assert result["passed"] is False
  assert any("instruction boundary" in finding for finding in result["findings"])


def test_static_audit_flags_indirect_control_flow():
  listing = "s_setpc_b64 s[4:5] // 0000000000000100: BEF40004\n" \
            "s_endpgm // 0000000000000104: BFB00000"
  result = audit_hsaco(_binary(), listing)
  assert result["control_flow"]["indirect_control_flow_flag"] is True
  assert result["control_flow"]["indirect"]
  assert result["passed"] is False
  assert "unexpected indirect control flow is present" in result["findings"]


def test_static_audit_rejects_decoded_or_raw_nonpadding_after_endpgm():
  decoded = audit_hsaco(_binary(), _listing(trailing="v_mov_b32_e32 v0, 0"))
  assert decoded["passed"] is False
  assert any("non-padding instruction" in finding for finding in decoded["findings"])

  binary = bytearray(_binary())
  struct.pack_into("<I", binary, 0x104, 0xDEADBEEF)
  raw = audit_hsaco(bytes(binary), _listing())
  assert raw["passed"] is False
  assert any("non-padding words" in finding for finding in raw["findings"])


def test_static_audit_fails_closed_on_invalid_inputs():
  for binary, disassembly in ((b"not-an-elf", _listing()), (_binary()[:80], _listing()), (_binary(), "")):
    result = audit_hsaco(binary, disassembly)
    assert result["passed"] is False and result["verdict"] == "BLOCKED"
    assert result["findings"]
