# RA2 DEFINE_REG accumulator opt-in

**Verdict:** AMD_ISA_REGALLOC_ACCUM_RA2_PASS_DEFINE_REG_OPT_IN

native tile DS load/store: **31 (LDS, flag off) -> 9 (flag on)**; v_pin refs on=22
PC/source lds_accum_stage static: 31 -> 9
in-model token_match (flag on): True; deterministic: True; ladder: all PASS
flag-off: verified: _vpool + isel pinned branch gated on AMD_ISA_REG_ACCUM; INC0/Phase B PASS flag-off