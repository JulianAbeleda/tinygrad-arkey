# Cooperative MMQ MFMA lowering boundary

The exact current compiler path is: contraction selection creates `Ops.WMMA`
in `tinygrad/codegen/opt/postrange.py`; register/stage pipelines preserve its
three operands; `tinygrad/renderer/llvmir.py::render_wmma_amd` lowers CDNA
WMMA to `llvm.amdgcn.mfma.*`; HIP rendering uses the corresponding
`__builtin_amdgcn_mfma_*` spelling.  For the supported CDNA candidate the
descriptor is `N,M,K = 16,16,16`, wave size 64, A/B are `half.vec(4)` and C/D
are `float.vec(4)` per lane.  The resulting instruction must be
`llvm.amdgcn.mfma.f32.16x16x16.f16` (or disassembly
`v_mfma_f32_16x16x16_f16`).

`extra/qk/mmq_mfma_lowering.py` is an evidence adapter only. It does not claim
that a cooperative tile has been lowered from geometry, and it does not touch
Q4/Q6 emitters or route selectors. Missing target, operand, lowering, or final
instruction evidence fails closed. A cooperative-dot-specific compiler rewrite,
lane/owner map, LDS synchronization proof, and edge-predicate lowering remain
unimplemented; this adapter must not be used as a substitute for those.
