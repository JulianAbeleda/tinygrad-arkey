# NVIDIA Q4 weight-as-A x4 substrate (2026-09-06)

The typed compiler context can reverse the internal Q4/Q8 IMMA operands, compute weight@activation.T, and interpret C as the transpose back to the requested MxN result. Admission fails closed outside the exact NVIDIA Q4_K/Q8 descriptor and the order is part of canonical candidate identity.

The native Q4 A carrier uses one byte-addressed `ldmatrix.x4` per cooperative char16 fragment. Its LOCAL pointer/index contract and the lane0-31 x K32-substep0-1 address map are unit checked. The scalar/native-x4/scalar R9 medians are 309.231/301.656/309.973 us; x4 saves 7.946 us (2.57%) against the scalar midpoint. Every arm checks all 2,097,152 outputs; x4 max_abs is 4.88e-4 with canonical read-only inputs, 32 signed IMMA, 8 LDSM, and zero local spill.

The initial generated CUDA source expanded from about 64 KB to 1.49 MB because carrier projections inherited the full 20,480-byte LDS shape. Scalar pointer formation at the specialized marker boundary severs that irrelevant shape: the requalified source is 59,211 bytes, with identical correctness and SASS structure. Its R9 median is 303.680 us, retaining a 5.922 us win against the original scalar midpoint. Rollback is omitting the typed `native_weight_fragment="q4_a_x4"` context value.
