# Shared-attention compiler evidence

The `tinygrad.shared_attention_compiler_capture.v2` records in this directory
supersede the original v1 captures. V2 preserves the same full-output numeric,
WMMA-role, allocation, and resource proof and adds an explicit synchronization
contract. For the admitted gfx1100 launch, that contract proves one wave per
workgroup, one ordered LDS completion wait, and zero workgroup barriers.

V1 remains available through repository history only and must not be used as
current proof authority.
