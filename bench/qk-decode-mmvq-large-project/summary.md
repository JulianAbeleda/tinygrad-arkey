# Decode MMVQ large project P0

- commit: `e72957644`
- object exists: `True`
- Q4_K/Q6_K candidate functions: `22`
- Q4_K/Q6_K descriptors: `22`
- verdict: `P0_PASS__SOURCE_IMPORT_P1_IS_LOADABLE_DESCRIPTOR_SMOKE`

## Candidate Snapshot

- `Q6_K` ncols `2` bools `0/0`: VGPR `36`, SGPR `30`, kernarg `144`, wgmax `32`
- `Q6_K` ncols `7` bools `0/0`: VGPR `66`, SGPR `30`, kernarg `144`, wgmax `32`
- `Q6_K` ncols `1` bools `0/1`: VGPR `26`, SGPR `24`, kernarg `144`, wgmax `64`
- `Q4_K` ncols `2` bools `0/0`: VGPR `40`, SGPR `28`, kernarg `144`, wgmax `32`
- `Q4_K` ncols `7` bools `0/0`: VGPR `78`, SGPR `28`, kernarg `144`, wgmax `32`
- `Q4_K` ncols `1` bools `0/1`: VGPR `23`, SGPR `24`, kernarg `144`, wgmax `32`
- `Q6_K` ncols `5` bools `0/0`: VGPR `52`, SGPR `30`, kernarg `144`, wgmax `32`
- `Q4_K` ncols `5` bools `0/0`: VGPR `65`, SGPR `28`, kernarg `144`, wgmax `32`
- `Q6_K` ncols `1` bools `1/1`: VGPR `33`, SGPR `42`, kernarg `144`, wgmax `64`
- `Q6_K` ncols `3` bools `0/0`: VGPR `42`, SGPR `30`, kernarg `144`, wgmax `32`
- `Q6_K` ncols `8` bools `0/0`: VGPR `72`, SGPR `30`, kernarg `144`, wgmax `32`
- `Q4_K` ncols `1` bools `1/1`: VGPR `34`, SGPR `42`, kernarg `144`, wgmax `32`

## Decision

Start with P1 source/object import. The object already has .kd descriptors and AMDGPU metadata; the next gate is
whether tinygrad HCQ can load a selected descriptor by name without HIP runtime and without unsupported relocation issues.
