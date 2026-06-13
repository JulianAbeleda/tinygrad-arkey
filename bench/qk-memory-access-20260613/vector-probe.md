# QK Integer Vector Load Probe

Capability probe for integer vector global loads on the AMD path.

## Summary

- device: `AMD`
- n_words: `4096`
- normal UOp uint4 load supported: `False`
- raw custom uint4 escape supported: `True`

| mode | exact | mismatches | first mismatch | device ms | interpretation |
|---|---:|---:|---:|---:|---|
| `scalar` | `True` | 0 | n/a | n/a | Scalar correctness baseline. |
| `uop_vec_request` | `False` | 3072 | 1 | n/a | A requested uint32.vec(4) UOp copy is expected to be non-exact today if codegen keeps only the scalar lane. |
| `custom_uint4` | `True` | 0 | n/a | n/a | Raw custom C can force a uint4 load/store when it supplies its own vector typedef. |

First values:

- `scalar` got `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` expected `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- `uop_vec_request` got `[0, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 0]` expected `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
- `custom_uint4` got `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]` expected `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]`
