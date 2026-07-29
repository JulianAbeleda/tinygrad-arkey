::: tinygrad.dtype.DType

::: tinygrad.dtype.dtypes
    options:
        members: true
        members_order: source
        show_labels: false

::: tinygrad.dtype.ConstType

## Ownership

At renderer boundaries, `DType` describes the scalar numeric type. A UOp's shape determines its value
lanes, and each target renderer owns the source-language spelling. Call `render_dtype(dtype)` for a
scalar, `render_type(uop)` for a shaped value, or `render_vector_dtype(dtype, lanes)` when an ABI requires
an explicit native vector type.

Legacy vector and pointer state still exists inside `DType` while EXP migrates the remaining codegen and
AMD fragment contracts. The staged migration, gates, and current audit are recorded in
[DType orthogonality migration](dtype-orthogonality-migration-20260729.md).
