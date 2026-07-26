# ctx128 scalar output-projection store

The 14B ctx128 prefill compile can leave an output projection as a vector `STORE`
whose target is duplicated scalar `LOAD(INDEX(GLOBAL,...))` nodes. C-style rendering
turns that target into `make_floatN(...) = ...`, which is not an assignable lvalue.

`_devec_output_projection_store` recognizes this scalar form as lane zero of the
existing GLOBAL output-load route. It remains fail-closed: only contiguous, equal-size
duplicate groups with distinct values are reduced, and the reduction is ADD-only.
No LOCAL or REG load is accepted by this repair.
