# Q4 native-fragment provider transfer seam

The exact selected generated gate/up cubin has the same 256 useful IMMA
instructions as the native reference, but its static body has 640 LDS, 168 LDG,
544 PRMT, and 26 STL / 21 LDL instructions.  The extracted native cubin has 440
LDS, 114 LDG, and no corresponding spill class.  This selected a narrow attempt
to reuse tinygrad's existing cooperative `native_fragment_x2` substrate for the
Q4 B fragment.

The current generated Q4 provider is
`tinygrad.codegen.opt.packed_weight.Q4KInt8FragmentProvider`.  It scalarizes 16
codes before shared staging.  `PackedFragmentSpec` and `native_fragment_x2`
exist, but `kernel_lds` admits a packed native fragment only for Q6, and the
retained `fragment_b_spec` has no Q4 consumer.

A local, reverted prototype added an explicit Q4 K32 spec and marked the B
`CONTRACT` after `apply_opts`, while descriptor element ownership was still
available.  The marker fired on the selected legacy LDS path.  Source
construction then failed before GPU execution at final UOp verification:

```
Ops.CAST dtypes.weakint <- Ops.STACK dtypes.int
```

The expander converts the K-substep/subtile address to a `STACK` while retaining
the opaque 8-byte fragment as one node.  Native x2 requires one scalar shared
address per lane and outer unroll instance.  Selecting element zero would
silently assume a descriptor mapping and was rejected.  A correct transfer
requires an expander representation/rule that distributes each outer UNROLL
address element to one scalar x2 carrier while preserving the inner `char8`
fragment.  No source containing `ldmatrix.x2`, correctness result, or timing is
claimed.

This closes direct provider reuse as an integration-only change.  Re-enter only
with the nested-vector/outer-UNROLL compiler hook and a 32-lane, two-K32-substep
mapping proof before GPU execution.
