# Packed Q4 fragment postrange corruption audit

## Finding

The first incorrect representation appears in the tensor-core permutation in
`KernelOpt.apply_tensor_cores`, before CONTRACT expansion, devectorization, or
CUDA rendering.

The Q4 graph forms eight logical groups by stacking low and high nibbles and
then reshaping.  Its correct scalar provider is:

```text
byte  = qs[(group >> 1) * 32 + k]
value = (group & 1) ? byte >> 4 : byte & 15
```

Postrange currently substitutes the tensor-core permutation through the
arbitrary STACK/reshape/ALU operand DAG.  The STACK selector and its padding
predicate are rebound to the outer group range.  The captured CUDA contains:

```text
predicate = (gidx0 < 1)
low  = predicate ? byte & 15 : 0
high = predicate ? 0 : byte >> 4
fragment_byte = low + high
```

This produces low nibble for group 0 and high nibble for groups 1--7.  Correct
parity alternates low/high for every group.  A direct int8 load has no STACK
selector and therefore survives the same permutation.

## Phase trace

1. Pre-TC graph: correct STACK(low, high), reshape, and int8 cast.
2. Postrange TC rewrite: selector becomes `gidx0 < 1`; first incorrect node.
3. CONTRACT/UNROLL expansion: preserves the incorrect conditional values.
4. Linearizer/devectorizer: constructs `signed_char8` from those values.
5. CUDA renderer: faithfully emits the incorrect predicate and a valid
   `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`.

Instruction presence is consequently not a correctness proof.

## Generic resolution

Fail closed for tensor-core operands with range-dependent STACK/WHERE/ALU
providers until a typed provider is installed.  The provider must expose a
logical matrix/K scalar query and fragment dtype; postrange must query it after
selecting TC axes and lane coordinates instead of substituting permutations
through its implementation DAG.

The Q4 typed query and a discriminator are pinned in
`extra/llm_research/prefill/q4k_q8_imma_oracle.py`.  For packed byte `0xD2`,
correct groups are `[2,13,2,13,2,13,2,13]`; the observed rewrite produces
`[2,13,13,13,13,13,13,13]`.  Six CPU-oracle tests pass.

No compiler or production route was edited.  The implementation agent owns
the fail-closed admission and composite provider/native microgate.
