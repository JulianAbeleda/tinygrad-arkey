# Pipe transport ABI inventory

The current insertion chain is:

1. `rangeify` creates a `SINK` carrying `KernelInfo`.
2. `postrange` preserves `KernelInfo.candidate_context` and applies warmstart
   contexts keyed by `(frozenset({M,N}), K)`.
3. `codegen.to_program` lowers the sink to a `PROGRAM` with `ProgramInfo`; the
   candidate context is read from `prg.src[0].arg` for cache/resource identity.
4. `engine.realize.get_runtime` turns the program into the runtime call tuple.
5. `HCQGraph` captures those calls and profiles each dispatch as a
   `ProfileGraphEntry`.

The safe pipe insertion point is before step 3: attach a typed
`KernelCandidateContext` to the ordinary `A@B` sink, then let normal
`to_program`/`get_runtime`/HCQGraph machinery run. The context must not contain
native instructions; `ProgramInfo` and runtime launch metadata remain compiler
owned. Cache identity is derived in `codegen/__init__.py` from the context and
program source. Any implementation that returns an `Ops.INS` tuple bypasses
this ABI and is not a pure transport.
