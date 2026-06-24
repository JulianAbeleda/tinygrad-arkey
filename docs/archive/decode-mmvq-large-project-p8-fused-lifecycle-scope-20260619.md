# Decode MMVQ large project P8 fused lifecycle scope - 2026-06-19

Purpose: execute the full 1-4 sequence after P7d/P7e closed imported Q4 artifact routing as a local timing win.

## Questions

1. **Lower-bound model:** is a fused q8 producer + multi-consumer Q4 MMVQ lifecycle worth building?
2. **Current native expressibility:** can current tinygrad UOps/COMGR/AMD DSL express it well enough?
3. **Handwritten prototype:** is there an existing handwritten/artifact route that proves the primitive can clear the
   local gate?
4. **Decision:** should decode proceed through artifact routing, native renderer transfer, or stop?

## Gates

- P8a passes if the additive lower bound for one q8 producer plus gate/up consumers is below the `1.10x` local gate.
- P8b passes as a native stop if current native COMGR/DSL attempts fail the same gate.
- P8c passes if the handwritten artifact route clears the local gate and has graph-route evidence.
- P8d must state one of: stop, research artifact route, or native renderer project.

## Inputs

- P7e `ffn_gate/up` baseline and imported timing.
- P5/P6 producer and imported-consumer device timings.
- q8 handwritten lifecycle artifacts.
- AMD scheduler/codegen capability map.
- W==D decode baseline and q8-route rows.
