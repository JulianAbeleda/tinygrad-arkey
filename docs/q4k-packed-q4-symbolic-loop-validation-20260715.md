# Packed-Q4 symbolic-loop validation

`extra/qk/q4k_symbolic_loop_validation.py` is a small compiler-independent
witness for the packed-Q4 loop geometry `M=32, N=32, K=512` with 16×16 output
tiles. It enumerates the four symbolic output tiles and their logical Q4 group
addresses, then fails if weight or activation addresses alias or if a tile is
stored more than once.

The witness expects 32×16 unique logical weight addresses and 32×16 unique
activation addresses. Q4_K's 512 K elements form two 256-element superblocks
and sixteen 32-element groups. It validates indexing ownership only; it does
not alter or depend on compiler, scheduler, emitter, or route-selection code.

Run it directly:

```bash
python3 extra/qk/q4k_symbolic_loop_validation.py
python3 -m pytest -q test/unit/test_q4k_symbolic_loop_validation.py
```
