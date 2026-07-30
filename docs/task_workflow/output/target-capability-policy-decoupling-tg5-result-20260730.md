# TG5 result — quant path proven on Metal

Date: 2026-07-30

Status: **TG0-TG5 complete and measured on Metal.** Diagnostic-free: the path is reached through the production
capability/policy split, not through a patch. Not pushed, not promoted to `dev`/`master`.

Scope: `docs/task_workflow/input/target-capability-policy-decoupling-scope-20260730.md`.

## Measurement

Apple M4 10-core / Metal, Qwen3-8B-Q4_K_M (`d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`),
depth 128, one measured decode token after two warmups, `METAL_HYBRID_REPLAY=1`.

| arm | reps (tok/s) | median | spread |
| --- | --- | ---: | ---: |
| baseline (TG0, `e01bac5fe`) | 5.3862 / 5.3816 / 5.4050 | 5.386 | 0.43% |
| after TG1-TG3 + TG2 amendment | 16.9947 / 17.1353 / 16.9099 | **16.995** | 1.33% |

**Delta: +215% (3.16x).** Both spreads sit inside this machine's 0.4-2.6% band, and the gap is two orders of
magnitude larger than the band.

**Correctness**: every rep in both arms produced prelude token `13876` and generated token `38835`, matching the
frozen baseline and llama.cpp. `prompt_evidence` sha256 identical throughout. This is a correctness-preserving
change, not a speed/accuracy trade.

**Binding**: 253 primitives bound (216 Q4_K + 37 Q6_K), reproducing the earlier throwaway-patch diagnostic through
the production path.

## Two findings this run turned from inference into evidence

### 1. MR3 hybrid replay is a hard prerequisite, not a neutral change

The first TG5 attempt failed outright:

```text
tinygrad.engine.jit.GraphException: Metal ICB offset exceeds 0xffffffff in partitioned replay
```

`storage_mode="shared"` aliases packed weights into the 4.68 GB GGUF buffer, and Metal's indirect command buffers
encode offsets in 32 bits. The run only completes under `METAL_HYBRID_REPLAY=1`.

MR4 classified hybrid replay `INCONCLUSIVE` with a paired wall delta indistinguishable from zero, measured in
isolation. That classification stands and is not amended here. What is added: **in combination with installed quant
primitives on Metal, hybrid replay is required for the workload to run at all.** A change can be neutral in
isolation and load-bearing in combination. This is TG6's record.

### 2. "No promotion record" means undecided, not denied

TG3 made `promoted_targets is None` mean no restriction recorded, so a target clearing capability is admitted. This
was necessary to preserve AMD, which has no policy artifact loaded in production today — denying by default would
have disabled the primary target.

Consequence: **TG4 was not a prerequisite for TG5.** Once Metal satisfied capability, it bound immediately, and TG4
becomes a matter of recording evidence rather than enabling a path. This default deserves an explicit decision: any
future target that clears capability is admitted silently until a record says otherwise. That is the opposite of
fail-closed, and it is a deliberate trade for AMD compatibility, not an oversight.

## Cross-package seam found and repaired

TG2 left `MetalRenderer.wave_size` unreported and TG3 required strict `wave_size == 32`. Each followed scope §3.3
correctly in isolation; together they made Metal permanently unadmittable — it failed capability before policy was
ever consulted, so no TG4 record could have helped.

Repaired in `4ca4afc4a` by reporting Metal's verified 32-wide simdgroup. §3.3 bars inventing a width for an unknown
target, not reporting a known one, and the unknown-target guarantee is now pinned against a CPU renderer instead of
against Metal. A device-level `threadExecutionWidth` query would be stricter and remains a follow-up.

This is the second integration seam of the day where two individually-correct halves produced a silent null result.
Packages that meet at an interface need the interface itself asserted, not just each side.

## Commits (branch `exp`, not pushed)

| commit | package |
| --- | --- |
| `e01bac5fe` | scope |
| `d117d0dd0` | TG0 baseline + `kernel_log_diff` parser with captured fixture |
| `b3c2022a1` | TG1 cross-lane shuffle as a renderer-lowered operation |
| `b69817531` | TG2 capability facts: shuffle availability, wave_size, ICB offset limit |
| `b29fbb806` | TG3 capability/policy split with distinct census reasons |
| `4ca4afc4a` | TG2 amendment: Metal's verified simdgroup width |

## Standing caveats

- **No AMD hardware.** AMD non-regression is structural throughout: rendered-source byte-equality for TG1, admitted-set
  equality for TG3. None of it is an execution result, and it must not be quoted as one.
- Absolute numbers carry the `90e93875c` devectorizer regression (2.03x, still unrepaired, out of scope per §7). The
  3.16x delta is valid; the absolutes will move when that packet lands.
- TG7-TG10 remain **unmeasured hypotheses**. TG5 proves the pattern for one gate; it does not license assuming the
  other four behave the same way.
