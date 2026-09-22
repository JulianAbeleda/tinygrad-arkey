# NV native/CUDA token-divergence triage

Date: 2026-08-04.  Diagnostic only; no production route or default was changed.

## Superseding P1-B addendum

The interim interpretation below is superseded by
`nv-decode-parity-p1b-shared-logits-record-20260804.md`.  On the exact shared
d512 token IDs, native NV and CUDA produce bitwise-identical full logits and
both select llama's argmax `13876`.  Direct native production sampling instead
returns invalid one-past-vocabulary sentinel `151936`.  The divergence is
therefore localized after logits in the temperature-zero sampling/scalar path,
not in prompt-final model numerics.  Retain the hashes below as provenance,
but do not call `151936` a valid prompt-final sampled token or an approved
correctness pin.

## Verdict

This section records the initial triage before the superseding shared-logits
probe.  Its conclusion that no native failure was established is now refuted:
the native fixed-depth stream is deterministic, but its `151936` value is an
invalid post-logits sampling result.

| arm | fused-prefill setting | d512 prelude | first measured decode token | stream |
| --- | --- | ---: | ---: | --- |
| native NV | forced off by `/tmp/b3_runner.py` | 151936 | 151936 | native pin |
| CUDA | default | 13876 | 38835 | CUDA loop |
| CUDA | forced off by the same wrapper | 13876 | 38835 | CUDA loop |

The last row is the decisive configuration control.  Turning off the broken
fused-prefill route does not change CUDA's result, so
`PACKED_FRAGMENT_LOAD` / the wrapper is not the owner of this divergence.

## Provenance check

The native ten-token stream is `[151936] * 10`.  Its SHA-256 is
`13cb9c480fba287939e74663336233941a23c27c5fbbd3954daeb9c69e0875fd`.
At the historical qualification length, `[151936] * 20` hashes to
`9d6b3787cef8c4a7b208df30c05c049f692a5ebc80dd19c2994dd54c18e789b9`,
exactly the NV fixed-depth pin used by the existing qualification records.

CUDA's observed d512 loop begins
`38835,34208,13,279,3974,13876`; its ten-token hash is
`317cec68399bbba606b7a6e5645d326e5d8734167672d24e63a750b95c909a09`.
That is the same CUDA divergence explicitly recorded in
`nv-decode-overlap-route-b-viability-record-20260804.md`, not a new native
regression.

The controls used the same model, prompt evidence hash
`e536e95f0307d714bc77b286f8ee4a6f7f38dc5ebc3c7321eda2170149308c49`,
seed, `sdpa` decode route, and authority harness.  The only relevant changed
axis in the failed comparison is `DEV`.

## Consequence

Do not block native-NV wall diagnostics on equality with CUDA token IDs, and
do not treat CUDA as a numeric control for native decode.  Native's old pin is
only an internal regression pin, however; it is not a substitute for a fresh
cross-engine semantic oracle.

The falsifiable next correctness gate is a shared-input logits comparison at
the exact prompt-final boundary: materialize the d512 prompt-final activation
or logits from llama.cpp and native NV, compare a fixed sampled row and argmax,
then separately compare CUDA.  If native differs from that shared oracle, the
native runtime/kernel path is genuinely faulty; if native agrees and CUDA does
not, this triage is confirmed and CUDA must remain a diagnostic-only route.
