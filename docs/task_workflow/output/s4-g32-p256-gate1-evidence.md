# S4_G32_P256 Gate-1 substrate evidence

This is a research-only WP1 artifact. It does not alter production admission,
routes, loaders, or kernels. `posthoc_q4=True` sidecars are explicitly marked
`promotable: false` and are plumbing/performance-smoke artifacts only.

## Geometry and checks

- Fixed block: 256 weights, eight groups of 32.
- Payload: eight FP16 scales (16 bytes) + 128 packed signed-nibble bytes = 144 bytes.
- Deterministic pack/decode and independent CPU block-dot oracle.
- Manifest uses canonical JSON, SHA-256 payload hashes, explicit offsets,
  padded lengths, 16-byte alignment, source/config hashes, and tie owner.
- Validation rejects truncation, corruption, unsupported format, and bad
  alignment/padding.

## Focused test result

`PYTHONPATH=extra/llm_research/quant pytest -q extra/llm_research/quant/test_s4_g32_p256.py`

Result: **4 passed** (pytest emitted only unrelated unknown-config-option
warnings for timeout settings).
