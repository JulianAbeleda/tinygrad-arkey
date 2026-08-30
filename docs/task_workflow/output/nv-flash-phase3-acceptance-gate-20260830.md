# Phase 3 query-tiled Flash acceptance gate

The standalone runner [nv_cleanroom_flash_fixture_runner.py](/home/ubuntu/tinygrad-arkey/extra/llm_research/prefill/nv_cleanroom_flash_fixture_runner.py) is the acceptance boundary for a Phase 3 CUDA executable.

The executable must produce a full contiguous fp32 output `.npy` and provide:

- `--timing-json` with `per_layer_us` (or `layer_us`)
- `--census-json` with the observed launch census
- canary status in the timing JSON (`canaries`)

Acceptance requires finite full-output allclose against the half-rounded Q/K/V
oracle, intact canaries, and a populated launch census. Timing is a hard
rejection at `>=79.8 us/layer`; missing timing or census is STOP. The runner
accepts a standalone CUDA executable through `--command` using `{fixture}` and
`{output}` substitutions, so no production seam is implied.
