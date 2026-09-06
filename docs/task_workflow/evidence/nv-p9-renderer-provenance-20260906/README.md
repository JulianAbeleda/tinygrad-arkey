# NV P9 selected-PROGRAM construction provenance

Commit `d307c3e6f`, ordinary decode defaults, one fresh context-512 request with
request-scoped prewarming. The captured production graph selects
`rollout_jit_flash_live[6]` and contains 418 PROGRAM launches representing 29
unique PROGRAMs. Every launch records the exact typed construction provenance
`tinygrad_renderer`, `tinygrad.renderer.cuda.CUDARenderer`, `NV`; the audit has
29 renderer-owned unique programs and zero unknown or native-precompiled
programs.

Every retained SOURCE text also hashes to its census key. Fresh NVRTC 13.2
compilation at `sm_120` reproduces 23 cubins byte for byte. The other six map
from their exact SOURCE keys to the captured cubins in the production
`compile_nv_sm_120` cache. Thus construction ownership and current
SOURCE-to-binary transport are positive for all 29 selected programs.

This closes selected-graph ownership for this context-512 normal-route census.
It does not prove fresh compiler reproducibility for the six cache-bound
binaries, identify semantic roles for every generic program, or establish
endpoint parity. The single measured token was 4.4505 ms at P0 and is census
service only, not a replicated latency comparison.
