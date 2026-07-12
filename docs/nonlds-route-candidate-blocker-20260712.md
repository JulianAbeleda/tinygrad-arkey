# Non-LDS route candidate blocker

Under gate/up-only policy, `attn_qo` resolves to the existing lean pipe route
with exact shape `M=512,N=4096,K=4096`. Its bounded surface is available via
`extract_wmma_pipe_spec`, including tile, pipeline depth, and pipe load knobs.

Candidate promotion is currently blocked at the backend boundary:
`lower_wmma_pipe_spec` intentionally raises `NotImplementedError` because the
backend-owned generated pipe lowerer does not yet exist. The runtime therefore
cannot produce a distinct exact candidate identity on this surface without
falling back to the raw oracle or introducing a new emitter, both outside this
scope. Host validation and the exact blocker test are in
`test_nonlds_route_candidate_blocker.py`.
