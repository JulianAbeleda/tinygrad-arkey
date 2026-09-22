# NV P9 continuous ctx2048 prefill failure

A fresh normal-route process used max context 2560 and request-scoped
prewarming for a 2048-token prompt. No promotion flag changed. Its first prompt
prefill failed before decode census or timing with `NV_ERR_NO_MEMORY`: an
allocation of 10.62 MiB failed while the allocator reported 29.90 GiB used.

Unlike the ctx1024 independent-request failures, this is a first-request
failure. Normal-default intrinsic coverage therefore currently passes one
ctx1024 request but is blocked by prefill memory before ctx2048. The exact
retained buffer/workspace owner still requires attribution. ctx4096 was not
launched because it crosses the same unresolved normal-prefill boundary.
