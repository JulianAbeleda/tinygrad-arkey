# NV P9 continuous ctx2048 R7 after memory-lifecycle fixes

Commit `959809bec`, ordinary defaults, one continuous request, request-scoped
decode prewarm, max-context 2560. The previously failing ctx2048 prefill now
completes after the nested-resolution memo was scoped per schedule and ordinary
no-reuse concrete prefill was kept eager.

The three ten-token windows cover logical contexts 2051 through 2081. They use
`rollout_jit_flash_live[18]` with 418 selected programs per token. Total samples
are 43.631822, 43.715830, and 43.692084 ms; aggregate median is 4.369208 ms/token
or 228.874 tok/s. GPU state is P0, 2572 MHz SM, 14001 MHz memory, 48 C, with no
reported throttle reason. First prefill token is 13876. This is continuous
coverage and does not replace an independent multi-request test.
