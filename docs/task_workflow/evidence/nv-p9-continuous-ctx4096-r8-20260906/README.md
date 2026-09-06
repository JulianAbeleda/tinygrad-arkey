# NV P9 continuous ctx4096 R8 after memory-lifecycle fixes

Commit `b0d1f4f23`, ordinary defaults, one continuous request, request-scoped
decode prewarm, max-context 4608. Prompt prefill and all measured windows pass.

The three ten-token windows cover logical contexts 4099 through 4129. They use
`rollout_jit_flash_live[34]` with 418 selected programs per token. Total samples
are 46.074111, 46.095462, and 46.130968 ms; aggregate median is 4.609546 ms/token
or 216.941 tok/s. GPU state is P0, 2572 MHz SM, 14001 MHz memory, 48--49 C, with
no reported throttle reason. First prefill token is 34208. This is continuous
coverage and does not replace an independent multi-request test.

Offline NVRTC 13.2 recompilation at sm_120 exactly matches 23 of 29 unique
retained SOURCE/cubin pairs. Six generic E/r programs mismatch deterministically.
No recognized native-precompiled marker is present. This proves transport for
the 23 matches; it does not prove generator lineage or universal absence of an
unknown native transport.
