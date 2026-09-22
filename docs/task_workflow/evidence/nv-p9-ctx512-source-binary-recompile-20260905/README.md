# NV P9 ctx512 source-to-binary recompilation audit

Every retained source text was verified against its SHA-256 table key, then
compiled directly with production `NVRTCCompiler`: NVRTC 13.2, `sm_120`, cubin
output, `--minimal`, and the recorded include paths. The result positively
matches 23 of 29 unique selected PROGRAM binaries byte for byte. Six generic
`E_*`/`r_*` programs compile deterministically with current NVRTC but do not
match their captured binary hashes.

The 23 matches prove their retained SOURCE-to-binary transport. They do not by
themselves prove generator/registry lineage. The six mismatches remain
fail-closed; they may reflect a historical compiler cache/version, but that is
not established. All 418 launches still contain zero recognized
`native_nv_program` marker, so no known llama packed binding is recognized,
but the universal no-llama-cubin gate remains open pending the six mismatches
and generator/build-path lineage.
