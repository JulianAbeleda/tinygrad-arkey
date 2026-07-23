# G2 shared K/V LDS replay verdict

The exact G2 Q32/Hq4/Hkv2/Hd128 candidate was measured with preallocated outputs and synchronized TinyJit replay. It remained bit-identical to the established single-wave route, but did not pass performance/resource admission.

The candidate reduced each compiled loop body's K and V traffic from 128 scalar half load sites to four half8/b128 sites. That saving required two workgroup barriers per KV-tile iteration and raised resources from 246 to 250 VGPRs, from 512 to 9,216 LDS bytes, and from zero to 36 private bytes.

KV64 was only 0.38% faster, KV512 was 0.20% slower, and KV4096 was 1.65% slower by median. The staged path was reverted; the generalized G2 fence/lane ABI remains. G4/G5 were not extended because G2 failed the prerequisite admission gate.
