# Deterministic staged `attn_qo` certification artifacts

This directory durably retains the exact frozen bundle and C1-C6 evidence for the selected
`attn_qo (512,5120,5120)` staged family generated from clean revision
`951d3615c2357d2bb0ef2f4b62339b45ce39597f`.

- Family identity: `sha256:2cfc30075f8024cee8a927c2c3de2e87eef3db6d83882da69faa0fe0a3cc1e4f`
- PROGRAM key: `3f478e6d89a2de467f6b7d1ca18418cdfd0cdb19de05db1d66608e65a5e6475f`
- HSACO SHA256: `dfb213624287a8dec10f8646d8c16e49651efee8e0ca27c67ff982b0d6b050bf`
- Staged-family manifest SHA256:
  `ca673be2a2989aa29a184d28d440121e4d1a2bd321de70e676b1b578e76bb322`

The bundle files are byte-for-byte copies of the independently reproduced `r1` bundle. The `evidence/` directory
retains the reproducibility result, static C1-C3 certificates, and isolated PM4/AQL C4-C6 results. C7 memory
admission, C8 matched timing, C9 whole-model validation, and production promotion are not claimed here.
