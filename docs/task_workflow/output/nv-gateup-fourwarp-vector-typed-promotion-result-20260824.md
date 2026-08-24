# NV gate/up four-warp vector typed-output promotion

Date: 2026-08-24  
Verdict: `PROMOTED_BOOKED`

The original four-warp closure was incomplete: it tested scalar loads, and
the first vector follow-up entered through an opaque output boundary that
added 36 materialization nodes.  Preserving the installed declared typed
output contract removes those nodes and converts the kernel win to wall.

## Gates

| gate | control | candidate | result |
| --- | ---: | ---: | --- |
| isolated hot kernel | 21.968 us | 21.021 us | bit-exact, -4.31% |
| opaque candidate census | 462 nodes | 498 nodes | reject; extra 18-group materialization |
| typed candidate census | 462 nodes | 462 nodes | pass; extra group absent |
| typed candidate device union | 4259.750 us retained control | 4216.750 us | about -43 us |

The typed output is `fp16`, shape `(12288,)`, with
`epilogue_absorption_admitted=True`, matching the installed one-warp vector
route.  The candidate uses 128 threads per row, four warps, vector Q4/half4
loads, 47 registers in the isolated compile, 32 bytes shared memory, and no
spills.

## Decisive wall bracket

Fresh d512, count 32, reps 9, control/candidate/control, pinned clocks:

```text
control A       4500.911 us/token
candidate       4443.726 us/token
control C       4493.199 us/token
midpoint        4497.055 us/token
recovery          53.329 us/token
speedup             1.200%
```

All 27 windows share token-stream SHA
`28b0923439dde9076100800bfaed6a9a8a7e00e396691776a16514a609e0543a`.
The candidate beats both controls.

## Promotion and endpoint

The route policy promotes only `NV/sm_120`.
`TINYGRAD_Q4K_GATE_UP_FOUR_WARP_DISABLE=1` restores the one-warp vector route.
A no-override production verification measures `4464.577 us/token` median
over seven windows with the expected token SHA.

Conservative booking:

```text
prior endpoint       4515.395719 us/token = 221.464532 tok/s
booked recovery        53.329000 us/token
new endpoint         4462.066719 us/token = 224.111405 tok/s
remaining to 227       56.780375 us/token =   2.888595 tok/s
```

Evidence is under
`docs/task_workflow/evidence/nv-ranked-parity-campaign-20260824/`.
