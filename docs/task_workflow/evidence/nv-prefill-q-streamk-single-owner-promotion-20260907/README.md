# Generated Q Stream-K promotion (2026-09-07)

The pp512 generated Q projection now selects the existing single-owner Stream-K capture in ordinary compiler mode. The capture realizes its opaque multi-output main once, retains only the output AFTER owner, and feeds raw workspace allocations to the fixup. Q retains the qualified shared flat Q8 record with Q4 V. `NV_COMPILER_Q4_Q_STREAMK=0` is the explicit rollback.

The isolated canonical Q lifecycle (`q_streamk_singleowner_r15.json`) improved from 221.957 us to 200.036 us per call (21.921 us, 0.789 ms projected across 36 layers), with read-only inputs and max absolute difference 6.68e-6 from reduction order.

The matched warmup-9 A/B/C model bracket measured controls at 44.651553/44.635352 ms median and 44.523272/44.408405 ms minimum. The candidate measured 43.787468 ms median and 43.637316 ms minimum: midpoint wins of 0.8559845 ms median and 0.8285225 ms minimum, both above the frozen 0.5 ms gate.

The final ordinary-selector deep smoke (`qstream_ordinary_deep3.json`) is PASS: generated Q/O mains 36/36, total generated mains 252, Q8 producers 234, active fixups 162, canonical/unique weights 252/252, zero copies/overlays, token 198, and exact logits plus every captured stage over three replay cycles.

The promotion also keys capture reuse by stable binding/program contract rather than JIT alone. Focused tests prove gate, Q, and O bindings cannot collide while repeated lookup of the same binding reuses its capture.
