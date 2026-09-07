# Q4-down single-owner x4 qualification

The generated Q4-down Stream-K main now retains only output's lazy AFTER owner;
raw partial/ID allocations feed fixup. A canonical one-call fixture contains
exactly producer1/main1/fixup1, is allclose (`max_abs=1.907e-5`,
`mean_abs=4.753e-7`), read-only, and measures wide481.144us versus x4
Stream-K363.824us in alternating R15.

The composed deep model smoke passes: Q4-down producer/main/owners18, O
main/fixup/owners36, total generated mains252, Q8234, active fixups126,
canonical weights, no copies/overlays, token198, and exact logits/every stage
through three cycles.

Stable control/candidate/control R9 on promoted O authority measured control
median midpoint46.093750ms against candidate45.917660ms, a0.176090ms win.
Minimum midpoint45.7572735ms against candidate45.252720ms is a0.5045535ms win.
The median fails the frozen0.5ms gate, so Q4-down x4 remains default-off.
