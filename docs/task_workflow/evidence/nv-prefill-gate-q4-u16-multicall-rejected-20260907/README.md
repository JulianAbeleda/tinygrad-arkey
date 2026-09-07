# Q4 metadata uint16 carrier: rejected

An aligned uint16 LDS load followed by an explicit bitcast to half retained PRMT48, IMMA256, LDS384 and zero spill. Single-call R15 improved 342.634 to333.998 us and a sequential three-activation/10-cycle fixture was exact.

The required composed eight-call fixture failed every candidate cycle, with maximum errors0.637–1.179, while all control calls stayed exact. This reproduces the full-graph hazard without model complexity and proves nonvolatile ushort aliasing is unsafe under multi-call scheduling. No code was integrated.
