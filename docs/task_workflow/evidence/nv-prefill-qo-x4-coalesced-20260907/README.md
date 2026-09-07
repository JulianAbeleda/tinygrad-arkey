# Q/O coalesced Q4-A/x4 transfer

The gate/up coalesced swapped-output contract generalizes to physical `(4096,512)` Q/O Stream-K geometry. Canonical block0 Q and O comparisons pass: max absolute errors are `5.7220459e-06` and `9.5367432e-06` respectively.

The no-share full current252 smoke is PASS with token198, exact replay3, canonical252/252 weights, generated Q/O mains72, active fixups162, and zero overlays/copies. Capture-local records, outputs, partials, and IDs are retained and reset across traces.

Q/Q4V flat-record sharing is deliberately fail-closed for Q/O Stream-K. Two attempted ownership forms were nondeterministic in full replay (max logits differences `0.109998` and `0.171922`) even though no-share replay is exact. The previously qualified wide Q/Q4V sharing remains unchanged. Timing must compare matched no-share arms and debit its measured ~0.226 ms benefit before selection.
