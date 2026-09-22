# Current234 ordinary-path diagnostic

The ordinary `Transformer.__call__` path was run after Q4-down promotion with
the explicit compiler pp512 stack and its normal defaults.  It selected no
FFN-down FP16 overlays across all 36 layers and returned the same finite token
on all calls.  This confirms that the production loader removes both the 18 Q4
and 18 Q6 down overlays.

The diagnostic status remains FAIL because `prefill_workload_reuse=False`
resets the concrete prefill TinyJit after every request.  Consequently the
post-run object exposes zero captured PROGRAM calls and repeated compilation
makes the 2.77--3.11 s samples non-authoritative.  This known diagnostic
lifecycle limitation does not replace the direct current234 structural census
or its fresh-process performance bracket.
