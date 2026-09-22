# Current234 gate/up Q8 pair reuse

One graph-owned Q8 record was shared across each ordered gate/up pair inside the
promoted Stream-K capture. The candidate retains all 72 mains and fixups while
reducing the full current234 producer count from 234 to 198. It passes the
structural census, token 198, and 20/20 exact recurrent replay cycles.

A matched no-reuse/reuse/no-reuse bracket measured 54.935668, 54.852787, and
55.076140 ms medians. The candidate improves 0.153117 ms or 0.278% against the
mean control median. This remains below the retained 0.5 ms standalone threshold,
so `NV_COMPILER_Q4_GATE_Q8_REUSE` stays default off. The implementation is kept
as a bounded research option because it changes no main or fixup arithmetic.
