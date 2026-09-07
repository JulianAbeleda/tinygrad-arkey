# Swapped Q4-A/Q8-B gate Stream-K substrate

The generated wide Q4-A/x4 body is bit-exact with the conventional generated body at the canonical gate shape `(512,12288,4096)` (`max_abs=0`). After converting the same body to physical `(12288,512)` Stream-K ownership, the direct and split/fixup tiles match the wide x4 body (`max_abs=3.8146973e-06`, no tile above `2e-3`). The source exports the semantic runtime ABI `(out, partials, ids, weight[7077888], record[594432])`, contains the native x4 carrier, launches 170 owners, and fixes up physical `(12288,512)` output.

The initial numerical gate exposed a cascading textual offset rewrite: with stride 512, `16384 -> 4096` was subsequently rewritten as `4096 -> 1024`. The transform now remaps every original offset in one regex pass, guarded by a focused regression.

This checkpoint validates compiler/source/body ownership only. Model chain selection remains default-off pending gate/up elementwise/down topology and full-model A/B/C qualification.
