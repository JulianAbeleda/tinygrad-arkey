# Post-Q promotion native-gate diagnostic

The refreshed ordinary generated authority is 43.837075 ms median and 43.631489 ms minimum, with exact replay and the qualified 252/234/162 census.

A native-gate timing arm measured 41.313938 ms median, but is invalid for debt accounting: recurrent replay drifted. Separate deep3 processes show native gate with wide Q/O is internally exact, and native gate with ordinary O plus wide Q is internally exact. The full native-gate + ordinary Q + ordinary O composition alternates corruption: cycles 0/2 differ while cycle1 matches, with first mismatches already at gate outputs/records and propagating through the graph. This is a diagnostic composition/ownership incompatibility; no timing is booked and production generated selection is unaffected.

The harness now recognizes the native gate main ABI (`outs=(2,3), ins=(0,1)`) so its output/record buffers can be included in deep replay. No native binary enters ordinary selection.
