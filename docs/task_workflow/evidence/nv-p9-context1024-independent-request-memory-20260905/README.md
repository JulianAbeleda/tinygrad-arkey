# NV P9 ctx1024 independent-request memory boundary

Two fresh-process runs captured the normal ctx1024 decode census, then failed
while constructing/replaying the separate measured request's prefill graph.
The max-context-4608 arm failed at 30.17 GiB used on a 20.89 MiB allocation.
The max-context-1536 arm failed at 30.12 GiB used on a 60.89 MiB allocation.
Both used request-scoped prewarming; no promotion flag changed.

Reducing physical context capacity therefore did not remove the failure. The
census request itself completed in both processes. The failure is tied to
retaining its prefill/decode graphs and then constructing a second independent
1024-token prompt request. It does not prove that one ordinary ctx1024 request
cannot execute. Continuous-request coverage is the next bounded test: exclude
capture/warm tokens, then measure sequential windows without duplicating the
prompt graph. Independent-request coverage remains blocked until retained
buffer ownership is measured and reduced.
