# Current252 native Q-only role delta

Generated/native-Q/generated medians are 47.811322 / 47.183995 / 47.824791 ms on the exact current252 graph. Native Q saves 0.634061 ms against the mean generated controls. Thus Q accounts for only 20.2% of the prior 3.133466 ms native-QKV advantage; K plus V retain an interaction-aware 2.499404 ms target.

Every arm reports PASS, token 198, canonical 252/252 weights, zero overlays, 216 Q8 producers, and five exact replay cycles. The native-Q candidate explicitly proves 36 native main and 36 native fixup calls while retaining 36 generated O calls. Native binaries remain diagnostic only.
