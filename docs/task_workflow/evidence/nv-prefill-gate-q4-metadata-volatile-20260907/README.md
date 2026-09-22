# Volatile Q4 metadata alias discriminator

Loading the 32 exact Q4 metadata pairs through volatile uint16 carriers preserves bit exactness and prevents compiler motion across the char-backed LDS publication. It retains PRMT 48 but increases LDS instructions from 384 to 512. Alternating R15 measured 333.898 us for the promoted control and 333.818 us for the volatile candidate, only 0.080 us/call. This safe form is neutral and was not integrated.
