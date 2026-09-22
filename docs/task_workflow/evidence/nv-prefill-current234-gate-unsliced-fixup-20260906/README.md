# Current234 gate Stream-K unsliced fixup

The candidate replaces the promoted four-slice 128-thread gate/up fixup with
one 256-thread block per active tile. It consumes the same partial layout and
three-slot deterministic map. Structural census, token 198, and 20/20 exact
recurrent replay cycles pass.

The candidate median is 55.775212 ms against matched current234 controls of
54.935668 and 55.076140 ms. This is a 0.769308 ms or 1.399% regression against
their mean. The option was reverted; the four-slice fixup remains selected.
Further gate work must improve the main body rather than fixup geometry.
