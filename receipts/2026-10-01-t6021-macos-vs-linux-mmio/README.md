# T6021 ANE DART tunables

`dart-tunables.tsv` lists, for each ANE DART tunable register of the M2 Max (T6021), the ADT mask and value, the value macOS writes
(`not written` where macOS leaves the register alone), the iBoot value, and what Linux does. The kernel patch
`kernel/patches/apple-dart-ane-tunables.patch` and `receipts/2026-10-01-t6021-dart-tunables/scripts/dart_compare.py` read it.

The notes that produced the table are kept private.
