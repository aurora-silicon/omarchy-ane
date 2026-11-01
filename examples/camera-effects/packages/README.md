# YuNet linked H13 candidate

`yunet-h13-linked-v1.tar.gz` contains the source-built ANEC, its MIL/weights,
model/compiler licences, artifact identities and the 30-portrait CPU-simulation
receipt. It contains **one program, one dispatch, 113 tasks**, with 34,324,480
bytes of scratch. All 30 portraits pass the unchanged t-4019 output gate.
There are no portrait images, machine pins or device qualification receipts.

Archive SHA-256:
`4919b1d38943450a92439b10d5aa5847d8192d89e3caeb6243f1c8b08d8626f3`.

The program consumes a 16-channel phase layout of the original BGR 320 image
and returns 64 channels at 40x40. The backend performs layout conversion and
score/head decode outside the program; the NN has no host partitions. Timing
includes these boundary operations. Package-MIL simulation is not hardware
execution and its timings are not ANE inference timings.

Verify `SHA256SUMS`, preserve any previous linked directory, then extract the
archive into `MODEL_ROOT/yunet/compiled/H13/linked`. Keep the checked original
CPU model export in `MODEL_ROOT/yunet`. Use both feature branches together.

The backend stays on CPU until the controller-approved M1 Pro qualification
passes. Follow [the qualification command](../linked/README.md) with
`--model yunet`; require 30 cold passes, measured ANE p50 below measured CPU
p50, one resident LOAD, one EXECUTE per frame, and `wedged` zero throughout.
Device parity, p50/p99 and speedup remain unverified. Selfie has no linked
package and blur remains on CPU.
