# YuNet linked H13 candidate

`yunet-h13-linked-v1.tar.gz` contains the source-built ANEC, its MIL/weights,
model/compiler licences, artifact identities and the 30-portrait decoded-DRAM CPU-simulation
receipt. It contains **one program, one dispatch, 113 tasks**, with 34,324,480
bytes of scratch. All 30 portraits pass the unchanged t-4019 output gate.
There are no portrait images, machine pins or device qualification receipts.

Archive SHA-256:
`2e310398b93aca137870798dbb36b10cd6f0f67afc685114cbc62940659c4dc9`.

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

The v1 DRAM archive subsequently failed device parity. Its host receipt is
not a device qualification; keep CPU fallback selected. Use the exact-prefix
and isolated-layer diagnostic workflow in `../linked/README.md` before any
new device qualification. A resident-graph candidate also requires the full
unchanged 30-portrait and measured-speed gates.

Two resident YuNet candidates are retained for device comparison:

| Archive | Tasks | L2 edges | Scratch bytes | Estimated descriptor DMA/frame |
| --- | ---: | ---: | ---: | ---: |
| `yunet-h13-linked-v2.tar.gz` | 113 | 41 | 25,067,520 | 68,007,936 bytes |
| `yunet-h13-linked-v3.tar.gz` | 109 | 42 | 26,050,560 | 68,667,392 bytes |

The failed v1 DRAM baseline has 113 tasks, 34,324,480 scratch bytes and an
86,358,016-byte descriptor estimate. These estimates sum enabled input/output
TileDMA extents and kernel-DMA lengths; they exclude cache and transaction
behavior and are not measured bandwidth or latency. v3 removes four separate
zero-producing tasks by padding the four stride-8 heads; v2 has slightly less
estimated traffic. Choose by measured device results, not these estimates.

Both resident candidates pass the unchanged output-tensor gate on all 30
portraits with decoded DRAM/L2 transport and package-MIL primitive arithmetic.
Each archive contains that CPU-simulation receipt and model/compiler/runtime
licences. Neither has a passing device receipt; both retain normal CPU
fallback. Both must still pass cold device parity, measured p50/p99, speedup
and wedged checks. Selfie still has no linked package.

Rebuild v2 with `tools/yunet_linked.py --schedule linked-l2 --head-padding legacy`
and v3 with `--schedule linked-l2 --head-padding compact` in the compiler
repository, supplying the audited model root and compiler/output arguments as
in its camera-linked guide. Archive hashes are in `SHA256SUMS`. Archives use
normalized ownership, permissions and timestamps and include no evaluation
images, derived image tensors, device pins or device receipts.

For the lead, preserve the installed candidate directory before extracting
one archive under `MODEL_ROOT/yunet/compiled/H13/linked`, then use the existing
one-command `linked/qualify.py --qualify --model yunet ...` from the linked
runbook with a new approval manifest and results directory. Compare v2/v3 in
separate qualified runs; preserve every receipt and failure.
