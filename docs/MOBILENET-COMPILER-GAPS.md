# MobileNetV1 compiler gaps

Compiler: public `joshuaswarren/mil-hwx-compiler` at
`927013f5480768b07c8e62a866e1514c41aba778`, built unchanged on Linux.
`tools/mobilenet_baseline.py --compiler PATH` compiles the full model and each
isolated operation for H13 and H14, preserving return codes and diagnostics in
`compiler-gaps.json`. The adjacent [CSV](mobilenet-compiler-gaps.csv) contains
exact convolution/head shapes and named diagnostic codes, with no device logs or
private captures.

The valid convolution and head probes currently establish:

| Operation | H13, batch 1 and 8 | H14, batch 1 and 8 |
| --- | --- | --- |
| All 27 convolutions, including all 13 depthwise convolutions | `h13.conv-outside-envelope` | `h14.conv-outside-envelope` |
| Spatial mean `[B,1024,7,7]` to `[B,1024]`, axes 2/3 | `h13.norm-outside-envelope` | `h14.norm-outside-envelope` |
| Biased linear `[B,1024]` to `[B,1000]` | compiles, not device-run | `h14.linear-bias-unsupported` |
| Softmax `[B,1000]`, axis -1 | `h13.norm-outside-envelope` | `h14.norm-outside-envelope` |

These are exact-shape refusals, not a claim that the compiler lacks all depthwise
or regular convolution. Its published backends cover decoded geometries. The
MobileNet points are outside them:

- Stem: `[B,3,224,224]`, weight `[32,3,3,3]`, bias `[32]`, stride 2, SAME padding.
  It needs 3 input/32 output channels and spatial coverage past 64.
- First depthwise: `[B,32,112,112]`, weight `[32,1,3,3]`, groups 32, stride 1.
  Small channels and 112x112 both need coverage.
- First pointwise: `[B,32,112,112]` to `[B,64,112,112]`, kernel 1, groups 1.
- Later blocks use spatial sides 56, 28, 14 and 7, with channel transitions
  64/128/256/512/1024, depthwise kernels 3 and pointwise kernels 1. The CSV
  lists each stride's resulting shapes. Being under 64 is insufficient; these
  exact geometries must be added to the decoded/validated envelopes.
- Batch 8 needs batched convolution support. Both backend diagnostics require
  batch one for rank-4 convolution at this pin.
- The reduction and 1000-wide softmax need their own validated geometry or a
  compiler-owned lowering. H14 also needs biased linear lowering.

H13 supports ReLU6 expressed as clip with finite fp32 scalar bounds 0 and 6, via
its existing min/max lowering. An initial exporter used fp16 scalar bounds and
hit `h13.inexact-constant`; that was an exporter error and is excluded from the
CSV. The corrected exporter emits fp32 bounds and reruns exact-shape clip probes.
H14 has no clip lowering in this pin and its clip probes report
`h14.outside-parity-envelope`.

A practical full-model path also needs a reasonable program count: H13's generic
clip lowering can split a large tensor into per-64-lane min/max programs. That
is compiler scheduling work, not a reason to clone a scheduler into applications.
Successful isolated probe payloads are not retained by the exporter; their program
counts remain in the report. Failed probe diagnostics and partial output remain.
A 180-second compiler-probe timeout is reported separately as rc 124, never as a
named compiler refusal.

The existing H13 Linux reference runner is another boundary: its convolution
reference currently accepts batch 1 only. Native batch 8 needs that reference
updated even after all compiler geometries are available. H14/H16 native runs
cannot use an H13-only runner. CPU batch 1 and 8 already execute with the same
open fp16-rounded weights; those measurements belong in CPU comparison rows.

Reproduce on a Linux build host:

```sh
python3 tools/mobilenet_baseline.py /tmp/mnv1-b1 --batch 1 --compiler /path/to/mil-hwxc
python3 tools/mobilenet_baseline.py /tmp/mnv1-b8 --batch 8 --compiler /path/to/mil-hwxc
```

No Linux ANE MobileNet throughput or energy claim follows from compilation,
structural parity, a mock or an x86_64 CPU smoke run. Keep those target cells TBD
until an authorized device run passes numerical checks.
