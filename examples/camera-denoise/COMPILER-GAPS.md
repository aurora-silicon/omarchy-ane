# TinyLuma compiler coverage

Corrected controller request **h-46549** passed export, byte-identity checks and
the independent fp16 MIL reference on 2026-10-31. The compiler was built from
Aurora revision `45558fb76e08f4c12d9b7d3e333dbac7595f2e80`. Input is the actual
coremltools 9.0 `ios18` / program-1.3 graph, with released weights. No compiler
source, envelope, driver check or ownership rule was changed.

[All 42 full-model/per-operation results](receipts/compiler-gaps.json) retain
return codes, exact shapes and diagnostics. Temporary build paths in commands
are normalized to `${MODEL_ROOT}`. The unmodified logs remain in controller
artifacts. Compiler diagnostic text is MIT; its
[copyright/licence notice](receipts/compiler-LICENSE.txt) is retained.

**No full-model package compiled on H13, H14 or H16G.** All full-model
compilations returned 65. `native_targets` is empty in the shipped model
metadata, so this model uses libane's CPU fallback and never submits native
programs. Individual operation compilation is not full-model support or device
qualification.

| Operation / NCHW geometry | H13 | H14 | H16G |
| --- | --- | --- | --- |
| enc: 1x1x64x64 -> 1x8x64x64, k3 s1 | conv envelope refusal | conv envelope refusal | depthwise form refusal |
| down: 1x8x64x64 -> 1x8x32x32, k3 s2 | conv envelope refusal | conv envelope refusal | regular-conv form refusal |
| mid: 1x8x32x32 -> 1x8x32x32, k3 s1 | conv envelope refusal | conv envelope refusal | regular-conv form refusal |
| dec: 1x8x64x64 -> 1x8x64x64, k3 s1 | conv envelope refusal | conv envelope refusal | regular-conv form refusal |
| head: 1x8x64x64 -> 1x1x64x64, k3 s1 | conv envelope refusal | conv envelope refusal | fp16-conv form refusal |
| ReLU at 32x32/64x64, C8 | compiled | parity envelope refusal | binary-ALU form refusal |
| nearest 2x, 1x8x32x32 -> 1x8x64x64 | no qualified encoder | parity envelope refusal | no decoded primitive |
| skip add, 1x8x64x64 | compiled | parity envelope refusal | binary-ALU geometry refusal |
| residual subtract, 1x1x64x64 | compiled | compiled | binary-ALU geometry refusal |
| clip [0,1], fp16 input and constants | constant-type refusal | parity envelope refusal | no decoded primitive |

H13's full-model first diagnostic is `h13.inexact-constant` at clip. The values
0 and 1 are exactly representable; the compiler specifically requires fp32
scalar operands. Core ML's typed clip contract uses the input type for both
limits: [definition](https://github.com/apple/coremltools/blob/9.0/coremltools/converters/mil/mil/ops/defs/iOS15/elementwise_unary.py),
[iOS17+ variant](https://github.com/apple/coremltools/blob/9.0/coremltools/converters/mil/mil/ops/defs/iOS17/elementwise_unary.py).
The exporter preserves that valid fp16 graph. It does not rewrite the constants
to an incompatible type to pass the compiler's predicate. The isolated probes
also establish all five H13 convolution refusals and the missing upsampler,
which the full-model diagnostic alone would hide.

H14 stops at `h14.conv-outside-envelope`; most isolated activation/movement
shapes also refuse. H16G stops at
`h16g.legalize.unsupported-operation` for nearest upsampling, with additional
per-op convolution/ALU and clip gaps. These diagnostics identify exact missing
cases; they do not establish the reason an encoder form is absent.

## Handoff to the compiler/runtime work

Add regressions using the shipped graph, weights and exporter-generated probes.
Support the valid fp16 clip constants, these exact small-channel k3 convolution
shapes including stride 2, and nearest movement. Preserve the existing refusal
rules until each form is proved. A performant native path also needs fused
programs and a reusable graph owner without per-tile CPU reference evaluation.

Then reproduce the full-model capture and host seam/reference checks, retain
all identities, and follow [the lead plan](LEAD-TEST-PLAN.md) for a separately
approved M1 Pro device run. H14/H16 graph-owner integration and H17 ownership
remain separate prerequisites. No generation override or Apple program bytes
are a substitute for those steps.
