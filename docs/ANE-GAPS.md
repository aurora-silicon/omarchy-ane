# Camera H13 coverage

Compiler: mil-hwx-compiler PR #5 head
`72e1d99c6f51eaf97ad61280936d42dbb0b6a1ae`. No compiler op encoder changed. [Reader-only compatibility PR #9](https://github.com/aurora-silicon/mil-hwx-compiler/pull/9) is required to validate channel-changing convolution packages; it preserves existing ANEC task, span, layout and allocation checks.
Compile and model-parity numbers below were measured on a Darwin/arm64 build host using CPU
simulation. Native Fedora 44 aarch64 VM validation also passes: 23 tests, both ARM64 ELF libraries and the dynamic C bridge smoke test, on 4 KiB pages. No M1 Pro, M2 or other ANE execution was performed by this worker. See [the Fedora receipt](receipts/camera-h13/fedora-arm64.json).

| Model | Original MIL probes | Rewritten H13 islands | CPU simulation, 30 open portraits |
| --- | --- | --- | --- |
| MediaPipe general selfie, Apache-2.0, 256×256 | 94/181 operator probes compile; full graph refuses pad | 54 conv + 1 transpose conv + 3 bilinear resize; 63 convolution stages compile | Mean mask IoU 0.998528, minimum 0.986569 against original CPU model |
| YuNet, MIT, 320×320 | 45/106 operator probes compile; full graph refuses materialized transpose | 53 conv + 2 nearest resize; 55 convolution stages compile | All 12 outputs on all 30 images pass `abs <= .03 + .02*abs(CPU)` |

These are hybrid bundles. Host ONNX partitions retain activations, pooling,
padding and output layout/decode. Normal execution requires a plan-bound H13
device qualification receipt; otherwise the original CPU model runs.
There is no ANE speed, power or camera-frame-rate claim.

Exact rewrites used:

- Zero-pad channels to at least 64 and uniform power-of-two lane groups, then
  discard padded outputs. Padding the 96-channel 5×5 gather to 384 initially
  failed; padding it to 512 compiles without changing its live channels.
- Run stride-1 SAME convolution over an explicitly padded input, then crop
  the halo and select the original stride on the host. This preserves
  asymmetric borders and avoids the refused stride-2 multi-tap form.
- Represent every 5×5 kernel by a 3×3 gather of four shifts per input channel
  and a 3×3 mix. No low-rank approximation or extra activation is introduced.
- Kernel-2/stride-2 transpose convolution becomes 1×1 convolution producing
  four phases per output channel plus host pixel shuffle, with repeated bias.
- Nearest 2× resize becomes a 1×1 phase-copy convolution plus pixel shuffle.
  Half-pixel bilinear 2× resize uses four exact 3×3 phase filters, replicated
  host border and pixel shuffle. Tests compare edge pixels with ONNX Resize.
- Export-time Concat operations construct constant shapes, not feature maps;
  basic ONNX constant folding removes them. These models need no feature-map
  concat or channel-offset-write workaround.

Remaining work for the compiler lane is below. Shapes are exact representative
NCHW shapes; every original op/shape/diagnostic is in
[direct-ops.json](receipts/camera-h13/direct-ops.json).

| Op | Shape | Rewrite tried / remaining refusal |
| --- | --- | --- |
| pad | `[1,3,256,256] -> [1,3,257,257]` | Host materialization preserves zeros/asymmetric edges. Full MIL refuses `h13.unsupported-pad`; needs a data-movement encoder or folding into the consumer. |
| hard-swish / clip(0,6) | `[1,16,128,128]`, `[1,96,32,32]`, `[1,128,16,16]` | Source already decomposes to add/clip/multiply/divide. Retyping the exact 0/6 clip scalars from fp16 to fp32 still returns `h13.inexact-constant`; keep the original CPU equations. |
| sigmoid, SE gates | `[1,C,1,1]`, C=16/24/96/128 | Padding to 64 or 512 compiles all 10 point gates in probes. These tiny gates remain in their host partitions to avoid extra dispatch; they are not a remaining lowering blocker. |
| sigmoid, final mask | `[1,1,256,256]` | Padding channels to 64 still refuses `h13.unary-outside-envelope`; spatial LUT geometry is missing. |
| sigmoid, YuNet scores | `[1,1600,1]`, `[1,400,1]`, `[1,100,1]` | Flattened score outputs lie outside the decoded unary envelope. Host sigmoid preserves the public decode contract; needs a vector/spatial LUT form. |
| broadcast multiply | `[1,16,64,64] * [1,16,1,1]` | SE-channel broadcasting is exact on the host. Direct MIL requires equal static operand shapes; needs broadcast lowering/fusion. |
| average pool | `[1,16,64,64] -> [1,16,1,1]` | No source-qualified `avg_pool` encoder. Keep the exact original mean; a conv/reduction decomposition would need extra rounding qualification. |
| max pool | `[1,16,160,160] -> [1,16,80,80]` | No source-qualified `max_pool` encoder. A linear convolution cannot replace a max exactly. |
| transpose / returned reshape alias | `[1,64,40,40] -> [1,40,40,64] -> [1,1600,64]` | Host output layout is exact. Whole YuNet refuses fastest-axis movement; standalone aliases refuse `h13.returned-input-alias`. Needs materialized layout/output support. |

The raw direct refusals for small-channel/depthwise/5×5/strided convolution,
resize and transpose convolution are resolved by the rewrites above. Add,
ReLU and many same-shape multiply probes already compile and stay in host
partitions in this first integration. Compiler lane tasks can fuse them later.

Validation receipts:
[compile status](receipts/camera-h13/compilation.json),
[selfie simulation](receipts/camera-h13/selfie-simulation.json),
[YuNet simulation](receipts/camera-h13/yunet-simulation.json),
[scalar rewrites](receipts/camera-h13/scalar-rewrites-v2.json).
Every compiled island is compared with its original ONNX operation on all
30 images, with fp16 input and `abs <= .002 + .002*abs(reference)`. This checks
the mathematical rewrite and boundary rounding, not emitted task execution.
The shared graph owner validates task packages and bindings on device setup. With reader PR #9, all 118 stage packages pass those structural checks and NCHW surface round trips offline. A fresh setup on PR #5 base `f7879f1` emits all 118 ANEC programs and all 88 host ONNX partitions byte-identically to the parity-tested baseline; see [the equivalence receipt](receipts/camera-h13/fresh-setup-equivalence.json).

For masks, every image also passes confidence MAE <= .005 and binary pixel
disagreement <= .005; measured maxima are 0.0007564 and 0.001236 respectively
on the Darwin/arm64 build host. An earlier convolution-only candidate failed
a per-image IoU gate at 18/19 pixels in a nearly empty mask. That failure is
[preserved](receipts/camera-h13/initial-iou-failure.json). The final resize
rewrites pass even that original per-image 0.98 IoU gate; the evaluator retains
absolute disagreement checks for near-empty masks.

Keep the existing general model: it preserves PR #3's audited weights and CPU
reference, and all its convolutions/resizes now lower after rewriting. Switching
to the landscape model would retain the same activation/resize families.
Device qualification, allocation/residency behavior, live-camera behavior,
latency and power remain unmeasured. See
[LEAD-RUN.md](../examples/camera-effects/LEAD-RUN.md).


## Exact layer table for t-3991

Baseline is omarchy-ane PR #3 revision `64327b7633c98e10130f008b9802bc827114a4de`. Layer names below are the exact names in each exported `models/<model>/model.mil` and `model.json`. Shapes include batch. Dtype is the activation dtype. `—` means the operation has no stride/groups/kernel parameter. Direct status is measured on the Darwin/arm64 build host using PR #5, before rewrites. No new op encoders are implemented here; t-3991 owns the listed compiler op work.

PR #4 (`5f529f5da1e63e587b7c84569e85c5886a9b58ef`) already has a separate spatial schedule for movement/resize/concat; the table describes PR #5 per-op coverage and does not claim those operations are absent from every compiler branch. The hybrid path keeps the exact rewrites described above.

| Model / MIL layer | Op | Input shape(s) | Output shape | Stride H,W | Groups | Kernel H,W | Dtype | Direct PR #5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selfie / `ce_op_3` | pad | `[1, 3, 256, 256]` | `[1, 3, 257, 257]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_11` | conv | `[1, 3, 257, 257]` | `[1, 16, 128, 128]` | `[2, 2]` | `1` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_16` | clip | `[1, 16, 128, 128]` | `[1, 16, 128, 128]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_27` | conv | `[1, 16, 128, 128]` | `[1, 16, 128, 128]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_32` | pad | `[1, 16, 128, 128]` | `[1, 16, 129, 129]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_40` | conv | `[1, 16, 129, 129]` | `[1, 16, 64, 64]` | `[2, 2]` | `16` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_48` | avg_pool | `[1, 16, 64, 64]` | `[1, 16, 1, 1]` | `[64, 64]` | — | `[64, 64]` | fp16 | refuse |
| selfie / `ce_op_56` | conv | `[1, 16, 1, 1]` | `[1, 8, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_65` | conv | `[1, 8, 1, 1]` | `[1, 16, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_66` | sigmoid | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_67` | mul | `[1, 16, 64, 64]`; `[1, 16, 1, 1]` | `[1, 16, 64, 64]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_75` | conv | `[1, 16, 64, 64]` | `[1, 16, 64, 64]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_83` | conv | `[1, 16, 64, 64]` | `[1, 72, 64, 64]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_88` | pad | `[1, 72, 64, 64]` | `[1, 72, 65, 65]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_96` | conv | `[1, 72, 65, 65]` | `[1, 72, 32, 32]` | `[2, 2]` | `72` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_105` | conv | `[1, 72, 32, 32]` | `[1, 24, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_113` | conv | `[1, 24, 32, 32]` | `[1, 88, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_118` | pad | `[1, 88, 32, 32]` | `[1, 88, 34, 34]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_126` | conv | `[1, 88, 34, 34]` | `[1, 88, 32, 32]` | `[1, 1]` | `88` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_135` | conv | `[1, 88, 32, 32]` | `[1, 24, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_144` | conv | `[1, 24, 32, 32]` | `[1, 96, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_149` | clip | `[1, 96, 32, 32]` | `[1, 96, 32, 32]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_156` | pad | `[1, 96, 32, 32]` | `[1, 96, 35, 35]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_164` | conv | `[1, 96, 35, 35]` | `[1, 96, 16, 16]` | `[2, 2]` | `96` | `[5, 5]` | fp16 | refuse |
| selfie / `ce_op_169` | clip | `[1, 96, 16, 16]` | `[1, 96, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_179` | avg_pool | `[1, 96, 16, 16]` | `[1, 96, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_187` | conv | `[1, 96, 1, 1]` | `[1, 24, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_196` | conv | `[1, 24, 1, 1]` | `[1, 96, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_197` | sigmoid | `[1, 96, 1, 1]` | `[1, 96, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_206` | conv | `[1, 96, 16, 16]` | `[1, 32, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_214` | conv | `[1, 32, 16, 16]` | `[1, 128, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_219` | clip | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_226` | pad | `[1, 128, 16, 16]` | `[1, 128, 20, 20]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_234` | conv | `[1, 128, 20, 20]` | `[1, 128, 16, 16]` | `[1, 1]` | `128` | `[5, 5]` | fp16 | refuse |
| selfie / `ce_op_239` | clip | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_249` | avg_pool | `[1, 128, 16, 16]` | `[1, 128, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_257` | conv | `[1, 128, 1, 1]` | `[1, 32, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_266` | conv | `[1, 32, 1, 1]` | `[1, 128, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_267` | sigmoid | `[1, 128, 1, 1]` | `[1, 128, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_268` | mul | `[1, 128, 16, 16]`; `[1, 128, 1, 1]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_276` | conv | `[1, 128, 16, 16]` | `[1, 32, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_285` | conv | `[1, 32, 16, 16]` | `[1, 128, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_290` | clip | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_297` | pad | `[1, 128, 16, 16]` | `[1, 128, 20, 20]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_305` | conv | `[1, 128, 20, 20]` | `[1, 128, 16, 16]` | `[1, 1]` | `128` | `[5, 5]` | fp16 | refuse |
| selfie / `ce_op_310` | clip | `[1, 128, 16, 16]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_320` | avg_pool | `[1, 128, 16, 16]` | `[1, 128, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_328` | conv | `[1, 128, 1, 1]` | `[1, 32, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_337` | conv | `[1, 32, 1, 1]` | `[1, 128, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_338` | sigmoid | `[1, 128, 1, 1]` | `[1, 128, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_339` | mul | `[1, 128, 16, 16]`; `[1, 128, 1, 1]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_347` | conv | `[1, 128, 16, 16]` | `[1, 32, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_356` | conv | `[1, 32, 16, 16]` | `[1, 96, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_361` | clip | `[1, 96, 16, 16]` | `[1, 96, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_368` | pad | `[1, 96, 16, 16]` | `[1, 96, 20, 20]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_376` | conv | `[1, 96, 20, 20]` | `[1, 96, 16, 16]` | `[1, 1]` | `96` | `[5, 5]` | fp16 | refuse |
| selfie / `ce_op_381` | clip | `[1, 96, 16, 16]` | `[1, 96, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_391` | avg_pool | `[1, 96, 16, 16]` | `[1, 96, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_399` | conv | `[1, 96, 1, 1]` | `[1, 24, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_408` | conv | `[1, 24, 1, 1]` | `[1, 96, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_409` | sigmoid | `[1, 96, 1, 1]` | `[1, 96, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_418` | conv | `[1, 96, 16, 16]` | `[1, 32, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_427` | conv | `[1, 32, 16, 16]` | `[1, 96, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_432` | clip | `[1, 96, 16, 16]` | `[1, 96, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_439` | pad | `[1, 96, 16, 16]` | `[1, 96, 20, 20]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_447` | conv | `[1, 96, 20, 20]` | `[1, 96, 16, 16]` | `[1, 1]` | `96` | `[5, 5]` | fp16 | refuse |
| selfie / `ce_op_452` | clip | `[1, 96, 16, 16]` | `[1, 96, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_462` | avg_pool | `[1, 96, 16, 16]` | `[1, 96, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_470` | conv | `[1, 96, 1, 1]` | `[1, 24, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_479` | conv | `[1, 24, 1, 1]` | `[1, 96, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_480` | sigmoid | `[1, 96, 1, 1]` | `[1, 96, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_489` | conv | `[1, 96, 16, 16]` | `[1, 32, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_497` | avg_pool | `[1, 32, 16, 16]` | `[1, 32, 1, 1]` | `[16, 16]` | — | `[16, 16]` | fp16 | refuse |
| selfie / `ce_op_505` | conv | `[1, 32, 16, 16]` | `[1, 128, 16, 16]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_513` | conv | `[1, 32, 1, 1]` | `[1, 128, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_515` | sigmoid | `[1, 128, 1, 1]` | `[1, 128, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_516` | mul | `[1, 128, 16, 16]`; `[1, 128, 1, 1]` | `[1, 128, 16, 16]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_520` | upsample_bilinear | `[1, 128, 16, 16]` | `[1, 128, 32, 32]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_528` | conv | `[1, 128, 32, 32]` | `[1, 24, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_536` | avg_pool | `[1, 24, 32, 32]` | `[1, 24, 1, 1]` | `[32, 32]` | — | `[32, 32]` | fp16 | refuse |
| selfie / `ce_op_544` | conv | `[1, 24, 1, 1]` | `[1, 24, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_553` | conv | `[1, 24, 1, 1]` | `[1, 24, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_554` | sigmoid | `[1, 24, 1, 1]` | `[1, 24, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_555` | mul | `[1, 24, 32, 32]`; `[1, 24, 1, 1]` | `[1, 24, 32, 32]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_564` | conv | `[1, 24, 32, 32]` | `[1, 24, 32, 32]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_569` | pad | `[1, 24, 32, 32]` | `[1, 24, 34, 34]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_577` | conv | `[1, 24, 34, 34]` | `[1, 24, 32, 32]` | `[1, 1]` | `24` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_583` | upsample_bilinear | `[1, 24, 32, 32]` | `[1, 24, 64, 64]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_591` | conv | `[1, 24, 64, 64]` | `[1, 16, 64, 64]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| selfie / `ce_op_599` | avg_pool | `[1, 16, 64, 64]` | `[1, 16, 1, 1]` | `[64, 64]` | — | `[64, 64]` | fp16 | refuse |
| selfie / `ce_op_607` | conv | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_616` | conv | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_617` | sigmoid | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_618` | mul | `[1, 16, 64, 64]`; `[1, 16, 1, 1]` | `[1, 16, 64, 64]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_627` | conv | `[1, 16, 64, 64]` | `[1, 16, 64, 64]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_632` | pad | `[1, 16, 64, 64]` | `[1, 16, 66, 66]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_640` | conv | `[1, 16, 66, 66]` | `[1, 16, 64, 64]` | `[1, 1]` | `16` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_646` | upsample_bilinear | `[1, 16, 64, 64]` | `[1, 16, 128, 128]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_654` | conv | `[1, 16, 128, 128]` | `[1, 16, 128, 128]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_662` | avg_pool | `[1, 16, 128, 128]` | `[1, 16, 1, 1]` | `[128, 128]` | — | `[128, 128]` | fp16 | refuse |
| selfie / `ce_op_670` | conv | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_679` | conv | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_680` | sigmoid | `[1, 16, 1, 1]` | `[1, 16, 1, 1]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_681` | mul | `[1, 16, 128, 128]`; `[1, 16, 1, 1]` | `[1, 16, 128, 128]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_690` | conv | `[1, 16, 128, 128]` | `[1, 16, 128, 128]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| selfie / `ce_op_695` | pad | `[1, 16, 128, 128]` | `[1, 16, 130, 130]` | — | — | — | fp16 | refuse |
| selfie / `ce_op_703` | conv | `[1, 16, 130, 130]` | `[1, 16, 128, 128]` | `[1, 1]` | `16` | `[3, 3]` | fp16 | refuse |
| selfie / `ce_op_714` | conv_transpose | `[1, 16, 128, 128]` | `[1, 1, 256, 256]` | `[2, 2]` | `1` | `[2, 2]` | fp16 | refuse |
| selfie / `ce_op_715` | sigmoid | `[1, 1, 256, 256]` | `[1, 1, 256, 256]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_7` | conv | `[1, 3, 320, 320]` | `[1, 16, 160, 160]` | `[2, 2]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_16` | conv | `[1, 16, 160, 160]` | `[1, 16, 160, 160]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_24` | conv | `[1, 16, 160, 160]` | `[1, 16, 160, 160]` | `[1, 1]` | `16` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_31` | max_pool | `[1, 16, 160, 160]` | `[1, 16, 80, 80]` | `[2, 2]` | — | `[2, 2]` | fp16 | refuse |
| yunet / `ce_op_39` | conv | `[1, 16, 80, 80]` | `[1, 16, 80, 80]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_47` | conv | `[1, 16, 80, 80]` | `[1, 16, 80, 80]` | `[1, 1]` | `16` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_56` | conv | `[1, 16, 80, 80]` | `[1, 32, 80, 80]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_64` | conv | `[1, 32, 80, 80]` | `[1, 32, 80, 80]` | `[1, 1]` | `32` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_73` | conv | `[1, 32, 80, 80]` | `[1, 32, 80, 80]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_81` | conv | `[1, 32, 80, 80]` | `[1, 32, 80, 80]` | `[1, 1]` | `32` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_90` | conv | `[1, 32, 80, 80]` | `[1, 64, 80, 80]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_98` | conv | `[1, 64, 80, 80]` | `[1, 64, 80, 80]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_105` | max_pool | `[1, 64, 80, 80]` | `[1, 64, 40, 40]` | `[2, 2]` | — | `[2, 2]` | fp16 | refuse |
| yunet / `ce_op_113` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_121` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_130` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_138` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_145` | max_pool | `[1, 64, 40, 40]` | `[1, 64, 20, 20]` | `[2, 2]` | — | `[2, 2]` | fp16 | refuse |
| yunet / `ce_op_153` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_161` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_170` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_178` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_185` | max_pool | `[1, 64, 20, 20]` | `[1, 64, 10, 10]` | `[2, 2]` | — | `[2, 2]` | fp16 | refuse |
| yunet / `ce_op_193` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_201` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_210` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_218` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_227` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_235` | conv | `[1, 64, 10, 10]` | `[1, 64, 10, 10]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_239` | upsample_nearest_neighbor | `[1, 64, 10, 10]` | `[1, 64, 20, 20]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_248` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_256` | conv | `[1, 64, 20, 20]` | `[1, 64, 20, 20]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_260` | upsample_nearest_neighbor | `[1, 64, 20, 20]` | `[1, 64, 40, 40]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_269` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | compile |
| yunet / `ce_op_277` | conv | `[1, 64, 40, 40]` | `[1, 64, 40, 40]` | `[1, 1]` | `64` | `[3, 3]` | fp16 | compile |
| yunet / `ce_op_286` | conv | `[1, 64, 40, 40]` | `[1, 1, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_294` | conv | `[1, 1, 40, 40]` | `[1, 1, 40, 40]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_302` | conv | `[1, 64, 20, 20]` | `[1, 1, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_310` | conv | `[1, 1, 20, 20]` | `[1, 1, 20, 20]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_318` | conv | `[1, 64, 10, 10]` | `[1, 1, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_326` | conv | `[1, 1, 10, 10]` | `[1, 1, 10, 10]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_334` | conv | `[1, 64, 40, 40]` | `[1, 4, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_342` | conv | `[1, 4, 40, 40]` | `[1, 4, 40, 40]` | `[1, 1]` | `4` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_350` | conv | `[1, 64, 20, 20]` | `[1, 4, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_358` | conv | `[1, 4, 20, 20]` | `[1, 4, 20, 20]` | `[1, 1]` | `4` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_366` | conv | `[1, 64, 10, 10]` | `[1, 4, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_374` | conv | `[1, 4, 10, 10]` | `[1, 4, 10, 10]` | `[1, 1]` | `4` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_382` | conv | `[1, 64, 40, 40]` | `[1, 1, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_390` | conv | `[1, 1, 40, 40]` | `[1, 1, 40, 40]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_398` | conv | `[1, 64, 20, 20]` | `[1, 1, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_406` | conv | `[1, 1, 20, 20]` | `[1, 1, 20, 20]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_414` | conv | `[1, 64, 10, 10]` | `[1, 1, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_422` | conv | `[1, 1, 10, 10]` | `[1, 1, 10, 10]` | `[1, 1]` | `1` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_430` | conv | `[1, 64, 40, 40]` | `[1, 10, 40, 40]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_438` | conv | `[1, 10, 40, 40]` | `[1, 10, 40, 40]` | `[1, 1]` | `10` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_446` | conv | `[1, 64, 20, 20]` | `[1, 10, 20, 20]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_454` | conv | `[1, 10, 20, 20]` | `[1, 10, 20, 20]` | `[1, 1]` | `10` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_462` | conv | `[1, 64, 10, 10]` | `[1, 10, 10, 10]` | `[1, 1]` | `1` | `[1, 1]` | fp16 | refuse |
| yunet / `ce_op_470` | conv | `[1, 10, 10, 10]` | `[1, 10, 10, 10]` | `[1, 1]` | `10` | `[3, 3]` | fp16 | refuse |
| yunet / `ce_op_473` | transpose | `[1, 1, 40, 40]` | `[1, 40, 40, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_474` | reshape | `[1, 40, 40, 1]` | `[1, 1600, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_475` | sigmoid | `[1, 1600, 1]` | `[1, 1600, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_478` | transpose | `[1, 1, 20, 20]` | `[1, 20, 20, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_479` | reshape | `[1, 20, 20, 1]` | `[1, 400, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_480` | sigmoid | `[1, 400, 1]` | `[1, 400, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_483` | transpose | `[1, 1, 10, 10]` | `[1, 10, 10, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_484` | reshape | `[1, 10, 10, 1]` | `[1, 100, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_485` | sigmoid | `[1, 100, 1]` | `[1, 100, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_488` | transpose | `[1, 1, 40, 40]` | `[1, 40, 40, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_489` | reshape | `[1, 40, 40, 1]` | `[1, 1600, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_490` | sigmoid | `[1, 1600, 1]` | `[1, 1600, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_493` | transpose | `[1, 1, 20, 20]` | `[1, 20, 20, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_494` | reshape | `[1, 20, 20, 1]` | `[1, 400, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_495` | sigmoid | `[1, 400, 1]` | `[1, 400, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_498` | transpose | `[1, 1, 10, 10]` | `[1, 10, 10, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_499` | reshape | `[1, 10, 10, 1]` | `[1, 100, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_500` | sigmoid | `[1, 100, 1]` | `[1, 100, 1]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_503` | transpose | `[1, 4, 40, 40]` | `[1, 40, 40, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_504` | reshape | `[1, 40, 40, 4]` | `[1, 1600, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_507` | transpose | `[1, 4, 20, 20]` | `[1, 20, 20, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_508` | reshape | `[1, 20, 20, 4]` | `[1, 400, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_511` | transpose | `[1, 4, 10, 10]` | `[1, 10, 10, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_512` | reshape | `[1, 10, 10, 4]` | `[1, 100, 4]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_515` | transpose | `[1, 10, 40, 40]` | `[1, 40, 40, 10]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_516` | reshape | `[1, 40, 40, 10]` | `[1, 1600, 10]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_519` | transpose | `[1, 10, 20, 20]` | `[1, 20, 20, 10]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_520` | reshape | `[1, 20, 20, 10]` | `[1, 400, 10]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_523` | transpose | `[1, 10, 10, 10]` | `[1, 10, 10, 10]` | — | — | — | fp16 | refuse |
| yunet / `ce_op_524` | reshape | `[1, 10, 10, 10]` | `[1, 100, 10]` | — | — | — | fp16 | refuse |

Concat: all 57 Concat nodes in the original selfie ONNX construct constant shape expressions and disappear under basic constant folding. Neither model has a remaining runtime feature-map concat layer; there is no runtime concat shape to supply.

