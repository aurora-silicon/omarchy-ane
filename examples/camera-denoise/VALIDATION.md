# Host validation and limits

Controller buildbox request **h-46448**, 2026-10-31. CPU: Intel Core i9-13900K,
32 logical CPUs, eight ONNX intra-op threads, one inter-op thread, sequential
execution. The host was shared: initial load average was 33.56 / 31.93 / 32.46.
There was no idle gate. These measurements describe this loaded x86_64 host,
not an ARM camera system or the ANE.

The released 1,905 fp16 parameters have SHA-256
`94727af5fc6e6bc1cf84d8915e5341d26e87f8dfd46da1f1d112d7cac1bb3874`.
The CPU graph has SHA-256
`e89353526c8750818034676de5a522903d823551a881d901b08bc4b71fbe6b5c`.
The [raw host receipt](receipts/host-cpu-quality.json) contains every sample,
decoded image identity, noise seed formula and metric convention.

## CPU latency

Five warmups followed by 20 measured frames per size. CPU graph execution uses
the t-3650 libane fallback callback. No native callback participates in timing.

| Luma size | Model p50 | Model p90 | Model min | Plane API p50 |
| --- | ---: | ---: | ---: | ---: |
| 1280x720 | 70.776 ms | 73.151 ms | 67.581 ms | 75.174 ms |
| 1920x1080 | 252.679 ms | 262.943 ms | 133.473 ms | 270.564 ms |

Model time includes edge padding, full-frame ONNX inference, fp16 operation
rounding and output cropping. Plane API time additionally includes uint8
normalization, strength blending and quantization. The C bridge's extra copies
are not included in this Python plane API measurement. Initialization and
loading are outside timing. The wide 1080p spread includes host contention.

**This model CPU path does not meet 30 fps on this host.** It is a correctness
fallback and a source for future ANE qualification. These results do not justify
replacing the filter's existing NEON CPU denoiser or enabling the candidate by
default. No ANE frame latency, power or acceleration claim is made.

## Quality smoke check

`open-luma-awgn-v1` has two held-out images, Chelsea and Hubble, with clipped
AWGN at sigma 15, 25 and 40 followed by uint8 quantization. Training uses two
different images. Each row below is the arithmetic mean of the two test cases.
PSNR uses range 1; SSIM uses scikit-image's default 7x7 window and sample
covariance. Results are luma only, after output quantization.

| Sigma /255 | Noisy PSNR | Model PSNR | Noisy SSIM | Model SSIM |
| --- | ---: | ---: | ---: | ---: |
| 15 | 25.2569 dB | 30.8580 dB | 0.4989 | 0.7588 |
| 25 | 21.1703 dB | 30.0189 dB | 0.3097 | 0.7368 |
| 40 | 17.3446 dB | 27.4843 dB | 0.1753 | 0.6371 |

All six individual cases improved in PSNR and SSIM. This tiny synthetic test
does not establish camera quality, preservation of fine texture, color denoise,
temporal consistency, motion handling or performance on real sensor noise.
The model needs a larger open training/evaluation corpus and the filter owner's
private camera A/B comparison before any default change.

## Checks already passed

- 13 pytest cases covering stride/padding, in-place operation, zero-strength
  bypass, invalid inputs, model identity, native failure retirement, complete
  CPU recomputation, dense-versus-tiled seams and private failure evidence.
- The C vtable bridge and its executable harness build with
  `-std=c11 -O2 -Wall -Wextra -Werror`; bridge assertions pass.
- Independent compiler fp16 MIL reference against ONNX: random and ramp
  64x64 cases, maximum absolute error **0.00048828125** in each case, within
  atol 0.002 / rtol 0.002. This is CPU reference agreement, not device parity.
- Training source hashes match the released checkpoint provenance. Export
  checks verify identical trained arrays in the typed MIL and ONNX constants.
- Public-text policy and its 69-case harness pass; source review found no
  remaining concrete issue after lifecycle and private-evidence fixes.

The first compiler capture in h-46448 refused the `ios16` declaration before
operation lowering. That was an interchange issue, not evidence about model
geometry. The exporter now constructs/serializes MIL at `ios18`, the compiler's
accepted program-1.3 opset. Corrected request **h-46549 passed** byte-identical
ONNX/weight-blob checks and the independent MIL reference, again with maximum
absolute error 0.00048828125. The earlier CPU receipt therefore applies to the
final export. The [corrected coverage capture](COMPILER-GAPS.md) shows no full
model package on H13/H14/H16G; all return 65 with named lowering refusals.
Earlier diagnostics remain preserved in the controller run.

Native execution and filter integration still require the exact lead steps in
[LEAD-TEST-PLAN.md](LEAD-TEST-PLAN.md). The provisional vtable must be adapted
to the filter's final ABI. No kernel, filter node or distro packaging is changed.
