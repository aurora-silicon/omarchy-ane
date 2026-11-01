# Public camera implementation sources

The full weight, export, implementation and evaluation licence audit is in
[examples/camera-effects/LICENCE-AUDIT.md](examples/camera-effects/LICENCE-AUDIT.md).
Linked mode uses the same Apache-2.0 MediaPipe general selfie weights and
MIT YuNet 2023mar weights, without retraining or another network. CPU fallback
uses the original checked ONNX export with the same weights.

The linked runtime, qualification tooling and procedural tests are MIT.
MIL numerical simulation uses NumPy (BSD-3-Clause), ONNX and ONNX Runtime
(Apache-2.0 and MIT), and the existing MIT compiler's parser and numerical
references. It never invokes an Apple compiler or includes Apple program
bytes. Generated packages and weights remain install-time data under their
original licences and are excluded from source control. Evaluation uses
only the existing 30 audited CC BY 2.0 COCO portraits; no images are committed.
