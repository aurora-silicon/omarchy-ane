# Notices

MediaPipe selfie segmentation: Copyright 2021 The MediaPipe Authors / Google.
The released weights and their derivatives are Apache-2.0; see
`models/SELFIE-LICENSE` and the official model card linked in `LICENCE-AUDIT.md`.
The derived NCHW graph replaces TFLite tensor layout, operators and storage with
PyTorch/ONNX/MIL equivalents; learned selfie parameters are unchanged.

YuNet: Copyright (c) 2020 Shiqi Yu. The original and fp16-rounded derived weights
are MIT; see `models/YUNET-LICENSE`. The input interface is changed from 640×640
to 320×320. No training or extra weights are used.

YuNet postprocessing refers to OpenCV's public equations. OpenCV is
Copyright (C) 2000-2024, Intel Corporation, Willow Garage, Itseez, OpenCV
Foundation and other contributors. See `models/OPENCV-LICENSE`. The runtime also
uses OpenCV's implementation of resize, Gaussian filtering, box filtering and
non-maximum suppression under that licence.
