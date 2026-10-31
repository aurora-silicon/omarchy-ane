# Model, implementation and evaluation licence audit

No Apple code, models, compiled programs or private calibration records are
inputs to this implementation. The framing range is board configuration data.

| Input | Public source and identity | Licence / use |
| --- | --- | --- |
| MediaPipe general selfie segmentation weights | [Official flatbuffer](https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite); SHA-256 `9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c` | Apache-2.0 explicitly identified on page 1 of the [Google model card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Selfie%20Segmentation.pdf). Full licence in `models/SELFIE-LICENSE`. Authors: Tingbo Hou, Siargey Pisarchyk, Karthik Raveendran; Google, May 2021. |
| MediaPipe processing contract | [CPU graph](https://github.com/google-ai-edge/mediapipe/blob/68392c4d368d14186566256d80067d8fc146f5c9/mediapipe/modules/selfie_segmentation/selfie_segmentation_cpu.pbtxt), revision `68392c4d368d14186566256d80067d8fc146f5c9` | Apache-2.0. RGB [0,1], 256×256 stretch, person confidence mask. The original importer reads the audited flatbuffer and refuses unsupported operators/options. |
| YuNet weights | [OpenCV Zoo source](https://github.com/opencv/opencv_zoo/tree/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet), `face_detection_yunet_2023mar.onnx`; SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` | Directory README explicitly licenses all files, including weights, under MIT. Full copyright/licence in `models/YUNET-LICENSE`; Copyright 2020 Shiqi Yu. Original 640×640 interface is re-exported at 320×320; learned weights are fp16-rounded without retraining. |
| YuNet decode equations | [OpenCV 4.11 implementation](https://github.com/opencv/opencv/blob/4.11.0/modules/objdetect/src/face_detect.cpp) | OpenCV Apache-2.0; independent vector decode of public stride/offset/score equations. OpenCV attribution/licence in `NOTICES.md` and `models/OPENCV-LICENSE`. |
| This integration, C bridge, importer and postprocessing | This repository | MIT, inherited repository licence. Guided upsampling implements local linear regression with box filters; no third-party guided-filter source is copied. |
| Shared runtime | `libane/ane_fallback.c`, `bindings/python/python/ane/fallback.py`; t-3650 | Existing runtime, reused without another scheduler or device recovery implementation. |
| Export tooling | coremltools 9.0; PyTorch 2.7.1; torchvision 0.22.1; ONNX 1.17; onnx2torch 1.5.15; tflite 2.18 | BSD-3-Clause, BSD-3-Clause, BSD-3-Clause, Apache-2.0, Apache-2.0, Apache-2.0 respectively. Build-only; no compiled Apple program bytes. |
| CPU inference and image operations | ONNX Runtime; OpenCV; NumPy; construct | MIT, Apache-2.0, BSD-3-Clause, MIT respectively. Dependency packages retain their own notices. |
| Quality annotations | [COCO terms](https://cocodataset.org/#termsofuse), COCO 2017 validation instance annotations | CC BY 4.0 annotations. Original dataset: Tsung-Yi Lin et al., *Microsoft COCO: Common Objects in Context*, 2014. Evaluation only; no annotation archive redistributed. |
| Evaluation images | Original COCO image metadata, filtered to licence ID 4; metadata URL is checked by the evaluator | CC BY 2.0 only. Deterministic person subset; image IDs, content hashes and COCO download URLs are numeric provenance, not republished images. No images are installed or committed. |

Source models and converted derivatives remain under their respective weight
licences. `export.py` records the original model hash and derived artifact hashes
in each `model.json`. The original selfie model uses consented but unreleased
training data according to its model card; its released weights are Apache-2.0.
No claim is made that the original training dataset can be redistributed.

The general selfie model is optimized for nearby prominent people; distant
people, occlusion, thin structures, different scales, low light and motion can
reduce quality. COCO is an open, independently annotated stress sample, not a
representative video-call or temporal/fairness qualification dataset.
