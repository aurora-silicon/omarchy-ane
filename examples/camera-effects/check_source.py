#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Independent comparisons with upstream MediaPipe and OpenCV implementations."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from aurora_camera_effects.backend import ROOT, Model
from aurora_camera_effects.framing import faces

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
p.add_argument('--images',type=Path)
a=p.parse_args()
# The legacy graph uses its packaged flatbuffer. Verify exact identity before
# treating it as an independent implementation of the audited public model.
model=Path(mp.__file__).parent/'modules/selfie_segmentation/selfie_segmentation.tflite'
assert hashlib.sha256(model.read_bytes()).digest()==hashlib.sha256((ROOT/'models/selfie.tflite').read_bytes()).digest()
seg=mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
ours=ort.InferenceSession(str(ROOT/'models/selfie/model.onnx'),options,providers=['CPUExecutionProvider'])
rng=np.random.default_rng(144)
errors=[]
samples=[rng.integers(0,256,(256,256,3),dtype=np.uint8),
         np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(256,1,3)),
         np.full((256,256,3),127,np.uint8)]
image_names=[]
if a.images:
    for path in sorted(a.images.glob('*.jpg'))[:5]:
        bgr=cv2.imread(str(path))
        if bgr is None:
            raise ValueError('unreadable reference image')
        samples.append(cv2.resize(bgr[...,::-1],(256,256),interpolation=cv2.INTER_LINEAR))
        image_names.append(path.name)
for rgb in samples:
    expected=seg.process(rgb).segmentation_mask
    actual=ours.run(None,{'x':rgb.transpose(2,0,1)[None].astype(np.float32)/255})[0][0,0]
    np.testing.assert_allclose(actual,expected,atol=2e-4,rtol=2e-3)
    errors.append(float(np.max(abs(actual-expected))))
seg.close()
# Compare raw graph outputs on a structured input. Source YuNet is fp32;
# the shipped CPU/MIL weights are rounded fp16, so record the numerical delta.
rgb=rng.integers(0,256,(320,320,3),dtype=np.uint8)
original=cv2.dnn.readNet(str(ROOT/'models/yunet.onnx'))
original.setInput(cv2.dnn.blobFromImage(rgb))
model=Model('yunet')
reference=original.forward(model.names)
actual,_=model(rgb.transpose(2,0,1)[None])
delta=[float(np.max(abs(x-y))) for x,y in zip(actual,reference)]
assert all(np.isfinite(x) and x < .1 for x in delta), delta
model.close()
a.output.write_text(json.dumps(dict(selfie_mediapipe_max_abs=errors,coco_reference_images=image_names,yunet_fp16_weight_vs_source_max_abs=delta),indent=2)+'\n')
