#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Download pinned public open weights/exports and check every SHA-256.

The converted exports are the public PR #3 baseline, built from the audited
Apache-2.0 MediaPipe and MIT YuNet weights. No Apple model/program is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parent
REVISION='64327b7633c98e10130f008b9802bc827114a4de'
BASE=f'https://raw.githubusercontent.com/aurora-silicon/omarchy-ane/{REVISION}/examples/camera-effects/models'
SOURCES={
    'selfie.tflite':('https://storage.googleapis.com/mediapipe-assets/selfie_segmentation.tflite',
                     '9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c'),
    'yunet.onnx':('https://media.githubusercontent.com/media/opencv/opencv_zoo/48b10fa505aeb51fbd127fc60a5c660040e823b7/models/face_detection_yunet/face_detection_yunet_2023mar.onnx',
                  '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4')}


def fetch(url,path,digest):
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
        return
    with urllib.request.urlopen(url,timeout=120) as response:
        data=response.read()
    if hashlib.sha256(data).hexdigest()!=digest:
        raise ValueError(f'download identity mismatch: {path.name}')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.download')
    temporary.write_bytes(data)
    temporary.replace(path)


def install(root):
    root.mkdir(parents=True,exist_ok=True)
    for file,(url,digest) in SOURCES.items():
        fetch(url,root/file,digest)
    for model in ('selfie','yunet'):
        metadata=(ROOT/'models'/model/'model.json').read_bytes()
        record=json.loads(metadata)
        for file,digest in record['artifacts'].items():
            if file not in ('model.onnx','model.mil','weights.bin','probes.zip'):
                raise ValueError('unexpected baseline artifact')
            fetch(f'{BASE}/{model}/{file}',root/model/file,digest)
        (root/model/'model.json').write_bytes(metadata)
    for name in ('SELFIE-LICENSE','YUNET-LICENSE','OPENCV-LICENSE'):
        (root/name).write_bytes((ROOT/'models'/name).read_bytes())
    print(f'camera models verified: {root}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'models')
    install(p.parse_args().output.resolve())
