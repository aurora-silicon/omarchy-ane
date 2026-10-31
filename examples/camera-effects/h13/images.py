#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fetch 30 deterministic CC BY 2.0 COCO validation portraits for parity.

No images or weights are redistributed. Keep COCO image IDs/URLs and hashes
with any numeric receipts. This is model parity, not a quality/fairness study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coco_annotations import validation_json

BASE='https://s3.amazonaws.com/images.cocodataset.org'


def fetch(root):
    root.mkdir(parents=True,exist_ok=True)
    annotations=root/'instances_val2017.json'
    if not annotations.exists():
        annotations.write_bytes(validation_json(BASE+'/annotations/annotations_trainval2017.zip'))
    data=json.loads(annotations.read_text())
    license=next(v for v in data['licenses'] if v['id']==4)
    if 'creativecommons.org/licenses/by/2.0' not in license['url']:
        raise ValueError('unexpected dataset licence')
    people={v['image_id'] for v in data['annotations'] if v['category_id']==1 and not v['iscrowd']}
    images=[v for v in sorted(data['images'],key=lambda v:v['id'])
            if v['license']==4 and v['id'] in people and v['height']>=v['width']][:30]
    if len(images)!=30:
        raise ValueError('insufficient portraits')
    for image in images:
        target=root/image['file_name']
        if not target.exists():
            with urllib.request.urlopen(BASE+'/val2017/'+image['file_name'],timeout=120) as response:
                target.write_bytes(response.read())
        image['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    (root/'images.json').write_text(json.dumps(images,indent=2)+'\n')
    (root/'dataset.json').write_text(json.dumps(dict(license=license,
        annotations_sha256=hashlib.sha256(annotations.read_bytes()).hexdigest(),
        selection='first 30 ascending image IDs, license 4, at least one noncrowd person, height >= width'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    fetch(p.parse_args().output)
