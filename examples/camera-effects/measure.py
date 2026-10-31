#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CPU cost and mask quality on a deterministic CC BY 2.0 COCO person subset.

Only aggregate/per-image numeric receipts leave the build slot. Images remain
in its temporary output directory. No camera or private input is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import urllib.request
from coco_annotations import validation_json
import cv2
import numpy as np
from pycocotools.coco import COCO
from aurora_camera_effects.backend import Backend
from aurora_camera_effects.image import replacement_nv12, model_rgb

BASE='https://s3.amazonaws.com/images.cocodataset.org'


def download(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=120) as response:
            path.write_bytes(response.read())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quality(work,count):
    archive_url=BASE+'/annotations/annotations_trainval2017.zip'
    path=work/'instances_val2017.json'
    if not path.exists():
        path.write_bytes(validation_json(archive_url))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    coco=COCO(str(path))
    license=coco.dataset['licenses']
    # COCO license ID 4 is Attribution (CC BY 2.0), checked against metadata.
    chosen=next(x for x in license if x['id']==4)
    assert 'creativecommons.org/licenses/by/2.0' in chosen['url']
    images=[]
    for image in sorted(coco.dataset['images'],key=lambda x:x['id']):
        people=coco.loadAnns(coco.getAnnIds(imgIds=[image['id']],catIds=[1]))
        if image['license']!=4 or not people or any(a['iscrowd'] for a in people):
            continue
        if sum(a['area'] for a in people)<.1*image['width']*image['height']:
            continue
        images.append((image,people))
        if len(images)==count:
            break
    if len(images)!=count:
        raise ValueError('insufficient audited dataset images')
    backend=Backend(640,480,'blur')
    records=[]
    sample=None
    for image,people in images:
        path=work/image['file_name']
        image_sha=download(BASE+'/val2017/'+image['file_name'],path)
        bgr=cv2.imread(str(path)); rgb=bgr[...,::-1].copy()
        # Exercise the filter's actual NV12 preprocessing. Trim at most one
        # final row/column because NV12 requires even dimensions.
        h,w=rgb.shape[:2];h-=h%2;w-=w%2
        original_rgb=rgb
        rgb=rgb[:h,:w]
        luma,chroma=replacement_nv12(rgb,w,h)
        small=model_rgb(luma,chroma,256)
        mask,_=backend.mask(small,luma)
        gt=np.zeros(luma.shape,np.uint8)
        for ann in people:
            gt |= coco.annToMask(ann)[:h,:w]
        pred=mask>=.5
        intersection=int(np.logical_and(pred,gt).sum())
        union=int(np.logical_or(pred,gt).sum())
        total=int(pred.sum()+gt.sum())
        # Boundary F1 at a fixed two-pixel tolerance, both masks at source size.
        kernel=np.ones((3,3),np.uint8)
        pb=pred.astype(np.uint8)-cv2.erode(pred.astype(np.uint8),kernel)
        gb=gt-cv2.erode(gt,kernel)
        near_g=cv2.dilate(gb,np.ones((5,5),np.uint8))
        near_p=cv2.dilate(pb,np.ones((5,5),np.uint8))
        precision=float((pb*near_g).sum()/max(1,pb.sum()))
        recall=float((gb*near_p).sum()/max(1,gb.sum()))
        f1=2*precision*recall/max(1e-12,precision+recall)
        records.append(dict(id=image['id'],file=image['file_name'],sha256=image_sha,
                            license_id=4,source=BASE+'/val2017/'+image['file_name'],
                            iou=intersection/max(1,union),dice=2*intersection/max(1,total),boundary_f1=f1,
                            intersection=intersection,union=union))
        if sample is None:
            sample=original_rgb
    backend.close()
    return dict(dataset='COCO 2017 validation',license=chosen,
                selection=f'first {count} ascending image IDs; license 4; person area sum >=10%; no crowd person annotations',
                preprocessing='BT.601 limited NV12 round-trip and production model_rgb; trim final odd row/column',
                annotations_json_sha256=digest,annotations_archive=archive_url,count=len(records),
                mean_iou=float(np.mean([r['iou'] for r in records])),
                global_iou=sum(r['intersection'] for r in records)/sum(r['union'] for r in records),
                mean_dice=float(np.mean([r['dice'] for r in records])),
                mean_boundary_f1=float(np.mean([r['boundary_f1'] for r in records])),images=records),sample


def timings(rgb,frames,work):
    results=[]
    replacement=work/'background.png'
    cv2.imwrite(str(replacement),np.full((64,64,3),(120,80,40),np.uint8))
    for w,h in [(1280,720),(1920,1080)]:
        y,uv=replacement_nv12(rgb,w,h)
        source=y.tobytes()+uv.tobytes()
        for effect,frame in [('off',False),('blur',False),('replace',False),('off',True),('blur',True)]:
            backend=Backend(w,h,effect,frame,1,6,1.5,replacement=str(replacement))
            for i in range(3):
                backend.process_bytes(source,1_000_000_000+i*33_333_333)
            wall,cpu=[],[]
            for i in range(frames):
                start=time.perf_counter_ns();cstart=time.process_time_ns()
                output=backend.process_bytes(source,2_000_000_000+i*33_333_333)
                cpu.append((time.process_time_ns()-cstart)/1e6)
                wall.append((time.perf_counter_ns()-start)/1e6)
                assert len(output)==len(source)
            models={name:model.runtime.stats() for name,model in [('segmentation',backend.segmenter),('face',backend.detector)] if model}
            backend.close()
            results.append(dict(width=w,height=h,background=effect,framing=frame,frames=frames,
                                wall_p50_ms=float(np.median(wall)),wall_p95_ms=float(np.percentile(wall,95)),
                                process_cpu_mean_ms=float(np.mean(cpu)),process_cpu_p50_ms=float(np.median(cpu)),
                                cores_at_30fps=float(np.mean(cpu)*30/1000),runtime=models))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--count',type=int,default=50)
    p.add_argument('--frames',type=int,default=30)
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(1)
    q,sample=quality(a.work,a.count)
    data=dict(platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].split('\n')[0].split(':',1)[1].strip(),
              ort_threads=2,opencv_threads=1,quality=q,timings=timings(sample,a.frames,a.work))
    a.output.write_text(json.dumps(data,indent=2)+'\n')
