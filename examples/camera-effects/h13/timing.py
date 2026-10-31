#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Time the complete blur/framing pipeline at 256x256 and preview resolution.

Includes NV12 conversion, inference, guided mask, compositing and framing; excludes
JPEG decode, initialization, warm-up and PNG writing. No camera/audio is opened.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aurora_camera_effects.backend import Backend,ROOT
from aurora_camera_effects.image import replacement_nv12


def measure(bgr,width,height,engine,frames,warmup,models,framing):
    y,uv=replacement_nv12(bgr[...,::-1],width,height)
    source=y.tobytes()+uv.tobytes()
    start=time.perf_counter()
    effect=Backend(width,height,'blur',framing,1,4,1.25,model_root=models,
                   engine='ane' if engine=='ane-hybrid' else 'cpu')
    init_ms=(time.perf_counter()-start)*1000
    active={name:model for name,model in [('selfie',effect.segmenter),('yunet',effect.detector)] if model}
    def check():
        if any(model.engine!=engine for model in active.values()):
            raise RuntimeError(f'requested {engine}; actual engines: '+str({n:m.engine for n,m in active.items()}))
    try:
        check()
        for i in range(warmup):
            effect.process_bytes(source,(i+1)*33_333_333)
        check()
        before={name:model.runtime.stats() for name,model in active.items()}
        times=[]
        for i in range(frames):
            start=time.perf_counter()
            effect.process_bytes(source,(warmup+i+1)*33_333_333)
            times.append((time.perf_counter()-start)*1000)
            check()
        per_model={}
        for name,model in active.items():
            stats=model.runtime.stats()
            calls=stats['calls']-before[name]['calls']
            per_model[name]=dict(engine=model.engine,calls=calls,
                native_calls=stats['native_calls']-before[name]['native_calls'],
                mean_ms=(stats['total_ns']-before[name]['total_ns'])/max(1,calls)/1e6,
                islands_on_ane=len(model.native.qualified_islands) if model.engine=='ane-hybrid' else 0,
                islands_total=model.native.total_islands if model.engine=='ane-hybrid' else None)
        return dict(engine=engine,width=width,height=height,frames=frames,warmup=warmup,
                    init_ms=init_ms,frame_ms=times,mean_ms=statistics.mean(times),
                    median_ms=statistics.median(times),per_model=per_model)
    finally:
        effect.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('images',type=Path,nargs='+')
    p.add_argument('--engine',choices=['both','ane-hybrid','cpu'],default='both')
    p.add_argument('--models',type=Path,default=ROOT/'models')
    p.add_argument('--compiler-root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=10)
    p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--framing',action='store_true',help='include YuNet and framing')
    a=p.parse_args()
    if a.frames<1 or a.warmup<0:
        p.error('frames must be positive and warmup nonnegative')
    if a.compiler_root:
        os.environ['AURORA_CAMERA_COMPILER_ROOT']=str(a.compiler_root.resolve())
    results=[]
    try:
        for image in a.images:
            bgr=cv2.imread(str(image))
            if bgr is None:
                raise ValueError(f'cannot decode {image}')
            h,w=bgr.shape[:2]
            scale=min(1,1280/w,720/h)
            sizes=[('256x256',256,256),('preview',max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2))]
            for label,width,height in sizes:
                for engine in (['ane-hybrid','cpu'] if a.engine=='both' else [a.engine]):
                    record=measure(bgr,width,height,engine,a.frames,a.warmup,a.models,a.framing)
                    record.update(image=image.name,resolution=label)
                    results.append(record)
                    print(f'{image.name} {label} {width}x{height} {engine}: {record["median_ms"]:.2f} ms/frame (median), {record["mean_ms"]:.2f} ms/frame (mean)',flush=True)
    finally:
        a.output.write_text(json.dumps(dict(records=results),indent=2)+'\n')


if __name__=='__main__':
    main()
