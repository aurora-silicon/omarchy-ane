#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare H13 islands and final outputs on 30 licensed portrait images.

--simulate is a CPU mathematical check, never evidence of device execution.
--qualify explicitly executes the H13 package and writes its qualification only
after all island checks and all image criteria pass. No camera/audio is opened.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aurora_camera_effects.backend import verified
from aurora_camera_effects.hybrid import HybridGraph,session,sha


def verify(root,package,images,simulate,library,compiler_root):
    name=root.name
    metadata=verified(root,name)
    files=sorted(images.glob('*.jpg'))
    if len(files)!=30:
        raise ValueError('expected exactly 30 audited portrait JPEGs')
    provenance=json.loads((images/'images.json').read_text())
    by_file={v['file_name']:v for v in provenance}
    if set(by_file)!={p.name for p in files} or any(v['license']!=4 or v['height']<v['width'] for v in provenance):
        raise ValueError('expected COCO CC BY 2.0 portrait provenance')
    graph=HybridGraph(root,metadata,library,simulate=simulate,qualification=not simulate,
                      package=package,compiler_root=compiler_root)
    reference=session(root/'model.onnx')
    records=[]
    try:
        for index,path in enumerate(files):
            if sha(path)!=by_file[path.name].get('sha256'):
                raise ValueError('portrait identity mismatch')
            bgr=cv2.imread(str(path))
            size=256 if name=='selfie' else 320
            pixels=cv2.resize(bgr,(size,size),interpolation=cv2.INTER_AREA)
            if name=='selfie':
                pixels=pixels[...,::-1].astype(np.float32)/255
            x=np.ascontiguousarray(pixels.transpose(2,0,1)[None],dtype=np.float16)
            cpu=reference.run(None,{'x':x.astype(np.float32)})
            start=time.perf_counter()
            actual=graph(x,compare_islands=True)
            elapsed=time.perf_counter()-start
            errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,cpu)]
            record=dict(file=path.name,id=by_file[path.name]['id'],sha256=sha(path),
                        source='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+path.name,
                        license='CC-BY-2.0',seconds=elapsed,output_max_abs=errors)
            if name=='selfie':
                a,b=actual[0]>=.5,cpu[0]>=.5
                union=int(np.logical_or(a,b).sum())
                iou=float(np.logical_and(a,b).sum()/union) if union else 1.
                disagreement=float(np.mean(a!=b))
                mae=float(np.mean(np.abs(actual[0]-cpu[0])))
                record.update(mask_iou=iou,mask_union=union,mask_disagreement=disagreement,mask_mean_abs=mae)
                # IoU alone is unstable for nearly empty masks: changing one
                # pixel in a 19-pixel union changes it by over five percent.
                # Keep every IoU, and also bound the absolute image error.
                if disagreement>.005 or mae>.005:
                    raise ValueError(f'{path.name}: mask disagreement/MAE exceeds .005')
            else:
                # Bounding-box/logit output criterion, not a claim of AP parity.
                for a,b in zip(actual,cpu):
                    np.testing.assert_allclose(a,b,atol=.03,rtol=.02)
            records.append(record)
            print(name,index+1,record.get('mask_iou',max(errors)),flush=True)
        if name=='selfie' and np.mean([r['mask_iou'] for r in records])<.98:
            raise ValueError('mean mask IoU < .98')
        return dict(status='pass',target='H13',mode='cpu-simulation' if simulate else 'libane-device',
                    abi=None if simulate else 1,chip=None if simulate else 13,
                    host=platform.system()+'/'+platform.machine()+' build host',platform=platform.platform(),plan_sha256=sha(package/'plan.json'),
                    images=len(records),islands_compared=len(graph.island_errors),island_max_abs=graph.island_errors,
                    criterion='islands: abs <= .002 + .002*abs(reference); mean mask IoU >= .98, every image mask disagreement and MAE <= .005; YuNet abs <= .03 + .02*abs(reference)',
                    mean_mask_iou=float(np.mean([r['mask_iou'] for r in records])) if name=='selfie' else None,
                    min_mask_iou=min(r['mask_iou'] for r in records) if name=='selfie' else None,
                    records=records)
    finally:
        graph.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--simulate',action='store_true')
    mode.add_argument('--qualify',action='store_true')
    p.add_argument('--library',type=Path,default=Path('/usr/lib/libane_python.so'))
    p.add_argument('--compiler-root',type=Path,default=Path('/usr/lib/mil-hwx-compiler'))
    a=p.parse_args()
    try:
        result=verify(a.model.resolve(),a.package.resolve(),a.images.resolve(),a.simulate,a.library,a.compiler_root)
    except Exception as error:
        # A failed requalification must retire a formerly qualified package.
        if a.qualify:
            (a.package/'qualification.json').write_text(json.dumps(dict(status='fail',error=str(error)))+'\n')
        a.output.write_text(json.dumps(dict(status='fail',mode='cpu-simulation' if a.simulate else 'libane-device',
                                          host=platform.system()+'/'+platform.machine()+' build host',error=str(error)),indent=2)+'\n')
        raise
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    if a.qualify:
        (a.package/'qualification.json').write_text(json.dumps(result,indent=2)+'\n')
