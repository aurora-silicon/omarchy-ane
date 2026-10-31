#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Qualify every H13 island on 30 portraits, then gate the selected hybrid.

--simulate never loads libane and cannot authorize device execution.
--qualify writes a device receipt only after the final mixed graph passes.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aurora_camera_effects.backend import verified
from aurora_camera_effects.hybrid import HybridGraph,session,sha


def output_metrics(name, actual, cpu, *, record=None):
    if len(actual)!=len(cpu) or any(a.shape!=b.shape for a,b in zip(actual,cpu)):
        raise ValueError('final output contract mismatch')
    errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,cpu)]
    if not all(np.isfinite(a).all() for a in [*actual,*cpu]):
        raise ValueError('nonfinite final output')
    if record is None:
        record={}
    record['output_max_abs']=errors
    if name=='selfie':
        a,b=actual[0]>=.5,cpu[0]>=.5
        union=int(np.logical_or(a,b).sum())
        iou=float(np.logical_and(a,b).sum()/union) if union else 1.
        disagreement=float(np.mean(a!=b))
        mae=float(np.mean(np.abs(actual[0]-cpu[0])))
        record.update(mask_iou=iou,mask_union=union,mask_disagreement=disagreement,mask_mean_abs=mae)
        if iou<.98 or disagreement>.005 or mae>.005:
            raise ValueError(f'mask gate failed: IoU={iou:.6f} (>=.98), disagreement={disagreement:.6f}, MAE={mae:.6f} (<=.005)')
    else:
        for a,b in zip(actual,cpu):
            np.testing.assert_allclose(a,b,atol=.03,rtol=.02)
    return record


def verify(root,package,images,simulate,library,compiler_root,*,inject_failure=None,progress=None,max_output_fallbacks=6):
    if max_output_fallbacks<0:
        raise ValueError('max_output_fallbacks must be nonnegative')
    if inject_failure and not simulate:
        raise ValueError('fault injection requires --simulate')
    name=root.name
    metadata=verified(root,name)
    files=sorted(images.glob('*.jpg'))
    if len(files)!=30:
        raise ValueError('expected exactly 30 audited portrait JPEGs')
    provenance=json.loads((images/'images.json').read_text())
    by_file={v['file_name']:v for v in provenance}
    if (len(provenance)!=30 or set(by_file)!={p.name for p in files} or
        any(v['license']!=4 or v['height']<v['width'] for v in provenance)):
        raise ValueError('expected COCO CC BY 2.0 portrait provenance')
    # Check all inputs before executing any candidate.
    inputs=[]
    for path in files:
        if sha(path)!=by_file[path.name].get('sha256'):
            raise ValueError('portrait identity mismatch')
        bgr=cv2.imread(str(path))
        if bgr is None:
            raise ValueError(f'cannot decode {path.name}')
        size=256 if name=='selfie' else 320
        pixels=cv2.resize(bgr,(size,size),interpolation=cv2.INTER_AREA)
        if name=='selfie':
            pixels=pixels[...,::-1].astype(np.float32)/255
        inputs.append(np.ascontiguousarray(pixels.transpose(2,0,1)[None],dtype=np.float16))
    graph=HybridGraph(root,metadata,library,simulate=simulate,qualification=True,
                      package=package,compiler_root=compiler_root)
    records=[]
    result=dict(schema=2,status='fail',target='H13',mode='cpu-simulation' if simulate else 'libane-device',
                abi=None if simulate else 1,chip=None if simulate else 13,
                host=platform.system()+'/'+platform.machine()+' build host',platform=platform.platform(),
                plan_sha256=sha(package/'plan.json'),images=30,islands_compared=len(graph.island_results),
                islands=graph.island_results,island_max_abs=graph.island_errors,
                criterion='islands: abs <= .002 + .002*abs(reference) on all 30 images; every image mask IoU >= .98, disagreement and MAE <= .005; YuNet abs <= .03 + .02*abs(reference)',
                records=records)
    try:
        if inject_failure:
            if inject_failure not in graph.island_results:
                raise ValueError('injected island is absent from the plan')
            candidate=graph.island
            def faulty(step,stages,value):
                output=candidate(step,stages,value)
                return output+13.1 if step['original']==inject_failure else output
            graph.island=faulty
            result['injected_failure']=inject_failure
        for index,(path,x) in enumerate(zip(files,inputs)):
            graph(x,compare_islands=True,sample=path.name,deadline_seconds=300 if simulate else 30)
            if progress:
                progress(result)
            print(name,'islands',index+1,'of 30',flush=True)
        # Local parity is necessary but accumulated rounding can still fail the
        # final model gate. Retire a bounded suffix conservatively, preserving
        # every failed attempt and distinguishing it from a compiler failure.
        reference=session(root/'model.onnx')
        result['output_gate_attempts']=[]
        result['max_output_fallbacks']=max_output_fallbacks
        for attempt in range(max_output_fallbacks+1):
            qualified=[k for k,v in graph.island_results.items() if v['status']=='qualified']
            refused=[k for k,v in graph.island_results.items() if v['status']=='refused']
            result.update(qualified_islands=qualified,refused_islands=refused)
            graph.qualified_islands=set(qualified)
            print(('cpu-simulation routing: ' if simulate else 'engine: ')+graph.engine_description,flush=True)
            records=[]
            result['records']=records
            gate=dict(status='incomplete',qualified_islands=list(qualified),records=records)
            result['output_gate_attempts'].append(gate)
            for index,(path,x) in enumerate(zip(files,inputs)):
                record=dict(file=path.name,id=by_file[path.name]['id'],sha256=sha(path),
                            source='https://s3.amazonaws.com/images.cocodataset.org/val2017/'+path.name,
                            license='CC-BY-2.0')
                start=time.perf_counter()
                try:
                    actual=graph(x,deadline_seconds=300 if simulate else 30)
                    record['seconds']=time.perf_counter()-start
                    cpu=reference.run(None,{'x':x.astype(np.float32)})
                    output_metrics(name,actual,cpu,record=record)
                    record['status']='pass'
                except Exception as error:
                    record.update(status='fail',error=f'{type(error).__name__}: {error}')
                records.append(record)
                if progress:
                    progress(result)
                print(name,'final',attempt+1,index+1,record['status'],record.get('mask_iou'),flush=True)
            if name=='selfie' and all('mask_iou' in r for r in records):
                result['mean_mask_iou']=float(np.mean([r['mask_iou'] for r in records]))
                result['min_mask_iou']=min(r['mask_iou'] for r in records)
            failed=[r for r in records if r['status']!='pass']
            gate['status']='fail' if failed else 'pass'
            if not failed:
                result['status']='pass'
                return result
            # Retry only finite, shape-valid numerical failures. Device/host
            # execution errors must leave a failed receipt, not trigger retries.
            if attempt==max_output_fallbacks or not qualified or any('output_max_abs' not in r for r in failed):
                return result
            key=qualified[-1]
            graph.island_results[key].update(status='refused',
                refusal_reason='conservative CPU fallback for the end-to-end output gate; not an isolated island parity failure',
                output_gate_errors=[dict(file=r['file'],error=r['error'],
                                        mask_iou=r.get('mask_iou'),output_max_abs=r['output_max_abs']) for r in failed])
            print('refused',key,': end-to-end output gate fallback',flush=True)
        return result
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
        return result
    finally:
        try:
            graph.close()
        except Exception as error:
            result.update(status='fail',cleanup_error=f'{type(error).__name__}: {error}')


def write_receipt(path, result):
    """Atomic replacement; never expose a truncated passing receipt."""
    temporary=path.with_name('.'+path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--simulate',action='store_true')
    mode.add_argument('--qualify',action='store_true')
    p.add_argument('--max-output-fallbacks',type=int,default=6,help='maximum conservative CPU suffix fallbacks after a numerical final-output gate failure (default: 6; 0 disables)')
    p.add_argument('--inject-failure',help='simulation only: corrupt this original-NNN.onnx candidate')
    p.add_argument('--library',type=Path,default=Path('/usr/lib/libane_python.so'))
    p.add_argument('--compiler-root',type=Path,default=Path('/usr/lib/mil-hwx-compiler'))
    a=p.parse_args()
    if a.max_output_fallbacks<0:
        p.error('--max-output-fallbacks must be nonnegative')
    if a.inject_failure and not a.simulate:
        p.error('--inject-failure requires --simulate')
    def save(result):
        if a.qualify:
            write_receipt(a.package/'qualification.json',result)
        write_receipt(a.output,result)
    if a.qualify:
        previous=a.package/'qualification.json'
        if previous.exists():
            previous.replace(previous.with_name(f'qualification.previous-{time.time_ns()}.json'))
    save(dict(schema=2,status='fail',mode='cpu-simulation' if a.simulate else 'libane-device',
              error='qualification incomplete or interrupted'))
    try:
        result=verify(a.model.resolve(),a.package.resolve(),a.images.resolve(),a.simulate,a.library,a.compiler_root,
                      inject_failure=a.inject_failure,progress=save,max_output_fallbacks=a.max_output_fallbacks)
    except Exception as error:
        result=dict(schema=2,status='fail',mode='cpu-simulation' if a.simulate else 'libane-device',
                    error=f'{type(error).__name__}: {error}')
    save(result)
    return 0 if result['status']=='pass' else 1


if __name__=='__main__':
    sys.exit(main())
