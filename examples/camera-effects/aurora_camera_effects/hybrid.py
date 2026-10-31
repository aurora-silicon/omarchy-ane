# SPDX-License-Identifier: MIT
"""H13 convolution islands with ONNX host operations and shared CPU fallback.

The simulator never loads libane. Production requires a matching qualification
receipt before querying caps or opening any model handle.
"""
import ctypes as C
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(root, name):
    file=(root/name).resolve()
    if not file.is_relative_to(root) or not file.is_file():
        raise ValueError('package path escapes root or is missing')
    return file


def session(path):
    options=ort.SessionOptions()
    options.intra_op_num_threads=2
    options.inter_op_num_threads=1
    return ort.InferenceSession(str(path),options,providers=['CPUExecutionProvider'])


def surface_view(buffer,binding):
    """Bounded NCHW view of a validated single-conv surface, including padding."""
    n,c,height,width,plane,row=binding['nchw']
    if (n!=1 or binding.get('dtype')!='float16' or binding.get('shape')!=[n,c,height,width] or
        binding['logicalBytes']!=n*c*height*width*2 or row<width*2 or plane<row*height or
        len(buffer)!=binding['allocationBytes']):
        raise ValueError('unsupported convolution surface layout')
    return np.ndarray((n,c,height,width),dtype='<f2',buffer=buffer,
                      strides=(c*plane,plane,row,2))


def pack_surface(value,binding):
    buffer=bytearray(binding['allocationBytes'])
    view=surface_view(buffer,binding)
    if value.shape!=view.shape:
        raise ValueError('surface shape mismatch')
    view[:]=value
    return bytes(buffer)


def unpack_surface(buffer,binding):
    return surface_view(buffer,binding).astype(np.float32)


class HybridGraph:
    def __init__(self, root, metadata, library, *, simulate=False, qualification=False,
                 compiler_root=Path('/usr/lib/mil-hwx-compiler'), package=None, receipt=None):
        self.root=Path(root).resolve()
        self.directory=Path(package).resolve() if package else self.root/'compiled/H13/hybrid'
        self.closed=False
        self.records=[]
        self.simulate=simulate
        self.island_errors={}
        self.island_results={}
        self.originals={}
        self.preparation_errors={}
        self.qualification=qualification
        self.qualified_islands=None
        plan_file=local(self.directory,'plan.json')
        self.plan=json.loads(plan_file.read_text())
        if (self.plan.get('schema')!=1 or self.plan.get('model')!=metadata['model'] or
            self.plan.get('source_sha256')!=metadata['source_sha256'] or
            self.plan.get('onnx_sha256')!=sha(self.root/'model.onnx')):
            raise ValueError('hybrid package model identity mismatch')
        for name,digest in self.plan['files'].items():
            if sha(local(self.directory,name))!=digest:
                raise ValueError(f'hybrid package identity mismatch: {name}')
        if not any(s['kind']=='ane' for s in self.plan['steps']):
            raise OSError('hybrid package has no compiled islands')
        islands=[s['original'] for s in self.plan['steps'] if s['kind']=='ane' and 'original' in s]
        self.total_islands=sum(s['kind']=='ane' for s in self.plan['steps'])
        if receipt is not None:
            # Used by CPU tests to exercise exactly the production routing gate.
            self.select_islands(receipt, simulation=simulate)
        elif not simulate and not qualification:
            if not (self.directory/'qualification.json').is_file():
                raise OSError('no matching H13 hybrid qualification')
            self.select_islands(json.loads(local(self.directory,'qualification.json').read_text()))
        if qualification:
            self.island_results={key:dict(status='qualified',samples=[]) for key in islands}
        if not simulate:
            lib=C.CDLL(str(library))
            lib.pyane_query_caps.argtypes=[C.c_int,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)]
            lib.pyane_query_caps.restype=C.c_int
            abi,chip=C.c_uint32(),C.c_uint32()
            if lib.pyane_query_caps(0,C.byref(abi),C.byref(chip)) or (abi.value,chip.value)!=(1,13):
                raise OSError('H13 hybrid requires generation H13 and ABI 1')
            compiler_root=Path(compiler_root)
            sys.path[:0]=[str(compiler_root/'tools'),str(compiler_root)]
            self.runner=importlib.import_module('h13_run_linux')
            self.adapter=self.runner.LibANEAdapter(library)
        self.steps=[]
        try:
            for step in self.plan['steps']:
                if step['kind']=='cpu':
                    self.steps.append((step,session(local(self.directory,step['file']))))
                else:
                    key=step.get('original')
                    if key is not None and (qualification or
                        (self.qualified_islands is not None and key not in self.qualified_islands)):
                        self.originals[key]=session(local(self.directory,key))
                    if self.qualified_islands is not None and key not in self.qualified_islands:
                        self.steps.append((step,[]))
                        continue
                    stages=[]
                    try:
                        for name in step['stages']:
                            root=local(self.directory,name+'/reference.onnx').parent
                            if simulate:
                                stages.append((session(root/'reference.onnx'),None))
                            else:
                                manifest,_=self.runner.inspect_anec.load_package(root/'bundle')
                                if len(manifest['programs'])!=1 or manifest['dispatchPlan']!=[0]:
                                    raise ValueError('island must contain one formula-built program')
                                program=manifest['programs'][0]
                                if (program['encoder']!='h13-generic-conv' or len(program['inputs'])!=1 or
                                    len(program['outputs'])!=1):
                                    raise ValueError('unsupported island bindings')
                                prepared=self.runner.PreparedProgram(self.adapter,
                                    local(root/'bundle',program['file']).read_bytes(),
                                    [bytes(program['inputs'][0]['allocationBytes'])],
                                    [program['outputs'][0]['allocationBytes']])
                                self.records.append(prepared)
                                stages.append((program,prepared))
                    except Exception as error:
                        if not qualification:
                            raise
                        self.preparation_errors[key]=f'{type(error).__name__}: {error}'
                    self.steps.append((step,stages))
        except BaseException as error:
            self.close(primary=error)
            raise
        uses={}
        for step,_ in self.steps:
            for name in step['inputs']:
                uses[name]=uses.get(name,0)+1
        self.uses=uses

    @property
    def engine_description(self):
        count=self.total_islands if self.qualified_islands is None else len(self.qualified_islands)
        return f'ane-hybrid ({count} of {self.total_islands} islands on the ANE)'

    def select_islands(self, receipt, *, simulation=False):
        """Reject incomplete/stale receipts before any device access."""
        keys={s.get('original') for s in self.plan['steps'] if s['kind']=='ane'}
        results=receipt.get('islands',{})
        valid=(receipt.get('schema')==2 and receipt.get('status')=='pass' and
               receipt.get('target')=='H13' and
               receipt.get('mode')==('cpu-simulation' if simulation else 'libane-device') and
               (simulation or (receipt.get('abi'),receipt.get('chip'))==(1,13)) and
               receipt.get('plan_sha256')==sha(self.directory/'plan.json') and
               receipt.get('images')==30 and receipt.get('islands_compared')==len(keys) and
               isinstance(results,dict) and set(results)==keys and None not in keys)
        records=receipt.get('records',[])
        valid=valid and len(keys)==self.total_islands and len(records)==30
        valid=valid and len({r.get('file') for r in records})==30 and all(r.get('status')=='pass' for r in records)
        if self.plan.get('model')=='selfie':
            valid=valid and all(r.get('mask_iou',0)>=.98 and r.get('mask_disagreement',1)<=.005 and
                               r.get('mask_mean_abs',1)<=.005 for r in records)
        if not valid:
            raise OSError('no matching H13 hybrid qualification')
        for result in results.values():
            samples=result.get('samples',[])
            if (result.get('status') not in ('qualified','refused') or len(samples)!=30 or
                {s.get('file') for s in samples}!={r['file'] for r in records} or
                (result['status']=='qualified' and any(s.get('status')!='pass' for s in samples))):
                raise OSError('incomplete per-island qualification')
        self.qualified_islands={k for k,v in results.items() if v['status']=='qualified'}

    def island(self, step, stages, value):
        t=step['transform']
        pt,pl,pb,pr=t['pads']
        value=np.pad(value,((0,0),(0,0),(pt,pb),(pl,pr)),mode=t.get('pad_mode','constant'))
        for program,prepared in stages:
            channels=program.get_inputs()[0].shape[1] if self.simulate else program['inputs'][0]['nchw'][1]
            if channels<value.shape[1]:
                raise ValueError('island channel contract mismatch')
            value=np.pad(value,((0,0),(0,channels-value.shape[1]),(0,0),(0,0))).astype(np.float16)
            if self.simulate:
                value=program.run(None,{'x':value.astype(np.float32)})[0].astype(np.float16).astype(np.float32)
            else:
                binding=program['inputs'][0]
                prepared.refill([pack_surface(value,binding)])
                raw=prepared.run_once()[0]
                binding=program['outputs'][0]
                value=unpack_surface(raw,binding)
            if not np.isfinite(value).all():
                raise ValueError('nonfinite island output')
        r=t['crop']
        if r:
            value=value[:,:,r:-r,r:-r]
        if t['shuffle']==2:
            n,co,oh,ow=step['output_shape']
            height,width=value.shape[2:]
            value=value[:,:co*4].reshape(n,co,2,2,height,width).transpose(0,1,4,2,5,3).reshape(n,co,oh,ow)
        else:
            sy,sx=t['stride']
            value=value[:,:step['output_shape'][1],::sy,::sx]
        if list(value.shape)!=step['output_shape']:
            raise ValueError('island output shape mismatch')
        return np.ascontiguousarray(value,dtype=np.float32)

    def __call__(self, x, *, compare_islands=False, sample=None, deadline_seconds=30):
        if self.closed:
            raise RuntimeError('hybrid graph closed')
        deadline=time.monotonic()+deadline_seconds
        values={self.plan['inputs'][0]:np.ascontiguousarray(x,dtype=np.float32)}
        remaining=dict(self.uses)
        for step,operation in self.steps:
            if time.monotonic()>=deadline:
                raise TimeoutError('hybrid frame deadline before next island')
            if step['kind']=='cpu':
                result=operation.run(step['outputs'],{n:values[n] for n in step['inputs']})
            else:
                value=values[step['inputs'][0]]
                key=step.get('original')
                def original(*,rounded=False):
                    source=value.astype(np.float16).astype(np.float32) if rounded else value.astype(np.float32)
                    return self.originals[key].run(step['outputs'],{step['inputs'][0]:source})
                if compare_islands:
                    if not self.qualification:
                        raise ValueError('island comparison requires explicit qualification mode')
                    # Traverse the original graph so a faulty candidate cannot
                    # contaminate the inputs used to qualify later islands.
                    result=original()
                    record=dict(file=sample,status='pass',max_abs=None,error=None)
                    try:
                        if key in self.preparation_errors:
                            raise ValueError(self.preparation_errors[key])
                        actual=self.island(step,operation,value)
                        # Keep the existing parity boundary: compare with the
                        # same fp16 input consumed by the candidate. CPU fallback
                        # and qualification traversal retain the original input.
                        ref=original(rounded=True)[0]
                        if actual.shape!=ref.shape or not np.isfinite(actual).all() or not np.isfinite(ref).all():
                            raise ValueError('invalid island output')
                        diff=np.abs(actual-ref)
                        record['max_abs']=float(diff.max())
                        self.island_errors[key]=max(self.island_errors.get(key,0.),record['max_abs'])
                        if np.any(diff>.002+.002*np.abs(ref)):
                            raise ValueError(f'island parity failed; max_abs={record["max_abs"]}')
                    except Exception as error:
                        record.update(status='fail',error=f'{type(error).__name__}: {error}')
                        self.island_results[key].update(status='refused',refusal_reason='island parity or candidate execution failure')
                    self.island_results[key]['samples'].append(record)
                elif self.qualified_islands is not None and key not in self.qualified_islands:
                    result=original()
                else:
                    result=[self.island(step,operation,value)]
            values.update(zip(step['outputs'],result))
            for name in step['inputs']:
                remaining[name]-=1
                if remaining[name]==0 and name not in self.plan['outputs']:
                    values.pop(name,None)
        return [values[n] for n in self.plan['outputs']]

    def close(self, primary=None):
        self.closed=True
        try:
            if self.records:
                self.runner._close_all(self.records,primary)
        finally:
            self.records.clear()
            # Stream teardown releases the host partitions and their weights,
            # including when shared graph-owner cleanup raises an error.
            self.steps.clear()
            self.originals.clear()
