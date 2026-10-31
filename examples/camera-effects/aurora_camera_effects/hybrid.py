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
                 compiler_root=Path('/usr/lib/mil-hwx-compiler'), package=None):
        self.root=Path(root).resolve()
        self.directory=Path(package).resolve() if package else self.root/'compiled/H13/hybrid'
        self.closed=False
        self.records=[]
        self.simulate=simulate
        self.island_errors={}
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
        if not simulate:
            # Explicit qualification execution is a separate lead-only command.
            # A normal backend call cannot self-qualify a package.
            if not qualification:
                if not (self.directory/'qualification.json').is_file():
                    raise OSError('no matching H13 hybrid qualification')
                receipt=json.loads(local(self.directory,'qualification.json').read_text())
                if (receipt.get('status')!='pass' or receipt.get('target')!='H13' or
                    receipt.get('mode')!='libane-device' or receipt.get('abi')!=1 or receipt.get('chip')!=13 or
                    receipt.get('plan_sha256')!=sha(plan_file) or receipt.get('images',0)<30 or
                    receipt.get('islands_compared')!=sum(s['kind']=='ane' for s in self.plan['steps'])):
                    raise OSError('no matching H13 hybrid qualification')
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
                    stages=[]
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
                    self.steps.append((step,stages))
        except BaseException:
            self.close()
            raise
        uses={}
        for step,_ in self.steps:
            for name in step['inputs']:
                uses[name]=uses.get(name,0)+1
        self.uses=uses

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

    def __call__(self, x, *, compare_islands=False, deadline_seconds=30):
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
                result=[self.island(step,operation,value)]
                if compare_islands:
                    ref=session(self.directory/step['original']).run(None,
                        {step['inputs'][0]:value.astype(np.float16).astype(np.float32)})[0]
                    diff=np.abs(result[0]-ref)
                    # One fp16 boundary rounding plus float32 accumulation order.
                    tolerance=.002+.002*np.abs(ref)
                    if not np.isfinite(ref).all() or np.any(diff>tolerance):
                        raise ValueError(f"island parity failed: {step['original']}; max_abs={diff.max()}")
                    key=step['original']
                    self.island_errors[key]=max(self.island_errors.get(key,0.),float(diff.max()))
            values.update(zip(step['outputs'],result))
            for name in step['inputs']:
                remaining[name]-=1
                if remaining[name]==0 and name not in self.plan['outputs']:
                    values.pop(name,None)
        return [values[n] for n in self.plan['outputs']]

    def close(self):
        self.closed=True
        try:
            if self.records:
                self.runner._close_all(self.records,None)
                self.records.clear()
        finally:
            # Stream teardown releases the host partitions and their weights,
            # including when shared graph-owner cleanup raises an error.
            self.steps.clear()
