# SPDX-License-Identifier: MIT
"""Worker-only effects backend; libane owns inference fallback policy."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import onnxruntime as ort
from .image import model_rgb, guided_mask, composite, crop_nv12, replacement_nv12
from .framing import Framer, faces

# Two bounded ORT threads per model; avoid another implicit OpenCV pool.
cv2.setNumThreads(1)

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO/'bindings/python/python'))
from ane import Fallback
# Source builds, private prefixes and distro libdirs share the same runtime.
_LIBRARIES=(REPO/'bindings/python/dylib/libane_python.so',
            Path(__file__).resolve().parents[2]/'libane_python.so',
            Path('/usr/lib64/libane_python.so'),Path('/usr/lib/libane_python.so'))
LIBRARY=next((path for path in _LIBRARIES if path.is_file()),_LIBRARIES[-1])


def verified(root, name):
    root=Path(root).resolve()
    metadata=json.loads((root/'model.json').read_text())
    shape=[1,3,256,256] if name=='selfie' else [1,3,320,320]
    if metadata.get('model')!=name or metadata.get('input')!=shape:
        raise ValueError('unsupported model contract')
    if not {'model.onnx','model.mil','weights.bin'} <= metadata['artifacts'].keys():
        raise ValueError('incomplete model identities')
    for item,digest in metadata['artifacts'].items():
        path=(root/item).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'model identity mismatch: {item}')
    return metadata


class Model:
    def __init__(self, name, root=ROOT/'models', library=LIBRARY, native_factory=None, *, engine='auto'):
        if engine not in ('auto','cpu','ane'):
            raise ValueError('camera effects engine must be auto, cpu or ane; no GPU consumer is installed')
        self.closed=False
        self.root=Path(root)/name
        self.metadata=verified(self.root,name)
        options=ort.SessionOptions()
        options.intra_op_num_threads=2
        options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(self.root/'model.onnx'),options,providers=['CPUExecutionProvider'])
        self.names=[o.name for o in self.session.get_outputs()]
        self.shapes=[tuple(o.shape) for o in self.session.get_outputs()]
        if self.names!=self.metadata['output_names']:
            raise ValueError('model output contract mismatch')
        self.native=None
        def load():
            from .native import NativeGraph
            self.native=native_factory() if native_factory else NativeGraph(self.root,self.metadata,library)
            if self.native is None:
                return None
            return lambda x: self.validate(self.native(x))
        self.runtime=Fallback(self.cpu,library=library,native_factory=load if engine!='cpu' else None)
        if engine=='ane' and self.runtime.native is None:
            self.close()
            raise RuntimeError('requested ANE camera graph is unavailable')
        print(f'camera-effects: engine: {"ane" if self.runtime.native is not None else "cpu"} ({name})',
              file=sys.stderr,flush=True)

    def validate(self, outputs):
        if len(outputs)!=len(self.shapes):
            raise ValueError('missing model outputs')
        for value,shape in zip(outputs,self.shapes):
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('invalid model output')
        return outputs

    def cpu(self, x):
        return self.validate(self.session.run(self.names,{'x':x.astype(np.float32)}))

    def __call__(self, x):
        if self.closed:
            raise RuntimeError('model closed')
        # Both graph boundaries use fp16; CPU convolution accumulates fp32.
        x=np.ascontiguousarray(x,dtype=np.float16)
        values,call=self.runtime.run(x)
        return self.validate(values),call

    def close(self):
        if self.closed:
            return
        self.closed=True
        try:
            if self.native is not None:
                self.native.close()
        finally:
            # Fallback callbacks form a Python reference cycle. Release ORT's
            # resident graphs at stream teardown rather than waiting for GC.
            self.session=None


class Backend:
    def __init__(self, width, height, background='off', framing=False,
                 minimum=1., maximum=1., default=1., zoom=0., replacement='',
                 matrix=601, full=False, *, model_root=ROOT/'models', library=LIBRARY, engine='auto'):
        if type(width)!=int or type(height)!=int or width<2 or height<2 or width>4096 or height>2160 or (width|height)&1:
            raise ValueError('expected even NV12 dimensions up to 4096x2160')
        if background not in ('off','blur','replace') or matrix not in (601,709):
            raise ValueError('unsupported effect or color matrix')
        self.width,self.height=width,height
        self.background,self.framing=background,bool(framing)
        self.matrix,self.full=matrix,bool(full)
        self.segmenter=Model('selfie',model_root,library,engine=engine) if background!='off' else None
        self.detector=None
        try:
            self.detector=Model('yunet',model_root,library,engine=engine) if framing else None
            self.framer=Framer(minimum,maximum,default,zoom) if framing else None
            self.replacement=None
            if background=='replace':
                image=cv2.imread(str(replacement),cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('cannot read replacement image')
                self.replacement=replacement_nv12(image[...,::-1],width,height,matrix,full)
        except BaseException:
            self.close()
            raise
        self.closed=False

    def reset(self):
        if self.framer:
            self.framer.reset()

    def mask(self, rgb, luma):
        x=cv2.resize(rgb,(256,256),interpolation=cv2.INTER_AREA).transpose(2,0,1)[None]
        values,call=self.segmenter(x)
        raw=np.clip(values[0][0,0],0,1)
        return guided_mask(raw,luma),call

    def process_bytes(self, source, pts_ns):
        if self.closed:
            raise RuntimeError('stream closed')
        n=self.width*self.height
        if len(source)!=n*3//2:
            raise ValueError('invalid NV12 byte length')
        if self.background=='off' and not self.framing:
            return source
        y=np.frombuffer(source,np.uint8,count=n).reshape(self.height,self.width)
        uv=np.frombuffer(source,np.uint8,offset=n).reshape(self.height//2,self.width//2,2)
        window=None
        if self.framing:
            scale=320/max(self.width,self.height)
            rw=max(1,round(self.width*scale)); rh=max(1,round(self.height*scale))
            rgb=model_rgb(y,uv,(rw,rh),self.matrix,self.full)
            bgr=np.zeros((320,320,3),np.float32)
            bgr[:rh,:rw]=rgb[...,::-1]*255
            outputs,_=self.detector(bgr.transpose(2,0,1)[None])
            detected=faces(outputs)
            if len(detected):
                detected/=np.array([rw/320,rh/320,rw/320,rh/320])
                right=np.minimum(1,detected[:,:2]+detected[:,2:])
                detected[:,:2]=np.clip(detected[:,:2],0,1)
                detected[:,2:]=right-detected[:,:2]
                detected=detected[np.all(detected[:,2:]>0,axis=1)]
            pts=pts_ns/1e9 if pts_ns>0 else time.monotonic()
            window=self.framer.step(detected,pts)
        if self.background!='off':
            rgb=model_rgb(y,uv,256,self.matrix,self.full)
            mask,_=self.mask(rgb,y)
            y,uv=composite(y,uv,mask,self.background,self.replacement)
        if window is not None:
            y,uv=crop_nv12(y,uv,window)
        return y.tobytes()+uv.tobytes()

    def close(self):
        self.closed=True
        errors=[]
        for model in (self.segmenter,self.detector):
            if model:
                try:
                    model.close()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]


def create(width, height, background, framing, minimum, maximum, default, zoom,
           replacement, matrix, full):
    return Backend(width,height,('off','blur','replace')[background],bool(framing),
                   minimum,maximum,default,zoom,replacement,matrix,bool(full),
                   engine=os.environ.get('AURORA_CAMERA_ENGINE','auto'))
