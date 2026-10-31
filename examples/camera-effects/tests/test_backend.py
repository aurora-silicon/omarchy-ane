# SPDX-License-Identifier: MIT
import ctypes as C
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from aurora_camera_effects.backend import Backend, Model, ROOT, verified
from aurora_camera_effects.framing import Framer
from aurora_camera_effects.image import guided_mask, composite, model_rgb, crop_nv12, replacement_nv12


def test_off_is_exact_and_does_not_load_models():
    backend=Backend(16,8)
    pixels=np.arange(16*8*3//2,dtype=np.uint8).tobytes()
    assert backend.segmenter is None and backend.detector is None
    assert backend.process_bytes(pixels,123)==pixels
    backend.close()
    with pytest.raises(RuntimeError):
        backend.process_bytes(pixels,124)


def test_guided_edges_and_mask_bounds():
    y=np.zeros((128,256),np.uint8);y[:,128:]=255
    low=cv2.resize(y,(32,16),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    coarse=cv2.GaussianBlur(low,(9,9),1.5)
    result=guided_mask(coarse,y)
    linear=cv2.resize(coarse,(256,128))
    # Full-resolution guidance should sharpen the known luma boundary.
    # Global soft-mask MAE need not decrease when error is redistributed.
    contrast=float(np.mean(result[:,128]-result[:,127]))
    baseline=float(np.mean(linear[:,128]-linear[:,127]))
    assert contrast>2*baseline
    assert np.isfinite(result).all() and np.min(result)>=0 and np.max(result)<=1
    np.testing.assert_allclose(guided_mask(np.full((16,32),.7,np.float32),y),.7,atol=2e-5)


def test_native_failure_retires_owner_and_uses_same_cpu_model():
    class Broken:
        calls=0
        def __call__(self,x):
            self.calls+=1
            raise OSError('injected native execution failure')
        def close(self): pass
    broken=Broken()
    model=Model('selfie',native_factory=lambda:broken)
    x=np.random.default_rng(4).random((1,3,256,256),dtype=np.float32)
    expected=model.cpu(x.astype(np.float16))
    for _ in range(2):
        actual,call=model(x)
        np.testing.assert_array_equal(actual[0],expected[0])
        assert call.backend==0 and call.reason==2
    assert broken.calls==1
    model.close()


def test_invalid_native_output_also_falls_back():
    class Broken:
        calls=0
        def __call__(self,x):
            self.calls+=1
            return [np.full((1,1,256,256),np.nan)]
        def close(self):pass
    b=Broken(); model=Model('selfie',native_factory=lambda:b)
    actual,call=model(np.zeros((1,3,256,256),np.float32))
    assert np.isfinite(actual[0]).all() and call.reason==2
    model.close()


def test_framing_range_smoothing_loss_and_isolation():
    a=Framer(1,6,1.5);b=Framer(1,6,1.5)
    assert a.step([],1)==(.5,.5,1.5)
    detection=np.array([[.75,.35,.08,.12]])
    first=a.step(detection,1+1/30)
    assert .5<first[0]<.79 and 1.5<first[2]<6
    for i in range(180):
        x,y,z=a.step(detection,2+i/30)
        assert 1<=z<=6 and .5/z<=x<=1-.5/z and .5/z<=y<=1-.5/z
    assert tuple(b.window)==(.5,.5,1.5)
    for i in range(300):a.step([],8+i/30)
    np.testing.assert_allclose(a.window,[.5,.5,1.5],atol=1e-6)
    a.step(detection,20);a.reset();assert tuple(a.window)==(.5,.5,1.5)
    with pytest.raises(ValueError):Framer(1,6,1.5,6.1)
    with pytest.raises(ValueError):Framer(1,6,float('nan'))


def test_nv12_channels_color_and_in_place_shape():
    y=np.arange(64*32,dtype=np.uint8).reshape(32,64)
    uv=np.empty((16,32,2),np.uint8);uv[:]=[40,210]
    yy,uu=composite(y,uv,np.ones_like(y,dtype=np.float32),'blur')
    np.testing.assert_array_equal(yy,y);np.testing.assert_array_equal(uu,uv)
    yy,uu=crop_nv12(y,uv,(.9,.1,6))
    assert yy.shape==y.shape and uu.shape==uv.shape
    np.testing.assert_array_equal(uu,uv)
    for matrix in (601,709):
        for full in (True,False):
            rgb=np.full((32,64,3),(.2,.6,.8),np.float32)
            ly,lu=replacement_nv12(np.rint(rgb*255).astype(np.uint8),64,32,matrix,full)
            np.testing.assert_allclose(model_rgb(ly,lu,16,matrix,full),rgb[:16,:16],atol=.008)


def test_both_models_and_replacement_file(tmp_path):
    background=tmp_path/'bg.png';cv2.imwrite(str(background),np.full((32,64,3),150,np.uint8))
    src=np.full(64*32*3//2,128,np.uint8).tobytes()
    stream=Backend(64,32,'replace',True,1,6,1.5,replacement=str(background))
    result=stream.process_bytes(src,1_000_000_000)
    assert len(result)==len(src) and result!=src
    assert stream.segmenter.runtime.stats()['cpu_calls']==1
    assert stream.detector.runtime.stats()['cpu_calls']==1
    stream.close()


def test_exported_model_identity_is_checked(tmp_path):
    import shutil
    shutil.copytree(ROOT/'models/selfie',tmp_path/'selfie',ignore=shutil.ignore_patterns('compiled'))
    with (tmp_path/'selfie/model.onnx').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='identity mismatch'):verified(tmp_path/'selfie','selfie')


def test_dormant_native_never_probes_device(monkeypatch):
    from aurora_camera_effects.native import NativeGraph
    monkeypatch.setattr(C,'CDLL',lambda *args:pytest.fail('unqualified package probed a device'))
    with pytest.raises(OSError,match='no qualified'):NativeGraph(ROOT/'models/selfie',{'native_targets':[]},'unused')
