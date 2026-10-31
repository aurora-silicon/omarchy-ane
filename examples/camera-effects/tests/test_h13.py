# SPDX-License-Identifier: MIT
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
import pytest

from aurora_camera_effects.backend import Model,ROOT
from aurora_camera_effects.hybrid import HybridGraph,session,pack_surface,unpack_surface

spec=importlib.util.spec_from_file_location('camera_h13_build',ROOT/'h13/build.py')
builder=importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def synthetic(tmp_path,cin,cout,k,groups,stride,pads,transpose=False):
    rng=np.random.default_rng(3926)
    shape=[1,cin,9,11]
    w=(rng.standard_normal((cin,cout,k,k) if transpose else (cout,cin//groups,k,k))*.1).astype(np.float16)
    b=(rng.standard_normal(cout)*.1).astype(np.float16)
    oh,ow=(18,22) if transpose else ((9+pads[0]+pads[2]-k)//stride+1,(11+pads[1]+pads[3]-k)//stride+1)
    outshape=[1,cout,oh,ow]
    node=h.make_node('ConvTranspose' if transpose else 'Conv',['x','w','b'],['y'],
                     group=groups,pads=pads,strides=[stride,stride])
    graph=h.make_graph([node],'test',[h.make_tensor_value_info('x',1,shape)],
                       [h.make_tensor_value_info('y',1,outshape)],
                       [nh.from_array(w.astype(np.float32),'w'),nh.from_array(b.astype(np.float32),'b')])
    onnx.save(h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=10),tmp_path/'model.onnx')
    layers,transform=builder.layers_for(node,{'w':w,'b':b},shape)
    package=tmp_path/'package';package.mkdir()
    stages=[]
    for i,(weight,bias,channels,g) in enumerate(layers):
        weight,bias,pin,pg=builder.padded(weight,bias,channels,g)
        path=package/f'stage{i}'
        builder.write_conv(path,[1,pin,shape[2]+pads[0]+pads[2],shape[3]+pads[1]+pads[3]],weight,bias,pg)
        stages.append(path.name)
    metadata=dict(model='synthetic',source_sha256='synthetic-open-test')
    plan=dict(schema=1,**metadata,onnx_sha256=hashlib.sha256((tmp_path/'model.onnx').read_bytes()).hexdigest(),
              inputs=['x'],outputs=['y'],steps=[dict(kind='ane',inputs=['x'],outputs=['y'],stages=stages,
              shape=shape,output_shape=outshape,transform=transform)],files={})
    for file in package.rglob('*'):
        if file.is_file():plan['files'][str(file.relative_to(package))]=builder.sha(file)
    (package/'plan.json').write_text(json.dumps(plan))
    return metadata,package,rng.standard_normal(shape).astype(np.float16)


@pytest.mark.parametrize('cin,cout,k,groups,stride,pads',[
    (3,7,3,1,2,[0,1,2,0]),
    (7,7,3,7,2,[1,1,1,1]),
    (3,5,5,1,1,[2,2,2,2]),
    (7,7,5,7,2,[2,2,2,2]),
    (3,1,1,1,1,[0,0,0,0]),
])
def test_exact_rewrites_with_nonmultiple_channels_and_borders(tmp_path,cin,cout,k,groups,stride,pads):
    metadata,package,x=synthetic(tmp_path,cin,cout,k,groups,stride,pads)
    g=HybridGraph(tmp_path,metadata,'unused',simulate=True,package=package)
    expected=session(tmp_path/'model.onnx').run(None,{'x':x.astype(np.float32)})[0]
    try:
        actual=g(x)[0]
        np.testing.assert_allclose(actual,expected,atol=.002,rtol=.002)
    finally:g.close()


def test_transpose_conv_pixel_shuffle_preserves_kernel_order_and_bias(tmp_path):
    metadata,package,x=synthetic(tmp_path,3,2,2,1,2,[0,0,0,0],transpose=True)
    g=HybridGraph(tmp_path,metadata,'unused',simulate=True,package=package)
    expected=session(tmp_path/'model.onnx').run(None,{'x':x.astype(np.float32)})[0]
    try:np.testing.assert_allclose(g(x)[0],expected,atol=.002,rtol=.002)
    finally:g.close()


@pytest.mark.parametrize('mode,coordinates',[('linear','half_pixel'),('nearest','asymmetric')])
def test_resize_phase_filters_preserve_onnx_edges(tmp_path,mode,coordinates):
    shape=[1,3,5,7];outshape=[1,3,10,14]
    scales=nh.from_array(np.array([1,1,2,2],np.float32),'scales')
    node=h.make_node('Resize',['x','','scales'],['y'],mode=mode,
                     coordinate_transformation_mode=coordinates,nearest_mode='floor')
    graph=h.make_graph([node],'resize',[h.make_tensor_value_info('x',1,shape)],
                       [h.make_tensor_value_info('y',1,outshape)],[scales])
    onnx.save(h.make_model(graph,opset_imports=[h.make_opsetid('',17)],ir_version=10),tmp_path/'model.onnx')
    layers,transform=builder.layers_for(node,{},shape,outshape)
    package=tmp_path/'package';package.mkdir()
    w,b,pin,pg=builder.padded(*layers[0])
    pad=transform['pads']
    builder.write_conv(package/'stage',[1,pin,shape[2]+pad[0]+pad[2],shape[3]+pad[1]+pad[3]],w,b,pg)
    metadata=dict(model='synthetic',source_sha256='synthetic-open-test')
    plan=dict(schema=1,**metadata,onnx_sha256=builder.sha(tmp_path/'model.onnx'),inputs=['x'],outputs=['y'],
              files={},steps=[dict(kind='ane',inputs=['x'],outputs=['y'],stages=['stage'],shape=shape,
                                   output_shape=outshape,transform=transform)])
    (package/'plan.json').write_text(json.dumps(plan))
    g=HybridGraph(tmp_path,metadata,'unused',simulate=True,package=package)
    x=np.random.default_rng(2).standard_normal(shape).astype(np.float16)
    try:
        actual=g(x)[0]
        expected=session(tmp_path/'model.onnx').run(None,{'x':x.astype(np.float32)})[0]
        np.testing.assert_allclose(actual,expected,atol=.001,rtol=.001)
    finally:g.close()


def test_unqualified_or_simulation_receipt_never_queries_device(tmp_path,monkeypatch):
    metadata,package,_=synthetic(tmp_path,3,7,3,1,1,[1]*4)
    monkeypatch.setattr(C,'CDLL',lambda *a,**kw:pytest.fail('qualification gate queried a device'))
    with pytest.raises(OSError,match='qualification'):
        HybridGraph(tmp_path,metadata,'unused',package=package)
    receipt=dict(status='pass',target='H13',mode='cpu-simulation',abi=1,chip=13,
                 plan_sha256=builder.sha(package/'plan.json'),images=30,islands_compared=1)
    (package/'qualification.json').write_text(json.dumps(receipt))
    with pytest.raises(OSError,match='qualification'):
        HybridGraph(tmp_path,metadata,'unused',package=package)


def test_package_tamper_is_rejected_before_device_query(tmp_path,monkeypatch):
    metadata,package,_=synthetic(tmp_path,3,7,3,1,1,[1]*4)
    with (package/'stage0/weights.bin').open('ab') as f:f.write(b'bad')
    monkeypatch.setattr(C,'CDLL',lambda *a,**kw:pytest.fail('tampered package queried a device'))
    with pytest.raises(ValueError,match='identity mismatch'):
        HybridGraph(tmp_path,metadata,'unused',package=package)


def test_explicit_cpu_never_loads_libane_or_native(monkeypatch):
    monkeypatch.setattr(C,'CDLL',lambda *a,**kw:pytest.fail('CPU engine loaded libane'))
    model=Model('selfie',engine='cpu',native_factory=lambda:pytest.fail('CPU loaded native'))
    try:
        _,call=model(np.zeros((1,3,256,256),np.float16))
        assert call.backend==0 and model.engine=='cpu'
    finally:model.close()


def test_absent_libane_still_runs_cpu(tmp_path):
    model=Model('selfie',library=tmp_path/'absent.so')
    try:
        _,call=model(np.zeros((1,3,256,256),np.float16))
        assert call.backend==0 and model.engine=='cpu'
    finally:model.close()


def test_surface_padding_is_zero_and_rows_planes_keep_their_order():
    binding=dict(nchw=[1,3,2,5,64,32],shape=[1,3,2,5],dtype='float16',logicalBytes=60,allocationBytes=192)
    x=np.arange(30,dtype=np.float16).reshape(1,3,2,5)
    packed=pack_surface(x,binding)
    np.testing.assert_array_equal(unpack_surface(packed,binding),x)
    for c in range(3):
        for y in range(2):
            assert packed[c*64+y*32+10:c*64+(y+1)*32]==bytes(22)
    with pytest.raises((ValueError,TypeError)):
        unpack_surface(packed[:-1],binding)
    with pytest.raises(ValueError):
        pack_surface(x,dict(binding,nchw=[1,3,2,5,64,8]))
