# SPDX-License-Identifier: MIT
"""Whole-graph ownership, qualification and per-model fallback regressions."""
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import pytest

from aurora_camera_effects import backend
from aurora_camera_effects.hybrid import sha
from aurora_camera_effects.whole_graph import (
    SCHEMA, RECEIPT_SCHEMA, WholeGraph, checked_package, check_receipt,
    pack_surface, unpack_surface,
)


@pytest.fixture
def package(tmp_path):
    root = tmp_path/'selfie'
    directory = root/'compiled/H13/linked'
    directory.mkdir(parents=True)
    shape = [1, 3, 256, 256]
    out = [1, 1, 256, 256]
    w = np.ones((1, 3, 1, 1), np.float32)/3
    graph = h.make_graph([h.make_node('Conv', ['x', 'w'], ['y'])], 'test',
        [h.make_tensor_value_info('x', 1, shape)], [h.make_tensor_value_info('y', 1, out)],
        [nh.from_array(w, 'w')])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=10), root/'model.onnx')
    for name in ('model.mil', 'weights.bin'):
        (root/name).write_bytes(b'procedural test fixture; never device executable')
        (directory/name).write_bytes((root/name).read_bytes())
    (directory/'program-0.anec').write_bytes(b'fake runner only; no hardware bytes')
    def binding(name, shape):
        n, c, height, width = shape
        row = width*2
        plane = height*row
        return dict(name=name, shape=shape, nchw=[n, c, height, width, plane, row],
                    dtype='float16', logicalBytes=n*c*height*width*2, allocationBytes=c*plane)
    program = dict(file='program-0.anec', operation='linked', encoder='h13-linked-conv',
                   inputs=[binding('x', shape)], outputs=[binding('y', out)])
    manifest = dict(target='H13', programs=[program], dispatchPlan=[0])
    (directory/'manifest.json').write_text(json.dumps(manifest))
    metadata = dict(model='selfie', source_sha256='procedural-test-only', input=shape,
                    output_shapes=[out], output_names=['y'], mil_outputs=['y'],
                    artifacts={n:sha(root/n) for n in ('model.onnx', 'model.mil', 'weights.bin')})
    (root/'model.json').write_text(json.dumps(metadata))
    plan = dict(schema=SCHEMA, target='H13', model='selfie', source_sha256=metadata['source_sha256'],
                onnx_sha256=sha(root/'model.onnx'), input_shape=shape,
                output_names=['y'], mil_outputs=['y'], output_shapes=[out],
                files={n:sha(directory/n) for n in ('manifest.json', 'model.mil', 'weights.bin', 'program-0.anec')})
    (directory/'camera-linked.json').write_text(json.dumps(plan))
    return root, directory, metadata, plan, manifest


def receipt(directory):
    return dict(schema=RECEIPT_SCHEMA, status='pass', target='H13', mode='libane-device', abi=1, chip=13,
                package_sha256=sha(directory/'camera-linked.json'), images=30, cold_passes=30,
                wedged_before='0', wedged_after='0',
                timing=dict(frames=100, ane_p50_ms=1., ane_p99_ms=1.1, cpu_p50_ms=5.),
                records=[dict(file=f'{i}.jpg', status='pass', mask_iou=1., mask_disagreement=0.,
                              mask_mean_abs=0.) for i in range(30)])


def test_reuses_one_load_and_exactly_one_execute_per_frame(package):
    root, directory, metadata, _, manifest = package
    calls = dict(load=0, execute=0, close=0)
    class Prepared:
        def __init__(self, adapter, data, inputs, outputs):
            calls['load'] += 1
        def refill(self, inputs):
            self.inputs = inputs
        def run_once(self):
            calls['execute'] += 1
            value = unpack_surface(self.inputs[0], manifest['programs'][0]['inputs'][0])
            return [pack_surface(value[:, :1], manifest['programs'][0]['outputs'][0])]
        def close(self):
            calls['close'] += 1
    runner = SimpleNamespace(inspect_anec=SimpleNamespace(load_package=lambda d:(manifest, {})),
                             LibANEAdapter=lambda lib:object(), PreparedProgram=Prepared)
    graph = WholeGraph(root, metadata, 'unused', qualification=True, runner=runner)
    x = np.ones(metadata['input'], np.float16)
    for _ in range(100):
        assert np.all(graph(x)[0] == 1)
    assert (graph.loads, graph.executes) == (1, 100)
    graph.close()
    graph.close()
    assert calls == dict(load=1, execute=100, close=1)
    with pytest.raises(RuntimeError, match='closed'):
        graph(x)


@pytest.mark.parametrize('mutation', ['hash', 'two-programs', 'two-dispatches', 'path', 'span'])
def test_bad_package_refused_before_device_access(package, monkeypatch, mutation):
    root, directory, metadata, plan, manifest = package
    monkeypatch.setattr(C, 'CDLL', lambda *a:pytest.fail('invalid package reached libane'))
    if mutation == 'hash':
        (directory/'weights.bin').write_bytes(b'wrong')
    elif mutation == 'path':
        plan['files']['../escape'] = '0'*64
    else:
        if mutation == 'two-programs':
            manifest['programs'] *= 2
        elif mutation == 'two-dispatches':
            manifest['dispatchPlan'] = [0, 0]
        else:
            manifest['programs'][0]['outputs'][0]['allocationBytes'] = 1
        (directory/'manifest.json').write_text(json.dumps(manifest))
        plan['files']['manifest.json'] = sha(directory/'manifest.json')
    (directory/'camera-linked.json').write_text(json.dumps(plan))
    with pytest.raises((ValueError, OSError)):
        WholeGraph(root, metadata, 'unused')


@pytest.mark.parametrize('mutation', ['simulation', 'stale', '29', 'duplicate', 'iou', 'nan', 'slow', 'wedged', 'warm'])
def test_receipt_cannot_authorize_incomplete_or_failing_results(package, mutation):
    _, directory, _, _, _ = package
    r = receipt(directory)
    check_receipt(r, directory)
    if mutation == 'simulation': r['mode'] = 'cpu-simulation'
    if mutation == 'stale': r['package_sha256'] = '0'*64
    if mutation == '29': r['records'].pop()
    if mutation == 'duplicate': r['records'][1]['file'] = r['records'][0]['file']
    if mutation == 'iou': r['records'][0]['mask_iou'] = .978
    if mutation == 'nan': r['timing']['ane_p99_ms'] = float('nan')
    if mutation == 'slow': r['timing']['ane_p50_ms'] = 6
    if mutation == 'wedged': r['wedged_after'] = '1'
    if mutation == 'warm': r['cold_passes'] = 0
    with pytest.raises(OSError, match='qualification'):
        check_receipt(r, directory)


def test_missing_receipt_never_opens_device(package, monkeypatch):
    root, directory, metadata, _, _ = package
    monkeypatch.setattr(C, 'CDLL', lambda *a:pytest.fail('unqualified package reached device'))
    with pytest.raises(ValueError, match='missing'):
        WholeGraph(root, metadata, 'unused')


def test_models_fall_back_independently(package, monkeypatch):
    root, _, metadata, _, _ = package
    class Policy:
        def __init__(self, cpu, library=None, native_factory=None):
            self.cpu = cpu
            self.native_failure = None
            try:
                self.native = native_factory()
            except Exception as error:
                self.native_failure = error
                self.native = None
        def run(self, x):
            call = backend.Call()
            if self.native:
                try:
                    out = self.native(x)
                    call.backend = 1
                    return out, call
                except Exception as error:
                    self.native_failure = error
                    self.native = None
            return self.cpu(x), call
    monkeypatch.setattr(backend, 'Fallback', Policy)
    class Native:
        def __call__(self, x): return [x[:, :1].astype(np.float32)]
        def close(self): pass
    selected = backend.Model('selfie', root.parent, native_factory=Native, engine='ane-whole')
    def unavailable(): raise OSError('package missing or qualification failed')
    fallback = backend.Model('selfie', root.parent, native_factory=unavailable, engine='ane-whole')
    x = np.ones(metadata['input'], np.float16)
    assert selected(x)[1].backend == 1
    assert fallback(x)[1].backend == 0
    assert selected(x)[1].backend == 1
    selected.close()
    fallback.close()


def test_package_mil_simulation_uses_its_weights_and_fp16_boundaries(tmp_path):
    import os
    import sys
    compiler = Path(os.environ.get('AURORA_CAMERA_COMPILER_ROOT', Path(__file__).resolve().parents[4]/'compiler'))
    if not (compiler/'tools/h13_reference.py').is_file():
        pytest.skip('set AURORA_CAMERA_COMPILER_ROOT for numerical simulation test')
    sys.path.insert(0, str(compiler/'tools'))
    from aurora_camera_effects.linked_sim import MILSimulation
    mil = '''program(1.3) {
      func main<ios18>(tensor<fp16, [1, 1, 2, 2]> x) {
        tensor<fp16, [1, 1, 1, 1]> w = const()[val = tensor<fp16, [1, 1, 1, 1]>([2.0])];
        string pt = const()[val = string("valid")];
        tensor<int32, [2]> st = const()[val = tensor<int32, [2]>([1, 1])];
        tensor<int32, [4]> pd = const()[val = tensor<int32, [4]>([0, 0, 0, 0])];
        int32 gp = const()[val = int32(1)];
        tensor<fp16, [1, 1, 2, 2]> c = conv(x = x, weight = w, strides = st, pad_type = pt, pad = pd, dilations = st, groups = gp);
        tensor<fp16, [1, 1, 2, 2]> y = relu(x = c);
      } -> (y);
    }'''
    path = tmp_path/'model.mil'
    path.write_text(mil)
    simulator = MILSimulation(path, tmp_path)
    x = np.array([[[[-1, 1], [2, 3]]]], np.float16)
    np.testing.assert_array_equal(simulator(x, ['y'])[0], np.maximum(x*2, 0))
    # Changing the package weights changes the result: no reference ONNX is
    # read by the simulator.
    path.write_text(mil.replace('[2.0]', '[3.0]'))
    simulator = MILSimulation(path, tmp_path)
    np.testing.assert_array_equal(simulator(x, ['y'])[0], np.maximum(x*3, 0))


def test_missing_binding_uses_original_cpu_model(package):
    root, _, metadata, _, _ = package
    class Native:
        def __call__(self, x): return [np.full((1, 1, 256, 256), np.nan)]
        def close(self): pass
    model = backend.Model('selfie', root.parent, library='missing', native_factory=Native, engine='ane-whole')
    assert np.isfinite(model(np.zeros(metadata['input'], np.float16))[0][0]).all()
    assert model.engine == 'cpu'
    model.close()


def test_resident_buffer_path_preserves_outputs_across_frames(package):
    root,_,metadata,_,manifest=package
    program=manifest['programs'][0]
    calls=dict(load=0,execute=0,close=0)
    class Prepared:
        def __init__(self,adapter,data,inputs,outputs):
            calls['load']+=1
            self.input_buffers=[C.create_string_buffer(v,len(v)) for v in inputs]
            self.output_buffers=[C.create_string_buffer(n) for n in outputs]
        def refill(self,inputs): pytest.fail('resident path allocated a refill payload')
        def run_once(self): pytest.fail('resident path copied all output bytes')
        def run_in_place(self):
            calls['execute']+=1
            value=unpack_surface(self.input_buffers[0],program['inputs'][0])
            payload=pack_surface(value[:,:1],program['outputs'][0])
            C.memmove(self.output_buffers[0],payload,len(payload))
            return self.output_buffers
        def close(self): calls['close']+=1
    runner=SimpleNamespace(inspect_anec=SimpleNamespace(load_package=lambda d:(manifest,{})),
                           LibANEAdapter=lambda lib:object(),PreparedProgram=Prepared)
    graph=WholeGraph(root,metadata,'unused',qualification=True,runner=runner)
    x=np.ones(metadata['input'],np.float16)
    first=graph(x)[0]
    for i in range(2,5): np.testing.assert_array_equal(graph(x*i)[0],x[:,:1]*i)
    np.testing.assert_array_equal(first,x[:,:1])
    assert graph.loads==1 and graph.executes==4
    graph.close()
    assert graph.input_views is None and calls==dict(load=1,execute=4,close=1)
