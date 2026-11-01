# SPDX-License-Identifier: MIT
"""Procedural qualification plumbing, without any device access."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from aurora_camera_effects.backend import ROOT
from aurora_camera_effects.whole_graph import check_receipt

spec = importlib.util.spec_from_file_location('linked_qualify', ROOT/'linked/qualify.py')
qualifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualifier)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path/'selfie'
    package = root/'compiled/H13/linked'
    package.mkdir(parents=True)
    (package/'camera-linked.json').write_text(json.dumps(dict(model='selfie')))
    (package/'qualification.json').write_text('previous receipt')
    compiler = tmp_path/'compiler'
    (compiler/'tools').mkdir(parents=True)
    (compiler/'tools/h13_run_linux.py').write_text('test reader identity')
    (compiler/'research').mkdir()
    for name in ('inspect_anec.py', 'h13_td.py'):
        (compiler/'research'/name).write_text('test inspector identity')
    library = tmp_path/'binding.so'
    library.write_text('test binding identity')
    wedged = tmp_path/'wedged'
    wedged.write_text('0')
    clock = [0]
    monkeypatch.setattr(qualifier.time, 'perf_counter_ns', lambda:clock[0])
    inputs = [(dict(file=f'{i:012}.jpg'), np.ones((1, 3, 2, 2), np.float16)) for i in range(30)]
    monkeypatch.setattr(qualifier, 'preflight', lambda *a:({}, package, inputs))
    class Reference:
        def run(self, *args):
            clock[0] += 5000000
            return [np.ones((1, 1, 2, 2), np.float32)]
    monkeypatch.setattr(qualifier, 'session', lambda path:Reference())
    owners = []
    class Graph:
        setup_ms = .1
        def __init__(self, *args, **kwargs):
            self.loads = 1
            self.executes = 0
            self.closed = False
            owners.append(self)
        def __call__(self, x):
            self.executes += 1
            clock[0] += 1000000
            return [np.ones((1, 1, 2, 2), np.float32)]
        def close(self):
            self.closed = True
    monkeypatch.setattr(qualifier, 'WholeGraph', Graph)
    return root, compiler, library, wedged, owners, Graph


def test_30_cold_runs_then_one_resident_timing_owner(rig, tmp_path):
    root, compiler, library, wedged, owners, _ = rig
    result = qualifier.qualify(root, tmp_path, compiler, library, tmp_path/'result.json',
        simulate=False, frames=100, approval=tmp_path/'approval', device='m1pro', wedged=wedged)
    assert result['status'] == 'pass'
    assert len(owners) == 31 and all(g.closed for g in owners)
    assert [g.executes for g in owners] == [1]*30 + [100]
    assert result['timing']['ane_p50_ms'] == 1
    assert result['timing']['cpu_p50_ms'] == 5
    check_receipt(result, root/'compiled/H13/linked')
    assert len(list((root/'compiled/H13/linked').glob('qualification.previous-*.json'))) == 1


def test_bad_cold_output_leaves_failed_receipt(rig, tmp_path, monkeypatch):
    root, compiler, library, wedged, owners, Graph = rig
    def bad(self, x):
        self.executes += 1
        return [np.full((1, 1, 2, 2), np.nan, np.float32)]
    monkeypatch.setattr(Graph, '__call__', bad)
    result = qualifier.qualify(root, tmp_path, compiler, library, tmp_path/'failure.json',
        simulate=False, frames=100, approval=tmp_path/'approval', device='m1pro', wedged=wedged)
    assert result['status'] == 'fail' and result['cold_passes'] == 0
    assert len(result['records']) == 30 and all(g.closed for g in owners)
    assert json.loads((root/'compiled/H13/linked/qualification.json').read_text())['status'] == 'fail'
    with pytest.raises(OSError):
        check_receipt(result, root/'compiled/H13/linked')


def test_simulation_does_not_replace_device_receipt(rig, tmp_path):
    root, compiler, library, _, _, _ = rig
    result = qualifier.qualify(root, tmp_path, compiler, library, tmp_path/'simulation.json',
        simulate=True)
    assert result['status'] == 'pass' and result['mode'] == 'cpu-simulation'
    assert (root/'compiled/H13/linked/qualification.json').read_text() == 'previous receipt'
    with pytest.raises(OSError):
        check_receipt(result, root/'compiled/H13/linked')


def test_cold_device_error_aborts_even_when_driver_reports_unwedged(rig,tmp_path,monkeypatch):
    root,compiler,library,wedged,owners,Graph=rig
    Graph.program={'taskDescriptors':109}
    def broken(self,x):
        self.executes+=1
        raise OSError('injected completion -5')
    monkeypatch.setattr(Graph,'__call__',broken)
    result=qualifier.qualify(root,tmp_path,compiler,library,tmp_path/'failed-submit.json',
        simulate=False,frames=100,approval=tmp_path/'approval',device='m1pro',wedged=wedged)
    assert result['status']=='fail' and result['cold_passes']==0
    assert len(owners)==1 and owners[0].closed and len(result['records'])==1
    assert 'completion -5' in result['records'][0]['error']
    assert result['active']==dict(phase='cold-execute',portrait='000000000000.jpg',task_count=109)
    assert result['wedged_after']=='0'


def test_load_failure_keeps_active_portrait_and_does_not_retry(rig,tmp_path,monkeypatch):
    root,compiler,library,wedged,owners,_=rig
    calls=[]
    def broken(*a,**kw):
        calls.append('load')
        raise OSError('injected LOAD failure')
    monkeypatch.setattr(qualifier,'WholeGraph',broken)
    result=qualifier.qualify(root,tmp_path,compiler,library,tmp_path/'failed-load.json',
        simulate=False,frames=100,approval=tmp_path/'approval',device='m1pro',wedged=wedged)
    assert calls==['load'] and result['status']=='fail'
    assert len(result['records'])==1 and result['records'][0]['status']=='fail'
    assert result['active']==dict(phase='cold-load',portrait='000000000000.jpg')


def test_wedged_timing_frame_stops_before_the_next_submit(rig,tmp_path,monkeypatch):
    root,compiler,library,wedged,owners,Graph=rig
    Graph.program={'taskDescriptors':109}
    original=Graph.__call__
    def wedge_on_timing(self,x):
        value=original(self,x)
        if len(owners)==31: wedged.write_text('1')
        return value
    monkeypatch.setattr(Graph,'__call__',wedge_on_timing)
    result=qualifier.qualify(root,tmp_path,compiler,library,tmp_path/'timing-wedge.json',
        simulate=False,frames=100,approval=tmp_path/'approval',device='m1pro',wedged=wedged)
    assert result['status']=='fail' and result['cold_passes']==30
    assert len(owners)==31 and owners[-1].executes==1 and all(g.closed for g in owners)
    assert result['active']['phase']=='timing-execute' and result['active']['frame']==0
    assert result['wedged_checks'][-1]==dict(phase='timing-after',frame=0,value='1')


@pytest.mark.parametrize('device_error,expected',[
    (True,['selfie']),(False,['selfie','yunet'])])
def test_both_model_command_stops_after_device_error_but_not_parity_failure(tmp_path,monkeypatch,device_error,expected):
    calls=[]
    def fake(root,*a,**kw):
        calls.append(root.name)
        return dict(status='fail',device_error=device_error)
    monkeypatch.setattr(qualifier,'qualify',fake)
    monkeypatch.setattr(qualifier.sys,'argv',['qualify.py','--qualify','--models',str(tmp_path/'models'),
        '--images',str(tmp_path/'images'),'--compiler-root',str(tmp_path/'compiler'),
        '--output',str(tmp_path/'results'),'--approval',str(tmp_path/'approval'),'--device','m1pro'])
    assert qualifier.main()==1 and calls==expected
