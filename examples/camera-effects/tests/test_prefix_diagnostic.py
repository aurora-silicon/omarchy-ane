# SPDX-License-Identifier: MIT
"""Diagnostic runs cannot authorize production and stop at device errors."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('camera_prefix_diagnostic',
    Path(__file__).resolve().parents[1]/'linked/bisect_camera.py')
diag=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diag)


def fixture(tmp_path,monkeypatch,*,fail=False,bias=0):
    probes=tmp_path/'probes';probes.mkdir();case=probes/'case';case.mkdir()
    (case/'program-0.anec').write_bytes(b'diagnostic')
    oracle=tmp_path/'oracle';oracle.mkdir()
    x=np.ones((1,64,1,1),np.float16)
    np.savez(oracle/'oracle.npz',x=x,y=x)
    (oracle/'oracle.json').write_text('{}')
    binding=dict(dtype='float16',shape=[1,64,1,1],nchw=[1,64,1,1,64,64],
                 logicalBytes=128,allocationBytes=16384,name='x',index=5)
    output=dict(binding,name='y',index=4)
    program=dict(inputs=[binding],outputs=[output],file='program-0.anec',taskDescriptors=9)
    record=dict(status='pass',directory='case',layer=7,result='y',mode='exact-prefix',
                boundary='x',source_first_task=8,source_tasks=1)
    index=dict(records=[record])
    (probes/'index.json').write_text('{}')
    monkeypatch.setattr(diag,'preflight',lambda *a:(index,dict(request='h-test')))
    calls=[]
    class Prepared:
        def __init__(self,*args): calls.append('load')
        def run_once(self):
            calls.append('execute')
            if fail: raise OSError('test task completion -5')
            return [diag.pack_surface(x+bias,output)]
        def close(self): calls.append('close')
    monkeypatch.setitem(sys.modules,'h13_run_linux',SimpleNamespace(
        LibANEAdapter=lambda library:calls.append('adapter'),PreparedProgram=Prepared))
    monkeypatch.setitem(sys.modules,'inspect_anec',SimpleNamespace(load_package=lambda p:({'programs':[program]},None)))
    wedged=tmp_path/'wedged';wedged.write_text('0\n')
    args=(probes,tmp_path/'source',oracle,tmp_path/'compiler',tmp_path/'library',tmp_path/'approval','m1pro',tmp_path/'output')
    return args,wedged,calls


def test_cold_ladder_and_numerical_failure_are_diagnostic_only(tmp_path,monkeypatch):
    args,wedged,calls=fixture(tmp_path,monkeypatch,bias=1)
    result=diag.run(*args,wedged=wedged,cold_runs=3)
    assert result['status']=='diagnostic-complete'
    assert calls==['adapter','load','execute','close','load','execute','close','load','execute','close']
    assert len(result['records'])==3
    assert all(r['fraction_over_gate']==1 for r in result['records'])
    assert not list(tmp_path.rglob('qualification.json'))


def test_device_error_preserves_active_source_tasks_and_aborts(tmp_path,monkeypatch):
    args,wedged,calls=fixture(tmp_path,monkeypatch,fail=True)
    with pytest.raises(OSError,match='completion -5'): diag.run(*args,wedged=wedged,cold_runs=3)
    result=json.loads((args[-1]/'diagnostic.json').read_text())
    assert result['status']=='device-error'
    assert result['active']['source_first_task']==8
    assert result['active']['submitted_tasks']==9
    assert result['records']==[] and calls==['adapter','load','execute','close']


def test_empty_selection_and_wedged_device_never_load(tmp_path,monkeypatch):
    args,wedged,calls=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='no matching'): diag.run(*args,wedged=wedged,layers=[999])
    assert calls==[]
    # A new output preserves the rejected diagnostic instead of overwriting it.
    args=(*args[:-1],tmp_path/'second-output');wedged.write_text('1')
    with pytest.raises(OSError,match='wedged before'): diag.run(*args,wedged=wedged)
    assert calls==[]


def test_tampered_package_rejected_before_device_preflight(tmp_path):
    probes=tmp_path/'probes';source=tmp_path/'source';probes.mkdir();source.mkdir()
    case=probes/'case';case.mkdir();(case/'manifest.json').write_text('original')
    index=dict(schema='aurora.camera-prefixes.v1',diagnostic_only=True,source_files={},
        records=[dict(status='pass',directory='case',files={'manifest.json':diag.sha(case/'manifest.json')})])
    (probes/'index.json').write_text(json.dumps(index))
    diag.check_index(probes,source)
    (case/'manifest.json').write_text('changed')
    with pytest.raises(ValueError,match='identity'): diag.check_index(probes,source)


def test_approval_mismatch_rejected_before_hardware_or_platform_access(tmp_path):
    probes=tmp_path/'probes';source=tmp_path/'source';oracle=tmp_path/'oracle'
    for p in (probes,source,oracle): p.mkdir()
    (probes/'index.json').write_text(json.dumps(dict(schema='aurora.camera-prefixes.v1',diagnostic_only=True,source_files={},records=[])))
    (oracle/'oracle.json').write_text('{}');approval=tmp_path/'approval';approval.write_text('{}')
    with pytest.raises(ValueError,match='approved'):
        diag.preflight(probes,source,oracle,tmp_path/'compiler',tmp_path/'library',approval,'m1pro')
