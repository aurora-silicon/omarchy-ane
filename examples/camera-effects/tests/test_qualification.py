# SPDX-License-Identifier: MIT
"""CPU-only fault injection through qualification, receipt loading and routing."""
import copy
import ctypes as C
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import helper as h,numpy_helper as nh
import pytest

from aurora_camera_effects.backend import ROOT,verified
from aurora_camera_effects.hybrid import HybridGraph,session,sha

spec=importlib.util.spec_from_file_location('h13_verify',ROOT/'h13/verify.py')
verifier=importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture(scope='module',params=['selfie','yunet'])
def package(tmp_path_factory,request):
    name=request.param
    tmp_path=tmp_path_factory.mktemp('qualification-'+name)
    root=tmp_path/name;root.mkdir()
    bundle=root/'compiled/H13/hybrid';bundle.mkdir(parents=True)
    shape=[1,3,*([256,256] if name=='selfie' else [320,320])]
    info=lambda name:h.make_tensor_value_info(name,1,shape)
    nodes=[];steps=[];all_weights=[]
    for i,index in enumerate([1,46,47]):
        src='x' if i==0 else f'v{i}'
        dst='y' if i==2 else f'v{i+1}'
        w=np.eye(3,dtype=np.float32).reshape(3,3,1,1)*(.1 if i==0 else 1)
        b=np.full(3,.65 if i==0 else 0,np.float32)
        weights=[nh.from_array(w,f'w{i}'),nh.from_array(b,f'b{i}')]
        node=h.make_node('Conv',[src,f'w{i}',f'b{i}'],[dst])
        original=f'original-{index:03}.onnx'
        model=h.make_model(h.make_graph([node],'original',[info(src)],[info(dst)],weights),
                           opset_imports=[h.make_opsetid('',17)],ir_version=10)
        onnx.save(model,bundle/original)
        stage=bundle/f'conv-{index:03}-0';stage.mkdir()
        stage_node=h.make_node('Conv',['x',f'w{i}',f'b{i}'],['y'])
        onnx.save(h.make_model(h.make_graph([stage_node],'stage',[info('x')],[info('y')],weights),
                   opset_imports=[h.make_opsetid('',17)],ir_version=10),stage/'reference.onnx')
        steps.append(dict(kind='ane',inputs=[src],outputs=[dst],original=original,
            stages=[stage.name],shape=shape,output_shape=shape,
            transform=dict(pads=[0]*4,stride=[1,1],crop=0,shuffle=1)))
        nodes.append(node);all_weights.extend(weights)
    onnx.save(h.make_model(h.make_graph(nodes,'full',[info('x')],[info('y')],all_weights),
               opset_imports=[h.make_opsetid('',17)],ir_version=10),root/'model.onnx')
    (root/'model.mil').write_text('synthetic test fixture')
    (root/'weights.bin').write_bytes(b'synthetic test fixture')
    metadata=dict(model=name,source_sha256='synthetic-test-only',input=shape,
        artifacts={n:sha(root/n) for n in ['model.onnx','model.mil','weights.bin']})
    (root/'model.json').write_text(json.dumps(metadata))
    plan=dict(schema=1,model=name,source_sha256=metadata['source_sha256'],
        onnx_sha256=sha(root/'model.onnx'),inputs=['x'],outputs=['y'],steps=steps,
        files={str(p.relative_to(bundle)):sha(p) for p in bundle.rglob('*') if p.is_file()})
    (bundle/'plan.json').write_text(json.dumps(plan))
    images=tmp_path/'images';images.mkdir()
    provenance=[]
    # Synthetic fixtures exercise validation plumbing, not COCO quality claims.
    for i in range(30):
        path=images/f'{i:012}.jpg'
        cv2.imwrite(str(path),np.full((32,16,3),i*8,np.uint8))
        provenance.append(dict(id=i,file_name=path.name,license=4,height=32,width=16,sha256=sha(path)))
    (images/'images.json').write_text(json.dumps(provenance))
    return root,bundle,images


def qualify(package,**kwargs):
    root,bundle,images=package
    return verifier.verify(root,bundle,images,True,Path('unused'),Path('unused'),**kwargs)


def test_one_bad_island_is_refused_all_portraits_complete_and_runtime_uses_original(package,monkeypatch):
    monkeypatch.setattr(C,'CDLL',lambda *a,**k:pytest.fail('CPU simulation loaded libane'))
    result=qualify(package,inject_failure='original-046.onnx')
    assert result['status']=='pass'
    assert result['qualified_islands']==['original-001.onnx','original-047.onnx']
    assert result['refused_islands']==['original-046.onnx']
    if package[0].name=='selfie':
        assert result['min_mask_iou']==1
    for key,data in result['islands'].items():
        assert len(data['samples'])==30
        assert all(s['status']==('fail' if key=='original-046.onnx' else 'pass') for s in data['samples'])
    assert all(s['max_abs']>13 for s in result['islands']['original-046.onnx']['samples'])
    root,bundle,_=package
    # Round trip the receipt; refusal must also skip candidate preparation.
    receipt=json.loads(json.dumps(result))
    from aurora_camera_effects import hybrid
    real_session=hybrid.session
    def checked_session(path):
        assert 'conv-046-0' not in str(path), 'refused island was prepared'
        return real_session(path)
    monkeypatch.setattr(hybrid,'session',checked_session)
    graph=HybridGraph(root,verified(root,root.name),'unused',simulate=True,package=bundle,receipt=receipt)
    candidate=graph.island
    calls=[]
    def counted(step,stages,value):
        assert step['original']!='original-046.onnx'
        calls.append(step['original'])
        return candidate(step,stages,value)
    graph.island=counted
    try:
        x=np.full(verified(root,root.name)['input'],.25,np.float32)
        actual=graph(x)
        expected=session(root/'model.onnx').run(None,{'x':x})
        assert verifier.output_metrics('selfie',actual,expected)['mask_iou']==1
        assert calls==['original-001.onnx','original-047.onnx']
        assert graph.engine_description=='ane-hybrid (2 of 3 islands on the ANE)'
    finally:
        graph.close()


@pytest.mark.parametrize('fault',['exception','nan'])
def test_candidate_execution_errors_do_not_abort_remaining_islands(package,monkeypatch,fault):
    original=HybridGraph.island
    def faulty(self,step,stages,value):
        if step['original']=='original-046.onnx':
            if fault=='exception':
                raise OSError('injected execution failure')
            return np.full(step['output_shape'],np.nan,np.float32)
        return original(self,step,stages,value)
    monkeypatch.setattr(HybridGraph,'island',faulty)
    result=qualify(package)
    assert result['status']=='pass'
    assert result['refused_islands']==['original-046.onnx']
    assert len(result['islands']['original-047.onnx']['samples'])==30
    assert all(s['error'] for s in result['islands']['original-046.onnx']['samples'])


@pytest.mark.parametrize('mutation',['simulation','stale','incomplete','bad_final','legacy','missing_island','failed_sample'])
def test_invalid_receipt_never_queries_device(package,monkeypatch,mutation):
    result=qualify(package,inject_failure='original-046.onnx')
    receipt=copy.deepcopy(result)
    receipt.update(mode='libane-device',abi=1,chip=13)
    if mutation=='simulation': receipt['mode']='cpu-simulation'
    elif mutation=='stale': receipt['plan_sha256']='stale'
    elif mutation=='incomplete': receipt['islands']['original-001.onnx']['samples'].pop()
    elif mutation=='bad_final':
        receipt['records'][0].update(mask_iou=.979,status='fail' if package[0].name=='yunet' else 'pass')
    elif mutation=='legacy': receipt.pop('schema')
    elif mutation=='missing_island': receipt['islands'].pop('original-047.onnx')
    elif mutation=='failed_sample': receipt['islands']['original-001.onnx']['samples'][0]['status']='fail'
    root,bundle,_=package
    (bundle/'qualification.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(C,'CDLL',lambda *a,**k:pytest.fail('invalid receipt queried a device'))
    with pytest.raises(OSError,match='qualification'):
        HybridGraph(root,verified(root,root.name),'unused',package=bundle)


def test_iou_gate_is_per_image_even_when_disagreement_and_mae_are_small():
    ref=np.zeros((1,1,100,100),np.float32);ref.flat[:100]=1
    actual=ref.copy();actual.flat[:3]=0
    assert np.mean(np.abs(actual-ref))<.005
    with pytest.raises(ValueError,match='IoU=0.970000'):
        verifier.output_metrics('selfie',[actual],[ref])


def test_final_gate_failure_retains_all_island_evidence_and_checks_all_images(package,monkeypatch):
    original=HybridGraph.__call__
    def bad_final(self,x,**kwargs):
        outputs=original(self,x,**kwargs)
        if not kwargs.get('compare_islands'):
            outputs[0].flat[:10000]=0
        return outputs
    monkeypatch.setattr(HybridGraph,'__call__',bad_final)
    result=qualify(package,inject_failure='original-046.onnx',max_output_fallbacks=0)
    assert result['status']=='fail'
    assert len(result['records'])==30
    assert all(r['status']=='fail' for r in result['records'])
    assert result['refused_islands']==['original-046.onnx']
    assert all(len(r['samples'])==30 for r in result['islands'].values())


def test_failed_requalification_retires_previous_success_before_execution(package,monkeypatch,tmp_path):
    root,bundle,images=package
    previous=bundle/'qualification.json'
    previous.write_text(json.dumps(dict(status='pass',old=True)))
    output=tmp_path/'failed.json'
    def fail(*a,**kw):
        assert json.loads(previous.read_text())['status']=='fail'
        raise ValueError('injected setup failure')
    monkeypatch.setattr(verifier,'verify',fail)
    monkeypatch.setattr(verifier.sys,'argv',['verify.py',str(root),'--package',str(bundle),
        '--images',str(images),'--qualify','--output',str(output)])
    assert verifier.main()==1
    archived=list(bundle.glob('qualification.previous-*.json'))
    assert len(archived)==1 and json.loads(archived[0].read_text())['old']
    assert json.loads(previous.read_text())==json.loads(output.read_text())
    assert 'injected setup failure' in json.loads(output.read_text())['error']


def test_preparation_error_refuses_only_that_island(package,monkeypatch):
    from aurora_camera_effects import hybrid
    original=hybrid.session
    def fail_stage(path):
        if Path(path).parent.name=='conv-046-0':
            raise OSError('injected preparation failure')
        return original(path)
    monkeypatch.setattr(hybrid,'session',fail_stage)
    result=qualify(package)
    assert result['status']=='pass'
    assert result['refused_islands']==['original-046.onnx']
    samples=result['islands']['original-046.onnx']['samples']
    assert len(samples)==30 and all('injected preparation failure' in s['error'] for s in samples)
    assert result['islands']['original-047.onnx']['status']=='qualified'


def test_all_refused_runs_original_cpu_operations(package,monkeypatch):
    def fail(*a,**kw):
        raise OSError('all candidates refused')
    monkeypatch.setattr(HybridGraph,'island',fail)
    result=qualify(package)
    assert result['status']=='pass'
    assert result['qualified_islands']==[]
    assert len(result['refused_islands'])==3
    assert len(result['records'])==30 and all(r['status']=='pass' for r in result['records'])


def test_zero_qualified_still_requires_h13_generation_before_runtime(package,monkeypatch):
    def fail(*a,**kw):
        raise OSError('all candidates refused')
    monkeypatch.setattr(HybridGraph,'island',fail)
    receipt=qualify(package)
    receipt.update(mode='libane-device',abi=1,chip=13)
    root,bundle,_=package
    (bundle/'qualification.json').write_text(json.dumps(receipt))
    queries=[]
    class Query:
        def __call__(self,device,abi,chip):
            queries.append(device)
            C.cast(abi,C.POINTER(C.c_uint32))[0]=1
            C.cast(chip,C.POINTER(C.c_uint32))[0]=14
            return 0
    class Library:
        pyane_query_caps=Query()
    monkeypatch.setattr(C,'CDLL',lambda *a,**kw:Library())
    with pytest.raises(OSError,match='generation H13 and ABI 1'):
        HybridGraph(root,verified(root,root.name),'unused',package=bundle)
    assert queries==[0]



def test_numeric_final_gate_can_conservatively_refuse_another_island(package,monkeypatch):
    original=HybridGraph.__call__
    def coupled(self,x,**kwargs):
        outputs=original(self,x,**kwargs)
        if not kwargs.get('compare_islands') and 'original-047.onnx' in self.qualified_islands:
            outputs[0].flat[:10000]=0
        return outputs
    monkeypatch.setattr(HybridGraph,'__call__',coupled)
    result=qualify(package,inject_failure='original-046.onnx')
    assert result['status']=='pass'
    assert result['qualified_islands']==['original-001.onnx']
    assert result['refused_islands']==['original-046.onnx','original-047.onnx']
    assert [a['status'] for a in result['output_gate_attempts']]==['fail','pass']
    assert all(len(a['records'])==30 for a in result['output_gate_attempts'])
    retired=result['islands']['original-047.onnx']
    assert all(s['status']=='pass' for s in retired['samples'])
    assert 'not an isolated island parity failure' in retired['refusal_reason']
    assert retired['output_gate_errors']
    root,bundle,_=package
    graph=HybridGraph(root,verified(root,root.name),'unused',simulate=True,package=bundle,receipt=result)
    assert graph.engine_description=='ane-hybrid (1 of 3 islands on the ANE)'
    graph.close()


def test_numeric_final_gate_fallback_budget_never_bypasses_gate(package,monkeypatch):
    original=HybridGraph.__call__
    def coupled(self,x,**kwargs):
        outputs=original(self,x,**kwargs)
        if not kwargs.get('compare_islands'):
            outputs[0].flat[:10000]=0
        return outputs
    monkeypatch.setattr(HybridGraph,'__call__',coupled)
    result=qualify(package,inject_failure='original-046.onnx',max_output_fallbacks=1)
    assert result['status']=='fail'
    assert len(result['output_gate_attempts'])==2
    assert all(a['status']=='fail' for a in result['output_gate_attempts'])
    assert result['qualified_islands']==['original-001.onnx']
    assert all(len(r['samples'])==30 for r in result['islands'].values())
