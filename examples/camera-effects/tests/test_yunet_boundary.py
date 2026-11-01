# SPDX-License-Identifier: MIT
import copy

import numpy as np
import pytest

from aurora_camera_effects import yunet_boundary as boundary


def contract():
    names=[kind+'_'+str(stride) for kind in ('cls','obj','bbox','kps') for stride in (8,16,32)]
    shapes=[];items=[];offset=0
    for i,name in enumerate(names):
        spatial=(40,20,10)[i%3];channels=1 if i<6 else 4 if i<9 else 10
        shape=[1,spatial*spatial,channels];shapes.append(shape)
        items.append(dict(name=name,channel_offset=offset,channels=channels,spatial=spatial,
                          sample_stride=40//spatial,sigmoid=i<6,shape=shape))
        offset+=channels
    metadata=dict(model='yunet',output_names=names,output_shapes=shapes)
    return dict(input_transform='yunet-space-to-depth-2',program_input_shape=[1,16,160,160],
                program_output_shapes=[[1,64,40,40]],program_outputs=['packed'],readback=items),metadata


def test_image_phase_layout_and_zero_channels_are_exact():
    x=np.arange(3*320*320,dtype=np.float32).reshape(1,3,320,320)%256
    actual=boundary.prepare(x.astype(np.float16))
    for c in range(3):
        for dy in range(2):
            for dx in range(2):
                np.testing.assert_array_equal(actual[:,c*4+dy*2+dx],x[:,c,dy::2,dx::2])
    assert np.all(actual[:,12:]==0)


def test_compact_head_readback_and_score_decode():
    plan,metadata=contract();boundary.validate(plan,metadata)
    packed=np.arange(64*40*40,dtype=np.float32).reshape(1,64,40,40)/1000-2
    actual=boundary.decode(packed,plan)
    for value,item in zip(actual,plan['readback']):
        c=item['channel_offset'];step=item['sample_stride']
        expected=packed[:,c:c+item['channels'],::step,::step].transpose(0,2,3,1).reshape(item['shape'])
        if item['sigmoid']: expected=1/(1+np.exp(-expected))
        np.testing.assert_allclose(value,expected,atol=1e-7,rtol=1e-6)


@pytest.mark.parametrize('field,value',[('channel_offset',64),('sample_stride',3),('sigmoid',False)])
def test_changed_head_contract_is_rejected(field,value):
    plan,metadata=contract();plan=copy.deepcopy(plan)
    plan['readback'][0][field]=value
    with pytest.raises(ValueError): boundary.validate(plan,metadata)
