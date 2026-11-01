# SPDX-License-Identifier: MIT
"""Static YuNet image-layout and score/head decode around one H13 program."""
import numpy as np


SCHEMA = 'aurora.camera-linked-lowered.v1'


def validate(plan, metadata):
    if (metadata['model']!='yunet' or plan.get('input_transform')!='yunet-space-to-depth-2' or
        plan.get('program_input_shape')!=[1,16,160,160] or
        plan.get('program_output_shapes')!=[[1,64,40,40]] or
        len(plan.get('program_outputs',[]))!=1 or len(plan.get('readback',[]))!=12):
        raise ValueError('unsupported YuNet linked boundary')
    offset=0
    for i,(item,name,shape) in enumerate(zip(plan['readback'],metadata['output_names'],metadata['output_shapes'])):
        spatial=(40,20,10)[i%3]
        channels=1 if i<6 else 4 if i<9 else 10
        expected=dict(name=name,channel_offset=offset,channels=channels,spatial=spatial,
                      sample_stride=40//spatial,sigmoid=i<6,shape=shape)
        if item!=expected or shape!=[1,spatial*spatial,channels]:
            raise ValueError('YuNet head decode contract changed')
        offset+=channels


def prepare(x):
    if x.shape!=(1,3,320,320):
        raise ValueError('YuNet layout requires NCHW 320 BGR')
    phase=x.reshape(1,3,160,2,160,2).transpose(0,1,3,5,2,4).reshape(1,12,160,160)
    result=np.zeros((1,16,160,160),np.float16)
    result[:,:12]=phase
    return result


def decode(packed, plan):
    if packed.shape!=(1,64,40,40):
        raise ValueError('invalid packed YuNet output')
    result=[]
    for item in plan['readback']:
        c=item['channel_offset'];step=item['sample_stride']
        value=packed[:,c:c+item['channels'],::step,::step].transpose(0,2,3,1)
        value=np.ascontiguousarray(value,dtype=np.float32).reshape(item['shape'])
        if item['sigmoid']:
            # Score activation belongs to decode; no intermediate graph value
            # is returned to the program or evaluated on the host.
            positive=value>=0
            out=np.empty_like(value)
            out[positive]=1/(1+np.exp(-value[positive]))
            exp=np.exp(value[~positive]);out[~positive]=exp/(1+exp)
            value=out
        result.append(value)
    return result
