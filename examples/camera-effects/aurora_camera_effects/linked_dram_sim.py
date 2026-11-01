# SPDX-License-Identifier: MIT
"""Simulate linked DRAM/L2 edges from decoded ANEC addresses/strides.

Convolution and resize arithmetic use the package MIL/weights. Physical input,
scratch, resident L2 and output movement uses the emitted task descriptors. This checks
linking and zero-copy placement; it is not an ANE microarchitecture emulator.
"""
import importlib
from pathlib import Path
import struct

import numpy as np

from .linked_sim import MILSimulation


def view(buffer, shape, *, row, plane, offset=0):
    n,c,h,w=shape
    end=offset+(c-1)*plane+(h-1)*row+w*2
    if n!=1 or offset<0 or row<w*2 or plane<row*h or end>len(buffer):
        raise ValueError('decoded DMA tensor exceeds its buffer')
    return np.ndarray(tuple(shape),dtype='<f2',buffer=buffer,offset=offset,
                      strides=(c*plane,plane,row,2))


L2_BYTES=0x200000


def resident(memory, shape, regs, *, source, value=None):
    """Logical NCHW <-> decoded circular L2 C/R strides (r2/r3)."""
    base,channel,row=(regs[a] for a in ((0x4808,0x480c,0x4810) if source else
                                      (0x4834,0x4838,0x483c)))
    n,c,h,w=shape
    if n!=1 or base%2 or channel<w*2 or row<c*channel or row%2 or channel%2:
        raise ValueError('invalid resident L2 geometry')
    addresses=(base+np.arange(c)[:,None,None]*channel+
               np.arange(h)[None,:,None]*row+np.arange(w)[None,None,:]*2)%L2_BYTES
    if source:
        return memory[addresses//2].view(np.float16)[None].astype(np.float32)
    memory[addresses//2]=np.asarray(value,dtype=np.float16)[0].view(np.uint16)


def ring_disjoint(base_a,size_a,base_b,size_b):
    if min(size_a,size_b)<0 or max(size_a,size_b)>L2_BYTES: return False
    def pieces(base,size):
        base%=L2_BYTES;end=base+size
        return [(base,min(end,L2_BYTES))]+([(0,end-L2_BYTES)] if end>L2_BYTES else [])
    return all(max(a,b)>=min(c,d) for a,c in pieces(base_a,size_a) for b,d in pieces(base_b,size_b))


class DRAMSimulation(MILSimulation):
    def __init__(self, directory, manifest, *, reader=None):
        directory=Path(directory)
        super().__init__(directory/'model.mil',directory)
        reader=reader or importlib.import_module('inspect_anec')
        self.program=manifest['programs'][0]
        self.regions=manifest['scratch']['regions']
        data=(directory/self.program['file']).read_bytes()
        header=reader.HEADER.unpack_from(data)
        tasks=reader.split_h13_tasks(data[4096:4096+header[3]],header[1]//4-1,header[2])
        self.tasks=[(struct.unpack_from('<10I',t),reader.h13_task_registers(t)) for t in tasks]
        self.nodes={op.result:op for op in self.operations}
        parameters,_,_=self.ref.Parser((directory/'model.mil').read_text()).parse()
        self.shapes={name:tuple(kind.shape) for kind,name in parameters}
        self.shapes.update({op.result:op.result_type.shape for op in self.operations})
        # Poison catches unwritten logical values. Padding is never arithmetic.
        self.scratch=bytearray(b'\x00\x7e')*(self.program['scratchBytes']//2)
        self.output=bytearray(b'\x00\x7e')*(self.program['outputs'][0]['allocationBytes']//2)
        self.l2=np.full(L2_BYTES//2,0x7e00,np.uint16)
        self.frame_count=0

    def stored(self, result):
        if result==self.program['outputs'][0]['name']:
            binding=self.program['outputs'][0]
            return view(self.output,binding['shape'],row=binding['nchw'][5],plane=binding['nchw'][4])
        region=self.regions[result]
        return view(self.scratch,self.shapes[result],row=region['nchw'][5],
                    plane=region['nchw'][4],offset=region['offset'])

    def __call__(self, x, outputs):
        if list(outputs)!=self.returns or len(outputs)!=1:
            raise ValueError('DRAM simulator requires one package result')
        binding=self.program['inputs'][0]
        data=bytearray(binding['allocationBytes'])
        view(data,binding['shape'],row=binding['nchw'][5],plane=binding['nchw'][4])[:]=x
        buffers={3:self.scratch,4:self.output,5:data}
        env=dict(self.constants)
        env[self.input]=np.asarray(x,np.float32)
        for layer in self.program['layers']:
            result=layer['result'];node=self.nodes[result]
            if layer['taskDescriptors']==0:
                if node.name!='concat' or not layer.get('zeroCopy'):
                    raise ValueError('unsupported task-free linked operation')
                operands=self.resolve(node.arguments['values'],env)
                expected=np.concatenate(operands,axis=int(self.resolve(node.arguments['axis'],env)))
                actual=self.stored(result).astype(np.float32)
                np.testing.assert_array_equal(actual,expected,
                    err_msg='decoded producer placement differs from concat semantics')
                env[result]=actual
                continue
            relu=layer['operation']=='conv+relu'
            if relu:
                if node.name!='relu': raise ValueError('fused relu result changed')
                node=self.nodes[node.arguments['x'].name]
            source=node.arguments['x'].name
            shape=self.shapes[source]
            selected=self.tasks[layer['firstTask']:layer['firstTask']+layer['taskDescriptors']]
            reads=[(header,regs) for header,regs in selected if
                   regs.get(0x13800,0)&1 and header[8]&31 in (3,5)]
            if layer['inputEdge']!='dram':
                header,regs=selected[0]
                if reads or regs[0x4804] not in (0x160,0x144,0x104,0x120):
                    raise ValueError('unqualified resident L2 source form')
                operand=resident(self.l2,shape,regs,source=True)
                np.testing.assert_array_equal(operand,env[source],
                    err_msg='decoded L2 transport differs from the producer value')
                for _,later in selected[1:]:
                    if later[0x4804]!=0x100 or later[0x13800]!=0x8880:
                        raise ValueError('unqualified split resident consumer')
                    np.testing.assert_array_equal(resident(self.l2,shape,later,source=True),operand)
                # Read the full live span; a row-band TileHeight cannot be
                # substituted for the inter-layer allocation.
                live=shape[2]*regs[0x4810]
                result_bytes=(layer['l2ResultBytes'] if layer['outputEdge']!='dram' else
                    regs[0x28]*regs[0x10]*max(16,(2*self.shapes[result][3]+15)//16*16))
                if not ring_disjoint(regs[0x4808],live,regs[0x4834],result_bytes):
                    raise ValueError('resident input overlaps result or output staging')
            else:
                if not reads: raise ValueError('DRAM layer has no active input DMA')
                header,regs=reads[0];channel=header[8]&31
                operand=view(buffers[channel],shape,row=regs[0x1380c],plane=regs[0x13810],
                             offset=regs[0x13808]).astype(np.float32)
            env[source]=operand
            args={name:self.resolve(value,env) for name,value in node.arguments.items()}
            if node.name=='conv':
                value=self.conv(node,args)
                if relu: value=np.maximum(value,0)
            elif node.name in ('upsample_nearest_neighbor','resize_nearest_neighbor'):
                value=self.layers.nearest2x(operand)
            else:
                raise ValueError('unsupported decoded DRAM primitive: '+node.name)
            value=self.layers.f16_ties_away(np.asarray(value,np.float64)).astype(np.float32)
            if tuple(value.shape)!=self.shapes[result] or not np.isfinite(value).all():
                raise ValueError('invalid decoded DRAM layer result')
            if layer['outputEdge']!='dram':
                first=0
                for header,regs in selected:
                    count=regs[0x10]
                    if regs[0x17800]&1 or count<1 or first+count>value.shape[1]:
                        raise ValueError('invalid resident producer slice')
                    resident(self.l2,(1,count,*value.shape[2:]),regs,source=False,
                             value=value[:,first:first+count])
                    first+=count
                if first!=value.shape[1]: raise ValueError('resident channels left unwritten')
                env[result]=value
                continue
            plane=((value.shape[3]*2+63)//64*64)*value.shape[2]
            covered=np.zeros(value.shape[1],dtype=bool)
            for header,regs in selected:
                if not regs.get(0x17800,0)&1: continue
                channel=(header[8]>>12)&31
                if channel not in (3,4): raise ValueError('unrouted linked output')
                offset=regs[0x17804]
                local=offset-layer['writeOffset']
                if local<0 or local%plane:
                    raise ValueError('decoded output slice is not channel aligned')
                first=local//plane;count=regs[0x10]
                if count<1 or first+count>value.shape[1] or covered[first:first+count].any():
                    raise ValueError('decoded output slices overlap or exceed their tensor')
                if node.name=='conv' and regs[0x14]!=((value.shape[2]<<16)|value.shape[3]):
                    raise ValueError('decoded convolution output geometry differs from MIL')
                target=view(buffers[channel],(1,count,*value.shape[2:]),row=regs[0x17808],
                            plane=regs[0x1780c],offset=offset)
                target[:]=value[:,first:first+count]
                covered[first:first+count]=True
            if not covered.all(): raise ValueError('decoded tasks leave output channels unwritten')
            env[result]=value
        self.frame_count+=1
        return [self.stored(outputs[0]).astype(np.float32)]
