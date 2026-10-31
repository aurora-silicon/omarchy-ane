#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build H13 convolution islands; retain exact ONNX host operations between them.

No device calls. All weights and generated programs remain install-time data.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aurora_camera_effects.backend import verified

COMPILER_REVISION = '72e1d99c6f51eaf97ad61280936d42dbb0b6a1ae'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compiler_revision(compiler):
    try:
        return subprocess.check_output(['git','-C',str(compiler.parent.parent),'rev-parse','HEAD'],
                                       stderr=subprocess.DEVNULL,text=True,timeout=5).strip()
    except (OSError,subprocess.SubprocessError):
        return None


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])


def blob(path, arrays):
    data = bytearray(struct.pack('<II', 1, 2) + bytes(56))
    offsets = []
    for array in arrays:
        payload = np.ascontiguousarray(array, dtype='<f2').tobytes()
        meta = len(data)
        offsets.append(meta)
        data += struct.pack('<IIQQ', 0xDEADBEEF, 1, len(payload), meta + 64) + bytes(40)
        data += payload
        data += bytes(-len(data) % 64)
    path.write_bytes(data)
    return offsets


def write_conv(root, shape, weight, bias, groups):
    """Formula-built stride-1 SAME conv; other border/stride work is explicit."""
    root.mkdir(parents=True)
    co, ci, k, _ = weight.shape
    offsets = blob(root/'weights.bin', [weight, bias])
    _, channels, height, width = shape
    p = k//2
    text = f'''program(1.3)
[buildInfo = dict<string, string>({{}})]
{{
  func main<ios18>(tensor<fp16, [1, {channels}, {height}, {width}]> x) {{
    string pt = const()[name = string("pt"), val = string("{'custom' if p else 'valid'}")];
    tensor<int32, [2]> st = const()[name = string("st"), val = tensor<int32, [2]>([1, 1])];
    tensor<int32, [4]> pd = const()[name = string("pd"), val = tensor<int32, [4]>([{p}, {p}, {p}, {p}])];
    tensor<int32, [2]> dl = const()[name = string("dl"), val = tensor<int32, [2]>([1, 1])];
    int32 gp = const()[name = string("gp"), val = int32({groups})];
    tensor<fp16, [{co}, {ci}, {k}, {k}]> w = const()[name = string("w"), val = tensor<fp16, [{co}, {ci}, {k}, {k}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[0]})))];
    tensor<fp16, [{co}]> b = const()[name = string("b"), val = tensor<fp16, [{co}]>(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offsets[1]})))];
    tensor<fp16, [1, {co}, {height}, {width}]> y = conv(x = x, weight = w, bias = b, strides = st, pad_type = pt, pad = pd, dilations = dl, groups = gp)[name = string("conv")];
  }} -> (y);
}}
'''
    (root/'model.mil').write_text(text)
    np.savez(root/'reference.npz', weight=weight, bias=bias, groups=groups, shape=shape)
    graph = h.make_graph([h.make_node('Conv', ['x','w','b'], ['y'], group=groups, pads=[p]*4)],
                         'padded-convolution', [h.make_tensor_value_info('x', 1, shape)],
                         [h.make_tensor_value_info('y', 1, [1,co,height,width])],
                         [nh.from_array(weight.astype(np.float32),'w'), nh.from_array(bias.astype(np.float32),'b')])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid('',17)], ir_version=10), root/'reference.onnx')


def padded(weight, bias, cin, groups):
    co, ci, k, _ = weight.shape
    # Uniform lane chunks also avoid the compiler's mixed 256+128 split at
    # 384 channels (the 5x5 gather for a 96-channel source).
    pin, pout = max(64,1<<(cin-1).bit_length()), max(64,1<<(co-1).bit_length())
    if groups == cin == co:
        result = np.zeros((pout,1,k,k),np.float16)
        result[:co] = weight
        pg = pin
    elif groups == 1:
        result = np.zeros((pout,pin,k,k),np.float16)
        result[:co,:ci] = weight
        pg = 1
    else:
        raise ValueError('only dense and depthwise convolution are supported')
    b = np.zeros(pout,np.float16)
    b[:co] = bias
    return result, b, pin, pg


def split5(weight, cin, groups):
    """Exact 5x5 = 3x3 gather + 3x3 mix, with four intermediates per input.

    Unlike a low-rank factorisation, this represents every possible 5x5 kernel.
    The gather only copies fp16 values, so adds no arithmetic rounding.
    """
    co, ci, _, _ = weight.shape
    gather = np.zeros((4*cin,cin,3,3),np.float16)
    for block,(dy,dx) in enumerate(((0,0),(0,2),(2,0),(2,2))):
        for c in range(cin):
            gather[block*cin+c,c,dy,dx] = 1
    mix = np.zeros((co,4*cin,3,3),np.float16)
    for out in range(co):
        for src in range(ci):
            original = out//(co//groups)*ci+src
            for y in range(5):
                for x in range(5):
                    by,bx = int(y>2),int(x>2)
                    mix[out,(by*2+bx)*cin+original,y-2*by,x-2*bx] = weight[out,src,y,x]
    return gather, mix


def layers_for(node, arrays, shape, output_shape=None):
    a = {v.name:h.get_attribute_value(v) for v in node.attribute}
    if node.op_type=='Resize':
        if output_shape!=[shape[0],shape[1],shape[2]*2,shape[3]*2]:
            raise ValueError('resize rewrite requires exactly 2x spatial scale')
        cin=shape[1]
        mode=a.get('mode',b'nearest')
        if mode==b'nearest' and a.get('coordinate_transformation_mode')==b'asymmetric' and a.get('nearest_mode')==b'floor':
            weight=np.zeros((4*cin,cin,1,1),np.float16)
            for c in range(cin):weight[c*4:c*4+4,c,0,0]=1
            return [(weight,np.zeros(4*cin),cin,1)],dict(pads=[0]*4,stride=[1,1],crop=0,shuffle=2)
        if mode==b'linear' and a.get('coordinate_transformation_mode')==b'half_pixel':
            # Four 3x3 phase filters implement align_corners=False bilinear
            # interpolation. Replicated host border matches ONNX edge clamping.
            weight=np.zeros((4*cin,cin,3,3),np.float16)
            for c in range(cin):
                for dy in range(2):
                    for dx in range(2):
                        ys=((0,.25),(1,.75)) if dy==0 else ((1,.75),(2,.25))
                        xs=((0,.25),(1,.75)) if dx==0 else ((1,.75),(2,.25))
                        for y,wy in ys:
                            for x,wx in xs:weight[c*4+dy*2+dx,c,y,x]=wy*wx
            return [(weight,np.zeros(4*cin),cin,1)],dict(pads=[1]*4,pad_mode='edge',stride=[1,1],crop=1,shuffle=2)
        raise ValueError('resize coordinate/mode requires a separate rewrite')
    weight = arrays[node.input[1]].astype(np.float16)
    cin = shape[1]
    groups = int(a.get('group',1))
    if a.get('dilations',[1,1]) != [1,1] or a.get('auto_pad',b'NOTSET') != b'NOTSET':
        raise ValueError('dilation/automatic padding requires a separate rewrite')
    if node.op_type == 'ConvTranspose':
        if (weight.shape[2:] != (2,2) or a.get('strides',[1,1]) != [2,2] or
            any(a.get('pads',[0]*4)) or any(a.get('output_padding',[0,0])) or groups != 1):
            raise ValueError('transpose convolution outside kernel-2 stride-2 form')
        co = weight.shape[1]
        bias = arrays[node.input[2]] if len(node.input)>2 else np.zeros(co)
        w = weight.transpose(1,2,3,0).reshape(co*4,cin,1,1)
        return [(w,np.repeat(bias,4),cin,1)], dict(pads=[0]*4, stride=[1,1], crop=0, shuffle=2)
    co = weight.shape[0]
    bias = arrays[node.input[2]] if len(node.input)>2 else np.zeros(co)
    k = weight.shape[2]
    if weight.shape[2:] not in ((1,1),(3,3),(5,5)):
        raise ValueError('kernel outside 1/3/5 rewrite')
    transform = dict(pads=list(a.get('pads',[0]*4)), stride=list(a.get('strides',[1,1])), crop=k//2, shuffle=1)
    if k == 5:
        gather,mix = split5(weight,cin,groups)
        return [(gather,np.zeros(4*cin),cin,1),(mix,bias,4*cin,1)],transform
    return [(weight,bias,cin,groups)],transform


def build(root, compiler, output):
    metadata = verified(root, root.name)
    output.mkdir(parents=True,exist_ok=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    options.optimized_model_filepath = str(output/'folded.onnx')
    ort.InferenceSession(str(root/'model.onnx'), options, providers=['CPUExecutionProvider'])
    model = onnx.shape_inference.infer_shapes(onnx.load(output/'folded.onnx'),data_prop=True)
    infos = {v.name:v for v in [*model.graph.input,*model.graph.value_info,*model.graph.output]}
    shapes = {n:[d.dim_value for d in v.type.tensor_type.shape.dim] for n,v in infos.items()}
    arrays = {v.name:nh.to_array(v) for v in model.graph.initializer}
    steps, records, pending = [], [], []
    all_nodes = list(model.graph.node)

    def flush():
        if not pending:
            return
        produced = {s for n in pending for s in n.output}
        needed = {s for n in all_nodes if n not in pending for s in n.input} | {s.name for s in model.graph.output}
        inputs = sorted({s for n in pending for s in n.input if s and s not in produced and s not in arrays})
        outputs = sorted(produced & needed)
        initial = [v for v in model.graph.initializer if any(v.name in n.input for n in pending)]
        graph = h.make_graph(list(pending),'host', [infos[n] for n in inputs], [infos[n] for n in outputs], initial)
        sub = h.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version)
        file = f'host-{len(steps):03}.onnx'
        onnx.checker.check_model(sub)
        onnx.save(sub,output/file)
        steps.append(dict(kind='cpu',file=file,inputs=inputs,outputs=outputs))
        pending.clear()

    for index,node in enumerate(all_nodes):
        if node.op_type not in ('Conv','ConvTranspose','Resize'):
            pending.append(node)
            continue
        record = dict(name=node.name, op=node.op_type, input=shapes.get(node.input[0]), output=shapes.get(node.output[0]), stages=[])
        try:
            shape = shapes[node.input[0]]
            if len(shape)!=4 or any(d<=0 for d in shape) or shape[0]!=1:
                raise ValueError('nonstatic convolution input')
            layers,transform = layers_for(node,arrays,shape,shapes[node.output[0]])
            pt,pl,pb,pr = transform['pads']
            height,width = shape[2]+pt+pb,shape[3]+pl+pr
            for j,(w,b,cin,g) in enumerate(layers):
                w,b,pin,pg = padded(w,b,cin,g)
                dest = output/f'conv-{index:03}-{j}'
                write_conv(dest,[1,pin,height,width],w,b,pg)
                command = [str(compiler),'--target','H13','--mil',str(dest/'model.mil'),
                           '--model-root',str(dest),'--output',str(dest/'bundle')]
                env = dict(os.environ,MILHWXC_H13_CONV_GENERIC='1')
                run = subprocess.run(command,capture_output=True,text=True,timeout=120,env=env)
                (dest/'compile.log').write_text(run.stdout+run.stderr)
                record['stages'].append(dict(directory=dest.name, rc=run.returncode, diagnostic=run.stderr[-3000:]))
                if run.returncode:
                    raise ValueError(run.stderr[-3000:])
                manifest = json.loads((dest/'bundle/manifest.json').read_text())
                if len(manifest['programs'])!=1 or manifest['programs'][0]['encoder']!='h13-generic-conv':
                    raise ValueError('requires one formula-built convolution program per stage')
            flush()
            original = f'original-{index:03}.onnx'
            graph = h.make_graph([node], 'original-island', [infos[node.input[0]]],
                                 [infos[node.output[0]]],
                                 [v for v in model.graph.initializer if v.name in node.input])
            onnx.save(h.make_model(graph,opset_imports=model.opset_import,ir_version=model.ir_version),output/original)
            steps.append(dict(kind='ane',inputs=[node.input[0]],outputs=list(node.output),
                              shape=shape,output_shape=shapes[node.output[0]], transform=transform,
                              original=original,
                              stages=[s['directory'] for s in record['stages']]))
            record['status']='compiled'
        except (ValueError,KeyError,subprocess.TimeoutExpired) as error:
            record.update(status='cpu',reason=str(error))
            pending.append(node)
        records.append(record)
        print(root.name,node.name,record['status'],flush=True)
    flush()
    plan = dict(schema=1,model=root.name,source_sha256=metadata['source_sha256'],
                onnx_sha256=sha(root/'model.onnx'),compiler_revision=compiler_revision(compiler),
                expected_compiler_base=COMPILER_REVISION,
                compiler_sha256=sha(compiler),build_host=platform.system()+'/'+platform.machine()+' build host',platform=platform.platform(),
                inputs=[v.name for v in model.graph.input],outputs=[v.name for v in model.graph.output],
                steps=steps,convolutions=records,qualified_targets=[])
    plan['files'] = {str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    return plan


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path)
    p.add_argument('compiler',type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    build(a.model.resolve(),a.compiler.resolve(),a.output.resolve())
