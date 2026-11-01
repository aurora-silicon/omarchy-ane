# SPDX-License-Identifier: MIT
"""Numerical simulation of package MIL, with fp16 surface boundaries.

This evaluates the package's MIL and weights, never the reference ONNX graph.
It does not emulate task descriptor fields or certify device arithmetic.
"""
import importlib
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as h, numpy_helper as nh
import onnxruntime as ort


class MILSimulation:
    def __init__(self, mil, model_root):
        self.ref = importlib.import_module('h13_reference')
        self.layers = importlib.import_module('layer_reference')
        self.root = Path(model_root)
        parameters, self.operations, self.returns = self.ref.Parser(Path(mil).read_text()).parse()
        if len(parameters) != 1:
            raise ValueError('camera simulator requires one input')
        self.input = parameters[0][1]
        self.constants = {}
        self.convolutions = {}
        for op in self.operations:
            if op.name == 'const':
                self.constants[op.result] = self.ref._execute(op, self.constants, self.root)

    def resolve(self, value, env):
        value = self.ref._resolve(value, env, self.root)
        if isinstance(value, self.ref.Tensor):
            return np.asarray(value.values).reshape(value.shape)
        if isinstance(value, list):
            return [self.resolve(v, env) for v in value]
        return value

    def conv(self, op, a):
        if op.result not in self.convolutions:
            x, w = a['x'], np.asarray(a['weight'], np.float32)
            inputs = [h.make_tensor_value_info('x', 1, list(x.shape))]
            weights = [nh.from_array(w, 'w')]
            names = ['x', 'w']
            if 'bias' in a:
                names.append('b')
                weights.append(nh.from_array(np.asarray(a['bias'], np.float32), 'b'))
            attrs = dict(strides=list(map(int, a['strides'])),
                         dilations=list(map(int, a['dilations'])), group=int(a['groups']))
            if a['pad_type'] == 'same':
                attrs['auto_pad'] = 'SAME_UPPER'
            elif a['pad_type'] == 'custom':
                pt, pb, pl, pr = map(int, a['pad'])
                attrs['pads'] = [pt, pl, pb, pr]
            elif a['pad_type'] != 'valid':
                raise ValueError('unsupported simulator padding')
            graph = h.make_graph([h.make_node('Conv', names, ['y'], **attrs)], 'linked-layer',
                inputs, [h.make_tensor_value_info('y', 1, list(op.result_type.shape))], weights)
            model = h.make_model(graph, opset_imports=[h.make_opsetid('', 17)], ir_version=10)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            self.convolutions[op.result] = ort.InferenceSession(model.SerializeToString(), options,
                providers=['CPUExecutionProvider'])
        return self.convolutions[op.result].run(None, {'x':np.asarray(a['x'], np.float32)})[0]

    def __call__(self, x, outputs):
        if list(outputs) != self.returns:
            raise ValueError('package simulation output names differ from MIL')
        env = dict(self.constants)
        env[self.input] = np.asarray(x, np.float32)
        for op in self.operations:
            if op.name == 'const':
                continue
            a = {name:self.resolve(value, env) for name, value in op.arguments.items()}
            name = op.name
            if name == 'conv':
                y = self.conv(op, a)
            elif name in ('add', 'sub', 'mul', 'real_div', 'maximum', 'minimum'):
                fn = dict(add=np.add, sub=np.subtract, mul=np.multiply, real_div=np.divide,
                          maximum=np.maximum, minimum=np.minimum)[name]
                y = fn(a['x'], a['y'])
            elif name in ('relu', 'relu6', 'clip'):
                y = np.clip(a['x'], a.get('alpha', 0), a.get('beta', 6 if name == 'relu6' else np.inf))
            elif name == 'sigmoid':
                y = self.layers.sigmoid(a['x'])
            elif name == 'concat':
                if a.get('interleave', False):
                    raise ValueError('interleaved concat is outside simulation')
                y = np.concatenate(a['values'], axis=int(a['axis']))
            elif name in ('upsample_nearest_neighbor', 'resize_nearest_neighbor',
                          'upsample_bilinear', 'resize_bilinear'):
                target = op.result_type.shape[-2:]
                if tuple(target) != tuple(v*2 for v in a['x'].shape[-2:]):
                    raise ValueError('simulator requires native 2x resize')
                if 'bilinear' in name:
                    if a.get('align_corners', False) or a.get('sampling_mode', 'UNALIGN_CORNERS') != 'UNALIGN_CORNERS':
                        raise ValueError('simulator requires half-pixel bilinear')
                    y = self.layers.bilinear_half2x(a['x'])
                else:
                    y = self.layers.nearest2x(a['x'])
            elif name in ('reshape', 'squeeze', 'expand_dims'):
                y = np.asarray(a['x']).reshape(op.result_type.shape)
            elif name == 'transpose':
                y = np.transpose(a['x'], list(map(int, a['perm'])))
            else:
                raise ValueError(f'unsupported linked simulation operation: {name}')
            if tuple(np.shape(y)) != op.result_type.shape:
                raise ValueError(f'simulator shape mismatch: {op.result}')
            # ANE surfaces round midpoint ties away from zero. Native internal
            # LUT/accumulator arithmetic remains a device qualification concern.
            env[op.result] = self.layers.f16_ties_away(np.asarray(y, np.float64)).astype(np.float32)
        return [env[name] for name in outputs]
