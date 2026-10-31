# SPDX-License-Identifier: MIT
"""Static, deliberately narrow importer for the audited MediaPipe flatbuffer.

Weights are read from the public fp16 TFLite buffers, never downloaded from a
conversion service. Unrecognized options/operators fail instead of guessing.
All activations use NCHW in the exported graph.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import tflite


class Selfie(nn.Module):
    def __init__(self, path):
        super().__init__()
        data = path.read_bytes()
        self.model = tflite.Model.GetRootAsModel(data, 0)
        g = self.model.Subgraphs(0)
        self.graph = g
        self.input = int(g.Inputs(0))
        self.output = int(g.Outputs(0))
        self.constants = {}
        names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v, int)}
        self.ops = []
        for i in range(g.TensorsLength()):
            t = g.Tensors(i)
            b = self.model.Buffers(t.Buffer())
            if not b.DataLength():
                continue
            dtype = {tflite.TensorType.FLOAT16: '<f2', tflite.TensorType.INT32: '<i4',
                     tflite.TensorType.FLOAT32: '<f4'}[t.Type()]
            value = np.frombuffer(b.DataAsNumpy().tobytes(), dtype=dtype).reshape(t.ShapeAsNumpy())
            name = f'tensor_{i}'
            self.register_buffer(name, torch.from_numpy(value.astype(np.float32).copy()))
            self.constants[i] = name
        for i in range(g.OperatorsLength()):
            o = g.Operators(i)
            code = self.model.OperatorCodes(o.OpcodeIndex())
            kind = names[code.BuiltinCode()]
            inputs = list(map(int, o.InputsAsNumpy()))
            output = int(o.Outputs(0))
            options = {}
            if kind in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
                cls = tflite.Conv2DOptions if kind == 'CONV_2D' else tflite.DepthwiseConv2DOptions
                opt = self.options(o, cls)
                if opt.DilationHFactor() != 1 or opt.DilationWFactor() != 1:
                    raise ValueError('dilated source convolution')
                stride = (opt.StrideH(), opt.StrideW())
                ih, iw = map(int, g.Tensors(inputs[0]).ShapeAsNumpy()[1:3])
                shape = g.Tensors(inputs[1]).ShapeAsNumpy()
                kh, kw = map(int, shape[1:3])
                pad = [0, 0, 0, 0]
                if opt.Padding() == tflite.Padding.SAME:
                    ph = max(0, ((ih + stride[0] - 1)//stride[0] - 1)*stride[0] + kh - ih)
                    pw = max(0, ((iw + stride[1] - 1)//stride[1] - 1)*stride[1] + kw - iw)
                    pad = [pw//2, pw-pw//2, ph//2, ph-ph//2]
                elif opt.Padding() != tflite.Padding.VALID:
                    raise ValueError('unknown padding')
                options = dict(stride=stride, pad=pad, activation=opt.FusedActivationFunction())
            elif kind == 'AVERAGE_POOL_2D':
                opt = self.options(o, tflite.Pool2DOptions)
                if opt.Padding() != tflite.Padding.VALID or opt.FusedActivationFunction() != 0:
                    raise ValueError('unexpected pool options')
                options = dict(kernel=(opt.FilterHeight(), opt.FilterWidth()),
                               stride=(opt.StrideH(), opt.StrideW()))
            elif kind == 'RESIZE_BILINEAR':
                opt = self.options(o, tflite.ResizeBilinearOptions)
                if not opt.HalfPixelCenters() or opt.AlignCorners():
                    raise ValueError('unsupported resize coordinates')
                options['size'] = tuple(map(int, g.Tensors(output).ShapeAsNumpy()[1:3]))
            elif kind == 'CUSTOM':
                if code.CustomCode() != b'Convolution2DTransposeBias':
                    raise ValueError('unrecognized custom operator')
                # This audited model has kernel 2, stride 2, output exactly 2x
                # input. There is no border overlap or implicit padding.
                ins = g.Tensors(inputs[0]).ShapeAsNumpy()
                outs = g.Tensors(output).ShapeAsNumpy()
                ws = g.Tensors(inputs[1]).ShapeAsNumpy()
                if tuple(ws) != (1, 2, 2, 16) or tuple(outs[1:3]) != tuple(ins[1:3]*2):
                    raise ValueError('unexpected custom transpose convolution')
            elif kind in ('ADD', 'MUL'):
                cls = tflite.AddOptions if kind == 'ADD' else tflite.MulOptions
                options['activation'] = self.options(o, cls).FusedActivationFunction()
            elif kind not in ('DEQUANTIZE', 'RELU', 'HARD_SWISH', 'LOGISTIC'):
                raise ValueError(f'unsupported source operator {kind}')
            self.ops.append((kind, inputs, output, options))

    @staticmethod
    def options(operator, cls):
        table = operator.BuiltinOptions()
        value = cls()
        value.Init(table.Bytes, table.Pos)
        return value

    @staticmethod
    def activate(x, kind):
        if kind == 0:
            return x
        if kind == tflite.ActivationFunctionType.RELU:
            return F.relu(x)
        if kind == tflite.ActivationFunctionType.RELU6:
            return torch.clamp(x, 0, 6)
        raise ValueError(f'unsupported fused activation {kind}')

    def forward(self, x):
        values = {i: getattr(self, n) for i, n in self.constants.items()}
        values[self.input] = x
        for kind, ids, dst, opt in self.ops:
            a = [values[i] for i in ids]
            if kind == 'DEQUANTIZE':
                y = a[0]  # Source constants are already fp16-rounded floats.
            elif kind in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
                if kind == 'CONV_2D':
                    weight, groups = a[1].permute(0, 3, 1, 2), 1
                else:
                    weight = a[1].permute(3, 0, 1, 2)
                    groups = int(self.graph.Tensors(ids[0]).ShapeAsNumpy()[3])
                y = F.conv2d(F.pad(a[0], opt['pad']), weight, a[2], opt['stride'], groups=groups)
                y = self.activate(y, opt['activation'])
            elif kind == 'AVERAGE_POOL_2D':
                y = F.avg_pool2d(a[0], opt['kernel'], opt['stride'])
            elif kind == 'RELU':
                y = F.relu(a[0])
            elif kind == 'HARD_SWISH':
                y = a[0]*torch.clamp(a[0]+3, 0, 6)/6
            elif kind == 'LOGISTIC':
                y = torch.sigmoid(a[0])
            elif kind == 'RESIZE_BILINEAR':
                y = F.interpolate(a[0], size=opt['size'], mode='bilinear', align_corners=False)
            elif kind == 'CUSTOM':
                y = F.conv_transpose2d(a[0], a[1].permute(3, 0, 1, 2), a[2], stride=2)
            else:
                y = a[0]+a[1] if kind == 'ADD' else a[0]*a[1]
                y = self.activate(y, opt['activation'])
            values[dst] = y
        return values[self.output]
