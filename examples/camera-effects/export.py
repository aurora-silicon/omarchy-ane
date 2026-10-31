#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Export open models to CPU ONNX and typed MIL using the same source weights.

coremltools performs graph conversion only. No Core ML compiler or Apple model
bytes are used. A successful compile is not automatically native qualification.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import shutil
import zipfile
import numpy as np
import torch
import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(root):
    import coremltools as ct
    from onnx2torch import convert
    from tflite_source import Selfie
    # Reuse the existing typed MIL/BLOBFILE serializer unchanged. These graphs
    # have straight-line, single-output operations and scalar/tensor operands.
    spec = importlib.util.spec_from_file_location('camera_mil_export', ROOT.parent/'camera-denoise/export.py')
    writer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(writer)
    torch.set_num_threads(2)
    torch.manual_seed(3653)
    source = ROOT/'models'
    expected_sources = {
        'selfie.tflite':'9ee168ec7c8f2a16c56fe8e1cfbc514974cbbb7e434051b455635f1bd1462f5c',
        'yunet.onnx':'8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'}
    for file,digest in expected_sources.items():
        if sha(source/file) != digest:
            raise ValueError(f'unaudited source model: {file}')
    results = {}
    for name, shape in [('selfie', (1, 3, 256, 256)), ('yunet', (1, 3, 320, 320))]:
        dest = root/name
        dest.mkdir(parents=True, exist_ok=False)
        if name == 'selfie':
            model = Selfie(source/'selfie.tflite').eval()
            output_names = ['mask']
        else:
            original = onnx.load(source/'yunet.onnx')
            # The source network is fully convolutional. Only its interface
            # metadata fixes it to 640. Export at 320 with the identical weights.
            original.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 320
            original.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 320
            from onnx import numpy_helper
            for tensor in original.graph.initializer:
                value = numpy_helper.to_array(tensor)
                if value.dtype == np.float32:
                    tensor.CopyFrom(numpy_helper.from_array(value.astype(np.float16).astype(np.float32), tensor.name))
            model = convert(original).eval()
            output_names = [o.name for o in original.graph.output]
        x = torch.rand(shape)* (255 if name == 'yunet' else 1)
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
        torch.onnx.export(model, x, str(dest/'model.onnx'), input_names=['x'],
                          output_names=output_names, opset_version=17, dynamo=False)
        session = ort.InferenceSession(str(dest/'model.onnx'), providers=['CPUExecutionProvider'])
        with torch.no_grad():
            expected = model(x)
        if isinstance(expected, torch.Tensor):
            expected = [expected]
        actual = session.run(None, {'x': x.numpy()})
        errors = [float(np.max(np.abs(a-b.detach().numpy()))) for a,b in zip(actual,expected)]
        for actual_value, expected_value in zip(actual,expected):
            np.testing.assert_allclose(actual_value, expected_value.detach().numpy(), atol=2e-4, rtol=1e-3)
        program = ct.convert(traced, source='pytorch',
                             inputs=[ct.TensorType(name='x', shape=shape, dtype=np.float16)],
                             minimum_deployment_target=ct.target.iOS18, convert_to='milinternal',
                             compute_precision=ct.precision.FLOAT16)
        # mil-hwxc identifiers contain letters/digits/underscores only, and
        # cannot start with a digit. coremltools' Torch names contain dots.
        for index, op in enumerate(program.functions['main'].operations):
            op.name = f'ce_op_{index}'
            for j, value in enumerate(op.outputs):
                value.set_name(f'ce_v_{index}_{j}')
        probes = writer.serialize_mil(program, dest)
        # Keep hundreds of reproducible operator probes in one source archive.
        # probe_compiler reads individual entries; it never extracts a path.
        with zipfile.ZipFile(dest/'probes.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted((dest/'probes').glob('*.mil')):
                entry=zipfile.ZipInfo(str(path.relative_to(dest)),date_time=(1980,1,1,0,0,0))
                entry.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(entry,path.read_bytes())
        shutil.rmtree(dest/'probes')
        metadata = dict(model=name, versions=dict(torch=torch.__version__,coremltools=ct.__version__,onnx=onnx.__version__,numpy=np.__version__), source_sha256=sha(source/('selfie.tflite' if name == 'selfie' else 'yunet.onnx')),
                        license='Apache-2.0' if name == 'selfie' else 'MIT', input=list(shape),
                        output_names=output_names, output_shapes=[list(v.shape) for v in actual], onnx_torch_max_abs=errors,
                        mil_outputs=[v.name for v in program.functions['main'].outputs],
                        native_targets=[], ops=probes,
                        artifacts={n:sha(dest/n) for n in ('model.onnx','model.mil','weights.bin','probes.zip')})
        (dest/'model.json').write_text(json.dumps(metadata, indent=2)+'\n')
        results[name] = metadata
    (root/'export.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    export(a.output.resolve())
