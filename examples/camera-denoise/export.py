#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build coremltools MIL and ONNX from the same released fp16 weight arrays.

MIL serialization walks the actual typed coremltools program. It does not use
Core ML compilation or any Apple program bytes. The small text writer emits
mil-hwxc's public textual interchange and documented BLOBFILE layout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mil_program(weights):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 64, 64), dtype=types.fp16)],
                opset_version=ct.target.iOS18)
    def program(x):
        def conv(x, name, stride=1):
            return mb.conv(x=x, weight=weights[name+".weight"], bias=weights[name+".bias"],
                           strides=[stride, stride], pad_type="same", name=name)
        skip = mb.relu(x=conv(x, "enc"), name="skip")
        down = mb.relu(x=conv(skip, "down", 2), name="down_relu")
        low = mb.relu(x=conv(down, "mid"), name="low")
        up = mb.upsample_nearest_neighbor(x=low, scale_factor_height=2.,
                                         scale_factor_width=2., name="up")
        merged = mb.add(x=up, y=skip, name="merged")
        decoded = mb.relu(x=conv(merged, "dec"), name="decoded")
        restored = mb.sub(x=x, y=conv(decoded, "head"), name="restored")
        return mb.clip(x=restored, alpha=np.float16(0), beta=np.float16(1), name="denoised")
    program.validate()
    return program


def serialize_mil(program, root):
    from coremltools.converters.mil.mil import types
    function = program.functions["main"]
    payload = bytearray(128)

    def typ(v):
        dtype = types.builtin_to_string(v.dtype)
        shape = tuple(v.shape) if types.is_tensor(v.sym_type) else ()
        return (f'tensor<{dtype}, [{", ".join(map(str, shape))}]>' if shape else dtype)

    def literal(v):
        val = v.val
        if isinstance(val, np.ndarray) and val.dtype == np.float16:
            payload.extend(bytes((-len(payload)) % 64))
            offset = len(payload)
            raw = val.astype("<f2").tobytes()
            payload.extend(struct.pack("<IIQQ", 0xDEADBEEF, 1, len(raw), offset+64)+bytes(40)+raw)
            return f'BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset}))'
        if isinstance(val, (np.generic, np.ndarray)):
            val = val.tolist()
        return json.dumps(val)

    args = ", ".join(f"{typ(v)} {v.name}" for v in function.inputs.values())
    lines = ['program(1.3)', '[buildInfo = dict<string, string>({})]', '{',
             f'  func main<ios18>({args}) {{']
    probes = []
    constants = {}
    for op in function.operations:
        output, = op.outputs
        prefix = f"    {typ(output)} {output.name} = {op.op_type}"
        if op.op_type == "const":
            line = prefix + f'()[name = string("{op.name}"), val = {typ(output)}({literal(output)})];'
            constants[output.name] = line
        else:
            if op.blocks:
                raise ValueError("only a straight-line MIL program is supported")
            if any(isinstance(v, (tuple, list)) for v in op.inputs.values()):
                raise ValueError("tuple MIL operands need a serializer extension")
            args = ", ".join(f"{key} = {v.name}" for key, v in op.inputs.items())
            line = prefix + f'({args})[name = string("{op.name}")];'
            # Each isolated op carries the actual MIL types and weight constants.
            runtime = [v for v in op.inputs.values() if v.name not in constants]
            declarations = "\n".join(constants[v.name] for v in op.inputs.values() if v.name in constants)
            declarations += "\n"+line
            parameters = ", ".join(f"{typ(v)} {v.name}" for v in runtime)
            probe = root/"probes"/(op.name+".mil")
            probe.parent.mkdir(exist_ok=True)
            probe.write_text('program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
                             f'  func main<ios18>({parameters}) {{\n{declarations}\n'
                             f'  }} -> ({output.name});\n}}\n')
            probes.append({"name": op.name, "op": op.op_type,
                           "inputs": {v.name: list(v.shape) for v in runtime},
                           "output": list(output.shape), "mil": str(probe.relative_to(root))})
        lines.append(line)
    lines += [f'  }} -> ({", ".join(v.name for v in function.outputs)});', '}']
    (root/"model.mil").write_text("\n".join(lines)+"\n")
    (root/"weights.bin").write_bytes(payload)
    return probes


def onnx_model(weights, root):
    nodes, initializers = [], []
    for name, value in weights.items():
        initializers.append(nh.from_array(value.astype(np.float32), name))

    def op(kind, inputs, name, **attributes):
        nodes.append(h.make_node(kind, inputs, [name+"_float"], **attributes))
        nodes.append(h.make_node("Cast", [name+"_float"], [name+"_half"], to=T.FLOAT16))
        nodes.append(h.make_node("Cast", [name+"_half"], [name], to=T.FLOAT))
        return name

    def conv(x, name, stride=1):
        return op("Conv", [x, name+".weight", name+".bias"], name,
                  auto_pad="SAME_UPPER", strides=[stride, stride])

    skip = op("Relu", [conv("x", "enc")], "skip")
    down = op("Relu", [conv(skip, "down", 2)], "down_relu")
    low = op("Relu", [conv(down, "mid")], "low")
    initializers += [nh.from_array(np.array([1, 1, 2, 2], np.float32), "scales"),
                     nh.from_array(np.array(0, np.float32), "zero"),
                     nh.from_array(np.array(1, np.float32), "one")]
    up = op("Resize", [low, "", "scales"], "up", mode="nearest",
            coordinate_transformation_mode="asymmetric", nearest_mode="floor")
    merged = op("Add", [up, skip], "merged")
    decoded = op("Relu", [conv(merged, "dec")], "decoded")
    restored = op("Sub", ["x", conv(decoded, "head")], "restored")
    op("Clip", [restored, "zero", "one"], "denoised")
    interface = [1, 1, "height", "width"]
    graph = h.make_graph(nodes, "tiny-luma-unet-v1",
                         [h.make_tensor_value_info("x", T.FLOAT, interface)],
                         [h.make_tensor_value_info("denoised", T.FLOAT, interface)], initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10,
                         producer_name="aurora-camera-denoise", producer_version="1")
    h.set_model_props(model, {"license": "MIT", "model": "tiny-luma-unet-v1",
                             "weights_sha256": sha(root/"weights.npz")})
    onnx.checker.check_model(model)
    onnx.save(model, root/"model.onnx")


def compile_probes(root, compiler, probes):
    records = []
    for target in ("H13", "H14", "H16G"):
        for record in [{"name": "full-model", "mil": "model.mil"}, *probes]:
            destination = root/"compiled"/target/record["name"]
            destination.mkdir(parents=True, exist_ok=True)
            command = [str(compiler), "--mil", str(root/record["mil"]),
                       "--model-root", str(root), "--target", target, "--output", str(destination)]
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                rc, stdout, stderr = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as error:
                rc, stdout, stderr = 124, str(error.stdout or ""), str(error.stderr or "")
            (destination/"stdout.log").write_text(stdout)
            (destination/"stderr.log").write_text(stderr)
            records.append({**record, "target": target, "rc": rc,
                            "diagnostic": stderr[-4000:], "command": command})
    (root/"compiler-gaps.json").write_text(json.dumps(records, indent=2)+"\n")
    return records


def compile_and_record(root, compiler, metadata):
    if (root/"compiled").exists():
        raise ValueError("compiler destination must be absent; preserve earlier attempts separately")
    results = compile_probes(root, compiler, metadata["ops"])
    metadata["native_targets"] = [r["target"] for r in results
                                  if r["name"] == "full-model" and r["rc"] == 0]
    for target in metadata["native_targets"]:
        for path in (root/"compiled"/target/"full-model").rglob("*"):
            if path.is_file() and path.suffix != ".log":
                metadata["artifacts"][str(path.relative_to(root))] = sha(path)
    (root/"model.json").write_text(json.dumps(metadata, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--compiler", type=Path)
    a = p.parse_args()
    root = a.model.resolve()
    if any((root/name).exists() for name in
           ("model.json", "model.mil", "model.onnx", "compiled", "probes", "source.mlpackage")):
        p.error("export needs a fresh directory containing only weights and training provenance")
    with np.load(root/"weights.npz", allow_pickle=False) as arrays:
        weights = dict(arrays)
    program = mil_program(weights)
    probes = serialize_mil(program, root)
    onnx_model(weights, root)
    # Validate using coremltools' MIL backend, without loading/compiling on Apple.
    import coremltools as ct
    converted = ct.convert(program, convert_to="mlprogram", skip_model_load=True,
                           minimum_deployment_target=ct.target.iOS18,
                           pass_pipeline=ct.PassPipeline.EMPTY)
    converted.license = "MIT"
    converted.author = "Aurora Silicon contributors"
    converted.save(str(root/"source.mlpackage"))
    metadata = {"model": "tiny-luma-unet-v1", "license": "MIT", "format": "GRAY8",
                "tile": 64, "halo": 12, "core": 40, "state": "stateless",
                "coremltools": ct.__version__, "ops": probes, "native_targets": [],
                "artifacts": {n: sha(root/n) for n in
                              ("weights.npz", "weights.bin", "model.onnx", "model.mil", "training.json")}}
    (root/"model.json").write_text(json.dumps(metadata, indent=2)+"\n")
    if a.compiler:
        compile_and_record(root, a.compiler.resolve(), metadata)


if __name__ == "__main__":
    main()
