#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Reproduce the MobileNetV1-style CoreML baseline without Apple program bytes.
NumPy RNG and layer order match mkmodel.py: seed 1, normal * .05, alpha 1,
224x224, bias on every conv and linear, ReLU6, no batch normalization.
"""
import argparse
import json
from pathlib import Path
import struct
import subprocess
import shutil
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper as nh
import hashlib

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

CFG = [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1), (512, 2)] + [(512, 1)]*5 + [(1024, 2), (1024, 1)]


def typ(dtype, shape):
    return f"tensor<{dtype}, [{','.join(map(str, shape))}]>"


class Export:
    def __init__(self, root, batch):
        self.root = root
        self.batch = batch
        self.rng = np.random.default_rng(1)
        self.nodes, self.initializers, self.lines, self.probes = [], [], [], []
        self.payload = bytearray(128)
        self.shapes = {"x": [batch, 3, 224, 224]}

    def attr(self, name, dtype, value):
        if isinstance(value, list):
            t = typ(dtype, [len(value)])
            literal = str(value)
        else:
            t = dtype
            literal = json.dumps(value) if dtype == "string" else str(value)
        line = f'    {t} {name} = const()[name = string("{name}"), val = {t}({literal})];'
        self.lines.append(line)
        return line

    def weight(self, name, shape):
        # Generate in float64 and cast to float32 before fp16, as baseline w().
        w = (self.rng.standard_normal(shape)*0.05).astype(np.float32).astype(np.float16)
        self.initializers.append(nh.from_array(w.astype(np.float32), name))
        self.payload.extend(bytes((-len(self.payload)) % 64))
        offset = len(self.payload)
        raw = w.astype('<f2').tobytes()
        self.payload.extend(struct.pack('<IIQQ', 0xDEADBEEF, 1, len(raw), offset+64) + bytes(40) + raw)
        t = typ("fp16", shape)
        line = f'    {t} {name} = const()[name = string("{name}"), val = {t}(BLOBFILE(path = string("@model_path/weights.bin"), offset = uint64({offset})))];'
        self.lines.append(line)
        return line

    def round_node(self, op, inputs, name, **attrs):
        self.nodes.append(h.make_node(op, inputs, [name+"_fp32"], **attrs))
        self.nodes.append(h.make_node("Cast", [name+"_fp32"], [name+"_fp16"], to=T.FLOAT16))
        self.nodes.append(h.make_node("Cast", [name+"_fp16"], [name], to=T.FLOAT))

    def probe(self, name, inputs, output, lines):
        args = ", ".join(f'{typ("fp16", self.shapes[n])} {n}' for n in inputs)
        text = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
        text += f'  func main<ios18>({args}) {{\n' + '\n'.join(lines) + f'\n  }} -> ({output});\n}}\n'
        path = self.root / "probes" / (name+".mil")
        path.write_text(text)
        self.probes.append({"name": name, "mil": str(path.relative_to(self.root)),
                            "inputs": {n: self.shapes[n] for n in inputs}, "output": self.shapes[output]})

    def conv(self, x, co, stride, groups, kernel, name, same):
        b, ci, hi, wi = self.shapes[x]
        oh, ow = (hi+stride-1)//stride, (wi+stride-1)//stride
        self.shapes[name] = [b, co, oh, ow]
        start = len(self.lines)
        self.weight(name+"_w", [co, ci//groups, kernel, kernel])
        self.weight(name+"_b", [co])
        self.attr(name+"_s", "int32", [stride, stride])
        self.attr(name+"_d", "int32", [1, 1])
        self.attr(name+"_p", "int32", [0, 0, 0, 0])
        self.attr(name+"_pt", "string", "same" if same else "valid")
        self.attr(name+"_g", "int32", groups)
        self.lines.append(f'    {typ("fp16", self.shapes[name])} {name} = conv(x = {x}, weight = {name}_w, bias = {name}_b, strides = {name}_s, dilations = {name}_d, pad = {name}_p, pad_type = {name}_pt, groups = {name}_g)[name = string("{name}")];')
        self.round_node("Conv", [x, name+"_w", name+"_b"], name, strides=[stride]*2,
                        group=groups, auto_pad="SAME_UPPER" if same else "VALID")
        self.probe(name, [x], name, self.lines[start:])
        return self.clip(name, name+"_relu6")

    def clip(self, x, name):
        self.shapes[name] = self.shapes[x]
        start = len(self.lines)
        self.attr(name+"_lo", "fp32", 0)
        self.attr(name+"_hi", "fp32", 6)
        self.lines.append(f'    {typ("fp16", self.shapes[name])} {name} = clip(x = {x}, alpha = {name}_lo, beta = {name}_hi)[name = string("{name}")];')
        self.initializers += [nh.from_array(np.array(v, dtype=np.float32), name+s) for s, v in (("_lo", 0), ("_hi", 6))]
        self.round_node("Clip", [x, name+"_lo", name+"_hi"], name)
        self.probe(name, [x], name, self.lines[start:])
        return name

    def build(self):
        x = self.conv("x", 32, 2, 1, 3, "stem", True)
        c = 32
        for i, (co, stride) in enumerate(CFG):
            x = self.conv(x, c, stride, c, 3, f"dw{i+1}", True)
            x = self.conv(x, co, 1, 1, 1, f"pw{i+1}", False)
            c = co
        self.shapes["mean"] = [self.batch, 1024]
        start = len(self.lines)
        self.attr("mean_axes", "int32", [2, 3])
        self.attr("mean_keep", "bool", "false")
        self.lines.append(f'    {typ("fp16", self.shapes["mean"])} mean = reduce_mean(x = {x}, axes = mean_axes, keep_dims = mean_keep)[name = string("mean")];')
        self.round_node("ReduceMean", [x], "mean", axes=[2, 3], keepdims=0)
        self.probe("mean", [x], "mean", self.lines[start:])
        self.shapes["logits"] = [self.batch, 1000]
        start = len(self.lines)
        self.weight("head_w", [1000, 1024]); self.weight("head_b", [1000])
        self.lines.append(f'    {typ("fp16", self.shapes["logits"])} logits = linear(x = mean, weight = head_w, bias = head_b)[name = string("logits")];')
        self.round_node("Gemm", ["mean", "head_w", "head_b"], "logits", transB=1)
        self.probe("linear", ["mean"], "logits", self.lines[start:])
        self.shapes["probabilities"] = [self.batch, 1000]
        start = len(self.lines)
        self.attr("softmax_axis", "int32", -1)
        self.lines.append(f'    {typ("fp16", self.shapes["probabilities"])} probabilities = softmax(x = logits, axis = softmax_axis)[name = string("probabilities")];')
        self.round_node("Softmax", ["logits"], "probabilities", axis=-1)
        self.probe("softmax", ["logits"], "probabilities", self.lines[start:])
        graph = h.make_graph(self.nodes, "mobilenetv1-baseline", [h.make_tensor_value_info("x", T.FLOAT, self.shapes["x"])],
                             [h.make_tensor_value_info("probabilities", T.FLOAT, self.shapes["probabilities"])], self.initializers)
        model = h.make_model(graph, opset_imports=[h.make_opsetid("", 13)], ir_version=10,
                             producer_name="anert-open-baseline")
        onnx.checker.check_model(model)
        onnx.save(model, self.root / "model.onnx")
        self.payload[:8] = struct.pack('<II', 1, 2)  # CoreML public weight-blob header
        (self.root / "weights.bin").write_bytes(self.payload)
        mil = 'program(1.3)\n[buildInfo = dict<string, string>({})]\n{\n'
        mil += f'  func main<ios18>({typ("fp16", self.shapes["x"])} x) {{\n' + '\n'.join(self.lines) + '\n  } -> (probabilities);\n}\n'
        (self.root / "model.mil").write_text(mil)
        metadata = {"model": "MobileNetV1-style alpha 1.0 224 fp16 seed 1",
                   "batch": self.batch, "cpu": {"file": "model.onnx", "sha256": sha(self.root / "model.onnx")},

                   "provenance": {"seed": 1, "weights_sha256": sha(self.root / "weights.bin"),
                                  "mil_sha256": sha(self.root / "model.mil"),
                                  "cpu_arithmetic": "fp32 accumulation with fp16 rounding at each MIL operation"}}
        (self.root / "model.json").write_text(json.dumps(metadata, indent=2)+"\n")


def compile_and_probe(root, compiler):
    report = {"compiler": str(compiler), "targets": {}}
    probes = json.loads((root / "probes.json").read_text())
    for target in ("H13", "H14"):
        records = []
        for item in [{"name": "full-model", "mil": "model.mil"}] + probes:
            output = root / "compiler-probes" / target / item["name"]
            output.parent.mkdir(parents=True, exist_ok=True)
            cmd = [str(compiler), "--target", target, "--mil", str(root/item["mil"]),
                   "--model-root", str(root), "--output", str(output)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                record = {**item, "rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
            except subprocess.TimeoutExpired as error:
                def text(value):
                    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                record = {**item, "rc": 124, "stdout": text(error.stdout),
                          "stderr": text(error.stderr) + "\nprobe exceeded 180s; not a compiler refusal"}
            if not record["rc"]:
                manifest = json.loads((output / "manifest.json").read_text())
                record["program_count"] = len(manifest.get("programs", []))
                # Isolated successful probes are reproducible; keep their
                # diagnostics/counts without retaining thousands of programs.
                if item["name"] != "full-model":
                    shutil.rmtree(output)
            records.append(record)
            report["targets"][target] = records
            (root / "compiler-gaps.json").write_text(json.dumps(report, indent=2)+"\n")
        report["targets"][target] = records
    (root / "compiler-gaps.json").write_text(json.dumps(report, indent=2)+"\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("destination", type=Path)
    p.add_argument("--batch", type=int, choices=(1, 8), required=True)
    p.add_argument("--compiler", type=Path)
    a = p.parse_args()
    a.destination.mkdir(parents=True, exist_ok=False)
    (a.destination / "probes").mkdir()
    e = Export(a.destination, a.batch)
    e.build()
    (a.destination / "probes.json").write_text(json.dumps(e.probes, indent=2)+"\n")
    if a.compiler:
        compile_and_probe(a.destination, a.compiler.resolve())
    print(a.destination)

if __name__ == "__main__": main()
