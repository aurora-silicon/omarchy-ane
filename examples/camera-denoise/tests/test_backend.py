# SPDX-License-Identifier: MIT
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from aurora_camera_ane.backend import Backend, FrameCPU, MODEL, LIBANE, verify


def backend(width=81, height=63, **options):
    return Backend(width, height, model_root=MODEL, library=LIBANE, **options)


def test_strength_stride_inplace_and_zero_cost():
    model = backend(native_factory=lambda: None)
    rng = np.random.default_rng(2)
    source = rng.integers(0, 256, (63, 97), np.uint8)
    output = np.full((63, 99), 73, np.uint8)
    try:
        assert model.process(source[:, :81], None, 0., output[:, :81]) is None
        np.testing.assert_array_equal(source[:, :81], output[:, :81])
        assert model.runtime.stats()["calls"] == 0
        model.process(source[:, :81], None, 1., output[:, :81])
        full = output[:, :81].copy()
        model.process(source[:, :81], None, .5, output[:, :81])
        np.testing.assert_allclose(output[:, :81].astype(float),
                                   (source[:, :81].astype(float)+full)/2, atol=1)
        model.process(source[:, :81], None, 1., source[:, :81])
        np.testing.assert_array_equal(source[:, :81], full)
        assert np.all(output[:, 81:] == 73)
    finally:
        model.fini()


def test_native_failure_recomputes_from_original_and_retires():
    calls = []
    def failed(frame):
        calls.append(frame.copy())
        raise OSError("simulated busy native owner")
    model = backend(native_factory=lambda: failed)
    source = np.full((63, 81), 128, np.uint8)
    output = np.full_like(source, 7)
    try:
        model.process(source, None, 1., output)
        wanted = np.rint(model.cpu(source.astype(np.float32)/255).astype(np.float32)*255).astype(np.uint8)
        np.testing.assert_array_equal(output, wanted)
        model.process(source, None, 1., output)
        assert len(calls) == 1
        assert model.runtime.stats()["cpu_calls"] == 2
        assert "simulated busy" in str(model.runtime.native_failure)
    finally:
        model.fini()


@pytest.mark.parametrize("strength", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_strength_is_transactional(strength):
    model = backend(native_factory=lambda: None)
    source = np.full((63, 81), 128, np.uint8)
    output = np.full_like(source, 7)
    try:
        with pytest.raises(ValueError):
            model.process(source, None, strength, output)
        assert np.all(output == 7)
    finally:
        model.fini()


def test_identity_tamper_before_inference(tmp_path):
    shutil.copytree(MODEL, tmp_path/"model")
    path = tmp_path/"model/model.onnx"
    path.write_bytes(path.read_bytes()+b"tampered")
    with pytest.raises(ValueError, match="identity mismatch"):
        verify(tmp_path/"model")


def test_halo_tiling_matches_full_frame():
    cpu = FrameCPU(MODEL)
    rng = np.random.default_rng(4)
    source = rng.uniform(0, 1, (81, 123)).astype(np.float32)
    full = cpu(source)
    padded = np.pad(source, ((12, 12+(-81) % 40), (12, 12+(-123) % 40)), mode="edge")
    tiled = np.empty_like(full)
    for y in range(0, 81, 40):
        for x in range(0, 123, 40):
            tile = cpu.tensor(padded[None, None, y:y+64, x:x+64])[0, 0]
            hh, ww = min(40, 81-y), min(40, 123-x)
            tiled[y:y+hh, x:x+ww] = tile[12:12+hh, 12:12+ww]
    # ORT can select different convolution accumulation kernels at each shape.
    np.testing.assert_allclose(tiled.astype(float), full.astype(float), atol=.002, rtol=.002)


def test_export_uses_identical_weights_and_mil_types():
    import onnx
    from onnx import numpy_helper
    from export import mil_program
    with np.load(MODEL/"weights.npz") as arrays:
        weights = dict(arrays)
    graph = onnx.load(MODEL/"model.onnx").graph
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    program = mil_program(weights)
    for op in program.functions["main"].operations:
        if op.op_type == "conv":
            for kind in ("weight", "bias"):
                expected = weights[op.name+"."+kind]
                np.testing.assert_array_equal(initializers[op.name+"."+kind].astype(np.float16), expected)
                np.testing.assert_array_equal(op.inputs[kind].val, expected)
        if op.op_type != "const":
            from coremltools.converters.mil.mil import types
            assert op.outputs[0].dtype == types.fp16


def test_close_and_invalid_format():
    model = backend(native_factory=lambda: None)
    model.fini()
    model.fini()
    with pytest.raises(RuntimeError):
        model.process(np.zeros((63,81), np.uint8), None, 0., np.zeros((63,81), np.uint8))
    with pytest.raises(ValueError):
        Backend(81, 63, "RGB")
