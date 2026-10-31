# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import numpy as np
import onnx
from onnx import numpy_helper as nh

from mobilenet_baseline import Export

class ExporterTests(unittest.TestCase):
    def test_baseline_architecture_and_shared_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'probes').mkdir()
            e = Export(root, 1); e.build()
            model = onnx.load(root/'model.onnx')
            convs = [n for n in model.graph.node if n.op_type == 'Conv']
            self.assertEqual(len(convs), 27)
            self.assertEqual(len([n for n in model.graph.node if n.op_type == 'Clip']), 27)
            self.assertEqual(len([n for n in convs if any(a.name == 'group' and a.i > 1 for a in n.attribute)]), 13)
            self.assertEqual(e.shapes['pw13_relu6'], [1,1024,7,7])
            self.assertEqual(e.shapes['probabilities'], [1,1000])
            constants = {t.name: nh.to_array(t) for t in model.graph.initializer}
            rng = np.random.default_rng(1)
            expected = (rng.standard_normal([32,3,3,3])*.05).astype(np.float32).astype(np.float16)
            np.testing.assert_array_equal(constants['stem_w'], expected.astype(np.float32))
            payload = (root/'weights.bin').read_bytes()
            mil = (root/'model.mil').read_text()
            self.assertIn('val = fp32(6)', mil)
            self.assertNotIn('val = fp16(6)', mil)
            import re
            for name, offset in re.findall(r' (\w+) = const\(\).*BLOBFILE.*offset = uint64\((\d+)\)', mil):
                magic, _, size, start = struct.unpack_from('<IIQQ', payload, int(offset))
                self.assertEqual(magic, 0xDEADBEEF)
                dense = np.frombuffer(payload[start:start+size], dtype='<f2').astype(np.float32)
                np.testing.assert_array_equal(dense, constants[name].ravel())
            runtime = json.loads((root/'model.json').read_text())
            self.assertEqual(runtime['batch'], 1)
            # Every arithmetic op has explicit fp16 activation rounding.
            self.assertEqual(len([n for n in model.graph.node if n.op_type == 'Cast']), 2*(27+27+3))

if __name__ == '__main__': unittest.main()
