# SPDX-License-Identifier: MIT
"""Real-package physical linking checks, without a device or image download."""
import copy
import json
import os
from pathlib import Path
import sys
import tarfile

import numpy as np
import pytest

from aurora_camera_effects.backend import ROOT
from aurora_camera_effects.yunet_boundary import prepare


@pytest.fixture(scope='module')
def package(tmp_path_factory):
    compiler=Path(os.environ.get('AURORA_CAMERA_COMPILER_ROOT',ROOT.parents[2]/'compiler'))
    if not (compiler/'tools/h13_reference.py').is_file():
        pytest.skip('set AURORA_CAMERA_COMPILER_ROOT for linked package simulation')
    sys.path[:0]=[str(compiler/'tools'),str(compiler/'research')]
    path=tmp_path_factory.mktemp('decoded-dram-package')
    archive=ROOT/'packages/yunet-h13-linked-v1.tar.gz'
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            assert Path(member.name).name==member.name and member.isfile()
            (path/member.name).write_bytes(tar.extractfile(member).read())
    import inspect_anec
    manifest,_=inspect_anec.load_package(path)
    return path,manifest


def input_image():
    return (np.arange(3*320*320,dtype=np.uint32)%251).astype(np.float16).reshape(1,3,320,320)


def test_decoded_dma_and_mil_simulation_agree_on_the_real_package(package):
    from aurora_camera_effects.linked_dram_sim import DRAMSimulation
    from aurora_camera_effects.linked_sim import MILSimulation
    path,manifest=package
    outputs=json.loads((path/'camera-linked.json').read_text())['program_outputs']
    physical=DRAMSimulation(path,manifest)
    semantic=MILSimulation(path/'model.mil',path)
    x=prepare(input_image())
    np.testing.assert_array_equal(physical(x,outputs)[0],semantic(x,outputs)[0])
    # The same scratch allocation must also handle a new frame.
    x=prepare(input_image()[:,:,::-1,::-1].copy())
    np.testing.assert_array_equal(physical(x,outputs)[0],semantic(x,outputs)[0])


def test_consistently_misplaced_producers_fail_concat_semantics(package):
    from aurora_camera_effects.linked_dram_sim import DRAMSimulation
    path,manifest=package
    sim=DRAMSimulation(path,copy.deepcopy(manifest))
    concat=next(v for v in sim.program['layers'] if v['operation']=='concat')
    names=[ref.name for ref in sim.nodes[concat['result']].arguments['values']]
    producers=[next(v for v in sim.program['layers'] if v['result']==name) for name in names]
    left,right=producers
    a,b=left['firstTask'],right['firstTask']
    # Change both the actual TD relocation and its matching layer metadata;
    # range/identity checks alone would not prove that this is the right view.
    sim.tasks[a][1][0x17804],sim.tasks[b][1][0x17804]=sim.tasks[b][1][0x17804],sim.tasks[a][1][0x17804]
    left['writeOffset'],right['writeOffset']=right['writeOffset'],left['writeOffset']
    outputs=json.loads((path/'camera-linked.json').read_text())['program_outputs']
    with pytest.raises(AssertionError,match='producer placement'):
        sim(prepare(input_image()),outputs)
