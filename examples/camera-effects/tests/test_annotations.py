# SPDX-License-Identifier: MIT
import io
import zipfile
from unittest.mock import patch
from coco_annotations import validation_json


def test_only_selected_annotation_member_is_downloaded():
    value=b'{"images":[],"annotations":[]}'
    storage=io.BytesIO()
    with zipfile.ZipFile(storage,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('annotations/instances_train2017.json',bytes(range(256))*4096,compress_type=zipfile.ZIP_STORED)
        archive.writestr('annotations/instances_val2017.json',value)
    payload=storage.getvalue();requested=[]
    class Response(io.BytesIO):
        pass
    def open(request,timeout):
        assert timeout==120
        if request.get_method()=='HEAD':
            r=Response();r.headers={'Content-Length':str(len(payload))};return r
        start,end=map(int,request.headers['Range'][6:].split('-'))
        requested.append((start,end))
        r=Response(payload[start:end+1]);r.status=206
        r.headers={'Content-Range':f'bytes {start}-{end}/{len(payload)}'}
        return r
    with patch('coco_annotations.urllib.request.urlopen',open):
        assert validation_json('https://example.invalid/official.zip')==value
    assert sum(b-a+1 for a,b in requested)<70000
