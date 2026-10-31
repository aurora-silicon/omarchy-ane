# SPDX-License-Identifier: MIT
"""Read one official COCO annotation ZIP member via verified HTTP byte ranges.
Avoid downloading unrelated training annotations for a validation-only run.
ZIP CRC validation remains in Python's standard zipfile reader.
"""
import io
import urllib.request
import zipfile


class RangeFile(io.RawIOBase):
    def __init__(self, url):
        self.url=url
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=120) as response:
            self.size=int(response.headers['Content-Length'])
        self.position=0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        value=offset+(0 if whence==0 else self.position if whence==1 else self.size)
        if whence not in (0,1,2) or value<0:
            raise ValueError('invalid annotation archive seek')
        self.position=value
        return value

    def read(self, size=-1):
        if size is None or size<0:
            size=self.size-self.position
        size=max(0,min(size,self.size-self.position))
        if not size:
            return b''
        start=self.position;end=start+size-1
        request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(request,timeout=120) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':
                raise ValueError('annotation server did not honor the requested byte range')
            value=response.read(size+1)
        if len(value)!=size:
            raise ValueError('incomplete annotation range')
        self.position+=size
        return value


def validation_json(url):
    with RangeFile(url) as source, zipfile.ZipFile(source) as archive:
        return archive.read('annotations/instances_val2017.json')
