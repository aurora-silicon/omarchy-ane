# SPDX-License-Identifier: MIT
"""YuNet decode follows its published stride/offset equations. No identity or
landmark history is kept. Only the current crop window persists per stream.
"""
import math
import cv2
import numpy as np


def faces(outputs, size=320):
    boxes,scores=[],[]
    for i,stride in enumerate((8,16,32)):
        n=size//stride
        cls=np.clip(outputs[i].reshape(-1),0,1)
        obj=np.clip(outputs[i+3].reshape(-1),0,1)
        score=np.sqrt(cls*obj)
        offsets=outputs[i+6].reshape(-1,4)
        rows,cols=np.divmod(np.arange(n*n),n)
        ids=np.flatnonzero(score>=.6)
        for k in ids:
            cx,cy=(np.array([cols[k],rows[k]])+offsets[k,:2])*stride
            bw,bh=np.exp(np.clip(offsets[k,2:],-10,10))*stride
            x,y=max(0,cx-bw/2),max(0,cy-bh/2)
            right,bottom=min(size,cx+bw/2),min(size,cy+bh/2)
            if right>x and bottom>y:
                boxes.append([float(x),float(y),float(right-x),float(bottom-y)])
                scores.append(float(score[k]))
    ids=cv2.dnn.NMSBoxes(boxes,scores,.6,.3,top_k=100)
    return np.array([boxes[int(k)] for k in np.asarray(ids).reshape(-1)],np.float32).reshape(-1,4)/size


class Framer:
    def __init__(self, minimum, maximum, default, zoom=0):
        if not all(math.isfinite(x) for x in (minimum,maximum,default,zoom)) or not (
                1 <= minimum <= default <= maximum <= 6 and (zoom == 0 or minimum <= zoom <= maximum)):
            raise ValueError('invalid board framing range or stream zoom')
        self.minimum,self.maximum,self.default,self.manual=minimum,maximum,default,zoom
        self.reset()

    def reset(self):
        self.window=np.array([.5,.5,self.manual or self.default],np.float64)
        self.target=self.window.copy()
        self.last=None
        self.missing=0.

    def step(self, detections, pts):
        dt=1/30 if self.last is None else max(0.,min(.2,pts-self.last))
        if self.last is not None and (pts<=self.last or pts-self.last>1):
            self.reset()
            dt=1/30
        self.last=pts
        if len(detections):
            self.missing=0
            left,top=np.min(detections[:,:2],axis=0)
            right,bottom=np.max(detections[:,:2]+detections[:,2:],axis=0)
            fw,fh=right-left,bottom-top
            # Keep room for head/shoulders around the union of visible faces.
            z=np.clip(min(1/max(fw*2.5,1e-6),1/max(fh*3.5,1e-6)),self.minimum,self.maximum)
            desired=np.array([(left+right)/2,(top+bottom)/2+fh*.5,self.manual or z])
            if np.max(np.abs(desired[:2]-self.target[:2]))>.02 or abs(desired[2]-self.target[2])>.05:
                self.target=desired
        else:
            self.missing+=dt
            if self.missing>1:
                self.target=np.array([.5,.5,self.manual or self.default])
        # Time-based easing is stable when inference skips captured frames.
        self.window+=(self.target-self.window)*(-math.expm1(-dt/.4))
        z=float(np.clip(self.window[2],self.minimum,self.maximum))
        half=.5/z
        self.window[:2]=np.clip(self.window[:2],half,1-half)
        self.window[2]=z
        return tuple(self.window)
