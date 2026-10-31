# SPDX-License-Identifier: MIT
"""NV12 operations preserve source colorimetry; RGB exists only at model size."""
import math
import cv2
import numpy as np


def model_rgb(y, uv, size, matrix=601, full=False):
    """Stretch as in MediaPipe's public graph. uv is (h/2,w/2,2)."""
    dims = (size,size) if isinstance(size,int) else size
    yy = cv2.resize(y, dims, interpolation=cv2.INTER_AREA).astype(np.float32)
    chroma = cv2.resize(uv, dims, interpolation=cv2.INTER_LINEAR).astype(np.float32)-128
    if full:
        yy /= 255
        chroma /= 255
    else:
        yy = (yy-16)/219
        chroma /= 224
    kr, kb = (0.2126, 0.0722) if matrix == 709 else (0.299, 0.114)
    kg = 1-kr-kb
    cb, cr = chroma[...,0], chroma[...,1]
    r = yy + (2-2*kr)*cr
    b = yy + (2-2*kb)*cb
    g = yy - kb*(2-2*kb)/kg*cb - kr*(2-2*kr)/kg*cr
    return np.clip(np.stack([r,g,b], axis=-1),0,1)


def guided_mask(mask, luma):
    """Fast guided upsampling: fit coefficients at mask resolution, evaluate
    them against the full-resolution luma. Radius 4, normalized epsilon .01.
    No previous frame's mask is ever used on the current frame.
    """
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError('invalid segmentation mask')
    h,w = luma.shape
    low = cv2.resize(luma, (mask.shape[1],mask.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    box = lambda a: cv2.boxFilter(a, -1, (9,9), borderType=cv2.BORDER_REFLECT)
    mean_i, mean_p = box(low), box(mask)
    a = (box(low*mask)-mean_i*mean_p)/(box(low*low)-mean_i*mean_i+.01)
    b = mean_p-a*mean_i
    a = cv2.resize(box(a), (w,h), interpolation=cv2.INTER_LINEAR)
    b = cv2.resize(box(b), (w,h), interpolation=cv2.INTER_LINEAR)
    return np.clip(a*(luma.astype(np.float32)/255)+b,0,1)


def composite(y, uv, mask, background, replacement=None):
    h,w = y.shape
    if background == 'blur':
        # Separable Gaussian in native plane units. U and V stay separate.
        sigma = max(2., min(w,h)/80)
        by = cv2.GaussianBlur(y, (0,0), sigma)
        buv = cv2.GaussianBlur(uv, (0,0), sigma/2)
    elif background == 'replace' and replacement is not None:
        by,buv = replacement
    else:
        raise ValueError('replacement image required')
    m = mask
    small = cv2.resize(m,(w//2,h//2),interpolation=cv2.INTER_AREA)[...,None]
    # uint8 subtraction would wrap. Commit only after both planes succeed.
    yy = np.rint(m*y+(1-m)*by).astype(np.uint8)
    uu = np.rint(small*uv+(1-small)*buv).astype(np.uint8)
    return yy,uu


def replacement_nv12(rgb, w, h, matrix=601, full=False):
    rgb = cv2.resize(rgb, (w,h), interpolation=cv2.INTER_AREA).astype(np.float32)/255
    kr,kb = (0.2126,0.0722) if matrix == 709 else (0.299,0.114)
    r,g,b = rgb[...,0],rgb[...,1],rgb[...,2]
    yy = kr*r+(1-kr-kb)*g+kb*b
    cb,cr = (b-yy)/(2-2*kb),(r-yy)/(2-2*kr)
    y = np.rint(yy*(255 if full else 219)+(0 if full else 16))
    uv = np.stack([cb,cr],axis=-1)*(255 if full else 224)+128
    uv = cv2.resize(uv, (w//2,h//2),interpolation=cv2.INTER_AREA)
    return np.clip(y,0,255).astype(np.uint8),np.clip(np.rint(uv),0,255).astype(np.uint8)


def crop_nv12(y, uv, window):
    h,w=y.shape
    cx,cy,zoom=window
    cw=max(2,min(w,math.ceil(w/zoom/2)*2)); ch=max(2,min(h,math.ceil(h/zoom/2)*2))
    x=max(0,min(w-cw,int(cx*w-cw/2)//2*2))
    v=max(0,min(h-ch,int(cy*h-ch/2)//2*2))
    return (cv2.resize(y[v:v+ch,x:x+cw],(w,h),interpolation=cv2.INTER_LINEAR),
            cv2.resize(uv[v//2:(v+ch)//2,x//2:(x+cw)//2],(w//2,h//2),interpolation=cv2.INTER_LINEAR))
