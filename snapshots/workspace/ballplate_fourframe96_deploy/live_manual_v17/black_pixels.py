"""Read only a root-window rectangle, not a whole 5560x3240 desktop crop.

No windows created or mapped, no painting/display setting or laser commands.
The XImage prefix follows installed /usr/include/X11/Xlib.h. Alpha/padding
bits are ignored; RGB masks and buffer bounds are strictly checked.
"""
import ctypes as C
import threading

import numpy as np


class Header(C.Structure):
    _fields_=[('width',C.c_int),('height',C.c_int),('xoffset',C.c_int),('format',C.c_int),
              ('data',C.c_void_p),('byte_order',C.c_int),('bitmap_unit',C.c_int),
              ('bitmap_bit_order',C.c_int),('bitmap_pad',C.c_int),('depth',C.c_int),
              ('bytes_per_line',C.c_int),('bits_per_pixel',C.c_int),('red_mask',C.c_ulong),
              ('green_mask',C.c_ulong),('blue_mask',C.c_ulong)]


def inspect_pixels(header,expected):
    w,h=expected
    if (header.width!=w or header.height!=h or header.xoffset!=0 or header.format!=2
        or header.bits_per_pixel!=32 or header.depth!=24 or header.byte_order not in (0,1)
        or header.bytes_per_line<w*4 or not 0<header.bytes_per_line*h<=100_000_000 or not header.data):
        raise RuntimeError('unsupported or unsafe root XImage layout')
    r,g,b=header.red_mask,header.green_mask,header.blue_mask
    if not r or not g or not b or r&g or r&b or g&b or (r|g|b)>0xffffffff:
        raise RuntimeError('unverified XImage RGB masks')
    raw=(C.c_uint8*(header.bytes_per_line*h)).from_address(header.data)
    values=np.ndarray((h,w),dtype='<u4' if header.byte_order==0 else '>u4',
                      buffer=raw,strides=(header.bytes_per_line,4))
    maximum=int(np.bitwise_and(values,r|g|b).max())
    return dict(all_RGB_zero=maximum==0,max_RGB_packed_bits=maximum,
                RGB_masks=[r,g,b],size=[w,h],alpha_and_padding_ignored=True)


def compare_pixels(header,expected):
    expected=np.asarray(expected)
    if expected.ndim!=2 or expected.dtype!=np.uint8:raise ValueError('uint8 expected display required')
    h,w=expected.shape;inspect_pixels(header,(w,h))
    masks=(header.red_mask,header.green_mask,header.blue_mask)
    shifts=[]
    for mask in masks:
        shift=(int(mask)&-int(mask)).bit_length()-1
        if mask!=255<<shift:raise RuntimeError('only verified 8-bit RGB masks supported')
        shifts.append(shift)
    raw=(C.c_uint8*(header.bytes_per_line*h)).from_address(header.data)
    values=np.ndarray((h,w),dtype='<u4' if header.byte_order==0 else '>u4',buffer=raw,
                      strides=(header.bytes_per_line,4))
    packed=np.zeros((h,w),np.uint32)
    for shift in shifts:packed|=expected.astype(np.uint32)<<shift
    mismatch=(values & sum(masks))!=packed
    count=int(np.count_nonzero(mismatch));bbox=None
    if count:
        ys=np.flatnonzero(mismatch.any(axis=1));xs=np.flatnonzero(mismatch.any(axis=0))
        bbox=[int(xs[0]),int(ys[0]),int(xs[-1]-xs[0]+1),int(ys[-1]-ys[0]+1)]
    return dict(matches=count==0,mismatched_pixels=count,mismatch_bbox=bbox,
                size=[w,h],alpha_and_padding_ignored=True)


class RootPixels:
    def __init__(self):
        self.x=C.CDLL('libX11.so.6');self.lock=threading.Lock();self.error=False
        self.x.XInitThreads.restype=C.c_int
        if not self.x.XInitThreads():raise RuntimeError('Xlib thread support unavailable')
        self.x.XOpenDisplay.argtypes=[C.c_char_p];self.x.XOpenDisplay.restype=C.c_void_p
        self.display=self.x.XOpenDisplay(None)
        if not self.display:raise RuntimeError('no current X display')
        self.x.XDefaultRootWindow.argtypes=[C.c_void_p];self.x.XDefaultRootWindow.restype=C.c_ulong
        self.root=self.x.XDefaultRootWindow(self.display)
        self.callback_type=C.CFUNCTYPE(C.c_int,C.c_void_p,C.c_void_p)
        def on_error(display,event):self.error=True;return 0
        self.callback=self.callback_type(on_error)
        self.x.XSetErrorHandler.argtypes=[C.c_void_p];self.x.XSetErrorHandler.restype=C.c_void_p
        self.old_handler=self.x.XSetErrorHandler(C.cast(self.callback,C.c_void_p))
        self.x.XGetImage.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_int,C.c_uint,C.c_uint,C.c_ulong,C.c_int]
        self.x.XGetImage.restype=C.POINTER(Header)
        self.x.XDestroyImage.argtypes=[C.POINTER(Header)];self.x.XDestroyImage.restype=C.c_int
        self.x.XSync.argtypes=[C.c_void_p,C.c_int]
        self.x.XCloseDisplay.argtypes=[C.c_void_p]

    def inspect(self,rectangle):
        x,y,w,h=rectangle
        if min(x,y)<0 or not (0<w<=10000 and 0<h<=10000):raise ValueError('invalid screen rectangle')
        with self.lock:
            self.error=False
            pointer=self.x.XGetImage(self.display,self.root,x,y,w,h,C.c_ulong(-1).value,2)
            self.x.XSync(self.display,False)
            try:
                if self.error or not pointer:raise RuntimeError('root rectangle read failed')
                return inspect_pixels(pointer.contents,(w,h))
            finally:
                if pointer:self.x.XDestroyImage(pointer)

    def compare(self,rectangle,expected):
        x,y,w,h=rectangle
        if min(x,y)<0 or not(0<w<=10000 and 0<h<=10000):raise ValueError('invalid screen rectangle')
        with self.lock:
            self.error=False
            pointer=self.x.XGetImage(self.display,self.root,x,y,w,h,C.c_ulong(-1).value,2)
            self.x.XSync(self.display,False)
            try:
                if self.error or not pointer:raise RuntimeError('root rectangle read failed')
                if np.asarray(expected).shape!=(h,w):raise ValueError('expected display shape differs')
                return compare_pixels(pointer.contents,expected)
            finally:
                if pointer:self.x.XDestroyImage(pointer)

    def close(self):
        with self.lock:
            if self.display:
                self.x.XCloseDisplay(self.display);self.display=None
                self.x.XSetErrorHandler(self.old_handler)
