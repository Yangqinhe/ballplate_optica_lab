"""Local UI input scoped to the live calibration window, never to optical outputs.

Uses the normal visible controls/dialogs, not an SDK side channel. Caller must
inspect the current UI and obtain the corresponding user permission first.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import subprocess
import time


def reject_owned_dialogs(tree, owner):
    """Never send a coordinate click through an unexpected modal dialog."""
    for line in tree.splitlines():
        if '("ballplate-calibration-workbench" "ballplate-calibration-workbench")' not in line:
            continue
        match=re.search(r'0x[0-9a-fA-F]+',line)
        if not match or int(match[0],16)==int(owner,16):continue
        props=subprocess.check_output(['xprop','-id',match[0],'WM_TRANSIENT_FOR'],text=True)
        transient=re.search(r'window id # (0x[0-9a-fA-F]+)',props)
        if transient and int(transient[1],16)==int(owner,16):
            raise RuntimeError('calibration dialog is open; resolve it in the original window before manual input')


def find_window(session):
    pid=json.loads((session/'heartbeat.json').read_text())['pid']
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True)
    candidates=[]
    for line in tree.splitlines():
        if '"光路分步标定 ·' not in line:continue
        wid=re.search(r'0x[0-9a-fA-F]+',line)[0]
        props=subprocess.check_output(['xprop','-id',wid,'_NET_WM_PID','WM_CLASS'],text=True)
        match=re.search(r'_NET_WM_PID\(CARDINAL\) = (\d+)',props)
        if match and int(match[1])==pid and 'ballplate-calibration-workbench' in props:candidates.append(wid)
    if len(candidates)!=1:raise RuntimeError('expected exactly one owned calibration control window')
    wid=candidates[0]
    reject_owned_dialogs(tree,wid)
    info=subprocess.check_output(['xwininfo','-id',wid],text=True)
    def value(name):return int(re.search(r'(?m)^\s*'+name+r':\s*(-?\d+)',info)[1])
    x,y,w,h=(value(k) for k in ('Absolute upper-left X','Absolute upper-left Y','Width','Height'))
    if not (1080<=x and 600<=y and x+w<=3640 and y+h<=2040):
        raise RuntimeError('control window must be entirely on the verified ordinary primary monitor')
    return int(wid,16),(x,y,w,h)


class Input:
    def __init__(self,wid):
        self.x=C.CDLL('libX11.so.6');self.t=C.CDLL('libXtst.so.6')
        self.x.XOpenDisplay.argtypes=[C.c_char_p];self.x.XOpenDisplay.restype=C.c_void_p
        self.display=self.x.XOpenDisplay(None)
        if not self.display:raise RuntimeError('no local X display')
        for name in ('XRaiseWindow','XSetInputFocus','XQueryTree','XSync','XFlush','XCloseDisplay','XKeysymToKeycode'):
            getattr(self.x,name).restype=C.c_int
        self.x.XRaiseWindow.argtypes=[C.c_void_p,C.c_ulong]
        self.x.XSetInputFocus.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_ulong]
        self.x.XQueryTree.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_ulong),C.POINTER(C.c_ulong),C.POINTER(C.c_void_p),C.POINTER(C.c_uint)]
        self.x.XSync.argtypes=[C.c_void_p,C.c_int];self.x.XFlush.argtypes=[C.c_void_p]
        self.x.XCloseDisplay.argtypes=[C.c_void_p];self.x.XKeysymToKeycode.argtypes=[C.c_void_p,C.c_ulong]
        self.x.XFree.argtypes=[C.c_void_p]
        self.t.XTestFakeMotionEvent.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_ulong]
        self.t.XTestFakeButtonEvent.argtypes=[C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
        self.t.XTestFakeKeyEvent.argtypes=[C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
        root=C.c_ulong();parent=C.c_ulong();children=C.c_void_p();count=C.c_uint()
        self.x.XQueryTree(self.display,wid,C.byref(root),C.byref(parent),C.byref(children),C.byref(count))
        if children:self.x.XFree(children)
        class Data(C.Union):
            _fields_=[('longs',C.c_long*5),('bytes',C.c_char*20)]
        class Message(C.Structure):
            _fields_=[('type',C.c_int),('serial',C.c_ulong),('send_event',C.c_int),('display',C.c_void_p),
                      ('window',C.c_ulong),('message_type',C.c_ulong),('format',C.c_int),('data',Data)]
        class Event(C.Union):
            _fields_=[('message',Message),('pad',C.c_long*24)]
        self.x.XInternAtom.argtypes=[C.c_void_p,C.c_char_p,C.c_int];self.x.XInternAtom.restype=C.c_ulong
        self.x.XSendEvent.argtypes=[C.c_void_p,C.c_ulong,C.c_int,C.c_long,C.POINTER(Event)]
        event=Event();event.message.type=33;event.message.send_event=1;event.message.display=self.display
        event.message.window=wid;event.message.message_type=self.x.XInternAtom(self.display,b'_NET_ACTIVE_WINDOW',False)
        event.message.format=32;event.message.data.longs[0]=2
        self.x.XRaiseWindow(self.display,parent.value if parent.value!=root.value else wid)
        self.x.XSendEvent(self.display,root.value,False,(1<<19)|(1<<20),C.byref(event))
        self.x.XSetInputFocus(self.display,wid,2,0);self.x.XSync(self.display,False)
        time.sleep(.3)

    def click(self,x,y):
        self.t.XTestFakeMotionEvent(self.display,-1,x,y,0)
        self.t.XTestFakeButtonEvent(self.display,1,True,0)
        self.t.XTestFakeButtonEvent(self.display,1,False,0)
        self.x.XSync(self.display,False)

    def key(self,keysym,down):
        code=self.x.XKeysymToKeycode(self.display,keysym)
        if not code:raise RuntimeError('missing keyboard keycode')
        self.t.XTestFakeKeyEvent(self.display,code,down,0);self.x.XFlush(self.display)

    def chord(self,keysym):
        self.key(0xffe3,True);self.key(keysym,True);self.key(keysym,False);self.key(0xffe3,False)

    def close(self):self.x.XCloseDisplay(self.display)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session',type=Path)
    parser.add_argument('--click',nargs=2,type=int,metavar=('CLIENT_X','CLIENT_Y'))
    parser.add_argument('--paste')
    args=parser.parse_args()
    wid,(x,y,w,h)=find_window(args.session)
    client=Input(wid)
    try:
        if args.click:
            cx,cy=args.click
            if not (0<=cx<w and 0<=cy<h):raise ValueError('click outside calibration controls')
            client.click(x+cx,y+cy)
        if args.paste is not None:
            from PyQt5 import QtCore,QtWidgets
            app=QtWidgets.QApplication(['calibration-local-text-input'])
            clipboard=app.clipboard();old=clipboard.mimeData();saved=QtCore.QMimeData()
            if old:
                for name in old.formats():saved.setData(name,old.data(name))
            clipboard.setText(args.paste);app.processEvents();client.chord(ord('a'));client.chord(ord('v'))
            for _ in range(15):app.processEvents();time.sleep(.02)
            clipboard.setMimeData(saved)
            for _ in range(10):app.processEvents();time.sleep(.02)
        print(json.dumps(dict(window=hex(wid),control_geometry=[x,y,w,h],action='text' if args.paste else 'click' if args.click else 'focus')))
    finally:client.close()


if __name__=='__main__':main()
