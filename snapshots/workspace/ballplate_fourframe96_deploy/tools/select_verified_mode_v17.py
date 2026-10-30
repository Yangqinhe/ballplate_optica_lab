"""Select a v17 mode through its real, PID-owned native combo popup.

Never trust hidden AT-SPI list-item SHOWING/rectangles. Never activate the main
window after opening a popup: that can dismiss its mouse grab. No SDK, output
windows, in-process injection, file loading, or projection permission changes.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import re
import subprocess
import sys
import time


OPTIONS={('dmd','black'):'全黑',('dmd','file'):'载入的二值图',
         ('slm','zero'):'零相位',('slm','file'):'载入灰度图'}


def check_state(state,now_ns):
    if (now_ns-state['unix_ns']>10_000_000_000
            or state['projection_armed'] or state['hardware_fault']
            or state.get('spatial_calibration',{}).get('busy')):
        raise RuntimeError('fresh disarmed, fault-free, non-spatial-scan state required')
    # v17 busy is literally worker.isRunning(), including normal free-running
    # CCD. Reject starting/stopping transitions, not a fresh disarmed stream.
    if state['busy'] and not state['live_camera']:
        raise RuntimeError('camera starting/stopping transition')
    if state['live_camera'] and (not state.get('latest_frame') or state['latest_frame']['age_seconds']>=1.5):
        raise RuntimeError('live CCD is stale/not ready')
    return state


def read_state(session):return check_state(json.loads((session/'heartbeat.json').read_text()),time.time_ns())


def inside(rect,outer):
    x,y,w,h=rect;l,t,ow,oh=outer
    return w>0 and h>0 and l<=x and t<=y and x+w<=l+ow and y+h<=t+oh


def target_point(item,popup):
    """Qt may report an unclipped item width; click only its visible intersection."""
    x,y,w,h=item;l,t,pw,ph=popup
    left=max(x,l)+2;right=min(x+w,l+pw)-2
    top=max(y,t)+2;bottom=min(y+h,t+ph)-2
    if right<=left or bottom<=top or h<=0 or h>ph:
        raise RuntimeError('item has no safe intersection with actual native popup')
    return ((left+right)//2,(top+bottom)//2)


def window(wid):
    info=subprocess.check_output(['xwininfo','-id',wid],text=True)
    def value(k):return int(re.search(r'(?m)^\s*'+k+r':\s*(-?\d+)',info)[1])
    props=subprocess.check_output(['xprop','-id',wid,'_NET_WM_PID','WM_TRANSIENT_FOR','_NET_WM_WINDOW_TYPE'],text=True)
    owner=re.search(r'_NET_WM_PID\(CARDINAL\) = (\d+)',props)
    parent=re.search(r'WM_TRANSIENT_FOR\(WINDOW\): window id # (0x[0-9a-fA-F]+)',props)
    return dict(wid=wid,pid=int(owner[1]) if owner else None,
        parent=int(parent[1],16) if parent else None,
        rect=[value(k) for k in ('Absolute upper-left X','Absolute upper-left Y','Width','Height')],
        viewable='Map State: IsViewable' in info,combo='_NET_WM_WINDOW_TYPE_COMBO' in props)


def owned_popups(pid,parent):
    tree=subprocess.check_output(['xwininfo','-root','-tree'],text=True);found=[]
    for line in tree.splitlines():
        if '"ballplate-realtime-manual"' not in line:continue
        wid=re.search(r'0x[0-9a-fA-F]+',line)[0];row=window(wid)
        if row['viewable'] and row['combo'] and row['pid']==pid and row['parent']==int(parent,16):
            if not inside(row['rect'],[1080,600,2560,1440]):raise RuntimeError('popup outside ordinary monitor')
            found.append(row)
    if len(found)>1:raise RuntimeError('ambiguous native combo popups')
    return found


def locate(pid,role):
    import gi
    gi.require_version('Atspi','2.0')
    from gi.repository import Atspi
    from collections import deque
    desktop=Atspi.get_desktop(0)
    apps=[desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    apps=[n for n in apps if n.get_process_id()==pid and n.get_name()=='ballplate-realtime-manual']
    if len(apps)!=1:raise RuntimeError('unexpected accessible owner')
    app=apps[0];queue=deque([app]);found=[]
    while queue:
        n=queue.popleft()
        if n.get_role_name()=='combo box':
            ancestor=n.get_parent()
            while ancestor and ancestor!=app:
                if ancestor.get_role_name()=='panel' and ancestor.get_name().startswith(role.upper()+' ·'):
                    found.append(n);break
                ancestor=ancestor.get_parent()
        for i in range(n.get_child_count()):queue.append(n.get_child_at_index(i))
    if len(found)!=1:raise RuntimeError('missing/ambiguous mode combo')
    return found[0],Atspi


def extents(node,Atspi):
    node.clear_cache();r=node.get_component_iface().get_extents(Atspi.CoordType.SCREEN)
    return [r.x,r.y,r.width,r.height]


class Pointer:
    """XTEST pointer only; intentionally does not raise/focus any window."""
    def __init__(self):
        self.x=C.CDLL('libX11.so.6');self.t=C.CDLL('libXtst.so.6')
        self.x.XOpenDisplay.argtypes=[C.c_char_p];self.x.XOpenDisplay.restype=C.c_void_p
        self.d=self.x.XOpenDisplay(None)
        if not self.d:raise RuntimeError('no X display')
        self.x.XDefaultRootWindow.argtypes=[C.c_void_p];self.x.XDefaultRootWindow.restype=C.c_ulong
        self.root=self.x.XDefaultRootWindow(self.d)
        self.x.XSync.argtypes=[C.c_void_p,C.c_int];self.x.XCloseDisplay.argtypes=[C.c_void_p]
        self.t.XTestFakeMotionEvent.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_ulong]
        self.t.XTestFakeButtonEvent.argtypes=[C.c_void_p,C.c_uint,C.c_int,C.c_ulong]
        self.x.XQueryPointer.argtypes=[C.c_void_p,C.c_ulong,C.POINTER(C.c_ulong),C.POINTER(C.c_ulong),
            C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_int),C.POINTER(C.c_uint)]

    def click(self,point,popup):
        x,y=point
        if not self.t.XTestFakeMotionEvent(self.d,-1,x,y,0):raise RuntimeError('pointer move rejected')
        self.x.XSync(self.d,False);time.sleep(.12)
        root=C.c_ulong();child=C.c_ulong();rx=C.c_int();ry=C.c_int();wx=C.c_int();wy=C.c_int();mask=C.c_uint()
        if not self.x.XQueryPointer(self.d,self.root,C.byref(root),C.byref(child),C.byref(rx),C.byref(ry),C.byref(wx),C.byref(wy),C.byref(mask)):
            raise RuntimeError('pointer query failed')
        if child.value!=int(popup['wid'],16) or (rx.value,ry.value)!=(x,y) or mask.value:
            raise RuntimeError('actual topmost pointer window/coordinates/modifiers differ')
        if not self.t.XTestFakeButtonEvent(self.d,1,True,0):raise RuntimeError('mouse press rejected')
        try:self.x.XSync(self.d,False);time.sleep(.08)
        finally:self.t.XTestFakeButtonEvent(self.d,1,False,0);self.x.XSync(self.d,False)

    def close(self):self.x.XCloseDisplay(self.d)


def select(session,role,mode,execute=False):
    state=read_state(session);pid=state['pid'];startup=json.loads((session/'STARTUP_VERIFICATION.json').read_text())
    wid=startup['windows']['control']['wid'];main=window(wid)
    if main['pid']!=pid or not main['viewable'] or not inside(main['rect'],[1080,600,2560,1440]):
        raise RuntimeError('main owner/visibility/ordinary-monitor bounds differ')
    combo,Atspi=locate(pid,role);rectangle=extents(combo,Atspi)
    if not inside(rectangle,main['rect']):raise RuntimeError('combo not actually within main window')
    for flag in (Atspi.StateType.ENABLED,Atspi.StateType.SENSITIVE,Atspi.StateType.SHOWING):
        if not combo.get_state_set().contains(flag):raise RuntimeError('combo disabled/not showing')
    name=OPTIONS[(role,mode)]
    result=dict(role=role,mode=mode,option=name,executed=execute,pid=pid,combo_rect=rectangle,
        native_popup_verified=False,heartbeat_committed=False,no_projection_or_camera_commands=True)
    if not execute:return result
    # This method never tabs through unrelated controls and never uses hidden-item
    # coordinates as evidence that a popup exists.
    popups=owned_popups(pid,wid)
    if not popups:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'live_manual_v17'))
        from calibration_desktop import Input
        focus=Input(int(wid,16));focus.close()
        active=subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True)
        if int(re.search(r'0x[0-9a-fA-F]+',active)[0],16)!=int(wid,16):raise RuntimeError('main not foreground')
        action=combo.get_action_iface()
        methods=[i for i in range(action.get_n_actions()) if action.get_action_name(i)=='ShowMenu']
        if len(methods)!=1 or not action.do_action(methods[0]):raise RuntimeError('show menu rejected')
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            read_state(session);popups=owned_popups(pid,wid)
            if popups:break
            time.sleep(.1)
        if not popups:raise RuntimeError('no actual visible owned combo popup')
    popup=popups[0]
    if not (popup['rect'][0]<=rectangle[0]+rectangle[2]//2<popup['rect'][0]+popup['rect'][2]
            and popup['rect'][1]<rectangle[1]+rectangle[3] and rectangle[1]<popup['rect'][1]+popup['rect'][3]):
        raise RuntimeError('visible popup is not attached to requested combo')
    combo,Atspi=locate(pid,role)
    items=[child.get_child_at_index(j) for i in range(combo.get_child_count())
           for child in [combo.get_child_at_index(i)] for j in range(child.get_child_count())]
    matches=[n for n in items if n.get_name()==name]
    if len(matches)!=1:raise RuntimeError('missing/ambiguous option')
    item=extents(matches[0],Atspi);point=target_point(item,popup['rect'])
    read_state(session);pointer=Pointer()
    try:pointer.click(point,popup)
    finally:pointer.close()
    result.update(native_popup_verified=True,popup=popup,item_rect=item,click_point=point)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        current=read_state(session)
        if current['pid']!=pid:raise RuntimeError('owner changed')
        if current['manual_settings'][role]==mode and not owned_popups(pid,wid):
            result.update(heartbeat_committed=True,display_revision=current['display_revision']);return result
        time.sleep(.15)
    raise RuntimeError('requested mode did not commit with popup dismissed')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True,type=Path);p.add_argument('--role',required=True,choices=('slm','dmd'))
    p.add_argument('--mode',required=True,choices=('zero','file','black'));p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if (a.role,a.mode) not in OPTIONS:raise ValueError('unsupported mode')
    print(json.dumps(select(a.session,a.role,a.mode,a.execute),ensure_ascii=False))


if __name__=='__main__':main()
