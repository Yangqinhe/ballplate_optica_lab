"""Select the new public tab and inspect retained state; no projection."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import core as c
import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--light-blocked-confirmed',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('shielding confirmation required')
    proof=c.read(c.ROOT/'POST_STARTUP_01/RESULT.json');cfg=c.read(c.ROOT/'CONFIG.json');pid=proof['pid']
    base=run.base_module();base.load('e107_display_client',base.DISPLAY/'run.py').install_display(c.read(c.checked(cfg['display_profile'])))
    from optical_lab import legacy_gui
    client=legacy_gui._load_clients(base.LEGACY);legacy_gui._install_accessibility_compatibility();client.install()
    base.load('e107_accessible_client',c.EXP/'E067_three_slm_20261026_1/client_compat.py').install()
    n=base.load('e107_tab_native',c.EXP/'E029_native_crash_recovery_20261024_1/native_buttons.py');n.ORDINARY=[1080,1080,2560,1440]
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    from black_pixels import RootPixels
    def state():
        h=c.read(c.ROOT/'GUI/heartbeat.json')
        if h['pid']!=pid or not 0<=time.time_ns()-h['unix_ns']<3e9 or h['hardware_fault'] or h['projection_armed']:
            raise ValueError('owner not fresh/zero/error-free')
        if h['camera_active_readback']!=cfg['expected_active'] or not h['geometry_frozen']:raise ValueError('retained camera/geometry differs')
        return h
    before=state();main=proof['windows']['control'];wid=n._wid(main['wid']);runtime=n._runtime()
    app,A=accessible(pid);tabs=[v for v in safe_nodes(app) if v.get_role_name()=='page tab']
    if a.audit_only:
        from PIL import ImageGrab
        if [v.get_name() for v in tabs]!=['主界面 · 光路与参数','读出稳定性测试','光路控制对比']:
            raise ValueError('three expected tabs required')
        n._same_window(runtime,wid,pid,main['rect'])
        names=['开始／继续','单步','暂停／全0','中心复位','同位置随机复位','同时扰动']
        visible={}
        for v in safe_nodes(app):
            if v.get_role_name()!='push button' or v.get_name() not in names:continue
            v.clear_cache()
            if v.get_state_set().contains(A.StateType.SHOWING):
                rect=list(runtime.extents(v,A))
                if v.get_process_id()!=pid or not runtime.inside(rect,main['rect']):raise ValueError('control ownership mismatch')
                visible[v.get_name()]=rect
        if set(visible)!=set(names):raise ValueError('new page controls not all visible')
        pixels=RootPixels()
        try:
            zeros={role:pixels.inspect(tuple(proof['windows'][role]['rect'])) for role in ('dmd','slm')}
            if not all(v['all_RGB_zero'] for v in zeros.values()):raise ValueError('outputs not zero')
        finally:pixels.close()
        after=state()
        out=c.ROOT/'LIVE_PAGE_CHECK_03';out.mkdir()
        n._reject_other_dialogs(pid,wid)
        focus=runtime.Input(wid)
        try:
            time.sleep(.2)
            if n._active_window()!=wid:raise ValueError('ordinary control window not foreground')
            n._same_window(runtime,wid,pid,main['rect']);state()
            x,y,w,h=main['rect'];ImageGrab.grab(bbox=(x,y,x+w,y+h),xdisplay=':1').save(out/'live_page.png')
        finally:focus.close()
        data=[dict(name=v.get_name(),selected=v.get_state_set().contains(A.StateType.SELECTED),rect=list(runtime.extents(v,A))) for v in tabs]
        c.write(out/'RESULT.json',dict(passed=True,tabs=data,before=before,after=after,hardware_actions=0,
            visible_page_controls=visible,zeros=zeros,no_control_started=True,no_projection=True,source=c.ref(__file__),
            ordinary_window_focused_only=True,earlier_screenshot='LIVE_PAGE_CHECK_02 screenshot was occluded by VS Code; retained as recorded.',
            earlier_assertion='Qt exposes SELECTED=false for every tab; initial selection assertion failed. Visible page controls and screenshot verified independently.'))
        print('Read-only live page audit PASS; all six controls visible and both physical display outputs zero',flush=True);return
    if [v.get_name() for v in tabs]!=['主界面 · 光路与参数','读出稳定性测试','光路控制对比']:
        raise ValueError('three expected tabs required')
    node=tabs[2];node.clear_cache()
    if node.get_process_id()!=pid or not all(node.get_state_set().contains(k) for k in (A.StateType.SHOWING,A.StateType.ENABLED,A.StateType.SENSITIVE)):
        raise ValueError('tab not visible/enabled')
    rect=list(runtime.extents(node,A));n._same_window(runtime,wid,pid,main['rect'])
    if not runtime.inside(rect,main['rect']):raise ValueError('tab outside ordinary window')
    n._reject_other_dialogs(pid,wid);focus=runtime.Input(wid);pointer=None
    try:
        if n._active_window()!=wid:raise ValueError('focus changed')
        if list(runtime.extents(node,A))!=rect:raise ValueError('tab moved')
        tree=n._root_child(focus,wid);state();pointer=runtime.Pointer()
        if type(pointer).__name__!='NumLockPointer':raise ValueError('unexpected pointer implementation')
        pointer.click((rect[0]+rect[2]//2,rect[1]+rect[3]//2),dict(wid=tree['pointer_root_child_wid']))
    finally:
        if pointer is not None:pointer.close()
        focus.close()
    time.sleep(.3);node.clear_cache()
    if not node.get_state_set().contains(A.StateType.SELECTED):raise ValueError('new tab not selected')
    pixels=RootPixels()
    try:
        zeros={role:pixels.inspect(tuple(proof['windows'][role]['rect'])) for role in ('dmd','slm')}
        if not all(v['all_RGB_zero'] for v in zeros.values()):raise ValueError('outputs not zero')
    finally:pixels.close()
    out=c.ROOT/'LIVE_PAGE_CHECK_01';out.mkdir()
    from PIL import ImageGrab
    x,y,w,h=main['rect'];ImageGrab.grab(bbox=(x,y,x+w,y+h),xdisplay=':1').save(out/'live_page.png')
    c.write(out/'RESULT.json',dict(passed=True,before=before,after=state(),tabs=[v.get_name() for v in tabs],
        selected='光路控制对比',zeros=zeros,public_tab_click_only=True,no_control_started=True,
        no_projection=True,source=c.ref(__file__)))
    print('New live tab selected; retained configuration/geometry and actual zero outputs PASS',flush=True)

if __name__=='__main__':main()
