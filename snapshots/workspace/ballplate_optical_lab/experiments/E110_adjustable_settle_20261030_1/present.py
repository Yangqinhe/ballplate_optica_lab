"""Show the adjustable-wait public tab; never start projection or control."""
import argparse
from types import SimpleNamespace as NS
import time
import run
from run import c,ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('shielding confirmation required')
    cfg=c.read(ROOT/'CONFIG.json');proof=c.read(ROOT/'POST_STARTUP_01/RESULT.json');pid=proof['pid']
    base=run.base_module();base.load('e110_present_display',base.DISPLAY/'run.py').install_display(c.read(c.checked(cfg['display_profile'])))
    from optical_lab import legacy_gui
    client=legacy_gui._load_clients(base.LEGACY);legacy_gui._install_accessibility_compatibility();client.install()
    base.load('e110_present_compat',c.EXP/'E067_three_slm_20261026_1/client_compat.py').install()
    n=base.load('e110_present_native',c.EXP/'E029_native_crash_recovery_20261024_1/native_buttons.py');n.ORDINARY=[1080,1080,2560,1440]
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    from black_pixels import RootPixels
    def state():
        h=c.read(ROOT/'GUI/heartbeat.json')
        if h['pid']!=pid or not 0<=time.time_ns()-h['unix_ns']<3e9 or h['hardware_fault'] or h['projection_armed']:
            raise ValueError('fresh zero owner required')
        if h['camera_active_readback']!=cfg['expected_active'] or not h['geometry_frozen']:raise ValueError('context differs')
        return h
    before=state();app,A=accessible(pid);runtime=n._runtime();main=proof['windows']['control'];wid=n._wid(main['wid'])
    tabs=[v for v in safe_nodes(app) if v.get_role_name()=='page tab' and v.get_name()=='光路控制对比']
    if len(tabs)!=1:raise ValueError('unique paired tab required')
    node=tabs[0];rect=list(runtime.extents(node,A))
    if node.get_process_id()!=pid or not runtime.inside(rect,main['rect']) or not node.get_state_set().contains(A.StateType.SHOWING):raise ValueError('tab outside owner')
    n._same_window(runtime,wid,pid,main['rect']);n._reject_other_dialogs(pid,wid)
    focus=runtime.Input(wid);pointer=None
    try:
        if n._active_window()!=wid or list(runtime.extents(node,A))!=rect:raise ValueError('focus/tab changed')
        tree=n._root_child(focus,wid);pointer=runtime.Pointer()
        if type(pointer).__name__!='NumLockPointer':raise ValueError('unexpected pointer')
        state();pointer.click((rect[0]+rect[2]//2,rect[1]+rect[3]//2),dict(wid=tree['pointer_root_child_wid']))
        deadline=time.monotonic()+3
        while state()['current_page']!=2:
            if time.monotonic()>deadline:raise ValueError('paired tab not selected')
            time.sleep(.1)
        pixels=RootPixels()
        try:zeros={k:pixels.inspect(tuple(proof['windows'][k]['rect'])) for k in ('dmd','slm')}
        finally:pixels.close()
        if not all(v['all_RGB_zero'] for v in zeros.values()):raise ValueError('outputs not zero')
        app,A=accessible(pid)
        waits=[v for v in safe_nodes(app) if v.get_role_name()=='spin button' and v.get_name()=='换图稳定等待（毫秒）']
        if len(waits)!=1 or not waits[0].get_state_set().contains(A.StateType.SHOWING):raise ValueError('wait control missing/hidden')
        q=waits[0].get_value_iface()
        wait_control=dict(minimum=q.get_minimum_value(),maximum=q.get_maximum_value(),value=q.get_current_value())
        if wait_control!=dict(minimum=10.,maximum=1000.,value=1000.):raise ValueError('wait control differs')
        out=ROOT/'LIVE_PAGE_01';out.mkdir()
        from PIL import ImageGrab
        x,y,w,h=main['rect'];time.sleep(.2)
        ImageGrab.grab(bbox=(x,y,x+w,y+h),xdisplay=':1').save(out/'page.png')
        c.write(out/'RESULT.json',dict(passed=True,before=before,after=state(),zeros=zeros,wait_control=wait_control,
            ordinary_tab_click_only=True,no_projection=True,no_control=True,source=c.ref(__file__)))
    finally:
        if pointer is not None:pointer.close()
        focus.close()
    print('Corrected tab presented; retained context and actual zero outputs PASS',flush=True)


if __name__=='__main__':main()
