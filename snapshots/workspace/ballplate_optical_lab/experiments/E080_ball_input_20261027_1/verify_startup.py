"""Public-button, shielded CCD-only startup check. No independent SDK owner."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
from types import SimpleNamespace
from run import ROOT,LAB,DISPLAY,read,write,sha,load

TITLE='光路工作台 · 四球尺寸/填孔 · 共用滤波'


def main():
    p=argparse.ArgumentParser();p.add_argument('--light-blocked-confirmed',action='store_true');a=p.parse_args()
    if not a.light_blocked_confirmed:raise ValueError('current shielding required')
    cfg=read(ROOT/'CONFIG.json');session=ROOT/'GUI';h=read(session/'heartbeat.json');pid=h['pid']
    proc=Path('/proc',str(pid));start=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    command=read(ROOT/'LAUNCH_PLAN.json')['command']
    if [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]!=command[command.index('--')+1:]:raise ValueError('owner command mismatch')
    for path,digest in cfg['source_sha256'].items():
        if sha(path)!=digest:raise ValueError('source changed '+path)
    out=ROOT/'POST_STARTUP_01';out.mkdir()
    load('e078_verify_display',DISPLAY/'run.py').install_display(read(cfg['display_profile']['path']))
    from optical_lab import legacy_gui
    c=legacy_gui._load_clients(LAB.parent/'ballplate_fourframe96_deploy');legacy_gui._install_accessibility_compatibility();c.install()
    load('e078_accessible_compat',LAB/'experiments/E067_three_slm_20261026_1/client_compat.py').install()
    n=load('e078_native_click',LAB/'experiments/E029_native_crash_recovery_20261024_1/native_buttons.py');n.ORDINARY=[1080,1080,2560,1440]
    from black_pixels import RootPixels
    from night_opt_20261020.manual_io_v17 import accessible,safe_nodes
    runtime=read(session/'runtime_sources.json')
    for group in ('sources','source_configs'):
        for path,digest in runtime[group].items():
            if sha(path)!=digest:raise ValueError('runtime source changed '+path)
    created=next(v for line in (session/'events.jsonl').read_text().splitlines() if (v:=json.loads(line)).get('state')=='created')
    roles=c.read_roles();c.same_signals(created['video'],roles)
    candidates=[]
    for line in subprocess.check_output(['xwininfo','-root','-tree'],text=True).splitlines():
        if '"'+TITLE not in line and '"ballplate-realtime-manual"' not in line:continue
        match=re.search(r'0x[0-9a-fA-F]+',line)
        if match:
            row=c.window(match[0])
            if row['pid']==pid and row['viewable']:candidates.append(dict(row,control='"'+TITLE in line))
    mains=[v for v in candidates if v['control']]
    if len(mains)!=1 or not c.inside(mains[0]['rect'],n.ORDINARY):raise ValueError('unique ordinary-screen main required')
    main=mains[0];windows={'control':main}
    for role,info in roles.items():
        rect=info['desktop_position']+info['desktop_size'];matches=[v for v in candidates if not v['control'] and v['rect']==rect]
        if len(matches)!=1:raise ValueError('missing output '+role)
        windows[role]=matches[0]
    def state(allow_fault=False):
        h=read(session/'heartbeat.json')
        if h['pid']!=pid or not 0<=time.time_ns()-h['unix_ns']<3e9 or (proc/'stat').read_text().rsplit(')',1)[1].split()[19]!=start:raise ValueError('owner freshness/lifetime differs')
        if h['projection_armed'] or any(h.get(k) for k in ('loop_active','deployment_busy','adaptation_active')):raise ValueError('must remain disarmed/idle')
        if h['hardware_fault'] and not allow_fault:raise ValueError(h['hardware_fault'])
        return h
    pixels=RootPixels()
    def zeros():
        c.same_signals(roles,c.read_roles());result={}
        for role in ('dmd','slm'):
            now=c.window(windows[role]['wid'])
            if now['pid']!=pid or not now['viewable'] or now['rect']!=windows[role]['rect']:raise ValueError('output window changed')
            result[role]=pixels.inspect(tuple(now['rect']))
            if not result[role]['all_RGB_zero']:raise ValueError('output not actually zero')
        return result
    ui=SimpleNamespace(pid=pid,main=main,state=state)
    def click(key):
        n.check_state(ui,key);app,A=accessible(pid)
        frames=[v for v in safe_nodes(app) if v.get_role_name()=='frame' and v.get_name().startswith(TITLE)]
        if len(frames)!=1:raise ValueError('unique accessibility frame required')
        buttons=[v for v in safe_nodes(frames[0]) if v.get_role_name()=='push button' and v.get_name()==n.NAMES[key]]
        if len(buttons)!=1:raise ValueError('unique visible button required')
        return n.click_visible_node(ui,buttons[0],A,execute=True)
    try:
        write(out/'BEFORE.json',dict(heartbeat=state(True),windows=windows,zeros=zeros()))
        write(out/'RECHECK_CLICK.json',click('recheck'));deadline=time.monotonic()+40
        while True:
            h=state(True)
            if not h['busy'] and h['hardware_fault'] is None:break
            if time.monotonic()>deadline:raise TimeoutError('baseline recheck failed')
            time.sleep(.2)
        write(out/'BASELINE_RECHECK.json',h)
        write(out/'CCD_CLICK.json',click('ccd'));deadline=time.monotonic()+40
        while True:
            h=state()
            if h['live_camera'] and h.get('latest_frame') and h['latest_frame']['age_seconds']<1:break
            if time.monotonic()>deadline:raise TimeoutError('CCD-only start failed')
            time.sleep(.2)
        snapshots=[]
        for i in range(20):
            h=state()
            if h['camera_active_readback']!=cfg['expected_active'] or h['latest_frame']['age_seconds']>=1:raise ValueError('active profile/freshness mismatch')
            snapshots.append(h)
            if i<19:time.sleep(1)
        result=dict(completed=True,pid=pid,start_ticks=start,frame_ids=[snapshots[0]['latest_frame']['frame_id'],snapshots[-1]['latest_frame']['frame_id']],
            snapshots=snapshots,windows=windows,final_zeros=zeros(),no_projection=True,control_qualified=False,
            physical_shielding_confirmed=True,external_SDK_connections=0,source_sha256=sha(__file__))
        write(out/'RESULT.json',result)
        print(json.dumps({k:result[k] for k in ('completed','pid','frame_ids','no_projection')},ensure_ascii=False))
    except BaseException as error:
        write(out/'FAILED.json',dict(completed=False,error=repr(error),no_projection_requested=True));raise
    finally:pixels.close()


if __name__=='__main__':main()
