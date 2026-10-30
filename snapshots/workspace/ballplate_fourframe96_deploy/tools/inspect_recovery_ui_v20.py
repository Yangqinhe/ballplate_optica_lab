"""v20 stopped-fault pixel/source inspection and original strict recheck button ONLY."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'live_manual_v17'))
from black_pixels import RootPixels
from display_linux import read_roles, same_signals
from select_verified_mode_v17 import inside, window, extents
from static_ui_v18 import nodes

ORDINARY = [1080, 600, 2560, 1440]
NAMES = dict(ccd='仅CCD（硬触发）', release='停采并恢复原参数',
             exposure='应用曝光', snapshot='保存当前对照', allzero='两路全0，CCD保持（Esc）')


def inspect(session, parked=False):
    session = Path(session)
    state = json.loads((session/'heartbeat.json').read_text())
    runtime = json.loads((session/'runtime_sources.json').read_text())
    if not 0 <= time.time_ns()-state['unix_ns'] <= 10_000_000_000:
        raise RuntimeError('stale/future owner heartbeat')
    if runtime.get('ui_version') != 19 or runtime.get('trigger_revision') != 6 or state['pid'] != runtime['pid']:
        raise RuntimeError('not the unique v19 owner')
    if state['projection_armed']:
        raise RuntimeError('DISARMED stopped outputs required')
    if parked and (state['busy'] or state['live_camera']):
        raise RuntimeError('verification requires stopped/released camera')
    for group in ('sources', 'source_configs'):
        for name, sha in runtime[group].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('loaded source/config changed: '+name)
    created = next(json.loads(line) for line in (session/'events.jsonl').read_text().splitlines()
                   if json.loads(line).get('state') == 'created')
    roles = read_roles(); same_signals(created['video'], roles)
    candidates = []
    for line in subprocess.check_output(['xwininfo', '-root', '-tree'], text=True).splitlines():
        if '"ballplate-realtime-manual"' not in line and '"实时光路对齐 v19 ' not in line:
            continue
        match = re.search(r'0x[0-9a-fA-F]+', line)
        if match:
            row = window(match[0])
            if row['pid'] == state['pid'] and row['viewable']:
                row['control'] = '"实时光路对齐 v19 ' in line
                candidates.append(row)
    main = [row for row in candidates if row['control']]
    if len(main) != 1 or not inside(main[0]['rect'], ORDINARY):
        raise RuntimeError('expected one PID-owned main window entirely on ordinary monitor')
    windows = dict(control=main[0])
    pixels = RootPixels()
    try:
        for role, row in roles.items():
            rectangle = row['desktop_position']+row['desktop_size']
            owned = [w for w in candidates if not w['control'] and w['rect'] == rectangle]
            if len(owned) != 1:
                raise RuntimeError('missing/moved/duplicate owned output: '+role)
            checked = pixels.inspect(tuple(rectangle))
            if not checked['all_RGB_zero']:
                raise RuntimeError('output not ACTUALLY all-zero: '+role)
            windows[role] = dict(owned[0], actual_pixels=checked)
    finally:
        pixels.close()
    return dict(pid=state['pid'], version=19, roles=roles, runtime=runtime, heartbeat=state,
        windows=windows, no_SDK_connection=True, no_projection=True,
        parked_verification_passed=bool(parked), release_approved=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',required=True,type=Path)
    p.add_argument('--save',type=Path)
    p.add_argument('--execute-recheck',action='store_true')
    a=p.parse_args();verified=inspect(a.session,parked=True)
    if a.save:
        with a.save.open('x') as f:json.dump(verified,f,ensure_ascii=False,indent=2)
    if a.execute_recheck:
        if not verified['heartbeat']['hardware_fault']:raise RuntimeError('no fault to recheck')
        import gi
        gi.require_version('Atspi','2.0');from gi.repository import Atspi
        d=Atspi.get_desktop(0);buttons=[]
        for i in range(d.get_child_count()):
            app=d.get_child_at_index(i)
            if app.get_process_id()!=verified['pid']:continue
            buttons += [n for n in nodes(app) if n.get_role_name()=='push button' and n.get_name()=='复核并恢复（不投图）']
        if len(buttons)!=1:raise RuntimeError('unique recovery button required')
        button=buttons[0]
        for flag in (Atspi.StateType.SENSITIVE,Atspi.StateType.ENABLED,Atspi.StateType.SHOWING):
            if not button.get_state_set().contains(flag):raise RuntimeError('recovery button disabled/hidden')
        if not inside(extents(button,Atspi),verified['windows']['control']['rect']):raise RuntimeError('button outside original main')
        act=button.get_action_iface()
        ids=[i for i in range(act.get_n_actions()) if act.get_action_name(i)=='Press']
        if len(ids)!=1 or not act.do_action(ids[0]):raise RuntimeError('recheck action rejected')
    print(json.dumps(dict(pid=verified['pid'],all_zero=True,stopped_before_action=True,
        fault=verified['heartbeat']['hardware_fault'],recheck_requested=a.execute_recheck,no_SDK=True,no_projection=True),ensure_ascii=False))

if __name__=='__main__':main()
