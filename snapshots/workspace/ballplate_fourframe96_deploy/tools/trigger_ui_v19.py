"""Read-only v19 verification and visible named buttons; NO SDK/projection."""
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
    if runtime.get('ui_version') != 19 or state['pid'] != runtime['pid']:
        raise RuntimeError('not the unique v19 owner')
    if state['hardware_fault'] or state['projection_armed']:
        raise RuntimeError('only fault-free DISARMED camera-only checks allowed')
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


def action(session, key, execute=False):
    result = inspect(session, parked=key == 'ccd')
    state, main = result['heartbeat'], result['windows']['control']
    if key in ('snapshot', 'exposure') and (not state['live_camera'] or not state.get('latest_frame')
                                           or state['latest_frame']['age_seconds'] > 1):
        raise RuntimeError('fresh running CCD required')
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi
    desktop = Atspi.get_desktop(0)
    apps = [desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    apps = [app for app in apps if app.get_process_id() == state['pid'] and app.get_name() == 'ballplate-realtime-manual']
    frames = [] if len(apps) != 1 else [node for node in nodes(apps[0])
        if node.get_role_name() == 'frame' and node.get_name().startswith('实时光路对齐 v19 ')]
    if len(frames) != 1:
        raise RuntimeError('ambiguous accessible v19 main')
    controls = [node for node in nodes(frames[0]) if node.get_role_name() == 'push button' and node.get_name() == NAMES[key]]
    if len(controls) != 1:
        raise RuntimeError('named visible button missing/ambiguous')
    button = controls[0]
    for flag in (Atspi.StateType.ENABLED, Atspi.StateType.SENSITIVE, Atspi.StateType.SHOWING):
        if not button.get_state_set().contains(flag):
            raise RuntimeError('button disabled/not visible')
    if not inside(extents(button, Atspi), main['rect']):
        raise RuntimeError('button outside original main')
    interface = button.get_action_iface()
    indices = [i for i in range(interface.get_n_actions()) if interface.get_action_name(i) == 'Press']
    if len(indices) != 1 or (execute and not interface.do_action(indices[0])):
        raise RuntimeError('visible button action not accepted')
    return dict(pid=state['pid'], key=key, executed=execute, no_SDK=True, no_projection=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--save', type=Path)
    p.add_argument('--parked', action='store_true')
    p.add_argument('--action', choices=NAMES)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    result = action(args.session, args.action, args.execute) if args.action else inspect(args.session, args.parked)
    if args.save:
        with args.save.open('x') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    if args.action:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps(dict(pid=result['pid'], windows=result['windows'],
            source_files=len(result['runtime']['sources']), parked=result['parked_verification_passed']), ensure_ascii=False))


if __name__ == '__main__':
    main()
