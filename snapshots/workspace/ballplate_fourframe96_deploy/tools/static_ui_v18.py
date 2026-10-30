"""External, visible-control-only v18 actions. No SDK or device windows.

Existing output-owner code is never patched or injected. Optical permission is
only toggled explicitly; emergency stop is always available by control name.
"""
import argparse
from collections import deque
import ctypes as C
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from select_verified_mode_v17 import (window, inside, locate, extents,
    owned_popups, target_point, Pointer)

PROJECT = Path(__file__).resolve().parents[1]
ORDINARY = [1080, 600, 2560, 1440]
NAMES = {'ccd': '仅CCD', 'snapshot': '保存当前对照',
         'stop': '停止／两路全0（Esc）', 'recheck': '故障只读复核（不投图）',
         'load_dmd': '载入图像', 'load_slm': '载入mask',
         'arm': '允许投图（ND／低功率条件已确认）', 'close': '遮光后退出'}
OPTIONS = {('slm', 'zero'): '固定灰度0', ('slm', 'file'): '载入灰度图',
           ('dmd', 'black'): '全黑', ('dmd', 'file'): '载入的二值图'}


def state(session, allow_fault=False, allow_armed=False):
    s = json.loads((session/'heartbeat.json').read_text())
    if not 0 <= time.time_ns()-s['unix_ns'] < 10_000_000_000:
        raise RuntimeError('stale/future heartbeat')
    if s['hardware_fault'] and not allow_fault:
        raise RuntimeError(s['hardware_fault'])
    if s['projection_armed'] and not allow_armed:
        raise RuntimeError('disarm before operation')
    return s


def owner(session):
    s = state(session, True, True)
    r = json.loads((session/'runtime_sources.json').read_text())
    if r['pid'] != s['pid'] or r.get('ui_version') != 18:
        raise RuntimeError('not the existing v18 owner')
    for group in ('sources', 'source_configs'):
        for name, sha in r[group].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != sha:
                raise RuntimeError('loaded source/config changed: '+name)
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
    found = []
    for line in tree.splitlines():
        if '"实时光路对齐 v18 ' not in line:
            continue
        w = window(re.search(r'0x[0-9a-fA-F]+', line)[0])
        if w['pid'] == s['pid'] and w['viewable']:
            found.append(w)
    if len(found) != 1:
        raise RuntimeError('expected exactly one existing main control window')
    return s, found[0]


def nodes(frame):
    q = deque([frame])
    while q:
        n = q.popleft()
        yield n
        for i in range(n.get_child_count()):
            q.append(n.get_child_at_index(i))


def control(session, key, execute=False, light_blocked=False):
    if key == 'close' and not light_blocked:
        raise RuntimeError('closing requires fresh physical-light-blocked confirmation')
    s, main = owner(session)
    if not inside(main['rect'], ORDINARY):
        raise RuntimeError('main outside qualified ordinary monitor')
    if key not in ('stop', 'recheck') and s['hardware_fault']:
        raise RuntimeError(s['hardware_fault'])
    if key in ('ccd', 'load_dmd', 'load_slm', 'recheck', 'close') and (
            s['busy'] or s['live_camera'] or s['projection_armed']):
        raise RuntimeError('operation requires stopped, disarmed owner')
    if key == 'snapshot' and (not s['live_camera'] or not s.get('latest_frame')
            or s['latest_frame']['age_seconds'] > 1.5):
        raise RuntimeError('snapshot requires fresh live CCD')
    if key == 'arm' and not s['projection_armed'] and (
            not s['live_camera'] or not s.get('latest_frame')
            or s['latest_frame']['age_seconds'] > 1.5):
        raise RuntimeError('arming requires fresh CCD-only stream')
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi
    d = Atspi.get_desktop(0)
    apps = [d.get_child_at_index(i) for i in range(d.get_child_count())]
    apps = [n for n in apps if n.get_name() == 'ballplate-realtime-manual'
            and n.get_process_id() == s['pid']]
    frames = [] if len(apps) != 1 else [n for n in nodes(apps[0])
        if n.get_role_name() == 'frame' and n.get_name().startswith('实时光路对齐 v18 ')]
    if len(frames) != 1:
        raise RuntimeError('ambiguous accessible v18 main')
    role = 'check box' if key == 'arm' else 'push button'
    found = [n for n in nodes(frames[0]) if n.get_role_name() == role and n.get_name() == NAMES[key]]
    if len(found) != 1:
        raise RuntimeError('missing/ambiguous control')
    n = found[0]
    for flag in (Atspi.StateType.ENABLED, Atspi.StateType.SENSITIVE, Atspi.StateType.SHOWING):
        if not n.get_state_set().contains(flag):
            raise RuntimeError('control disabled/not showing')
    if not inside(extents(n, Atspi), main['rect']):
        raise RuntimeError('control outside main')
    a = n.get_action_iface()
    methods = [i for i in range(a.get_n_actions()) if a.get_action_name(i) in ('Press', 'Toggle')]
    if len(methods) != 1:
        raise RuntimeError('ambiguous action')
    if execute and not a.do_action(methods[0]):
        raise RuntimeError('UI action rejected')
    return dict(pid=s['pid'], action=key, name=NAMES[key], executed=execute)


def select(session, role, mode, execute=False):
    s, main = owner(session)
    state(session)
    if not inside(main['rect'], ORDINARY) or (s['busy'] and not s['live_camera']):
        raise RuntimeError('main outside ordinary display or camera transition')
    if s['live_camera'] and (not s.get('latest_frame') or s['latest_frame']['age_seconds'] > 1.5):
        raise RuntimeError('stale live frame')
    combo, Atspi = locate(s['pid'], role)
    rect = extents(combo, Atspi)
    if not inside(rect, main['rect']):
        raise RuntimeError('combo outside main')
    name = OPTIONS[(role, mode)]
    result = dict(role=role, mode=mode, executed=execute, native_popup_verified=False)
    if not execute:
        return result
    if not owned_popups(s['pid'], main['wid']):
        sys.path.insert(0, str(PROJECT/'live_manual_v17'))
        from calibration_desktop import Input
        focus = Input(int(main['wid'], 16)); focus.close()
        action = combo.get_action_iface()
        methods = [i for i in range(action.get_n_actions()) if action.get_action_name(i) == 'ShowMenu']
        if len(methods) != 1 or not action.do_action(methods[0]):
            raise RuntimeError('show menu failed')
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        state(session)
        popups = owned_popups(s['pid'], main['wid'])
        if popups:
            break
        time.sleep(.1)
    if not popups:
        raise RuntimeError('no real owned popup')
    popup = popups[0]
    if not (popup['rect'][0] <= rect[0]+rect[2]//2 < popup['rect'][0]+popup['rect'][2]
            and popup['rect'][1] < rect[1]+rect[3] and rect[1] < popup['rect'][1]+popup['rect'][3]):
        raise RuntimeError('popup not attached to requested combo')
    combo, Atspi = locate(s['pid'], role)
    found = [n for n in nodes(combo) if n.get_name() == name and n.get_role_name() == 'list item']
    if len(found) != 1:
        raise RuntimeError('missing/ambiguous option')
    point = target_point(extents(found[0], Atspi), popup['rect'])
    state(session)
    pointer = Pointer()
    try:
        pointer.click(point, popup)
    finally:
        pointer.close()
    deadline = time.monotonic()+5
    while time.monotonic() < deadline:
        now = state(session)
        if now['pid'] != s['pid']:
            raise RuntimeError('owner changed')
        if now['manual_settings'][role] == mode and not owned_popups(s['pid'], main['wid']):
            return dict(result, native_popup_verified=True, heartbeat_committed=True,
                        display_revision=now['display_revision'])
        time.sleep(.15)
    raise RuntimeError('mode did not commit')


def move_main(session, execute=False):
    s, main = owner(session)
    if s['busy'] or s['live_camera'] or s['projection_armed']:
        raise RuntimeError('only move stopped/disarmed main')
    if not inside(main['rect'], [1080, 2040, 1920, 1200]):
        raise RuntimeError('source is not the observed ordinary DP-3 monitor')
    result = dict(before=main, target=[1150, 664, 1840, 1020], executed=execute,
                  device_windows_moved=False)
    if not execute:
        return result
    # EWMH client messages to this verified ordinary main window only.
    class Data(C.Union):
        _fields_ = [('longs', C.c_long*5), ('bytes', C.c_char*20)]
    class Message(C.Structure):
        _fields_ = [('type', C.c_int), ('serial', C.c_ulong), ('send_event', C.c_int),
            ('display', C.c_void_p), ('window', C.c_ulong), ('message_type', C.c_ulong),
            ('format', C.c_int), ('data', Data)]
    class Event(C.Union):
        _fields_ = [('message', Message), ('pad', C.c_long*24)]
    x = C.CDLL('libX11.so.6')
    x.XOpenDisplay.argtypes = [C.c_char_p]; x.XOpenDisplay.restype = C.c_void_p
    d = x.XOpenDisplay(None)
    if not d:
        raise RuntimeError('no display')
    x.XDefaultRootWindow.argtypes = [C.c_void_p]; x.XDefaultRootWindow.restype = C.c_ulong
    x.XInternAtom.argtypes = [C.c_void_p, C.c_char_p, C.c_int]; x.XInternAtom.restype = C.c_ulong
    x.XSendEvent.argtypes = [C.c_void_p, C.c_ulong, C.c_int, C.c_long, C.POINTER(Event)]
    x.XSync.argtypes = [C.c_void_p, C.c_int]; x.XCloseDisplay.argtypes = [C.c_void_p]
    def send(name, values):
        e = Event(); e.message.type = 33; e.message.send_event = 1; e.message.display = d
        e.message.window = int(main['wid'], 16)
        e.message.message_type = x.XInternAtom(d, name.encode(), False)
        e.message.format = 32
        for i, value in enumerate(values): e.message.data.longs[i] = value
        if not x.XSendEvent(d, x.XDefaultRootWindow(d), False, (1 << 19) | (1 << 20), C.byref(e)):
            raise RuntimeError('window-manager request rejected')
        x.XSync(d, False)
    try:
        send('_NET_WM_STATE', [0, x.XInternAtom(d, b'_NET_WM_STATE_MAXIMIZED_HORZ', False),
                             x.XInternAtom(d, b'_NET_WM_STATE_MAXIMIZED_VERT', False), 2])
        time.sleep(.3)
        send('_NET_MOVERESIZE_WINDOW', [10 | (15 << 8) | (2 << 12), *result['target']])
        time.sleep(.6)
        after = window(main['wid'])
        if after['pid'] != s['pid'] or not inside(after['rect'], ORDINARY):
            raise RuntimeError('main did not arrive within ordinary display')
        return dict(result, after=after)
    finally:
        x.XCloseDisplay(d)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--light-blocked-confirmed', action='store_true')
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('move-main')
    a = sub.add_parser('action'); a.add_argument('key', choices=NAMES)
    a = sub.add_parser('mode'); a.add_argument('role', choices=('dmd', 'slm')); a.add_argument('mode', choices=('zero', 'file', 'black'))
    a = p.parse_args()
    if a.command == 'move-main': result = move_main(a.session, a.execute)
    elif a.command == 'mode': result = select(a.session, a.role, a.mode, a.execute)
    else: result = control(a.session, a.key, a.execute, a.light_blocked_confirmed)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
