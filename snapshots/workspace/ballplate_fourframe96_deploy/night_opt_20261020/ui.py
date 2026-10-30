"""Visible v22 public controls only. Never SDK, injection or new device windows."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools')]
from trigger_ui_v19 import inspect, nodes, inside, extents
from live_adapt_v22.core import digest, write_new

SESSION = PROJECT/'outputs/live_calibration/MANUAL_RESTART_20261020_231007_257706/GUI'
PLAN = PROJECT/'night_opt_20261020/SESSION_PLAN.json'


def check_authority():
    plan = json.loads(PLAN.read_text())
    if time.time() >= plan['hardware_authorization_deadline_unix_seconds']:
        raise RuntimeError('ten-hour hardware authorization expired; no further projection')
    return plan


def accessible(pid):
    import gi
    gi.require_version('Atspi', '2.0'); from gi.repository import Atspi
    d = Atspi.get_desktop(0)
    apps = [d.get_child_at_index(i) for i in range(d.get_child_count())]
    apps = [a for a in apps if a.get_process_id() == pid and a.get_name() == 'ballplate-realtime-manual']
    if len(apps) != 1: raise RuntimeError('unique original accessible owner required')
    return apps[0], Atspi


def press(node):
    action = node.get_action_iface()
    indexes = [i for i in range(action.get_n_actions()) if action.get_action_name(i) in ('Press', 'click', 'activate')]
    if len(indexes) != 1 or not action.do_action(indexes[0]): raise RuntimeError('visible button rejected')


def real_dialog(pid, title, Atspi):
    # Qt's native GTK chooser registers a SECOND accessible application named
    # "python" under the SAME PID. The Qt dialog is only a childless stub.
    d = Atspi.get_desktop(0)
    apps = [d.get_child_at_index(i) for i in range(d.get_child_count())]
    dialogs = [n for app in apps if app.get_process_id() == pid for n in nodes(app)
               if n.get_role_name() in ('dialog', 'file chooser') and n.get_name() == title
               and n.get_child_count() > 0]
    if len(dialogs) > 1: raise RuntimeError('ambiguous native dialogs')
    return dialogs


def load_bundle(catalog, output, open_only=False):
    check_authority(); v = inspect(SESSION); h = v['heartbeat']
    if h['pid'] != 1530936 or h.get('adaptation_active'): raise RuntimeError('original idle optical owner required')
    data = json.loads(catalog.read_text())
    for path, sha in data['bound_sha256'].items():
        if digest(path) != sha: raise RuntimeError('candidate bound asset changed')
    app, Atspi = accessible(h['pid'])
    dialog = real_dialog(h['pid'], '单帧适配包', Atspi)
    if not dialog:
        buttons = [n for n in nodes(app) if n.get_role_name() == 'push button' and n.get_name() == '载入单帧适配包']
        if len(buttons) != 1: raise RuntimeError('unique v22 load button missing')
        if not inside(extents(buttons[0], Atspi), v['windows']['control']['rect']):
            raise RuntimeError('load button outside ordinary control window')
        press(buttons[0]); time.sleep(.4)
        app, Atspi = accessible(h['pid'])
        dialog = real_dialog(h['pid'], '单帧适配包', Atspi)
    if len(dialog) != 1: raise RuntimeError('one PID-owned native file dialog required; no blind keyboard input')
    dialog = dialog[0]; content = list(nodes(dialog))
    description = [(n.get_role_name(), n.get_name()) for n in content]
    if open_only:
        print(json.dumps(dict(dialog_nodes=description, no_hardware_projection=True), ensure_ascii=False)); return
    fields = [n for n in content if n.get_role_name() == 'text' and n.get_name().replace('&', '').strip() in ('File name:', '文件名:', '文件名：')]
    if not fields:
        # GTK's location entry is exposed by its standard Ctrl+L shortcut.
        import re
        from select_verified_mode_v17 import window
        from calibration_desktop import Input
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
        windows = [window(re.search(r'0x[0-9a-fA-F]+', line)[0]) for line in tree.splitlines()
                   if '"单帧适配包"' in line]
        windows = [w for w in windows if w['pid'] == h['pid'] and w['viewable']
                   and inside(w['rect'], [1080, 600, 2560, 1440])]
        if len(windows) != 1: raise RuntimeError('one actual PID-owned ordinary-monitor chooser required')
        keyboard = Input(int(windows[0]['wid'], 16))
        try:
            active = subprocess.check_output(['xprop', '-root', '_NET_ACTIVE_WINDOW'], text=True)
            if int(re.search(r'0x[0-9a-fA-F]+', active)[0], 16) != int(windows[0]['wid'], 16):
                raise RuntimeError('native chooser did not gain actual keyboard focus')
            keyboard.chord(ord('l')); time.sleep(.2)
        finally: keyboard.close()
        dialogs = real_dialog(h['pid'], '单帧适配包', Atspi)
        if len(dialogs) != 1: raise RuntimeError('chooser disappeared')
        dialog = dialogs[0]; content = list(nodes(dialog))
        fields = [n for n in content if n.get_role_name() == 'text' and
                  all(n.get_state_set().contains(s) for s in (Atspi.StateType.EDITABLE, Atspi.StateType.SHOWING, Atspi.StateType.FOCUSED))]
    if len(fields) != 1: raise RuntimeError('unique native filename field missing: '+str(description))
    if not fields[0].get_editable_text_iface().set_text_contents(str(catalog.resolve())):
        raise RuntimeError('native filename edit rejected')
    if fields[0].get_text_iface().get_text(0, -1) != str(catalog.resolve()):
        raise RuntimeError('filename readback mismatch')
    time.sleep(.2)
    opens = [n for n in content if n.get_role_name() == 'push button' and n.get_name().replace('&', '') in ('Open', '打开', '打开(O)')]
    if len(opens) != 1: raise RuntimeError('unique native Open missing')
    press(opens[0]); time.sleep(.8)
    app, Atspi = accessible(h['pid'])
    if real_dialog(h['pid'], '单帧适配包', Atspi):
        raise RuntimeError('file dialog did not commit; no retry')
    texts = [n.get_text_iface().get_text(0, -1) for n in nodes(app) if n.get_role_name() == 'text']
    if str(catalog.resolve()) not in texts: raise RuntimeError('new catalog not adopted by GUI')
    after = inspect(SESSION)
    if after['pid'] != h['pid']: raise RuntimeError('owner changed')
    write_new(output, dict(catalog=str(catalog.resolve()), sha256=digest(catalog),
        pid=h['pid'], action='native_file_dialog_load_bundle', new_output_windows=False,
        SDK_access=False, no_projection=True, completed_unix_ns=time.time_ns()))
    print(json.dumps(dict(loaded=str(catalog.resolve()), all_outputs_still_zero=True), ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('load', 'ccd', 'release', 'collect', 'pilot'))
    p.add_argument('--catalog', type=Path); p.add_argument('--save', type=Path, required=True)
    p.add_argument('--phase-mode', choices=('learned', 'flat'), default='learned')
    p.add_argument('--open-only', action='store_true'); a = p.parse_args()
    if a.action != 'release': check_authority()
    if a.action in ('collect', 'pilot'):
        plan = check_authority()
        required_seconds = 330 if a.action == 'collect' else 630
        if time.time()+required_seconds >= plan['hardware_authorization_deadline_unix_seconds']:
            raise RuntimeError('insufficient authorization time to finish and park this bounded task')
    if a.action == 'load':
        load_bundle(a.catalog, a.save, a.open_only); return
    if a.action in ('ccd', 'release'):
        cmd = ['/usr/bin/python3', str(PROJECT/'tools/trigger_ui_v19.py'), '--session', str(SESSION),
               '--action', a.action, '--execute', '--save', str(a.save)]
    else:
        cmd = ['/usr/bin/python3', str(PROJECT/'tools/run_single_frame_ui_v22.py'), '--session', str(SESSION),
               '--catalog', str(a.catalog.resolve()), '--phase-mode', a.phase_mode, '--action', a.action,
               '--execute', '--low-power-restored-confirmed', '--save', str(a.save)]
    subprocess.run(cmd, cwd=PROJECT, check=True, timeout=20)


if __name__ == '__main__': main()
