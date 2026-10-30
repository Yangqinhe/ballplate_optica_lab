"""Current-request, black-only native bundle chooser; no historical authority.

Same output owner is retained. Opening a native dialog is asynchronous so a
modal accessibility RPC cannot block its own filename entry. No SDK or Qt app.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from inspect_recovery_ui_v20 import inspect
from live_adapt_v22.core import Bundle, digest, write_new
from select_verified_mode_v17 import window, inside, extents
from calibration_desktop import Input
# These are stateless traversal helpers ONLY. No expired I/O driver is called.
from night_opt_20261020.manual_io_v17 import accessible, safe_nodes, real_dialog, focused_fields


def press(node):
    action = node.get_action_iface()
    choices = [i for i in range(action.get_n_actions()) if action.get_action_name(i).lower() in ('press','click','activate')]
    if len(choices) != 1 or not action.do_action(choices[0]):
        raise RuntimeError('visible action rejected')


def one_control(pid, role, name, rectangle):
    app, Atspi = accessible(pid)
    found = [n for n in safe_nodes(app) if n.get_role_name() == role and n.get_name() == name]
    if len(found) != 1:
        raise RuntimeError('unique native control missing: '+name)
    node = found[0]
    if any(not node.get_state_set().contains(s) for s in (Atspi.StateType.SENSITIVE, Atspi.StateType.SHOWING, Atspi.StateType.ENABLED)):
        raise RuntimeError('native control disabled/hidden')
    if not inside(extents(node, Atspi), rectangle):
        raise RuntimeError('control outside original main')
    return node, Atspi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', required=True, type=Path)
    p.add_argument('--catalog', type=Path)
    p.add_argument('--save', type=Path)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--open-dialog-child', action='store_true')
    a = p.parse_args()
    before = inspect(a.session, parked=True)
    if before['heartbeat']['hardware_fault']:
        raise RuntimeError('fault-free parked outputs required')
    main_rect = before['windows']['control']['rect']
    if a.open_dialog_child:
        button, _ = one_control(before['pid'], 'push button', '载入单帧适配包', main_rect)
        press(button)
        return
    if a.catalog is None or a.save is None:
        p.error('explicit catalog and new save path required')
    bundle = Bundle(a.catalog)
    if not a.execute:
        print(json.dumps(dict(catalog=str(bundle.path), sha256=bundle.sha, parked=True, no_projection=True), ensure_ascii=False))
        return
    if a.save.exists():
        raise ValueError('new output file required')
    tab, Atspi = one_control(before['pid'], 'page tab', '单帧适配 / MuJoCo', main_rect)
    press(tab)
    time.sleep(.2)
    record = dict(completed=False, error=None, catalog=str(bundle.path), catalog_sha256=bundle.sha,
        original_PID=before['pid'], authority='user 恢复到之前的情况了，继续之前的进度吧',
        no_SDK=True, no_controller_writes=True, no_projection=True, no_new_device_windows=True,
        source_sha256=digest(__file__), maximum_seconds=45, release_approved=False)
    child = None
    end = time.monotonic()+45
    try:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--session', str(a.session.resolve()),
            '--open-dialog-child'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        dialogs = []
        while time.monotonic() < end:
            dialogs = real_dialog(before['pid'], '单帧适配包', Atspi)
            if dialogs: break
            if child.poll() is not None:
                raise RuntimeError('native chooser did not open: '+child.communicate()[1])
            time.sleep(.1)
        if len(dialogs) != 1:
            raise RuntimeError('one PID-owned native chooser required')
        tree = subprocess.check_output(['xwininfo','-root','-tree'], text=True)
        candidates = [window(re.search(r'0x[0-9a-fA-F]+', line)[0])
                      for line in tree.splitlines() if '"单帧适配包"' in line]
        candidates = [w for w in candidates if w['pid'] == before['pid'] and w['viewable']
                      and inside(w['rect'], [1080,600,2560,1440])]
        if len(candidates) != 1:
            raise RuntimeError('one actual ordinary-monitor chooser required')
        chooser = candidates[0]
        keyboard = Input(int(chooser['wid'], 16))
        try:
            def focused():
                actual = subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'], text=True)
                if int(re.search(r'0x[0-9a-fA-F]+', actual)[0],16) != int(chooser['wid'],16):
                    raise RuntimeError('chooser lost actual keyboard focus')
            time.sleep(.5)
            focused()
            keyboard.chord(ord('l'))
            time.sleep(.3)
            dialogs = real_dialog(before['pid'], '单帧适配包', Atspi)
            if len(dialogs) != 1:
                raise RuntimeError('chooser disappeared')
            fields = focused_fields(dialogs[0], Atspi)
            if len(fields) != 1:
                raise RuntimeError('one focused native location field required')
            field = fields[0]
            if not field.get_editable_text_iface().set_text_contents(str(bundle.path)):
                raise RuntimeError('native filename edit rejected')
            time.sleep(.3)
            focused()
            if field.get_text_iface().get_text(0,-1) != str(bundle.path):
                raise RuntimeError('native filename readback differs')
            bundle.verify()
            keyboard.key(0xff0d, True)
            keyboard.key(0xff0d, False)
        finally:
            keyboard.close()
        while time.monotonic() < end:
            if not real_dialog(before['pid'], '单帧适配包', Atspi): break
            time.sleep(.1)
        else:
            raise TimeoutError('chooser did not commit once; no retry')
        out, err = child.communicate(timeout=max(.1, end-time.monotonic()))
        if child.returncode:
            raise RuntimeError('native chooser action failed: '+err)
        app, _ = accessible(before['pid'])
        texts = [n.get_text_iface().get_text(0,-1) for n in safe_nodes(app) if n.get_role_name() == 'text']
        if str(bundle.path) not in texts:
            raise RuntimeError('new catalog not adopted by original GUI')
        after = inspect(a.session, parked=True)
        if after['pid'] != before['pid']:
            raise RuntimeError('owner changed')
        record.update(completed=True, final_black=after)
    except BaseException as error:
        record['error'] = repr(error)
    finally:
        if child and child.poll() is None:
            # Terminates ONLY our blocked accessibility client, never the GUI.
            child.terminate()
            child.communicate(timeout=3)
        write_new(a.save, record)
        print(json.dumps({k:record[k] for k in ('completed','error','catalog','original_PID')}, ensure_ascii=False))
    if not record['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
