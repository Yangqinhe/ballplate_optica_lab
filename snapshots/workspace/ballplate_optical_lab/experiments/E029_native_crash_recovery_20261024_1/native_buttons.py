"""Explicit external, single native mouse clicks; never AT-SPI actions.

Import is standard-library only.  Caller still owns source/display/physical
conditions, operation authorization, heartbeat postconditions and result logs.
This alternative input path is NOT a demonstrated fix for the Qt native crash.
"""
from __future__ import annotations

import ctypes as C
import re
import subprocess
from types import SimpleNamespace


ORDINARY = [1080, 600, 2560, 1440]
NAMES = dict(recheck='复核并恢复（不投图）', ccd='仅CCD（硬触发）',
             allzero='两路全0，CCD保持（Esc）', snapshot='保存当前对照',
             arm='允许投图（ND／低功率条件已确认）')


def _runtime():
    # Public clients only, lazily imported after an explicit call.  In
    # particular, do not load the GUI, Qt, SDK, or legacy _load_clients here.
    import optical_ui_v19 as optical
    from calibration_desktop import Input
    return SimpleNamespace(optical=optical, Input=Input, Pointer=optical.Pointer,
                           window=optical.window, inside=optical.inside,
                           extents=optical.extents, nodes=optical.nodes)


def _wid(value):
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and re.fullmatch(r'0x[0-9a-fA-F]+', value) and int(value, 16) > 0:
        return int(value, 16)
    raise ValueError('explicit positive X client window id required')


def check_state(ui, key):
    if key not in NAMES:
        raise ValueError('unsupported native button key')
    h = ui.state(allow_fault=key in ('recheck', 'allzero'))
    if h.get('pid') != ui.pid:
        raise RuntimeError('original heartbeat PID changed')
    if any(h.get(k) is True for k in ('loop_active', 'deployment_busy', 'adaptation_active')) or h.get('spatial_calibration', {}).get('busy'):
        raise RuntimeError('competing operation; no native button click')
    if key in ('recheck', 'ccd') and any(h.get(k) is not False for k in ('busy', 'live_camera', 'projection_armed', 'camera_reserved')):
        raise RuntimeError('released/off/disarmed owner required')
    if key in ('arm', 'snapshot') and not ui.fresh():
        raise RuntimeError('fresh live CCD required')
    return h


def _node_rectangle(node, A, runtime, owner_rect):
    node.clear_cache()
    if node.get_role_name() not in ('push button', 'check box'):
        raise RuntimeError('only a visible native button/checkbox is accepted')
    for flag in (A.StateType.ENABLED, A.StateType.SENSITIVE, A.StateType.SHOWING):
        if not node.get_state_set().contains(flag):
            raise RuntimeError('native control disabled or hidden')
    rect = runtime.extents(node, A)
    if not runtime.inside(rect, owner_rect) or min(rect[2:]) < 6:
        raise RuntimeError('native control outside verified owner or too small')
    return list(rect)


def _active_window():
    value = subprocess.check_output(['xprop', '-root', '_NET_ACTIVE_WINDOW'], text=True)
    match = re.search(r'window id # (0x[0-9a-fA-F]+)', value)
    if not match:
        raise RuntimeError('missing exact active X client')
    return int(match[1], 16)


def _reject_other_dialogs(pid, allowed_owner):
    """Read the WM client list, never dismiss or click through another popup."""
    value = subprocess.check_output(['xprop', '-root', '_NET_CLIENT_LIST_STACKING'], text=True)
    if '_NET_CLIENT_LIST_STACKING(WINDOW)' not in value:
        raise RuntimeError('WM client list unavailable')
    # Override-redirect native popups need not be EWMH clients. Include the
    # original application's named root-tree entries as well as client list.
    tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
    ids = re.findall(r'0x[0-9a-fA-F]+', value)
    for line in tree.splitlines():
        if '"ballplate-realtime-manual"' in line:
            match = re.search(r'0x[0-9a-fA-F]+', line)
            if match:
                ids.append(match[0])
    checked = []
    for wid in dict.fromkeys(ids):
        if int(wid, 16) == allowed_owner:
            continue
        props = subprocess.check_output(['xprop', '-id', wid, '_NET_WM_PID',
            'WM_TRANSIENT_FOR', '_NET_WM_STATE', '_NET_WM_WINDOW_TYPE'], text=True)
        owner = re.search(r'_NET_WM_PID\(CARDINAL\) = (\d+)', props)
        if owner is None or int(owner[1]) != pid:
            continue
        info = subprocess.check_output(['xwininfo', '-id', wid], text=True)
        if 'Map State: IsViewable' not in info:
            continue
        checked.append(wid)
        if ('WM_TRANSIENT_FOR(WINDOW): window id #' in props
                or '_NET_WM_STATE_MODAL' in props
                or re.search(r'_NET_WM_WINDOW_TYPE_(?:DIALOG|MENU|DROPDOWN_MENU|POPUP_MENU|COMBO|TOOLTIP)\b', props)):
            raise RuntimeError('another visible PID-owned dialog/popup; no click')
    return checked


def _query_tree(focus, client):
    root = C.c_ulong(); parent = C.c_ulong(); children = C.c_void_p(); count = C.c_uint()
    ok = focus.x.XQueryTree(focus.display, client, C.byref(root), C.byref(parent),
                           C.byref(children), C.byref(count))
    try:
        if not ok or not root.value or not parent.value or parent.value == client:
            raise RuntimeError('owner XQueryTree did not prove its parent')
        return dict(queried_client_wid=hex(client), root_wid=hex(root.value),
                    returned_parent_wid=hex(parent.value), child_count=count.value)
    finally:
        if children:
            focus.x.XFree(children)


def _root_child(focus, owner):
    tree = _query_tree(focus, owner)
    root = int(tree['root_wid'], 16); parent = int(tree['returned_parent_wid'], 16)
    if parent == root:
        child = owner; parent_tree = None
    else:
        parent_tree = _query_tree(focus, parent)
        if int(parent_tree['root_wid'], 16) != root or int(parent_tree['returned_parent_wid'], 16) != root:
            raise RuntimeError('WM parent is not the verified root child; no guessing')
        child = parent
    return dict(owner_tree=tree, parent_tree=parent_tree, pointer_root_child_wid=hex(child),
                parent_was_returned_by_XQueryTree=True)


def _same_window(runtime, wid, pid, rect):
    row = runtime.window(hex(wid))
    if row.get('pid') != pid or not row.get('viewable') or list(row.get('rect', [])) != list(rect):
        raise RuntimeError('original X client PID/visibility/rectangle changed')
    if not runtime.inside(rect, ORDINARY):
        raise RuntimeError('owner outside verified ordinary monitor')
    return row


def click_visible_node(ui, node, A, *, owner_client_wid=None, owner_rect=None, execute=True):
    """Click one caller-verified node in the original main or exact dialog.

    For dialogs the caller must first verify unique PID, title, role and Yes
    control, then supply that actual client id/rectangle. This helper never
    searches for a confirmation, guesses a window, or retries a mouse action.
    """
    if type(execute) is not bool:
        raise ValueError('execute must be an explicit bool')
    if (owner_client_wid is None) != (owner_rect is None):
        raise ValueError('dialog client id and rectangle must be supplied together')
    runtime = _runtime()
    main = ui.main; main_wid = _wid(main['wid'])
    owner = main_wid if owner_client_wid is None else _wid(owner_client_wid)
    rectangle = list(main['rect'] if owner_rect is None else owner_rect)
    state = ui.state(allow_fault=True)
    if state.get('pid') != ui.pid or node.get_process_id() != ui.pid:
        raise RuntimeError('heartbeat/control must belong to original PID')
    _same_window(runtime, main_wid, ui.pid, main['rect'])
    _same_window(runtime, owner, ui.pid, rectangle)
    rect = _node_rectangle(node, A, runtime, rectangle)
    name = node.get_name(); point = [rect[0] + rect[2] // 2, rect[1] + rect[3] // 2]
    checked = _reject_other_dialogs(ui.pid, owner)
    result = dict(pid=ui.pid, name=name, role=node.get_role_name(), owner_client_wid=hex(owner),
        owner_rect=rectangle, control_rect=rect, click_point=point, executed=False, click_count=0,
        other_visible_PID_clients_checked=checked, atspi_actions_used=False, SDK_connections=0,
        new_windows_created=0, retries=0, Qt_crash_root_cause_fixed=False)
    if not execute:
        return result
    cls = runtime.Pointer
    if cls.__name__ != 'NumLockPointer' or cls.__module__ != 'workbench.unified.pointer_control':
        raise RuntimeError('caller must install original verified NumLockPointer first')
    focus = runtime.Input(owner)
    pointer = None
    try:
        if _active_window() != owner:
            raise RuntimeError('exact original owner is not foreground')
        _same_window(runtime, main_wid, ui.pid, main['rect'])
        _same_window(runtime, owner, ui.pid, rectangle)
        _reject_other_dialogs(ui.pid, owner)
        if _node_rectangle(node, A, runtime, rectangle) != rect or node.get_name() != name:
            raise RuntimeError('control changed after focusing; no click')
        if ui.state(allow_fault=True).get('pid') != ui.pid:
            raise RuntimeError('heartbeat PID changed before click')
        tree = _root_child(focus, owner)
        if _active_window() != owner:
            raise RuntimeError('foreground changed before click')
        pointer = cls()
        # Pointer queries the root's immediate child, which is a WM frame for
        # decorated clients. Never pass the client id blindly or skip its
        # coordinate/modifier guard. NumLock is the sole permitted modifier.
        pointer.click(tuple(point), dict(wid=tree['pointer_root_child_wid']))
        result.update(executed=True, click_count=1, active_client_wid=hex(owner),
            root_parent_proof=tree, pointer_guard='original NumLockPointer: exact root child/point; NumLock only')
        return result
    finally:
        if pointer is not None:
            pointer.close()
        focus.close()


def invoke(ui, key, execute=False):
    """Read-only exact PID/app/frame/control discovery, then one native click."""
    check_state(ui, key)
    import gi
    gi.require_version('Atspi', '2.0')
    from gi.repository import Atspi as A
    runtime = _runtime(); desktop = A.get_desktop(0)
    apps = [desktop.get_child_at_index(i) for i in range(desktop.get_child_count())]
    apps = [a for a in apps if a.get_process_id() == ui.pid and a.get_name() == 'ballplate-realtime-manual']
    if len(apps) != 1:
        raise RuntimeError('unique original PID-owned application required')
    frames = [n for n in runtime.nodes(apps[0]) if n.get_role_name() == 'frame'
              and n.get_name().startswith('实时光路对齐 v19 ')]
    if len(frames) != 1:
        raise RuntimeError('unique original main frame required')
    role = 'check box' if key == 'arm' else 'push button'
    matches = [n for n in runtime.nodes(frames[0]) if n.get_role_name() == role and n.get_name() == NAMES[key]]
    if len(matches) != 1:
        raise RuntimeError('exact named native control missing/ambiguous')
    check_state(ui, key)
    result = click_visible_node(ui, matches[0], A, execute=execute)
    result['key'] = key
    return result
