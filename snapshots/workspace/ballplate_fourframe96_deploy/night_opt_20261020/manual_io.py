"""Original GUI's PUBLIC file-load/snapshot controls for settled optical I/O.

No SDK, in-process injection, bound-file overwrite, new device windows,
calibration changes, simulator or actuator. All interactions are visible.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from night_opt_20261020.ui import SESSION, check_authority, accessible, real_dialog, press
from trigger_ui_v19 import nodes, extents, inside
from select_verified_mode_v17 import window
from black_pixels import RootPixels
from display_linux import read_roles, same_signals
from calibration_math import embed_phase
from calibration_desktop import Input
from live_adapt_v22.core import digest

PID = 1530936
PROCESS_START = 26830311
ORDINARY = [1080, 600, 2560, 1440]
ARM_NAME = '允许投图（ND／低功率条件已确认）'


def read_json_complete(path):
    # Original snapshot writer commits several separate small files. This
    # read waits ONLY for that file's initial write, never substitutes data.
    until = time.monotonic()+2
    while True:
        try: return json.loads(Path(path).read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if time.monotonic() >= until: raise
            time.sleep(.02)


def alive():
    stat = Path(f'/proc/{PID}/stat').read_text().rsplit(')', 1)[1].split()
    if int(stat[19]) != PROCESS_START: raise RuntimeError('original optical process identity changed')


class ManualIO:
    def __init__(self, bundle):
        self.bundle = bundle; self.runtime = read_json_complete(SESSION/'runtime_sources.json')
        if self.runtime['pid'] != PID or self.runtime.get('deployment_ui_revision') != 22:
            raise RuntimeError('original v22 owner required')
        created = next(json.loads(line) for line in (SESSION/'events.jsonl').read_text().splitlines()
            if json.loads(line).get('state') == 'created')
        self.video = created['video']; self.pixels = RootPixels(); self.last_id = -1; self.last_ts = -1
        startup = read_json_complete(SESSION/'STARTUP_VERIFICATION.json')
        self.windows = {role: row['wid'] for role, row in startup['windows'].items()}
        self.expected = None; self.phase_loaded = False; self.dmd = None
        self.guard(black=True)

    def guard(self, black=False, require_live=True):
        check_authority(); alive()
        for group in ('sources', 'source_configs'):
            for path, sha in self.runtime[group].items():
                if digest(path) != sha: raise RuntimeError('loaded GUI source/config changed: '+path)
        h = read_json_complete(SESSION/'heartbeat.json')
        if (h['pid'] != PID or not 0 <= time.time_ns()-h['unix_ns'] < 8e9
                or h['hardware_fault'] or h.get('adaptation_active') or h.get('deployment_busy')):
            raise RuntimeError('original owner stale, faulted, or running another bounded task')
        if require_live and (not h['live_camera'] or not h.get('latest_frame')
                             or h['latest_frame']['age_seconds'] > 1.):
            raise RuntimeError('fresh Line0 stream required; no camera fallback')
        c = self.bundle.contract; s = h['manual_settings']
        for key in ('full_panel', 'x', 'y', 'focal_mm', 'period', 'square'):
            if s[key] != c['settings'][key]: raise RuntimeError('fixed manual geometry changed: '+key)
        if (h['actual_roi'] != c['roi'] or h['optical_reference'] != c['optical_config']
                or not h['geometry_frozen']): raise RuntimeError('ROI/model/geometry lock changed')
        if require_live and h['camera_active_readback'] != c['expected_active']:
            raise RuntimeError('live camera nodes differ from frozen experiment')
        roles = read_roles(); same_signals(self.video, roles)
        main = window(self.windows['control'])
        if main['pid'] != PID or not main['viewable'] or not inside(main['rect'], ORDINARY):
            raise RuntimeError('original control window missing/moved')
        for role, row in roles.items():
            w = window(self.windows[role]); rectangle = row['desktop_position']+row['desktop_size']
            if w['pid'] != PID or not w['viewable'] or w['rect'] != rectangle:
                raise RuntimeError('original output identity/geometry changed: '+role)
            if black:
                if h['projection_armed'] or not self.pixels.inspect(tuple(rectangle))['all_RGB_zero']:
                    raise RuntimeError('actual disarmed all-zero output required')
            elif self.expected is not None:
                if not self.pixels.compare(tuple(rectangle), self.expected[role])['matches']:
                    raise RuntimeError('actual framebuffer differs from expected optical query: '+role)
        return h, main

    def find(self, role, name, showing=True):
        app, Atspi = accessible(PID)
        matches = [n for n in nodes(app) if n.get_role_name() == role and n.get_name() == name]
        if len(matches) != 1: raise RuntimeError('missing/ambiguous native control: '+name)
        n = matches[0]
        if showing:
            for flag in (Atspi.StateType.SHOWING, Atspi.StateType.SENSITIVE, Atspi.StateType.ENABLED):
                if not n.get_state_set().contains(flag): raise RuntimeError('control hidden/disabled: '+name)
            main = window(self.windows['control'])
            if not inside(extents(n, Atspi), main['rect']): raise RuntimeError('control outside main window')
        return n, Atspi

    def manual_tab(self):
        self.guard(black=True)
        n, _ = self.find('page tab', '手动对齐'); press(n); time.sleep(.15)
        self.find('push button', '载入图像')

    def load(self, path, phase=False):
        self.guard(black=not bool(self.expected))
        path = Path(path).resolve(); sha = digest(path)
        a = np.load(path, allow_pickle=False)
        if a.dtype != np.uint8 or a.shape != (600, 600): raise ValueError('exact uint8 600-square file required')
        if not phase and not np.isin(a, (0, 255)).all(): raise ValueError('binary DMD required')
        if phase and not np.array_equal(a, self.bundle.phase):
            raise RuntimeError('phase differs from frozen head bundle; do not load it')
        title = '载入规范SLM灰度图' if phase else '载入DMD图'
        button, Atspi = self.find('push button', '载入mask' if phase else '载入图像')
        press(button); time.sleep(.12)
        dialogs = real_dialog(PID, title, Atspi)
        if len(dialogs) != 1: raise RuntimeError('PID-owned native file chooser missing')
        tree = subprocess.check_output(['xwininfo', '-root', '-tree'], text=True)
        windows = [window(re.search(r'0x[0-9a-fA-F]+', line)[0]) for line in tree.splitlines() if '"'+title+'"' in line]
        windows = [w for w in windows if w['pid'] == PID and w['viewable'] and inside(w['rect'], ORDINARY)]
        if len(windows) != 1: raise RuntimeError('actual owned chooser not entirely on ordinary monitor')
        keyboard = Input(int(windows[0]['wid'], 16))
        try:
            active = subprocess.check_output(['xprop', '-root', '_NET_ACTIVE_WINDOW'], text=True)
            if int(re.search(r'0x[0-9a-fA-F]+', active)[0], 16) != int(windows[0]['wid'], 16):
                raise RuntimeError('file chooser did not obtain actual keyboard focus')
            keyboard.chord(ord('l')); time.sleep(.07)
        finally: keyboard.close()
        dialogs = real_dialog(PID, title, Atspi)
        if len(dialogs) != 1: raise RuntimeError('native dialog disappeared')
        content = list(nodes(dialogs[0])); entries = [n for n in content if n.get_role_name() == 'text'
            and all(n.get_state_set().contains(s) for s in (Atspi.StateType.EDITABLE, Atspi.StateType.SHOWING, Atspi.StateType.FOCUSED))]
        if len(entries) != 1: raise RuntimeError('one real focused location entry required')
        entry = entries[0]
        if not entry.get_editable_text_iface().set_text_contents(str(path)):
            raise RuntimeError('native filename field rejected edit')
        if entry.get_text_iface().get_text(0, -1) != str(path): raise RuntimeError('filename readback differs')
        opens = [n for n in content if n.get_role_name() == 'push button'
                 and n.get_name().replace('&', '') in ('Open', '打开', '打开(O)')]
        if len(opens) != 1: raise RuntimeError('one native Open button required')
        press(opens[0]); time.sleep(.12)
        if real_dialog(PID, title, Atspi): raise RuntimeError('file chooser did not commit')
        if digest(path) != sha: raise RuntimeError('input file changed during load')
        events = [json.loads(line) for line in (SESSION/'events.jsonl').read_text().splitlines()]
        if not any(e.get('state') == 'manual_file_loaded' and e.get('path') == str(path)
                   and e.get('sha256') == sha and e.get('SLM') == phase for e in events):
            raise RuntimeError('GUI did not acknowledge exact file/hash/type')
        if phase:
            self.phase_loaded = True
        else: self.dmd = a.copy()
        self.expected = None  # New dirty pattern not yet framebuffer-acknowledged.
        return dict(path=str(path), sha256=sha, phase=phase)

    def arm_and_verify(self):
        if self.dmd is None or not self.phase_loaded: raise RuntimeError('load both validated optical inputs first')
        check_authority(); alive()
        n, Atspi = self.find('check box', ARM_NAME)
        if not n.get_state_set().contains(Atspi.StateType.CHECKED):
            action = n.get_action_iface(); ids = [i for i in range(action.get_n_actions())
                if action.get_action_name(i) in ('Toggle', 'Press')]
            if len(ids) != 1 or not action.do_action(ids[0]): raise RuntimeError('arming checkbox rejected')
        s = self.bundle.contract['settings']
        expected = dict(dmd=np.pad(self.dmd, ((0, 0), (100, 100))),
            slm=np.ascontiguousarray(np.fliplr(embed_phase(self.bundle.phase, (s['x'], s['y'])))))
        until = time.monotonic()+3
        while time.monotonic() < until:
            self.expected = expected
            try:
                h, _ = self.guard()
                if h['projection_armed'] and h['last_verified_revision'] == h['display_revision']:
                    return dict(verified_unix_ns=time.time_ns(), revision=h['display_revision'],
                        actual_framebuffer_checked=True, hardware_pattern_sync_verified=False)
            except RuntimeError as error:
                if 'framebuffer differs' not in str(error): raise
            time.sleep(.05)
        raise TimeoutError('new expected actual video was not acknowledged')

    def snapshot_after(self, verified, output, settle=1.5):
        if settle != 1.5: raise ValueError('retain original settled-query protocol')
        # The public button saves the latest packet, rather than requesting a
        # brand-new sensor exposure. Leave two extra 60Hz frames of margin and
        # STILL require the saved acquisition-start timestamp to pass 1.5s.
        while time.time_ns() < verified['verified_unix_ns']+int((settle+.1)*1e9):
            self.guard(); time.sleep(.2)
        self.guard(); before = set(SESSION.glob('snapshot_*'))
        button, _ = self.find('push button', '保存当前对照'); press(button)
        until = time.monotonic()+3; folder = None
        while time.monotonic() < until:
            fresh = set(SESSION.glob('snapshot_*'))-before
            if len(fresh) > 1: raise RuntimeError('ambiguous snapshot provenance')
            if fresh and (next(iter(fresh))/'metadata.json').exists(): folder = next(iter(fresh)); break
            time.sleep(.02)
        if folder is None: raise TimeoutError('new public snapshot not saved')
        meta = read_json_complete(folder/'metadata.json'); frame = meta['frame']
        h, _ = self.guard()
        if (meta['camera_active'] != self.bundle.contract['expected_active']
                or meta['display_revision'] != verified['revision']
                or h['display_revision'] != verified['revision']
                or frame['FrameID'] <= self.last_id or frame['TimeStamp'] <= self.last_ts
                or frame['host_get_started_unix_ns'] < verified['verified_unix_ns']+int(settle*1e9)):
            raise RuntimeError('snapshot wrong revision, camera, order, or settle freshness')
        raw_path = folder/'CCD_raw.npy'; raw = np.load(raw_path, allow_pickle=False)
        if raw.dtype != np.uint8 or raw.shape != self.bundle.weights.shape[1:] or raw.max() >= 240:
            raise RuntimeError('snapshot violates native Mono8/headroom contract')
        self.last_id, self.last_ts = frame['FrameID'], frame['TimeStamp']
        record = dict(snapshot=str(folder), raw_sha256=digest(raw_path), metadata_sha256=digest(folder/'metadata.json'),
            verification=verified, frame=frame, software_settled_association_only=True,
            original_snapshot_metadata_NOT_modified=True,
            raw_integrals8=np.einsum('khw,hw->k', self.bundle.weights, raw.astype(float)).tolist())
        with Path(output).open('x') as stream: json.dump(record, stream, ensure_ascii=False, indent=2)
        return record

    def park(self):
        # De-arming is allowed even after the ten-hour projection authorization
        # expires. It never rebuilds windows or opens an SDK connection.
        alive(); n, Atspi = self.find('check box', ARM_NAME)
        if n.get_state_set().contains(Atspi.StateType.CHECKED):
            action = n.get_action_iface(); ids = [i for i in range(action.get_n_actions())
                if action.get_action_name(i) in ('Toggle', 'Press')]
            if len(ids) != 1 or not action.do_action(ids[0]): raise RuntimeError('de-arm failed')
        until = time.monotonic()+4
        while time.monotonic() < until:
            alive(); roles = read_roles(); same_signals(self.video, roles)
            if all(self.pixels.inspect(tuple(r['desktop_position']+r['desktop_size']))['all_RGB_zero'] for r in roles.values()):
                self.expected = None; return dict(actual_outputs_zero=True, original_pid=PID, windows_rebuilt=False)
            time.sleep(.05)
        raise RuntimeError('could not certify actual zero outputs; no replacement windows')

    def close(self): self.pixels.close()
