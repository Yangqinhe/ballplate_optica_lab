"""Bounded CCD-only transport observation; no projection or network writes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from static_ui_v18 import owner

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'live_manual_v17'))
from black_pixels import RootPixels


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def host_transport():
    fields = ('rx_packets','rx_errors','rx_crc_errors','rx_length_errors','rx_fifo_errors','rx_dropped','rx_missed_errors','collisions')
    nic = {n: int(Path('/sys/class/net/eno1/statistics', n).read_text()) for n in fields}
    lines = Path('/proc/net/snmp').read_text().splitlines(); udp = {}
    for i,line in enumerate(lines[:-1]):
        if line.startswith('Udp:'):
            udp = dict(zip(line.split()[1:], map(int, lines[i+1].split()[1:]))); break
    softnet = [line.split() for line in Path('/proc/net/softnet_stat').read_text().splitlines()]
    return dict(unix_ns=time.time_ns(), NIC=nic, UDP=udp,
        softnet_processed=sum(int(r[0],16) for r in softnet),
        softnet_dropped=sum(int(r[1],16) for r in softnet),
        softnet_time_squeeze=sum(int(r[2],16) for r in softnet),
        cumulative_counters_not_camera_specific=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--resume-contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = dict(purpose='CCD_ONLY_TRANSPORT_OBSERVATION_NO_PROJECTION', frames=3, maximum_seconds=30,
        source_sha256={str(p): digest(p) for p in (Path(__file__), PROJECT/'tools/static_ui_v18.py')},
        resume_contract_sha256=digest(a.resume_contract), no_direct_SDK=True,
        no_new_output_windows=True, no_arming=True, release_approved=False)
    if not a.execute:
        print(json.dumps(plan)); return
    session = a.session.resolve()
    contract = json.loads(a.resume_contract.read_text())
    startup = json.loads((session/'STARTUP_VERIFICATION.json').read_text())
    before, main_window = owner(session)
    if before['busy'] or before['live_camera'] or before['projection_armed'] or before['hardware_fault']:
        raise RuntimeError('start only parked/fault-free')
    a.output.mkdir(exist_ok=False)
    (a.output/'PROSPECTIVE_PROTOCOL.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    record = dict(completed=False, errors=[], snapshots=[], plan=plan, before=before)
    started = time.monotonic(); deadline = started+30
    reports_before = set(session.glob('live_*/report.json'))
    pixels = RootPixels(); camera_requested = False
    record['transport_snapshots'] = [host_transport()]
    def save():
        (a.output/'qualification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
    def black():
        result = {}
        for role, row in startup['roles'].items():
            result[role] = pixels.inspect(row['desktop_position']+row['desktop_size'])
            if not result[role]['all_RGB_zero']: raise RuntimeError('actual output not black: '+role)
        return result
    def state():
        s, _ = owner(session)
        if s['pid'] != before['pid'] or s['hardware_fault'] or s['projection_armed']:
            raise RuntimeError('owner/fault/arming changed during CCD-only qualification')
        black()
        if time.monotonic() > deadline: raise RuntimeError('30s bounded camera qualification expired')
        return s
    def action(key):
        r = subprocess.run(['/usr/bin/python3', str(PROJECT/'tools/static_ui_v18.py'),
            '--session', str(session), '--execute', 'action', key], capture_output=True, text=True, timeout=6)
        if r.returncode: raise RuntimeError(r.stderr[-2000:])
    try:
        record['before_black'] = black(); save()
        camera_requested = True; action('ccd')
        end = time.monotonic()+15
        while time.monotonic() < end:
            s = state()
            if s['live_camera'] and s.get('latest_frame') and s['latest_frame']['age_seconds'] < 1:
                break
            time.sleep(.1)
        else: raise RuntimeError('no fresh CCD frame within 15s')
        if s['actual_roi'] != contract['roi'] or not s['geometry_frozen']:
            raise RuntimeError('ROI/reference not retained on restart')
        # One declared 15s observation, never retry to obtain a clean run.
        hold_end = time.monotonic()+15
        while time.monotonic() < hold_end:
            state(); record['transport_snapshots'].append(host_transport()); save(); time.sleep(1)
        last_id = -1
        for _ in range(3):
            state(); snapshots_before = set(session.glob('snapshot_*'))
            action('snapshot'); end = time.monotonic()+3
            while time.monotonic() < end:
                state(); found = set(session.glob('snapshot_*'))-snapshots_before
                if len(found) == 1 and (next(iter(found))/'metadata.json').exists(): break
                time.sleep(.1)
            else: raise RuntimeError('no unique new raw snapshot')
            folder = next(iter(found)); meta = json.loads((folder/'metadata.json').read_text())
            raw = np.load(folder/'CCD_raw.npy', allow_pickle=False)
            expected = contract['expected_active']
            if any(meta['camera_active'].get(k) != value for k, value in expected.items()):
                raise RuntimeError('active camera parameters differ from original reference')
            fid = int(meta['frame']['FrameID'])
            if fid <= last_id or raw.shape != (contract['roi']['Height'], contract['roi']['Width']):
                raise RuntimeError('old/out-of-order snapshot or wrong ROI')
            if raw.dtype != np.uint8 or int(raw.max()) >= 240:
                raise RuntimeError('CCD qualification count ceiling exceeded')
            last_id = fid
            record['snapshots'].append(dict(folder=str(folder.resolve()), frame=meta['frame'],
                max_ADU=int(raw.max()), mean_ADU=float(raw.mean()), raw_sha256=digest(folder/'CCD_raw.npy')))
            save(); time.sleep(.2)
        record['completed'] = True
    except BaseException as error:
        record['errors'].append(str(error))
    finally:
        if camera_requested:
            try: action('stop')
            except Exception as error: record['errors'].append('stop: '+str(error))
        end = time.monotonic()+10
        while time.monotonic() < end:
            s, _ = owner(session)
            if not s['busy'] and not s['live_camera']: break
            time.sleep(.1)
        try:
            record['after_black'] = black(); record['after'] = s
            reports = set(session.glob('live_*/report.json'))-reports_before
            if len(reports) != 1: raise RuntimeError('missing unique new camera restoration report')
            report_path = next(iter(reports)); report = json.loads(report_path.read_text())
            record['camera_report'] = str(report_path.resolve())
            record['camera_report_sha256'] = digest(report_path)
            if (not report.get('camera_settings_restored') or report.get('restore_differences')
                    or report.get('restoration_error') or report.get('camera_before') != report.get('camera_after')
                    or report.get('error') or report.get('rejected_incomplete_frames')):
                raise RuntimeError('CCD stream/restore report did not qualify')
            if s['hardware_fault'] or s['projection_armed'] or s['busy'] or s['live_camera'] or not s['geometry_frozen']:
                raise RuntimeError('owner not stopped/disarmed/frozen after test')
        except Exception as error:
            record['errors'].append(str(error))
        record['transport_snapshots'].append(host_transport())
        first,last = record['transport_snapshots'][0],record['transport_snapshots'][-1]
        record['NIC_delta'] = {k:last['NIC'][k]-first['NIC'][k] for k in first['NIC']}
        record['UDP_delta'] = {k:last['UDP'][k]-first['UDP'][k] for k in first['UDP']}
        record['completed'] = bool(record['completed'] and not record['errors'])
        record['elapsed_seconds'] = time.monotonic()-started
        pixels.close(); save()
    print(json.dumps(dict(completed=record['completed'], errors=record['errors'],
        snapshots=len(record['snapshots']), elapsed_seconds=record['elapsed_seconds']), ensure_ascii=False))
    if not record['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
