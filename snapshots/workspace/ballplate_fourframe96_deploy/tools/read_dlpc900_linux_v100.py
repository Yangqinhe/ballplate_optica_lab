"""Linux hidraw DLPC900 READ requests only, using the frozen original codec.

No hidapi install, Windows driver, reset, pattern start/stop, LUT write,
arbitrary commands, kernel detach, permission changes or new output windows.
HID write() carries a USB READ request: configuration write count stays0.
Per-pattern bit depth/exposure are NOT claimed readable by this whitelist.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import select
import stat
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT), str(PROJECT/'tools'), str(PROJECT/'live_manual_v17')]
from diagnose_black_restart_v83 import TimedBlackHold
from inspect_recovery_ui_v20 import inspect
from live_adapt_v22.core import digest, write_new

CODEC = PROJECT.parent/'ballplate_minframes_optical_ppo/SDON_Linux_Handoff_20261015/src/sdon/hardware/dlpc900_readonly.py'
CODEC_SHA256 = '4938de25b2f0d94a19974ecc480d53decf64f2f06688b1652ee31495fc2b724b'
QUERIES = frozenset(('hardware_status', 'system_status', 'main_status', 'firmware_version',
    'display_mode', 'input_source', 'port_clock', 'trigger_out1', 'trigger_out2',
    'pattern_lut_config', 'pattern_invert', 'block_load'))


def codec():
    if digest(CODEC) != CODEC_SHA256:
        raise RuntimeError('frozen original READ-only protocol source changed')
    spec = importlib.util.spec_from_file_location('original_dlpc900_readonly_codec', CODEC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover(base=Path('/sys/class/hidraw')):
    found = []
    for item in base.glob('hidraw*'):
        device = (item/'device').resolve()
        metadata = dict(line.split('=', 1) for line in (device/'uevent').read_text().splitlines() if '=' in line)
        parts = metadata.get('HID_ID', '').split(':')
        if len(parts) != 3 or (int(parts[1], 16), int(parts[2], 16)) != (0x0451, 0xC900):
            continue
        interface = next((p for p in device.parents if (p/'bInterfaceNumber').exists()), None)
        usb = next((p for p in device.parents if (p/'idVendor').exists() and (p/'idProduct').exists()), None)
        if interface is None or usb is None:
            raise RuntimeError('cannot bind exact USB interface identity')
        if int((interface/'bInterfaceNumber').read_text().strip(), 16) != 0:
            continue
        if (usb/'product').read_text().strip() != 'DLPC900':
            raise RuntimeError('TI VID/PID matched but product identity differs')
        node = Path('/dev')/item.name
        found.append(dict(node=str(node), sysfs=str(device), usb=str(usb), product='DLPC900',
            vendor='0451', product_id='c900', interface=0,
            user_can_read_write=os.access(node, os.R_OK | os.W_OK)))
    return found


def read_packet(protocol, name, sequence):
    if name not in QUERIES:
        raise ValueError('command outside fixed READ-only whitelist')
    packet = protocol.read_request(name, sequence)
    command = protocol.READ_COMMANDS[name][0]
    if (len(packet) != 65 or packet[:2] != bytes([0, 0xC0])
            or packet[2] != sequence or packet[3:5] != b'\x02\x00'
            or int.from_bytes(packet[5:7], 'little') != command or any(packet[7:])):
        raise RuntimeError('READ flag, length, command or zero-payload invariant failed')
    return packet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--execute-readonly', action='store_true')
    args = p.parse_args()
    protocol = codec()
    devices = discover()
    if not args.execute_readonly:
        print(json.dumps(dict(discovered=devices, USB_opened=False, READ_requests_sent=0,
            controller_configuration_writes=0), ensure_ascii=False, indent=2))
        return
    if args.session is None or args.output is None:
        p.error('explicit existing session and NEW output directory required')
    if len(devices) != 1:
        raise RuntimeError('exactly one DLPC900 interface0 required')
    device = devices[0]
    if not device['user_can_read_write']:
        raise PermissionError('device access denied; no sudo, chmod or alternative backend attempted')
    args.output.mkdir(parents=True, exist_ok=False)
    write_new(args.output/'PROSPECTIVE_USB_READONLY.json', dict(source_sha256=digest(__file__),
        codec=str(CODEC), codec_sha256=CODEC_SHA256, device=device, maximum_seconds=45,
        HID_requests_are_READ_flag_C0=True, controller_configuration_writes=0,
        no_reset_or_sequence_start_stop=True, no_new_windows_or_projection=True,
        per_pattern_bit_depth_and_exposure_NOT_readable_by_this_whitelist=True, release_approved=False))
    preflight = inspect(args.session, parked=True)
    guard = TimedBlackHold(args.session, preflight, time.monotonic()+45)
    fd, lock = None, None
    result = dict(completed=False, error=None, transactions=[], controller_configuration_writes=0,
        camera_SDK_opened=False, new_device_output_windows=False, release_approved=False)
    try:
        lock = open(f'/run/user/{os.getuid()}/ballplate_dlpc900_readonly.lock', 'a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        guard.check(force=True)
        if discover() != devices:
            raise RuntimeError('USB device identity or access changed before open')
        fd = os.open(device['node'], os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
        if not stat.S_ISCHR(os.fstat(fd).st_mode):
            raise RuntimeError('not a character device')
        drained = []
        for _ in range(16):
            if not select.select([fd], [], [], 0)[0]:
                break
            drained.append(os.read(fd, 65).hex())
        result['preexisting_reports_discarded_NOT_accepted_as_responses'] = drained

        def query(name):
            guard.check(force=True)
            if digest(CODEC) != CODEC_SHA256 or discover() != devices:
                raise RuntimeError('frozen source or USB identity changed')
            seq = len(result['transactions'])+1
            packet = read_packet(protocol, name, seq)
            row = dict(name=name, sequence=seq, request_hex=packet.hex(), logical_operation='READ',
                       requested_unix_ns=time.time_ns())
            result['transactions'].append(row)
            if os.write(fd, packet) != len(packet):
                raise RuntimeError('short HID READ request; not retried')
            row['HID_read_request_sent'] = True
            if not select.select([fd], [], [], 1.5)[0]:
                raise TimeoutError('READ response deadline expired; no retries or writes')
            raw = os.read(fd, 65)
            row['response_hex'] = raw.hex()
            data = protocol.parse_reply(name, seq, raw)
            row['decoded'] = protocol.decode_reply(name, data)
            guard.check(force=True)
            print(json.dumps(dict(name=name, decoded=row['decoded']), ensure_ascii=False), flush=True)
            return row['decoded']

        for name in ('hardware_status', 'system_status', 'main_status', 'firmware_version'):
            query(name)
        mode = query('display_mode')['mode']
        result['mode'] = mode
        if mode in ('video', 'video_pattern'):
            query('input_source')
            query('port_clock')
        if mode in ('video_pattern', 'pre_stored_pattern', 'pattern_on_the_fly'):
            for name in ('trigger_out1', 'trigger_out2', 'pattern_lut_config', 'pattern_invert', 'block_load'):
                query(name)
        elif mode != 'video':
            raise RuntimeError('unknown controller mode; no further queries')
        for _ in range(2):
            time.sleep(.2)
            query('hardware_status')
            query('main_status')
        result['health'] = protocol.assess_status_series(result['transactions'])
        result['completed'] = True
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        if fd is not None:
            os.close(fd)
        if lock is not None:
            lock.close()
        result['final_black'] = inspect(args.session, parked=True)
        guard.close()
        result['USB_READ_requests_sent'] = sum(r.get('HID_read_request_sent', False) for r in result['transactions'])
        write_new(args.output/'USB_READONLY_RESULT.json', result)
        print(json.dumps({k: result[k] for k in ('completed', 'error', 'USB_READ_requests_sent', 'controller_configuration_writes')}, ensure_ascii=False), flush=True)
    if not result['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
